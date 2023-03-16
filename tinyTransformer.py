import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 128  # How many independent sequences will we process in parallel?
block_size = 64  # What is the maximum context length for predictions?
max_iters = 5000  # How many training steps will we take?
eval_interval = 500  # How often will we print results for training?
learning_rate = 1e-3  # What is our learning rate?
device = "cuda" if torch.cuda.is_available() else "cpu"  # Where will we train?
eval_iters = 200  # How many samples to use to estimate loss?
n_embed = 64  # How many dimensions do our embeddings have?
head_size = n_embed  # How many dimensions do Key, Query and Value get?
n_head = 8  # How many self-attention head does a multiheaded self attention block get?
n_blocks = 4  # How many sequential self-attention blocks does our model get?
# ------------

torch.manual_seed(1337)

# get dataset
paths = os.listdir("./data/maarten/")
files = []
for path in paths:
    with open("./data/maarten/" + path, "r") as file:
        files.append(file.read())

dataset = ""
for file in files:
    dataset += file


# here are all the unique characters that occur in this text
chars = sorted(list(set(dataset)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]  # encoder: take a string, output a list of integers


def decode(l):
    return "".join(
        [itos[i] for i in l]
    )  # decoder: take a list of integers, output a string


# Train and test splits
data = torch.tensor(encode(dataset), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()  # Estimating loss is not a procedure we will backprop over
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """
    A single head of self.attention.
    """

    def __init__(self, head_size):
        super().__init__()
        self.Key = nn.Linear(n_embed, head_size, bias=False)
        self.Query = nn.Linear(n_embed, head_size, bias=False)
        # Please note: The head size for the value layer is allowed to be different in size
        self.Value = nn.Linear(n_embed, head_size, bias=False)

        # register_buffer signals to Torch that this is an externally fed object, not to be optimised
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        # Perform initial query
        k = self.Key(x)  # (B, T, H)
        q = self.Query(x)  # (B, T, H)
        wei = (
            q @ k.transpose(-2, -1) * C**-0.5
        )  # (B, T, H) @ (B, H, T) ---> (B, T, T)

        # Normalize results from the previously seen tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        # Extract value using query result.
        v = self.Value(x)
        out = wei @ v
        return out


class MultiHead(nn.Module):
    """
    Multiple heads of self.attention.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


class FeedForward(nn.Module):
    """
    A straightforward fully connected layer
    """

    def __init__(self, n_layers=1):
        super().__init__()
        layers = []
        for layer in range(n_layers):
            layers.extend(
                [
                    nn.Linear(n_embed, n_embed),
                    nn.ReLU(),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MultiHeadAttenionBlock(nn.Module):
    """
    Create bloks attention followed by computation
    """

    def __init__(self):
        super().__init__()
        head_size = n_embed // n_head
        self.mhsa = MultiHead(num_heads=n_head, head_size=head_size)
        self.fforward = FeedForward()

    def forward(self, x):
        x = self.mhsa(x)  # (B, T, H)
        x = self.fforward(x)
        return x


# super simple bigram model
class tinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        blocks = [MultiHeadAttenionBlock() for _ in range(n_blocks)]
        self.mhsab = nn.Sequential(*blocks)
        self.lm_head = nn.Linear(head_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape  # (B: Batch_size, T: Block_size)

        # idx and targets are both (B,T) tensor of integers
        # C : Embedding_size
        token_embed = self.token_embedding_table(idx)  # (B,T,C)
        position_embed = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C)
        # For every sample from the batch, we broadcast the position embedding
        x = token_embed + position_embed  # (B, T, C)
        x = self.mhsab(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self.forward(idx[:, -block_size:])
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = tinyTransformer()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
