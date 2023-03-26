import os
import pickle

import torch

from utils.BPE import BPETokenizer
from utils.save import get_savefile_name
from utils.config import Config
from tinyTransformer import tinyTransformer

# -----------------------------------------------------------------------------------#
#                                 Hyperparameters                                    #
# -----------------------------------------------------------------------------------#
torch.manual_seed(2112)
config = Config(
    batch_size=256,  # How many independent sequences will we process in parallel?
    block_size=256,  # What is the maximum context length for predictions?
    n_head=6,  # How many self-attention head does a multiheaded self attention block get?
    n_embed=None,  # How many dimensions do our embeddings have?
    n_blocks=4,  # How many sequential self-attention blocks does our model get?
    epochs=30,  # For how many epochs will we train the model?
    steps_per_epoch=10000,  # How many training steps will we take per Epoch?
    eval_interval=1000,  # How often will we print results for training?
    learning_rate=1e-4,  # What is our learning rate?
    eval_iters=200,  # How many samples to use to estimate loss?
    model_precision=(
        torch.bfloat16
    ),  # Do you want to set the model_precision to float16 to be faster and reduce the memory?
    compile=True,  # Do you want to compile the model in Pytorch 2.0 to be faster?
    testrun=False,
    out_dir="./models/",
    device="cuda" if torch.cuda.is_available() else "cpu",  # Where will we train?
    # Set random seed for
    # Set the processing precision of the model: Either float32 or bfloat16
    precision=None,  # defaults to bfloat.16
)

# -----------------------------------------------------------------------------------#
#                                 Training settings                                  #
# -----------------------------------------------------------------------------------#
# -----------------------------------------------------------------------------------#
#                                 Load training data                                 #
# -----------------------------------------------------------------------------------#
print("Fetching data...")
# get dataset
paths = os.listdir("./data/maarten/")
files = []
for path in paths:
    with open("./data/maarten/" + path, "r") as file:
        files.append(file.read().lower())

dataset = ""
length_dataset = len(files) if not config.testrun else 3
for file in files[:length_dataset]:
    dataset += file


# -----------------------------------------------------------------------------------#
#                           Process training data                                    #
# -----------------------------------------------------------------------------------#
# here are all the unique characters that occur in this text
chars = sorted(list(set(dataset)))

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Load pretrained tokenizer
print("Loading BPETokenizer...")
with open("./utils/BPE.pickle", "rb") as f:
    vocab = pickle.load(f)
    rules = pickle.load(f)

tokenizer = BPETokenizer(vocab, rules)
vocab_size = len(tokenizer.vocab)

print("Encode Dataset...")
# Train and test splits
data = torch.tensor(tokenizer.encode(dataset), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i : i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + config.block_size + 1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y


# -----------------------------------------------------------------------------------#
#                                  Load and set model                                #
# -----------------------------------------------------------------------------------#

print("Training Model...")
model = tinyTransformer(config, tokenizer)
m = model.to(config.device)

# compile the model
if config.compile:
    print("Compiling the model for faster training and inference")
    model = torch.compile(model)  # requires PyTorch 2.0!

# Create an Adam Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)


# -----------------------------------------------------------------------------------#
#                                   Train model                                      #
# -----------------------------------------------------------------------------------#
@torch.no_grad()  # Estimating loss is not a procedure we will backprop over
def estimate_loss():
    """Retrieve both a training and a testing batch to estimate the current loss"""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Run our training epochs
for epoch in range(config.epochs):
    # Run each training step per epoch
    for iter in range(config.steps_per_epoch):
        # Every once in a while evaluate the loss on train and val sets
        if iter % config.eval_interval == 0:
            losses = estimate_loss()
            print(
                f"-epoch: {epoch}- step: {iter} | train loss {losses['train']:.4f}, val loss {losses['val']:.4f}\n\n"
            )
            context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
            print(f"Generative output at step {iter}:")
            print(
                tokenizer.decode(m.generate(context, max_new_tokens=100)[0].tolist())
                + "\n\n"
            )

        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        with config.precision:
            logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Learning-rate decay
    scheduler.step()

    # Save model
    if epoch > 0:
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        }
        print(f"saving checkpoint to {config.out_dir}")
        torch.save(checkpoint, os.path.join(config.out_dir, get_savefile_name(epoch)))


# -----------------------------------------------------------------------------------#
#                                   Save the model                                   #
# -----------------------------------------------------------------------------------#
torch.save(m.state_dict(), "./models/tinyTransformer1.0.torch")


# -----------------------------------------------------------------------------------#
#                                   Run the model                                    #
# -----------------------------------------------------------------------------------#
print("Finished training...")
print("Generative output after training:")
context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
print(tokenizer.decode(m.generate(context, max_new_tokens=100)[0].tolist()))


def prompt(input):
    """
    Given your own prompt, generate a text.
    """
    context = torch.tensor(
        tokenizer.encode(input), dtype=torch.long, device=config.device
    ).reshape(1, -1)
    print(tokenizer.decode(m.generate(context, max_new_tokens=256)[0].tolist()))
