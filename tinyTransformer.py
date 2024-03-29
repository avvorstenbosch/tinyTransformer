import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """
    A single head of self.attention.
    """

    def __init__(self, config):
        super().__init__()
        self.Key = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.Query = nn.Linear(config.n_embed, config.head_size, bias=False)
        # Please note: The head size for the value layer is allowed to be different in size
        self.Value = nn.Linear(config.n_embed, config.head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_percentage)

        # register_buffer signals to Torch that this is an externally fed object, not to be optimised
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )

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
        wei = self.dropout(wei)

        # Extract value using query result.
        v = self.Value(x)
        out = wei @ v
        return out


class MultiHead(nn.Module):
    """
    Multiple heads of self.attention.
    """

    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList(Head(config) for _ in range(config.n_head))
        self.projection = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout_percentage)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    """
    A straightforward fully connected layer
    """

    def __init__(self, config):
        super().__init__()
        layers = [
            nn.Linear(config.n_embed, config.compute_layer_scaling * config.n_embed),
            nn.ReLU(),
        ]
        for layer in range(config.n_layers - 1):
            layers.extend(
                [
                    nn.Linear(
                        config.compute_layer_scaling * config.n_embed,
                        config.compute_layer_scaling * config.n_embed,
                    ),
                    nn.ReLU(),
                ]
            )
        layers.extend(
            [
                nn.Linear(
                    config.compute_layer_scaling * config.n_embed, config.n_embed
                ),
                nn.Dropout(config.dropout_percentage),
            ]
        )  # Add projection for skip-connection
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MultiHeadAttenionBlock(nn.Module):
    """
    Create bloks attention followed by computation
    """

    def __init__(self, config):
        super().__init__()
        self.mhsa = MultiHead(config)
        self.fforward = FeedForward(config)
        # This layer normalization acts on a per token level (on H)
        self.layernorm1 = nn.LayerNorm(config.n_embed)
        self.layernorm2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        # Our forward pass has 2 skip connections implemented
        x = x + self.mhsa(self.layernorm1(x))  # (B, T, H)
        x = x + self.fforward(self.layernorm2(x))
        return x


# simple GPT-like model, except it's trainable on a single consumer-grade GPU.
class tinyTransformer(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        vocab_size = len(tokenizer.vocab)
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embed)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        blocks = [MultiHeadAttenionBlock(config) for _ in range(config.n_blocks)]
        self.mhsab = nn.Sequential(*blocks)
        self.lm_head = nn.Linear(config.n_embed, vocab_size)
        self.layernorm = nn.LayerNorm(config.n_embed)
        self.device = config.device

    def forward(self, idx, targets=None):
        B, T = idx.shape  # (B: Batch_size, T: Block_size)

        # idx and targets are both (B,T) tensor of integers
        # C : Embedding_size
        token_embed = self.token_embedding_table(idx)  # (B,T,C)
        position_embed = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T, C)
        # For every sample from the batch, we broadcast the position embedding
        x = token_embed + position_embed  # (B, T, C)
        x = self.mhsab(x)  # (B, T, C)
        x = self.layernorm(x)
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
            logits, loss = self.forward(idx[:, -self.config.block_size :])
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

    def prompt(self, input):
        """
        Given your own prompt, generate a text.
        """
        context = torch.tensor(
            self.tokenizer.encode(input), dtype=torch.long, device=self.device
        ).reshape(1, -1)
        print(
            self.tokenizer.decode(
                self.generate(context, max_new_tokens=256)[0].tolist()
            )
        )


if __name__ == "__main__":
    pass
