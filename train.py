import os
import pickle
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.BPE import BPETokenizer
from tinyTransformer import tinyTransformer

# -----------------------------------------------------------------------------------#
#                                 Hyperparameters                                    #
# -----------------------------------------------------------------------------------#
batch_size = 128  # How many independent sequences will we process in parallel?
block_size = 256  # What is the maximum context length for predictions?
epochs = 30  # For how many epochs will we train the model?
steps_per_epoch = 10000  # How many training steps will we take per Epoch?
eval_interval = 250  # How often will we print results for training?
learning_rate = 1e-4  # What is our learning rate?
eval_iters = 200  # How many samples to use to estimate loss?
n_embed = 64 * 8  # How many dimensions do our embeddings have?
head_size = n_embed  # How many dimensions do Key, Query and Value get?
n_head = 8  # How many self-attention head does a multiheaded self attention block get?
n_blocks = 8  # How many sequential self-attention blocks does our model get?
model_precision = torch.bfloat16  # Do you want to set the model_precision to float16 to be faster and reduce the memory?
compile = True  # Do you want to compile the model in Pytorch 2.0 to be faster?
# -----------------------------------------------------------------------------------#
#                                 Training settings                                  #
# -----------------------------------------------------------------------------------#
device = "cuda" if torch.cuda.is_available() else "cpu"  # Where will we train?
model_precision = (
    nullcontext()
    if device == "cpu"
    else torch.amp.autocast(device_type=device, dtype=model_precision)
)
# Set random seed for
torch.manual_seed(2112)
# Set the processing precision of the model: Either float32 or bfloat16
precision = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=model_precision)
)
# -----------------------------------------------------------------------------------#
# collect all relevant model creation parameters
model_args = dict(
    n_blocks=n_blocks,
    n_head=n_head,
    n_embed=n_embed,
    head_size=head_size,
    block_size=block_size,
)  # start with model_args from command line
# collect all relevant parameters and config settings in this module for saving
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
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
for file in files:
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
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# -----------------------------------------------------------------------------------#
#                                  Load and set model                                #
# -----------------------------------------------------------------------------------#

print("Training Model...")
model = tinyTransformer()
m = model.to(device)

# compile the model
if compile:
    print("Compiling the model for faster training and inference")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0!

# Create an Adam Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# -----------------------------------------------------------------------------------#
#                                   Train model                                      #
# -----------------------------------------------------------------------------------#


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


# Run our training epochs
for epoch in epochs:
    # Run each training step per epoch
    for iter in range(steps_per_epoch):
        # Every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}\n\n"
            )
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            print(f"Generative output at step {iter}:")
            print(
                tokenizer.decode(m.generate(context, max_new_tokens=100)[0].tolist())
                + "\n\n"
            )

        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        with precision:
            logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Save model
        if epoch > 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "config": config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))


# -----------------------------------------------------------------------------------#
#                                   Save the model                                   #
# -----------------------------------------------------------------------------------#

torch.save(m.state_dict(), "./models/tinyTransformer1.0.torch")


# -----------------------------------------------------------------------------------#
#                                   Run the model                                    #
# -----------------------------------------------------------------------------------#

print("Finished training...")
print("Generative output after training:")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tokenizer.decode(m.generate(context, max_new_tokens=100)[0].tolist()))


def prompt(input):
    """
    Given your own prompt, generate a text.
    """
    context = torch.tensor(
        tokenizer.encode(input), dtype=torch.long, device=device
    ).reshape(1, -1)
    print(tokenizer.decode(m.generate(context, max_new_tokens=256)[0].tolist()))
