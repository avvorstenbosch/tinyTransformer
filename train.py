import os
import pickle

import torch
import click

from utils.BPE import BPETokenizer
from utils.save import get_savefile_name
from utils.config import Config
from utils.logging import setup_logger
from tinyTransformer import tinyTransformer
from contextlib import nullcontext

logger = setup_logger(__name__)
torch.manual_seed(2112)


def prompt(input):
    """
    Given your own prompt, generate a text.
    """
    context = torch.tensor(
        tokenizer.encode(input), dtype=torch.long, device=config.device
    ).reshape(1, -1)
    logger.info(tokenizer.decode(m.generate(context, max_new_tokens=256)[0].tolist()))


# -----------------------------------------------------------------------------------#
#                                 Hyperparameters                                    #
# -----------------------------------------------------------------------------------#
@click.command()
@click.option(
    "--batch_size",
    default=128,
    help="How many independent sequences will we process in parallel?",
)
@click.option(
    "--block_size",
    default=512,
    help="What is the maximum context length for predictions?",
)
@click.option(
    "--n_head",
    default=4,
    help="How many self-attention head does a multiheaded self attention block get?",
)
@click.option(
    "--n_embed", default=None, help="How many dimensions do our embeddings have?"
)
@click.option(
    "--n_blocks",
    default=6,
    help="How many sequential self-attention blocks does our model get?",
)
@click.option(
    "--epochs", default=30, help="For how many epochs will we train the model?"
)
@click.option(
    "--steps_per_epoch",
    default=10000,
    help="How many training steps will we take per Epoch?",
)
@click.option(
    "--eval_interval",
    default=1000,
    help="How often will we print results for training?",
)
@click.option("--learning_rate", default=1e-3, help="What is our learning rate?")
@click.option(
    "--eval_iters", default=100, help="How many samples to use to estimate loss?"
)
@click.option(
    "--model_precision",
    default=torch.bfloat16,
    help="Do you want to run mixed_precision?",
)
@click.option(
    "--compile/--no-compile",
    default=True,
    help="Do you want to compile the model in Pytorch 2.0 to be faster?",
)
@click.option("--testrun/--no-testrun", default=True)
@click.option(
    "--out_dir",
    default="./models/",
    help="Output directory for model checkpoints and logs.",
)
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Where will we train?",
)
@click.option(
    "--dropout_percentage", default=0.3, help="Dropout percentage for regularization."
)
def main(**kwargs):
    config = Config(**kwargs)
    logger.info(
        "Config settings:\n\t\t\t\t"
        + ",\n\t\t\t\t".join(
            f"{key}={str(value)}" for key, value in config.__dict__.items()
        ),
    )

    # -----------------------------------------------------------------------------------#
    #                                 Load training data                                 #
    # -----------------------------------------------------------------------------------#
    logger.info("Loading the dataset for processing.")
    # get dataset
    paths = os.listdir("./data/maarten/")
    files = []
    for path in paths:
        with open("./data/maarten/" + path, "r") as file:
            files.append(file.read().lower())

    logger.info("Concatenating dataset into a single string for easy handling.")
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
    logger.info("Loading pretrained BPETokenizer.")
    with open("./utils/BPE.pickle", "rb") as f:
        vocab = pickle.load(f)
        rules = pickle.load(f)

    tokenizer = BPETokenizer(vocab, rules)
    vocab_size = len(tokenizer.vocab)

    logger.info("Encoding dataset with BPETokenizer.")
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

    logger.info("Loading tinyTransformer model.")
    model = tinyTransformer(config, tokenizer)
    m = model.to(config.device)

    # compile the model
    if config.compile:
        logger.info("Compiling the model for faster training and inference.")
        model = torch.compile(model)  # requires PyTorch 2.0!
    else:
        logger.debug("Using non-compiled model.")

    # Create an Adam Optimizer
    logger.info("Setting optimiser with exponential learning-rate decay scheduler.")
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

    precision = (
        nullcontext()
        if config.device == "cpu"
        else torch.amp.autocast(device_type=config.device, dtype=config.model_precision)
    )

    # Run our training epochs
    for epoch in range(config.epochs):
        # Run each training step per epoch
        for iter in range(config.steps_per_epoch):
            # Every once in a while evaluate the loss on train and val sets
            if iter % config.eval_interval == 0:
                losses = estimate_loss()
                logger.info(
                    f"-epoch: {epoch}- step: {iter} | train loss {losses['train']:.4f}, val loss {losses['val']:.4f}\n\n"
                )
                context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
                logger.info(f"Generative output at step {iter}:")
                logger.info(
                    tokenizer.decode(
                        m.generate(context, max_new_tokens=100)[0].tolist()
                    )
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

        # Learning-rate decay
        scheduler.step()

        # Save model
        if epoch > 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
            }
            checkpoint_name = get_savefile_name(epoch)
            logger.info(f"saving checkpoint to {config.out_dir}")
            torch.save(checkpoint, os.path.join(config.out_dir, checkpoint_name))

    # -----------------------------------------------------------------------------------#
    #                                   Save the model                                   #
    # -----------------------------------------------------------------------------------#
    torch.save(m.state_dict(), "./models/tinyTransformer1.0.torch")

    # -----------------------------------------------------------------------------------#
    #                                   Run the model                                    #
    # -----------------------------------------------------------------------------------#
    logger.info("Finished training...")
    logger.info("Generative output after training:")
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    logger.info(tokenizer.decode(m.generate(context, max_new_tokens=100)[0].tolist()))


if __name__ == "__main__":
    main()
