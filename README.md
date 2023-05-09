# Tiny Transformer
This repository contains an implementation of a simple GPT-like transformer model that can be trained on a single consumer-grade GPU. The code mostly follows along with the videos [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4764s&ab_channel=AndrejKarpathy) and [Building Makemore](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&ab_channel=AndrejKarpathy) by Andrej Karpathy. Further additions are logging, checkpointing, tensorboard monitoring and a custom implementation of BytePairEncoding (BPE). The script includes functions for evaluating the model's performance during training, saving model checkpoints, and generating text using the trained model.

## Requirements
- Python 3.9 
- PyTorch

## Model Implementation
The code is organized in the following modules:

1. Head: A single head of self-attention.
2. MultiHead: Multiple heads of self-attention.
3. FeedForward: A straightforward fully connected layer.
4. MultiHeadAttentionBlock: A block of attention followed by computation.
5. tinyTransformer: The main model class implementing the transformer architecture.

## Custom BytePairEncoding (BPE) Tokenizer
The tokenizer used in this project is a custom implementation of the BytePairEncoding (BPE) algorithm. You can find the code for the tokenizer in the bpe_tokenizer.py file.

Key features of the BPE tokenizer include:

- A custom vocabulary size, defined by the **LENGTH_VOCAB** hyperparameter.
- Special tokens and patterns for pre-tokenization.
- Functions to get pair frequencies, merge word frequencies, and apply BPE rules.

## Usage
You can import the tinyTransformer class and create an instance with a configuration object and a tokenizer. The configuration object can be controlled via de CLI and contains the following parameters:

- **`--batch_size`**: How many independent sequences will we process in parallel? (default: 128)
- **`--block_size`**: What is the maximum context length for predictions? (default: 256)
- **`--n_head`**: How many self-attention heads does a multiheaded self-attention block get? (default: 6)
- **`--n_embed`**: How many dimensions do our embeddings have? (default: None, type: int)
- **`--n_blocks`**: How many sequential self-attention blocks does our model get? (default: 3)
- **`--epochs`**: For how many epochs will we train the model? (default: 30)
- **`--steps_per_epoch`**: How many training steps will we take per epoch? (default: 1000)
- **`--eval_interval`**: How often will we print results for training? (default: 100)
- **`--learning_rate`**: What is our learning rate? (default: 1e-4)
- **`--eval_iters`**: How many batches to use to estimate loss? (default: 10)
- **`--model_precision`**: Do you want to run mixed_precision? (default: "bfloat16")
- **`--compile/--no-compile`**: Do you want to compile the model in Pytorch 2.0 to be faster? (default: True)
- **`--testrun/--no-testrun`**: Do you want to do a quick testrun instead of a full run? (default: False)
- **`--out_dir`**: Output directory for model checkpoints and logs. (default: "./models/")
- **`--device`**: Where will we train? (default: "cuda" if torch.cuda.is_available() else "cpu")
- **`--dropout_percentage`**: Dropout percentage for regularization. (default: 0.3)
- **`--data_dir`**: Directory where the dataset is located. (default: "./data/corpus/")
Here's an example of how to use the tinyTransformer class:

```python
from tiny_transformer import tinyTransformer
from bpe_tokenizer import BPETokenizer

# Create instances of tinyTransformer and BPETokenizer with the appropriate configurations
config = ...
tokenizer = ....

# Create a tinyTransformer instance
model = tinyTransformer(config, tokenizer)

# Train the model with your data
...

# Generate text using the trained model
input_prompt = "Once upon a time, "
model.prompt(input_prompt)
```
## License
This project is open source and available under the MIT License.