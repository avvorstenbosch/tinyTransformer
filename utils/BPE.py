import os
import re
from collections import defaultdict
import pickle

# Hyperparameters
LENGTH_VOCAB = 2**10  # How long is the vocab?
SPECTOKENS = ["<|END|>", "<|W|>", "\n"]
# How do we find pre-tokens?
PATTERN = "|".join(re.escape(token) for token in SPECTOKENS) + "|" + r"\b\w+\b|[^\w\s]"
ANTI_PATTERN_WORD = "|".join(re.escape(token) for token in SPECTOKENS) + "|[^\w\s]"
PATTERN_WHITESPACE = r"\s+([^\w\s])"
re_pattern_whitespace = re.compile(PATTERN_WHITESPACE)
# ------------


def get_pair_freq(word_freqs):
    """
    Get frequencies of vocab token bigrams.
    """
    pairs = defaultdict(int)
    for word, freq in word_freqs.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_word_freqs(pair, v_in):
    """
    Insert the new token into the word_freqs
    """
    v_out = {}
    bigram = re.escape(" ".join(pair))
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    for word in v_in:
        w_out = p.sub("".join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


class BPETokenizer:
    def __init__(self, vocab, rules):
        # create a mapping from characters to integers
        self.vocab = vocab
        self.encode_rules = rules
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for i, ch in enumerate(vocab)}

    def merge_tokens(self, text):
        # Apply all the merger rules to the corpus
        text_out = " <|JOIN|> ".join(text)
        for rule in self.encode_rules:
            rep_old = "".join([" ", " ".join(rule), " "])
            rep_new = "".join([" ", "".join(rule), " "])
            text_out = text_out.replace(rep_old, rep_new)
            # Originally we use re.sub, but replace is a lot faster.
            # bigram = re.escape(" ".join(rule))
            # p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
            # text_out = p.sub("".join(rule), " <|JOIN|> ".join(text))
        text_out = text_out.split(" <|JOIN|> ")
        return text_out

    def encode(self, s):
        # Pre-tokenize
        text = re.findall(PATTERN, s)
        text = map(
            lambda word: "".join([" ".join(word), " <|W|>"])
            if not re.search(ANTI_PATTERN_WORD, word)
            else word,
            text,
        )

        # Apply BPE rules:
        text = self.merge_tokens(text)

        text = " ".join(text)  # flatten list
        text = text.split(" ")  # split into tokens

        return list(
            map(self.stoi.get, text)
        )  # encoder: take a string, output a list of integers

    def decode(self, Inputs):
        # decoder: take a list of integers, output a string
        s = "".join([self.itos[i] for i in Inputs]).replace("<|W|>", " ")
        # remove whitespaces before punctuation
        return re_pattern_whitespace.sub(r"\1", s)


if __name__ == "__main__":
    # get dataset
    paths = os.listdir("../data/maarten/")
    files = []
    for path in paths:
        with open("../data/maarten/" + path, "r") as file:
            text = file.read()
        text = text.lower() + " <|END|>"
        files.append(text)

    # here are all the unique characters that occur in this text
    dataset = (" ".join(files)).lower()
    chars = sorted(list(set(dataset)))
    vocab = sorted(
        list(set(SPECTOKENS + chars.copy()))
    )  # Unique token for our embedding

    merge_rules = []

    # get word frequencies
    word_freqs = defaultdict(int)
    for text in files:
        words = re.findall(PATTERN, text)
        for word in words:
            if not re.search(ANTI_PATTERN_WORD, word):
                word = " ".join(word) + " <|W|>"
            word_freqs[word] += 1

    num_mergers = LENGTH_VOCAB - len(vocab)
    for i in range(num_mergers):
        pairs = get_pair_freq(word_freqs)
        best = max(pairs, key=pairs.get)
        word_freqs = merge_word_freqs(best, word_freqs)
        vocab.append(best[0] + best[1])
        merge_rules.append(best)
        if i % 100 == 0:
            print(best)

    tokenizer = BPETokenizer(vocab, merge_rules)
    with open("BPE.pickle", "wb") as f:
        pickle.dump(vocab, f)
        pickle.dump(merge_rules, f)
