#!/usr/bin/env python3
"""Tokenizer for SlopGPT - regex-based word extraction with vocab.json."""

import json
import os
import random
import re

# Set USE_PUNCT=1 to include punctuation tokens in the tokenizer
USE_PUNCT = os.environ.get("USE_PUNCT", "0") == "1"

CONTRACTIONS = [
    ("won't", "will not"),
    ("can't", "cannot"),
    ("shan't", "shall not"),
    ("ain't", "is not"),
    ("n't", " not"),
    ("'re", " are"),
    ("'ve", " have"),
    ("'ll", " will"),
    ("'d", " would"),
    ("'s", " is"),
    ("'m", " am"),
]

# Load vocabulary from vocab.json
with open("vocab.json", "r") as f:
    vocab_data = json.load(f)
    VOCAB = vocab_data["vocab"]

# Build mappings
stoi = {w: i for i, w in enumerate(VOCAB)}
itos = {i: w for i, w in enumerate(VOCAB)}
UNK_ID = stoi.get("<UNK>", 0)


def normalize_contraction(text):
    text = text.lower()
    text = text.replace("\u2019", "'")  # curly -> straight
    for contr, exp in CONTRACTIONS:
        text = text.replace(contr, exp)
    return text


PUNCT_RE = re.compile(
    r"\b\w+\b|[.,;:!?'\"\(\)\[\]{}\-—…]"
)

def encode(text):
    """Encode text to token IDs using regex word + optional punctuation extraction."""
    text = normalize_contraction(text)
    if USE_PUNCT:
        tokens = PUNCT_RE.findall(text)
    else:
        tokens = re.findall(r"\b\w+\b", text)
    return [stoi.get(w, UNK_ID) for w in tokens]


# Domain-specific UNK words for better readability when UNKs appear
BITCOIN_UNKS = [
    "cyberhornets",
    "hyperbitcoinization",
    "thermodynamic",
    "uncorruptible",
    "uninflatable",
    "cantillon",
    "hodlers",
    "plebs",
    "nocoiners",
]


CONTRACTION_FRAGMENT_MAP = {
    "s": "'s",
    "t": "'t",
    "m": "'m",
    "ve": "'ve",
    "ll": "'ll",
    "re": "'re",
    "d": "'d",
}


PUNCTUATION_CHARS = set(".,;:!?'\"()[]{}—…-")

def is_punct(w):
    return len(w) == 1 and w in PUNCTUATION_CHARS

def decode(ids):
    """Decode token IDs to text."""
    import random

    words = []
    for i in ids:
        w = itos.get(i, "<UNK>")
        if w == "<UNK>":
            words.append("[?]")
        elif is_punct(w):
            if w in ("'",) and words and not words[-1].endswith("'"):
                words[-1] = words[-1] + w
            elif w in (".", ",", ":", ";", "!", "?", ")", "]", "}", "'") and words:
                words[-1] = words[-1] + w
            elif w in ("(", "[", "{", '"'):
                words.append(w)
            else:
                words.append(w)
        else:
            if w in CONTRACTION_FRAGMENT_MAP and words:
                words[-1] = words[-1] + CONTRACTION_FRAGMENT_MAP[w]
            else:
                words.append(w)
    return " ".join(words)


def check_vocab():
    """Verify vocabulary integrity."""
    reserved = {
        "apex",
        "predator",
        "scarcity",
        "sovereign",
        "bitcoin",
        "satoshi",
        "nakamoto",
        "treasury",
        "collateral",
        "fiat",
        "laser",
        "maxi",
        "btc",
    }
    missing = [w for w in reserved if w not in stoi]
    if missing:
        print(f"Missing reserved: {missing}")
    else:
        print(f"✓ All {len(reserved)} reserved words present")
    print(f"Vocab size: {len(VOCAB)}")
    print(f"UNK_ID: {UNK_ID}")


def save_vocab(output_path="vocab.json"):
    """Save vocabulary to JSON file."""
    json.dump({"vocab": VOCAB}, open(output_path, "w"), indent=2)
    print(f"Vocab saved to {output_path}")
    print(f"Size: {len(VOCAB)} tokens")


if __name__ == "__main__":
    check_vocab()
