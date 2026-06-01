#!/usr/bin/env python3
"""
Training script for SlopGPT with an LLM-refined corpus.
Uses new tokenizer with regex-based word extraction.
"""

import argparse
import glob
import math
import random
from pathlib import Path

import torch

from model import CONTEXT_LEN, DEVICE, SlopGPT
from tokenizer import UNK_ID, VOCAB, encode

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train SlopGPT")
parser.add_argument(
    "--cpu",
    action="store_true",
    help="Force CPU training (even if CUDA is available)",
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=10000,
    help="Number of optimizer steps to run",
)
parser.add_argument(
    "--eval-every",
    type=int,
    default=200,
    help="Evaluate and checkpoint every N steps",
)
args = parser.parse_args()
if args.max_steps <= 0:
    parser.error("--max-steps must be greater than 0")
if args.eval_every <= 0:
    parser.error("--eval-every must be greater than 0")

# Override DEVICE if --cpu flag is passed
if args.cpu:
    DEVICE = "cpu"
    print(f"Force CPU mode (CUDA unavailable or --cpu flag set)")
else:
    print(f"using {DEVICE}")

# ── data ──────────────────────────────────────────────────────────────────────

# Load chunks — check claim_chunks first, then filtered_chunks, then llm_refined
for candidate in ["data_gen/claim_chunks", "data_gen/filtered_chunks", "data_gen/llm_refined"]:
    if list(Path(candidate).glob("chunk_*.txt")):
        chunk_dir = candidate
        break
else:
    parser.error("No chunk_*.txt files found. Run `python3 data_gen/prepare_corpus.py build` first.")

chunks = []
for f in sorted(glob.glob(f"{chunk_dir}/*.txt")):
    chunks.append(Path(f).read_text(encoding="utf-8"))

random.seed(33)
random.shuffle(chunks)
print(f"Loaded {len(chunks)} chunks from {chunk_dir}")

corpus = "\n".join(chunks)
data = torch.tensor(encode(corpus), dtype=torch.long)
print(f"Corpus: {len(data):,} tokens")

# Split train/val
n = int(0.9 * len(data))
train = data[:n]
val = data[n:]

print(f"Train: {len(train):,} tokens")
print(f"Val:   {len(val):,} tokens")

# ── batching ──────────────────────────────────────────────────────────────────

BATCH_SIZE = 64


def get_batch(split):
    d = train if split == "train" else val
    ix = torch.randint(len(d) - CONTEXT_LEN, (BATCH_SIZE,))
    x = torch.stack([d[i : i + CONTEXT_LEN] for i in ix])
    y = torch.stack([d[i + 1 : i + CONTEXT_LEN + 1] for i in ix])
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    return x, y


# ── training ──────────────────────────────────────────────────────────────────

WARMUP_STEPS = 100
LEARNING_RATE = 5e-4
MAX_STEPS = args.max_steps
EVAL_EVERY = args.eval_every

model = SlopGPT().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.01)

# Cosine LR schedule with warmup: start low, warmup to peak, then cosine decay to near-zero
def get_lr(step):
    if step < WARMUP_STEPS:
        return 1e-6 + (LEARNING_RATE - 1e-6) * (step / WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return 1e-6 + (LEARNING_RATE - 1e-6) * 0.5 * (1 + math.cos(math.pi * progress))

best_val_loss = float("inf")
best_step = 0


@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        batch_losses = []
        for _ in range(20):
            x, y = get_batch(split)
            _, loss = model(x, y)
            batch_losses.append(loss.item())
        losses[split] = sum(batch_losses) / len(batch_losses)
    model.train()
    return losses


def evaluate_and_checkpoint(step):
    global best_step, best_val_loss

    losses = estimate_loss()
    print(
        f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}",
        end="",
    )

    if losses["val"] < best_val_loss:
        best_val_loss = losses["val"]
        best_step = step
        torch.save(model.state_dict(), "weights_best.pt")
        print("  saved", end="")

    print()


print("\n" + "=" * 60)
print("TRAINING STARTED")
print("=" * 60)
print(f"Vocab size: {len(VOCAB)}")
print(f"UNK rate: {data.tolist().count(UNK_ID) / len(data) * 100:.2f}%")
print(f"Context length: {CONTEXT_LEN}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max steps: {MAX_STEPS}")
print("=" * 60 + "\n")

for step in range(MAX_STEPS):
    if step % EVAL_EVERY == 0:
        evaluate_and_checkpoint(step)

    x, y = get_batch("train")
    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Apply learning rate schedule
    for param_group in optimizer.param_groups:
        param_group["lr"] = get_lr(step)

    optimizer.step()

if MAX_STEPS % EVAL_EVERY != 0:
    evaluate_and_checkpoint(MAX_STEPS)

print(f"\n" + "=" * 60)
print(f"TRAINING COMPLETE")
print(f"best val loss {best_val_loss:.4f} at step {best_step}")
print(f"best weights saved to weights_best.pt")
print("=" * 60)
