import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path

# all the knobs for our tiny model
# we'll tune these once the architecture is working

VOCAB_TOKENS = json.loads(Path("vocab.json").read_text())["vocab"]
GENERATION_VOCAB_SIZE = len(VOCAB_TOKENS)
UNK_ID = VOCAB_TOKENS.index("<UNK>") if "<UNK>" in VOCAB_TOKENS else 0
# Existing checkpoints have one extra output row beyond vocab.json. Keep the row
# for checkpoint compatibility, but mask it during generation.
VOCAB_SIZE = GENERATION_VOCAB_SIZE + 1
ALLOW_ARCH_OVERRIDE = os.environ.get("ALLOW_ARCH_OVERRIDE", "0") == "1"
EMBED_DIM = int(os.environ.get("EMBED_DIM", "112")) if ALLOW_ARCH_OVERRIDE else 112
N_HEADS = 4  # number of attention heads
N_LAYERS = int(os.environ.get("N_LAYERS", "4")) if ALLOW_ARCH_OVERRIDE else 4
CONTEXT_LEN = 128  # words not chars — more context for sentence-level coherence
DROPOUT = 0.1  # regularization — randomly zero out 10% of activations
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 100
DEFAULT_TOP_P = 0.9
DEFAULT_FREQUENCY_PENALTY = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Note: DEVICE choice is controlled by train.py --cpu flag


class AttentionHead(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        # the three matrices — each projects EMBED_DIM down to head_dim
        self.W_q = nn.Linear(EMBED_DIM, head_dim, bias=False)
        self.W_k = nn.Linear(EMBED_DIM, head_dim, bias=False)
        self.W_v = nn.Linear(EMBED_DIM, head_dim, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

        # causal mask — upper triangle of ones, registered as a buffer
        # (not a parameter, won't be trained, but moves to GPU with the model)
        self.register_buffer(
            "mask", torch.triu(torch.ones(CONTEXT_LEN, CONTEXT_LEN), diagonal=1).bool()
        )

    def forward(self, x):
        B, T, C = x.shape  # batch size, sequence length, embed dim

        q = self.W_q(x)  # (B, T, head_dim)
        k = self.W_k(x)  # (B, T, head_dim)
        v = self.W_v(x)  # (B, T, head_dim)

        # attention scores — every token's Q against every token's K
        scale = k.shape[-1] ** -0.5
        scores = q @ k.transpose(-2, -1) * scale  # (B, T, T)

        # causal mask — tokens cannot attend to future tokens
        # (we're predicting the next char, so no cheating)
        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))

        # softmax to get weights that sum to 1
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # weighted sum of values
        out = weights @ v  # (B, T, head_dim)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # each head works in a lower-dim space, we split EMBED_DIM evenly
        head_dim = EMBED_DIM // N_HEADS  # 128 // 4 = 32

        self.heads = nn.ModuleList([AttentionHead(head_dim) for _ in range(N_HEADS)])

        # projects concatenated head outputs back to EMBED_DIM
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # run all heads in parallel, concatenate along the last dim
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # (B, T, head_dim * N_HEADS) = (B, T, EMBED_DIM)

        # project back and dropout
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, 4 * EMBED_DIM),  # expand
            nn.GELU(),  # activation
            nn.Linear(4 * EMBED_DIM, EMBED_DIM),  # contract
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = MultiHeadAttention()
        self.feedforward = FeedForward()
        # layernorm stabilizes the residual stream
        # normalizes each token's vector to mean=0, std=1
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.norm2 = nn.LayerNorm(EMBED_DIM)

    def forward(self, x):
        # attention with residual connection
        x = x + self.attention(self.norm1(x))
        # feedforward with residual connection
        x = x + self.feedforward(self.norm2(x))
        return x


class SlopGPT(nn.Module):
    def __init__(self):
        super().__init__()
        # token embedding — maps each char index to a vector
        self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        # position embedding — tells the model where in the sequence each token is
        self.position_embedding = nn.Embedding(CONTEXT_LEN, EMBED_DIM)

        # the stack of transformer blocks
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(N_LAYERS)])

        # final layernorm before output
        self.norm = nn.LayerNorm(EMBED_DIM)

        # project from EMBED_DIM to VOCAB_SIZE — gives a score for each possible next char
        self.output = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, tokens, targets=None):
        B, T = tokens.shape

        # look up token and position embeddings, add them together
        tok_emb = self.token_embedding(tokens)  # (B, T, EMBED_DIM)
        pos_emb = self.position_embedding(
            torch.arange(T, device=tokens.device)
        )  # (T, EMBED_DIM)
        x = tok_emb + pos_emb  # (B, T, EMBED_DIM)

        # through the transformer blocks
        x = self.blocks(x)
        x = self.norm(x)

        # project to vocab size — one score per possible character, per position
        logits = self.output(x)  # (B, T, VOCAB_SIZE)

        # if we have targets, compute loss
        loss = None
        if targets is not None:
            # cross entropy expects (B*T, VOCAB_SIZE) and (B*T,)
            loss = F.cross_entropy(logits.view(B * T, VOCAB_SIZE), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens,
        max_new_tokens,
        temperature=DEFAULT_TEMPERATURE,
        top_k=DEFAULT_TOP_K,
        top_p=DEFAULT_TOP_P,
        frequency_penalty=DEFAULT_FREQUENCY_PENALTY,
        mask_unk=True,
    ):
        temperature = max(float(temperature), 1e-6)
        top_k = None if top_k is None else max(1, min(int(top_k), GENERATION_VOCAB_SIZE))
        top_p = None if top_p is None else min(max(float(top_p), 0.0), 1.0)
        if tokens.shape[1] == 0:
            tokens = torch.zeros((tokens.shape[0], 1), dtype=tokens.dtype, device=tokens.device)

        for _ in range(max_new_tokens):
            ctx = tokens[:, -CONTEXT_LEN:]

            logits, _ = self(ctx)
            logits = logits[:, -1, :]  # (B, VOCAB_SIZE)
            logits[:, GENERATION_VOCAB_SIZE:] = float("-inf")
            if mask_unk:
                logits[:, UNK_ID] = float("-inf")

            if frequency_penalty > 0:
                counts = []
                for row in ctx:
                    valid = row[row < GENERATION_VOCAB_SIZE]
                    counts.append(torch.bincount(valid, minlength=VOCAB_SIZE).float())
                token_counts = torch.stack(counts, dim=0).to(logits.device)
                logits = logits - float(frequency_penalty) * token_counts

            if top_k is not None:
                topk_values, _ = torch.topk(logits, top_k, dim=-1)
                cutoff = topk_values[..., -1:]
                logits = torch.where(logits >= cutoff, logits, float("-inf"))

            if top_p is not None and top_p < 1:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits / temperature, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                logits = torch.where(indices_to_remove, float("-inf"), logits)

            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            probs = torch.clamp(probs, min=0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens


if __name__ == "__main__":
    model = SlopGPT().to(DEVICE)

    # count parameters
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    # fake batch — 4 sequences of 64 random token ids
    x = torch.randint(0, VOCAB_SIZE, (4, 64)).to(DEVICE)
    y = torch.randint(0, VOCAB_SIZE, (4, 64)).to(DEVICE)

    logits, loss = model(x, y)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    # expected loss at init — random model should be close to -log(1/33) = 3.5
    import math

    print(f"Expected random loss: {math.log(VOCAB_SIZE):.4f}")


# Backward-compatible name for old scripts/checkpoints.
SaylorGPT = SlopGPT
