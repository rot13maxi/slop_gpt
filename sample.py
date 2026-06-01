import argparse
from pathlib import Path

import torch

from model import (
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEVICE,
    SlopGPT,
)
from tokenizer import decode, encode

parser = argparse.ArgumentParser()
parser.add_argument("seed", nargs="?", default="bitcoin is")
parser.add_argument("--weights", default=None, help="Path to weights_best.pt")
parser.add_argument("--slop", default=None, help="Path to pleb.slop")
parser.add_argument("--temperature", "-t", type=float, default=DEFAULT_TEMPERATURE)
parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
parser.add_argument("--max-new", type=int, default=128)
parser.add_argument("--tokens", type=int, dest="max_new", default=argparse.SUPPRESS, help="Alias for --max-new")
parser.add_argument("--resample-unk", dest="mask_unk", action="store_true", default=True, help="Mask UNK from logits before sampling")
parser.add_argument("--allow-unk", dest="mask_unk", action="store_false", help="Allow UNK tokens to be sampled")
parser.add_argument("--freq-penalty", type=float, default=DEFAULT_FREQUENCY_PENALTY)
args = parser.parse_args()

model = SlopGPT().to(DEVICE)

weights_path = Path(args.weights) if args.weights else Path("weights_best.pt")
slop_path = Path(args.slop) if args.slop else Path("pleb.slop")

if args.slop or not weights_path.exists():
    from test_slop import load_slop

    model.load_state_dict(load_slop(str(slop_path)), strict=False)
else:
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))

model.eval()

tokens = torch.tensor([encode(args.seed)], dtype=torch.long).to(DEVICE)
output = model.generate(
    tokens,
    max_new_tokens=args.max_new,
    temperature=args.temperature,
    top_k=args.top_k,
    top_p=args.top_p,
    frequency_penalty=args.freq_penalty,
    mask_unk=args.mask_unk,
)
print(decode(output[0].tolist()))
