# Bitcoin Pleb Slop

A tiny browser-run Transformer that generates AI Bitcoin maximalist slop. The
deployment target is a Bitcoin inscription, so the quantized model artifact must
fit under the standard transaction-size limit.

## Project Layout

```text
slop_gpt/
├── model.py                  # SlopGPT model architecture
├── tokenizer.py              # Regex word tokenizer backed by vocab.json
├── train.py                  # Training script
├── sample.py                 # Local text generation
├── quantize.py               # Variable-bit .slop serializer
├── check.py                  # Environment/CUDA check
├── vocab.json                # 2,000-entry vocabulary
├── weights_best.pt           # Trained checkpoint, ignored by git
├── pleb.slop                 # Quantized model artifact for inscription
├── corpus.txt                # Combined training corpus (you'll generate your own)
├── data_gen/
│   ├── inbox/urls.txt        # URL inbox for corpus sources (you'll add your own)
│   ├── prepare_corpus.py     # Repeatable corpus/vocab pipeline
│   └── llm_refined/          # Training corpus chunks (generated, not committed)
├── docs/
│   └── corpus-strategy.md    # Corpus guidance
├── web-ui/
│   ├── inscription.html      # Single-file browser UI/inference runtime
│   ├── test.spec.js          # Playwright smoke/e2e tests
│   ├── inference.spec.js     # Inference runtime tests
│   ├── inline.spec.js        # Inline webui tests
│   └── ultra.spec.js         # Ultra inline webui tests
└── scripts/                  # Ordinal inscription/regtest helpers
```

## Quick Start

```bash
python check.py
python train.py
python sample.py "bitcoin is" --temperature 0.8
python quantize.py
```

`python quantize.py` writes `pleb.slop` from `weights_best.pt`.

To build a single-file copy of the browser UI with `pleb.slop` embedded:

```bash
npm run build:inline-webui
```

This writes `web-ui/inscription.inline.html`. The current artifact is larger than the 400 KB standard policy target because embedding the compressed model as base64 expands the weights before the page code is counted.

For a hard-cap single artifact, build the stripped ultra version instead:

```bash
npm run build:ultra-inline-webui
```

This writes `web-ui/inscription.ultra.html`, appends raw optimized model bytes and a compressed vocabulary after the HTML, and has the page load that embedded payload from its own response. The current build is about 392 KB. This raw-binary-tail format is useful for local experiments, but it is not the recommended `ordinals.com` deployment path because public HTML serving can treat invalid UTF-8 bytes as text replacement characters.

## Corpus Pipeline

The repeatable data path is:

1. Put one YouTube or article URL per line in `data_gen/inbox/urls.txt`.
2. Put pasted article/transcript text files in `data_gen/inbox/text/`.
3. Run:

```bash
npm run corpus:all
```

Useful commands:

```bash
npm run corpus:init   # create inbox folders/files
npm run corpus:fetch  # fetch URLs and import pasted text into raw/
npm run corpus:filter # score chunks and write filtered_chunks/
npm run corpus:rewrite # rewrite chunks into source-grounded claims
npm run corpus:build        # rebuild chunks, vocab, report, and web vocab
npm run corpus:build:clean  # rebuild corpus/vocab from filtered chunks
npm run corpus:build:claims # build corpus/vocab from rewritten claims
npm run corpus:all          # fetch + build
```

The pipeline writes fetched/plain sources to ignored `data_gen/raw/`, cleaned
training chunks to `data_gen/llm_refined/`, a combined `corpus.txt`, `vocab.json`,
`vocab.txt`, `data_gen/corpus_report.json`, and the embedded vocabulary in
`web-ui/inscription.html`.

YouTube URLs use `yt-dlp` when it is installed. Plain text files work without any
network tooling.

### Claim Rewrite Stage

`data_gen/rewrite_claims.py` converts transcript/article chunks into short,
source-grounded Bitcoin-maxi claims. It does not require a specific local LLM
server. Set `CLAIM_REWRITE_CMD` to any command that reads a prompt on stdin and
writes JSON or newline-delimited claims on stdout.

Example:

```bash
CLAIM_REWRITE_CMD='ollama run llama3.1:8b-instruct-q4_K_M' npm run corpus:rewrite
npm run corpus:build:claims
```

For a smoke test before a full job:

```bash
CLAIM_REWRITE_CMD='ollama run llama3.1:8b-instruct-q4_K_M' \
  python3 data_gen/prepare_corpus.py rewrite --rewrite-limit 25 --rewrite-force
```

To inspect the exact prompt before running the model:

```bash
python3 data_gen/rewrite_claims.py --limit 1 --print-prompt
```

The rewrite step writes ignored outputs to `data_gen/claim_chunks/`,
`data_gen/rewrite_responses/`, and `data_gen/rewrite_manifest.json`. It is
resumable by default; use `--rewrite-force` to regenerate existing chunks.

After rebuilding the corpus or vocabulary, retrain before using `pleb.slop`; a
changed vocabulary changes token IDs and invalidates old weights.

## Model

- **Architecture**: GPT-style Transformer
- **Vocabulary file**: `vocab.json`
- **Vocabulary entries**: 2,000
- **Checkpoint tensor rows**: 2,001, preserving compatibility with the current model artifact
- **Embedding dim**: 112
- **Attention heads**: 4
- **Transformer layers**: 4
- **Context length**: 128 tokens
- **Quantization**: Configurable via `BITS` env var (4-bit default, 2-bit with `BITS=2`, mixed precision with `MIXED=1`)
- **Target artifact size**: under 400 KB

## Training Data

The corpus should target generalized Bitcoin maximalist monologues rather than a
single person's voice. Prefer short, high-signal passages with first-person
conviction, fiat-collapse grievance, sovereignty larping, proof-of-work rhetoric,
and absurd metaphors.

See [docs/corpus-strategy.md](docs/corpus-strategy.md) for source and cleaning
guidance.

### Corpus Design Notes

The model is small enough that architecture tradeoffs are fixed — quality comes
from the data. A few hard-earned lessons from building this corpus:

- **Build vocabulary from actual corpus frequencies**, not a static word list.
  The tokenizer uses regex word extraction (`\b\w+\b`) with lowercase normalization
  and contraction splitting (`"we're"` → `"we"` + `"re"`).
- **Contraction expansion is the single biggest UNK reducer** (~60% of out-of-vocab
  tokens). Expand `"you're"` → `"you are"` before tokenization.
- **LLMs can't be constrained to a small vocabulary via prompting.** Don't rely
  on prompt engineering for vocab compliance — use deterministic methods (word
  mappings, regex rules, filtering).
- **~7-10% UNK rate is acceptable.** Most unknown words are rare enough that the
  model won't learn patterns for them anyway. Context from surrounding tokens
  provides most of the signal.
- **Quality comes from better data, not a larger architecture.** With a fixed
  quantized size budget, invest in clean, voice-consistent text over model scale.

## Inscription Flow

The UI and the weights are both on-chain in recursive Bitcoin inscriptions:

1. Inscribe `web-ui/inscription.html` as the parent inscription.
2. Inscribe a byte-identical `.bin` copy of `pleb.slop` as a child inscription.
3. The browser UI loads the latest child inscription as model weights.

```bash
./scripts/inscribe.sh
```

The script keeps `pleb.slop` as the tracked build artifact, creates a temporary
`.bin` copy for inscription, and lets `ord` classify that child as
`application/octet-stream`. The viewer remains normal `text/html`, while the
weights are served as binary bytes.

For local browser testing, `web-ui/inscription.html` falls back to `web-ui/pleb.slop`
when it is not running under `/content/<inscription_id>`.

## Validation

```bash
python3 -m py_compile data_gen/prepare_corpus.py model.py tokenizer.py train.py sample.py quantize.py test_slop.py build_vocab.py check.py
npm test
```

The Playwright suite includes a local smoke test and regtest inscription checks.
Regtest checks are skipped unless the required environment variables are set.

## Notes

- `weights_best.pt`, `weights.pt`, and local `web-ui/*.slop` files are ignored.
- Root `pleb.slop` is tracked because it is the deployable inscription artifact.
- `build_vocab.py` is now a compatibility wrapper around `data_gen/prepare_corpus.py`.

## License

Internal use only.
