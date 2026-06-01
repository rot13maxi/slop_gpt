#!/usr/bin/env python3
"""Rewrite corpus chunks into short source-grounded Bitcoin claims.

Supports two modes:
  1. --cmd         External command that reads prompt from stdin, writes JSON/text to stdout.
  2. --openai-api  OpenAI-compatible API endpoint. Requires --openai-key and optionally --openai-model.

Examples:
  CLAIM_REWRITE_CMD='ollama run llama3.1:8b-instruct-q4_K_M' python3 data_gen/rewrite_claims.py
  python3 data_gen/rewrite_claims.py --cmd ./local_rewriter.sh --limit 25
  python3 data_gen/rewrite_claims.py --openai-api https://forge.example.com/v1 --openai-key sk-... --openai-model gpt-4o
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_gen"
CHUNK_DIR = DATA_DIR / "llm_refined"
FILTERED_CHUNK_DIR = DATA_DIR / "filtered_chunks"
CLAIM_CHUNK_DIR = DATA_DIR / "claim_chunks"
REWRITE_RESPONSE_DIR = DATA_DIR / "rewrite_responses"
REWRITE_MANIFEST_PATH = DATA_DIR / "rewrite_manifest.json"
FILTER_CONFIG_PATH = DATA_DIR / "corpus_filters.json"

DEFAULT_MIN_WORDS = 4
DEFAULT_MAX_WORDS = 16
DEFAULT_MIN_CLAIMS = 3
DEFAULT_MAX_CLAIMS = 14

CLAIM_TYPES = [
    "definition",
    "cause",
    "contrast",
    "warning",
    "instruction",
    "prediction",
    "analogy",
]


def load_filter_config():
    if FILTER_CONFIG_PATH.exists():
        return json.loads(FILTER_CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        "banlist": [],
        "artifact_words": [],
        "filler_words": ["uh", "um", "like", "you know"],
        "bitcoin_signal_words": ["bitcoin", "btc", "satoshi", "fiat", "money"],
        "year_pattern": r"\b(?:19|20)\d{2}\b",
    }


def word_tokens(text):
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def normalize_claim(text):
    text = text.strip()
    text = re.sub(r"^[\-\*\d\.\)\s]+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n\"'")
    return text


def claim_key(text):
    return " ".join(word_tokens(text))


def choose_source_dir(prefer):
    if prefer:
        path = ROOT / prefer if not Path(prefer).is_absolute() else Path(prefer)
        return path
    if any(FILTERED_CHUNK_DIR.glob("chunk_*.txt")):
        return FILTERED_CHUNK_DIR
    return CHUNK_DIR


def build_prompt(source_text, source_name, min_claims, max_claims, min_words, max_words):
    return f"""Rewrite the source into {min_claims}-{max_claims} short Bitcoin maximalist claims.

Output JSON only: {{"claims": [{{"type": "definition", "claim": "Bitcoin is scarce digital money."}}]}}

Rules:
- Each claim: {min_words}-{max_words} words, one idea, direct statement.
- Convert conversation, questions, hedging into direct statements.
- Simple present tense. No years, no speaker names, no filler.
- Source-grounded only. No generic slogans.
- Types: definition, cause, contrast, warning, instruction, prediction, analogy.

SOURCE:
{source_text.strip()}
"""


def run_rewriter(command, prompt, timeout):
    if not command:
        raise RuntimeError(
            "No rewrite command configured. Pass --cmd or set CLAIM_REWRITE_CMD."
        )
    proc = subprocess.run(
        shlex.split(command),
        input=prompt,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        cwd=ROOT,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"rewrite command failed with exit {proc.returncode}: {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def run_openai_rewriter(api_url, api_key, model, prompt, timeout, max_retries=3, retry_delay=5):
    """Call an OpenAI-compatible /v1/chat/completions endpoint with retries."""
    url = f"{api_url.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that rewrites text into simple claims. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 8192,
        "enable_thinking": False,
    }).encode("utf-8")

    import time

    last_exc = None
    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            print(f"  [retry {attempt}/{max_retries} after {retry_delay}s delay]...")
            time.sleep(retry_delay)

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            last_exc = RuntimeError(f"API returned HTTP {exc.code}: {err_body[:500]}")
            continue
        except urllib.error.URLError as exc:
            last_exc = RuntimeError(f"API request failed: {exc.reason}")
            continue

        if "choices" not in body or not body["choices"]:
            last_exc = RuntimeError(f"Unexpected API response: {json.dumps(body)[:500]}")
            continue

        content = body["choices"][0].get("message", {}).get("content", "") or ""
        content = content.strip()
        if not content:
            last_exc = RuntimeError("API returned empty content")
            continue

        return content

    raise last_exc


def extract_json_object(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def parse_claims(response_text):
    parsed = extract_json_object(response_text)
    if isinstance(parsed, dict) and isinstance(parsed.get("claims"), list):
        claims = []
        for item in parsed["claims"]:
            if isinstance(item, dict):
                claims.append(
                    {
                        "type": str(item.get("type", "statement")).strip().lower(),
                        "claim": normalize_claim(str(item.get("claim", ""))),
                    }
                )
            elif isinstance(item, str):
                claims.append({"type": "statement", "claim": normalize_claim(item)})
        return claims

    claims = []
    for line in response_text.splitlines():
        line = normalize_claim(line)
        if not line or line in {"[", "]", "{", "}"}:
            continue
        if line.lower().startswith(("claims:", "claim:", "output:")):
            line = normalize_claim(line.split(":", 1)[-1])
        if line:
            claims.append({"type": "statement", "claim": line})
    return claims


def source_anchor_words(source_words, domain_words):
    stop = {
        "the", "a", "an", "and", "or", "but", "if", "then", "is", "are", "was",
        "were", "be", "been", "being", "to", "of", "in", "on", "for", "with",
        "as", "by", "from", "that", "this", "it", "its", "they", "them", "you",
        "your", "we", "our", "i", "he", "she", "not", "do", "does", "did",
        "can", "will", "would", "should", "could", "have", "has", "had",
    }
    counts = Counter(w for w in source_words if len(w) > 3 and w not in stop)
    anchors = {w for w, _ in counts.most_common(80)}
    return anchors | set(domain_words)


def validate_claim(item, source_anchors, config, min_words, max_words):
    claim = normalize_claim(item.get("claim", ""))
    lower = claim.lower()
    words = word_tokens(claim)
    reasons = []

    if not claim:
        reasons.append("empty")
    if "?" in claim:
        reasons.append("question")
    if len(words) < min_words:
        reasons.append("too_short")
    if len(words) > max_words:
        reasons.append("too_long")

    year_re = re.compile(config.get("year_pattern", r"\b(?:19|20)\d{2}\b"))
    if year_re.search(lower):
        reasons.append("year")

    banned = set(config.get("banlist", [])) | set(config.get("artifact_words", []))
    if any(b in lower for b in banned):
        reasons.append("banned_token")

    filler = set(config.get("filler_words", [])) | {"uh", "um", "umm", "you know"}
    if any(f" {f} " in f" {lower} " for f in filler):
        reasons.append("filler")

    if len(set(words)) <= max(2, len(words) // 2):
        reasons.append("low_word_diversity")

    if not (set(words) & source_anchors):
        reasons.append("no_source_anchor")

    return {
        "type": item.get("type", "statement"),
        "claim": claim,
        "word_count": len(words),
        "reasons": reasons,
        "keep": len(reasons) == 0,
    }


def _rewrite_single_chunk(
    index, path, args, command, api_url, api_key, api_model, domain_words, config
):
    """Process one chunk: call LLM, validate, return result dict."""
    output_path = CLAIM_CHUNK_DIR / path.name
    response_path = REWRITE_RESPONSE_DIR / path.name

    if output_path.exists() and not args.force:
        claims = [
            normalize_claim(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if normalize_claim(line)
        ]
        return {
            "chunk": path.name,
            "index": index,
            "skipped": True,
            "kept": claims,
            "dropped": [],
            "error": None,
        }

    source_text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not source_text:
        return {
            "chunk": path.name,
            "index": index,
            "skipped": False,
            "kept": [],
            "dropped": [],
            "error": None,
        }

    prompt = build_prompt(
        source_text,
        path.name,
        args.min_claims,
        args.max_claims,
        args.min_words,
        args.max_words,
    )
    if args.print_prompt:
        print(prompt)
        return None

    try:
        if api_url:
            response = run_openai_rewriter(
                api_url, api_key, api_model, prompt, args.timeout,
                max_retries=args.retries, retry_delay=args.retry_delay,
            )
        else:
            response = run_rewriter(command, prompt, args.timeout)
    except Exception as exc:
        return {
            "chunk": path.name,
            "index": index,
            "skipped": False,
            "kept": [],
            "dropped": [],
            "error": str(exc),
        }

    response_path.write_text(response + "\n", encoding="utf-8")
    source_words = word_tokens(source_text)
    anchors = source_anchor_words(source_words, domain_words)

    parsed = parse_claims(response)
    kept = []
    dropped = []
    for item in parsed:
        result = validate_claim(item, anchors, config, args.min_words, args.max_words)
        if result["keep"]:
            kept.append(result)
        else:
            dropped.append(result)

    return {
        "chunk": path.name,
        "index": index,
        "skipped": False,
        "kept": kept,
        "dropped": dropped,
        "input_words": len(source_words),
        "error": None,
    }


def rewrite_chunks(args):
    config = load_filter_config()
    source_dir = choose_source_dir(args.source_dir)
    source_files = sorted(source_dir.glob("chunk_*.txt"))
    if args.limit:
        source_files = source_files[: args.limit]
    if not source_files:
        raise RuntimeError(f"No chunk_*.txt files found in {source_dir}")

    CLAIM_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    REWRITE_RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    if args.force:
        for path in CLAIM_CHUNK_DIR.glob("chunk_*.txt"):
            path.unlink()
        for path in REWRITE_RESPONSE_DIR.glob("chunk_*.txt"):
            path.unlink()

    command = args.cmd or os.environ.get("CLAIM_REWRITE_CMD", "")
    api_url = args.openai_api
    api_key = args.openai_key or os.environ.get("OPENAI_API_KEY", "")
    api_model = args.openai_model or "current"
    workers = args.workers

    if not command and not api_url:
        raise RuntimeError(
            "No rewrite backend configured. Pass --cmd, set CLAIM_REWRITE_CMD, "
            "or pass --openai-api (plus --openai-key)."
        )

    domain_words = set(config.get("bitcoin_signal_words", [])) | {
        "bitcoin",
        "fiat",
        "money",
        "mining",
        "miner",
        "miners",
        "node",
        "nodes",
        "keys",
        "custody",
        "energy",
        "block",
        "blocks",
        "scarcity",
        "satoshi",
        "fees",
        "savings",
        "inflation",
        "debasement",
    }

    indexed_files = list(enumerate(source_files, start=1))

    # Phase 1: rewrite in parallel
    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _rewrite_single_chunk,
                idx,
                path,
                args,
                command,
                api_url,
                api_key,
                api_model,
                domain_words,
                config,
            ): (idx, path)
            for idx, path in indexed_files
        }

        for future in as_completed(futures):
            result = future.result()
            if result is None:
                return
            results.append(result)
            completed += 1
            status = "skipped" if result["skipped"] else (
                f"ERROR {result['chunk']}: {result['error']}"
                if result["error"]
                else f"kept {len(result['kept'])}, dropped {len(result['dropped'])}"
            )
            print(f"[{completed}/{len(indexed_files)}] {status}")

    # Phase 2: sequential dedup and write
    results.sort(key=lambda r: r["index"])
    manifest_rows = []
    reason_counts = Counter()
    type_counts = Counter()
    global_seen = set()
    kept_total = 0
    dropped_total = 0
    skipped_total = 0

    for result in results:
        if result["skipped"]:
            skipped_total += 1
            for claim in result["kept"]:
                global_seen.add(claim_key(claim))
            kept_total += len(result["kept"])
            manifest_rows.append(
                {
                    "chunk": result["chunk"],
                    "input_words": result.get("input_words", 0),
                    "kept": len(result["kept"]),
                    "dropped": 0,
                    "kept_sample": result["kept"][:3],
                }
            )
            continue

        if result["error"]:
            reason_counts["rewrite_error"] += 1
            manifest_rows.append(
                {
                    "chunk": result["chunk"],
                    "input_words": result.get("input_words", 0),
                    "kept": 0,
                    "dropped": 0,
                    "error": result["error"],
                }
            )
            continue

        # Deduplicate within this chunk and against global state
        final_kept = []
        final_dropped = list(result["dropped"])
        for item in result["kept"]:
            key = claim_key(item["claim"])
            if key in global_seen:
                item["reasons"].append("duplicate")
                final_dropped.append(item)
            else:
                final_kept.append(item)
                global_seen.add(key)

        # Write outputs
        output_path = CLAIM_CHUNK_DIR / result["chunk"]
        output_path.write_text(
            "\n".join(item["claim"] for item in final_kept)
            + ("\n" if final_kept else ""),
            encoding="utf-8",
        )

        for item in final_kept:
            type_counts[item["type"]] += 1
        for item in final_dropped:
            for reason in item["reasons"]:
                reason_counts[reason] += 1

        kept_total += len(final_kept)
        dropped_total += len(final_dropped)
        manifest_rows.append(
            {
                "chunk": result["chunk"],
                "input_words": result.get("input_words", 0),
                "kept": len(final_kept),
                "dropped": len(final_dropped),
                "drop_reasons": dict(
                    Counter(r for item in final_dropped for r in item["reasons"])
                ),
                "kept_sample": [item["claim"] for item in final_kept[:3]],
            }
        )

    manifest = {
        "source_dir": str(source_dir.relative_to(ROOT)),
        "output_dir": str(CLAIM_CHUNK_DIR.relative_to(ROOT)),
        "response_dir": str(REWRITE_RESPONSE_DIR.relative_to(ROOT)),
        "chunks_seen": len(source_files),
        "chunks_skipped_existing": skipped_total,
        "claims_kept": kept_total,
        "claims_dropped": dropped_total,
        "top_drop_reasons": dict(reason_counts.most_common(20)),
        "claim_types": dict(type_counts.most_common()),
        "settings": {
            "min_words": args.min_words,
            "max_words": args.max_words,
            "min_claims": args.min_claims,
            "max_claims": args.max_claims,
            "timeout": args.timeout,
            "command": command,
            "openai_api": api_url,
            "openai_model": api_model,
            "workers": workers,
        },
        "chunks": manifest_rows,
    }
    REWRITE_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {REWRITE_MANIFEST_PATH.relative_to(ROOT)}")
    print(
        f"Claims kept: {kept_total}; dropped: {dropped_total}; skipped chunks: {skipped_total}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmd", help="Rewrite command. Defaults to CLAIM_REWRITE_CMD.")
    parser.add_argument("--source-dir", help="Chunk directory. Defaults to filtered_chunks if present, else llm_refined.")
    parser.add_argument("--limit", type=int, help="Process only the first N chunks.")
    parser.add_argument("--force", action="store_true", help="Rewrite existing outputs.")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument("--min-claims", type=int, default=DEFAULT_MIN_CLAIMS)
    parser.add_argument("--max-claims", type=int, default=DEFAULT_MAX_CLAIMS)
    parser.add_argument("--print-prompt", action="store_true", help="Print the first rewrite prompt and exit.")
    parser.add_argument("--openai-api", help="OpenAI-compatible API base URL (e.g. https://forge.example.com/v1).")
    parser.add_argument("--openai-key", help="API key. Defaults to OPENAI_API_KEY env var.")
    parser.add_argument("--openai-model", default="current", help="Model name. Defaults to 'current'.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel API workers. Defaults to 4.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per chunk on empty/timeout. Defaults to 3.")
    parser.add_argument("--retry-delay", type=int, default=5, help="Seconds between retries. Defaults to 5.")
    args = parser.parse_args()
    rewrite_chunks(args)


if __name__ == "__main__":
    main()
