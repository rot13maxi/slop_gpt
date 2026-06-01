#!/usr/bin/env python3
"""Inbox-to-training-corpus pipeline for Bitcoin pleb slop.

Inputs:
  data_gen/inbox/urls.txt       one URL per line
  data_gen/inbox/text/*.txt     pasted articles, transcripts, notes

Outputs:
  data_gen/raw/*.txt
  data_gen/llm_refined/chunk_*.txt
  corpus.txt
  vocab.json
  vocab.txt
  data_gen/corpus_report.json
"""

import argparse
import hashlib
import os
import html.parser
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INBOX_DIR = ROOT / "data_gen" / "inbox"
INBOX_TEXT_DIR = INBOX_DIR / "text"
RAW_DIR = ROOT / "data_gen" / "raw"
CHUNK_DIR = ROOT / "data_gen" / "llm_refined"
FILTERED_CHUNK_DIR = ROOT / "data_gen" / "filtered_chunks"
CLAIM_CHUNK_DIR = ROOT / "data_gen" / "claim_chunks"
REPORT_PATH = ROOT / "data_gen" / "corpus_report.json"
FILTER_MANIFEST_PATH = ROOT / "data_gen" / "filter_manifest.json"
CORPUS_PATH = ROOT / "corpus.txt"
VOCAB_JSON_PATH = ROOT / "vocab.json"
VOCAB_TXT_PATH = ROOT / "vocab.txt"
WEB_UI_PATH = ROOT / "web-ui" / "inscription.html"
FILTER_CONFIG_PATH = ROOT / "data_gen" / "corpus_filters.json"

VOCAB_SIZE = 2000
SPECIAL_TOKENS = ["<UNK>"]
RESERVED_WORDS = {
    "bitcoin",
    "btc",
    "satoshi",
    "nakamoto",
    "fiat",
    "debasement",
    "cantillon",
    "scarcity",
    "scarce",
    "sovereign",
    "sovereignty",
    "self",
    "custody",
    "custodian",
    "custodial",
    "hash",
    "proof",
    "work",
    "node",
    "nodes",
    "miner",
    "miners",
    "mining",
    "block",
    "blocks",
    "lightning",
    "hodl",
    "hodler",
    "hodlers",
    "pleb",
    "plebs",
    "maxi",
    "maxis",
    "maximalist",
    "hyperbitcoinization",
    "thermodynamic",
    "uncorruptible",
    "uninflatable",
    "indestructible",
    "energy",
    "monetary",
    "collateral",
    "treasury",
    "orange",
    "laser",
}


class TextExtractor(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip_depth = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1
        if tag in {"p", "br", "div", "h1", "h2", "h3", "li"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self.skip_depth:
            self.skip_depth -= 1
        if tag in {"p", "div", "li"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)

    def text(self):
        return " ".join("".join(self.parts).split())


def slug_for(value):
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    words = re.findall(r"[a-zA-Z0-9]+", value.lower())
    slug = "-".join(words[-6:])[:70].strip("-") or "source"
    return f"{slug}-{digest}"


def is_youtube_url(url):
    return "youtube.com" in url or "youtu.be" in url


def fetch_url(url):
    if is_youtube_url(url):
        try:
            text = fetch_youtube_transcript(url)
        except RuntimeError as e:
            print(f"WARN: skipping — {e}", file=sys.stderr)
            return None
        if text:
            return text

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 corpus-prep/1.0",
            "Accept": "text/html,text/plain;q=0.9,*/*;q=0.5",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        body = response.read()
        content_type = response.headers.get("content-type", "")

    if "text/plain" in content_type:
        return body.decode("utf-8", errors="replace")

    parser = TextExtractor()
    parser.feed(body.decode("utf-8", errors="replace"))
    return parser.text()


def fetch_youtube_transcript(url):
    yt_dlp = shutil.which("yt-dlp")
    if not yt_dlp:
        print("WARN: yt-dlp not found; falling back to generic URL fetch", file=sys.stderr)
        return None

    with tempfile.TemporaryDirectory() as tmp:
        outtmpl = str(Path(tmp) / "%(id)s.%(ext)s")
        cmd = [
            yt_dlp,
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs",
            "en.*",
            "--sub-format",
            "vtt",
            "-o",
            outtmpl,
            url,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        vtt_files = sorted(Path(tmp).glob("*.vtt"))
        if not vtt_files:
            raise RuntimeError(f"yt-dlp did not produce subtitles for {url}")
        return parse_vtt(vtt_files[0].read_text(encoding="utf-8", errors="replace"))


def parse_vtt(text):
    lines = []
    seen = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line == "WEBVTT" or "-->" in line:
            continue
        if re.fullmatch(r"\d+", line):
            continue
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"&[a-zA-Z]+;", " ", line)
        line = " ".join(line.split())
        if line and line not in seen:
            seen.add(line)
            lines.append(line)
    return "\n".join(lines)


def fetch_sources():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    urls_path = INBOX_DIR / "urls.txt"
    fetched = 0

    if urls_path.exists():
        for raw_url in urls_path.read_text(encoding="utf-8").splitlines():
            url = raw_url.strip()
            if not url or url.startswith("#"):
                continue
            output = RAW_DIR / f"{slug_for(url)}.txt"
            if output.exists():
                continue
            print(f"fetch {url}")
            text = fetch_url(url)
            if text is None:
                print(f"SKIP: no content from {url}", file=sys.stderr)
                continue
            output.write_text(text, encoding="utf-8")
            fetched += 1

    INBOX_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    for source in sorted(INBOX_TEXT_DIR.glob("*.txt")):
        output = RAW_DIR / f"{slug_for(source.stem)}.txt"
        if not output.exists():
            output.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            fetched += 1

    return fetched


def clean_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", text)
    text = re.sub(r"\[[^\]]{0,80}\]", " ", text)
    text = re.sub(r"\([^)]{0,40}(music|applause|laughter|sponsor|ad)[^)]{0,40}\)", " ", text, flags=re.I)
    text = re.sub(r"(?m)^\s*[A-Z][A-Za-z ._-]{0,40}:\s+", "", text)
    text = re.sub(r"(?m)^\s*(host|interviewer|speaker|question|q):\s+", "", text, flags=re.I)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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


def normalize_contraction(text):
    text = text.lower()
    text = text.replace("\u2019", "'")  # curly -> straight
    for contr, exp in CONTRACTIONS:
        text = text.replace(contr, exp)
    return text


PUNCT_RE = re.compile(
    r"\b\w+\b|[.,;:!?'\"\(\)\[\]{}\-—…]"
)

PUNCTUATION_TOKENS = [".", ",", ";", ":", "!", "?", "'", '"', "(", ")", "[", "]", "-", "—", "…"]

def iter_words(text):
    text = normalize_contraction(text)
    if os.environ.get("USE_PUNCT", "0") == "1":
        return PUNCT_RE.findall(text)
    return re.findall(r"\b\w+\b", text)


def load_filter_config():
    if FILTER_CONFIG_PATH.exists():
        return json.loads(FILTER_CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        "banlist": [],
        "bitcoin_signal_words": ["bitcoin", "btc", "satoshi", "fiat", "money"],
        "artifact_words": [],
        "filler_words": ["uh", "um"],
        "interviewer_phrases": [],
        "year_pattern": r"\b(?:19|20)\d{2}\b",
        "url_pattern": r"https?://\S+",
        "handle_pattern": r"@\w+",
        "single_letters": True,
        "scoring": {
            "min_bitcoin_density": 0.02,
            "max_artifact_density": 0.03,
            "max_question_density": 0.10,
            "max_filler_density": 0.08,
            "max_repetition_score": 0.15,
            "min_words": 30
        }
    }


def count_bigrams(words):
    bigram_counts = Counter()
    for i in range(len(words) - 1):
        bigram_counts[(words[i], words[i + 1])] += 1
    return bigram_counts


def count_trigrams(words):
    tri_counts = Counter()
    for i in range(len(words) - 2):
        tri_counts[(words[i], words[i + 1], words[i + 2])] += 1
    return tri_counts


def repetition_score(words):
    if len(words) < 3:
        return 0.0
    bigrams = count_bigrams(words)
    trigrams = count_trigrams(words)
    bigram_total = len(words) - 1 or 1
    trigram_total = len(words) - 2 or 1
    repeated_bigrams = sum(c - 1 for c in bigrams.values() if c > 1)
    repeated_trigrams = sum(c - 1 for c in trigrams.values() if c > 1)
    return (repeated_bigrams / bigram_total + repeated_trigrams / trigram_total) / 2


def score_chunk(text, config):
    words = iter_words(text)
    total = len(words) or 1
    word_set = set(words)
    lower_text = text.lower()

    scoring = config.get("scoring", {})
    signal_words = set(config.get("bitcoin_signal_words", []))
    artifact_words = set(config.get("artifact_words", []))
    banlist = set(config.get("banlist", []))
    filler_words = set(config.get("filler_words", []))
    interviewer_phrases = config.get("interviewer_phrases", [])

    signal_count = sum(1 for w in words if w in signal_words)
    bitcoin_density = signal_count / total

    artifact_count = sum(1 for w in words if w in artifact_words)
    banlist_count = sum(1 for w in words if w in banlist)
    artifact_density = (artifact_count + banlist_count) / total

    question_marks = lower_text.count("?")
    phrase_hits = sum(1 for phrase in interviewer_phrases if phrase in lower_text)
    question_density = (question_marks + phrase_hits) / total

    filler_count = sum(1 for w in words if w in filler_words)
    filler_density = filler_count / total

    rep = repetition_score(words)

    reasons = []
    if bitcoin_density < scoring.get("min_bitcoin_density", 0.02):
        reasons.append("low_bitcoin_density")
    if artifact_density > scoring.get("max_artifact_density", 0.03):
        reasons.append("artifact_density")
    if question_density > scoring.get("max_question_density", 0.10):
        reasons.append("too_many_questions")
    if filler_density > scoring.get("max_filler_density", 0.08):
        reasons.append("too_much_filler")
    if rep > scoring.get("max_repetition_score", 0.15):
        reasons.append("high_repetition")
    if total < scoring.get("min_words", 30):
        reasons.append("too_short")

    return {
        "bitcoin_density": round(bitcoin_density, 4),
        "artifact_density": round(artifact_density, 4),
        "question_density": round(question_density, 4),
        "filler_density": round(filler_density, 4),
        "repetition_score": round(rep, 4),
        "word_count": total,
        "banlist_hits": banlist_count,
        "reasons": reasons,
        "keep": len(reasons) == 0,
    }


def filter_chunks(config=None):
    if config is None:
        config = load_filter_config()

    chunk_files = sorted(CHUNK_DIR.glob("chunk_*.txt"))
    if not chunk_files:
        print("No chunks to filter.", file=sys.stderr)
        return []

    FILTERED_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    for old in FILTERED_CHUNK_DIR.glob("chunk_*.txt"):
        old.unlink()

    kept = 0
    dropped = 0
    reason_counts = Counter()
    dropped_details = []

    for idx, path in enumerate(chunk_files, start=1):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue

        result = score_chunk(text, config)

        if result["keep"]:
            (FILTERED_CHUNK_DIR / path.name).write_text(text + "\n", encoding="utf-8")
            kept += 1
        else:
            dropped += 1
            for reason in result["reasons"]:
                reason_counts[reason] += 1
            dropped_details.append({
                "chunk": path.name,
                "reasons": result["reasons"],
                "scores": {
                    "bitcoin_density": result["bitcoin_density"],
                    "artifact_density": result["artifact_density"],
                    "question_density": result["question_density"],
                    "filler_density": result["filler_density"],
                    "repetition_score": result["repetition_score"],
                }
            })

    banlist = set(config.get("banlist", []))
    top_removed = find_top_removed_tokens(chunk_files, banlist, config)

    manifest = {
        "kept_chunks": kept,
        "dropped_chunks": dropped,
        "total_chunks": len(chunk_files),
        "top_dropped_reasons": dict(reason_counts.most_common(10)),
        "top_removed_tokens": top_removed,
        "dropped_sample": dropped_details[:20],
    }

    FILTER_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Filter complete: kept {kept}, dropped {dropped} of {len(chunk_files)} chunks")
    if reason_counts:
        print(f"Top drop reasons: {reason_counts.most_common(5)}")
    print(f"Wrote {FILTER_MANIFEST_PATH.relative_to(ROOT)}")

    return sorted(FILTERED_CHUNK_DIR.glob("chunk_*.txt"))


def find_top_removed_tokens(chunk_files, banlist, config):
    year_re = re.compile(config.get("year_pattern", r"\b(19|20)\d{2}\b"))
    removed_counts = Counter()

    for path in chunk_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        result = score_chunk(text, config)
        if not result["keep"]:
            words = iter_words(text)
            for w in words:
                if w in banlist:
                    removed_counts[w] += 1
            for m in year_re.findall(text.lower()):
                removed_counts[m] += 1

    return [word for word, _ in removed_counts.most_common(20)]


def should_exclude_from_vocab(word, counts, config):
    banlist = set(config.get("banlist", []))
    if word in banlist:
        return True
    year_re = re.compile(r"^\d{4}$")
    if year_re.match(word):
        return True
    if config.get("single_letters", True) and len(word) == 1 and word not in set(PUNCTUATION_TOKENS):
        return True
    if word.startswith("@"):
        return True
    url_re = re.compile(r"^https?://")
    if url_re.match(word):
        return True
    if word.isupper() and len(word) <= 3:
        return True
    return False


def build_chunks(chunk_words):
    raw_files = sorted(RAW_DIR.glob("*.txt"))
    if not raw_files:
        existing_chunks = sorted(CHUNK_DIR.glob("chunk_*.txt"))
        if existing_chunks:
            chunks = [
                path.read_text(encoding="utf-8", errors="replace").strip()
                for path in existing_chunks
            ]
            CORPUS_PATH.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
            return [chunk for chunk in chunks if chunk]

    chunks = []
    combined_parts = []

    for path in raw_files:
        text = clean_text(path.read_text(encoding="utf-8", errors="replace"))
        if not text:
            continue
        combined_parts.append(text)
        words = text.split()
        for i in range(0, len(words), chunk_words):
            chunk = " ".join(words[i : i + chunk_words]).strip()
            if len(iter_words(chunk)) >= 30:
                chunks.append(chunk)

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    for old in CHUNK_DIR.glob("chunk_*.txt"):
        old.unlink()
    for idx, chunk in enumerate(chunks, start=1):
        (CHUNK_DIR / f"chunk_{idx:04d}.txt").write_text(chunk + "\n", encoding="utf-8")

    CORPUS_PATH.write_text("\n\n".join(combined_parts) + "\n", encoding="utf-8")
    return chunks


def build_vocab(chunks, filter_config=None):
    if filter_config is None:
        filter_config = load_filter_config()

    counts = Counter()
    total_words = 0
    for chunk in chunks:
        words = iter_words(chunk)
        counts.update(words)
        total_words += len(words)

    excluded_words = set()
    for word in list(counts.keys()):
        if should_exclude_from_vocab(word, counts, filter_config):
            excluded_words.add(word)
            del counts[word]

    vocab = list(SPECIAL_TOKENS)
    for pt in PUNCTUATION_TOKENS:
        if pt in counts and pt not in vocab:
            vocab.append(pt)
    reserved = sorted(w for w in RESERVED_WORDS if w in counts and w not in vocab)
    vocab.extend(reserved)
    for word, _ in counts.most_common():
        if len(vocab) >= VOCAB_SIZE:
            break
        if word not in vocab:
            vocab.append(word)

    vocab_set = set(vocab)
    unk_tokens = sum(count for word, count in counts.items() if word not in vocab_set)
    unk_rate = (unk_tokens / total_words * 100) if total_words else 0.0

    VOCAB_JSON_PATH.write_text(json.dumps({"vocab": vocab}, indent=2) + "\n", encoding="utf-8")
    VOCAB_TXT_PATH.write_text("\n".join(vocab) + "\n", encoding="utf-8")

    return counts, vocab, {
        "total_tokens": total_words,
        "unique_words": len(counts),
        "vocab_size": len(vocab),
        "reserved_words": reserved,
        "excluded_vocab_words": sorted(excluded_words),
        "unk_tokens": unk_tokens,
        "unk_rate": unk_rate,
        "top_unknown": [
            {"word": word, "count": count}
            for word, count in counts.most_common()
            if word not in vocab_set
        ][:50],
        "top_words": [{"word": word, "count": count} for word, count in counts.most_common(50)],
    }


def sync_web_vocab(vocab):
    html = WEB_UI_PATH.read_text(encoding="utf-8")
    vocab_json = json.dumps(vocab, indent=4)

    # Replace VOCAB array — use string find/replace to avoid re.escape issues
    start = html.find("const VOCAB = [\n")
    if start != -1:
        end = html.find("];", start + len("const VOCAB = [\n"))
        if end != -1:
            html = html[:start] + "const VOCAB = " + vocab_json + ";" + html[end + 2:]

    # Replace vocab size constants. VOCAB_SIZE keeps the legacy extra checkpoint row.
    match = re.search(r"const GENERATION_VOCAB_SIZE = \d+;", html)
    if match:
        html = html[:match.start()] + f"const GENERATION_VOCAB_SIZE = {len(vocab)};" + html[match.end():]

    match = re.search(r"const VOCAB_SIZE = \d+;", html)
    if match:
        html = html[:match.start()] + f"const VOCAB_SIZE = {len(vocab) + 1};" + html[match.end():]

    WEB_UI_PATH.write_text(html, encoding="utf-8")


def ensure_inbox():
    INBOX_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    urls = INBOX_DIR / "urls.txt"
    if not urls.exists():
        urls.write_text(
            "# Drop one YouTube or article URL per line.\n"
            "# Plain text files go in data_gen/inbox/text/.\n",
            encoding="utf-8",
        )


def main():
    parser = argparse.ArgumentParser(description="Prepare corpus, vocab, and web UI vocabulary.")
    parser.add_argument("command", choices=["init", "fetch", "filter", "rewrite", "build", "all"], nargs="?", default="all")
    parser.add_argument("--chunk-words", type=int, default=220)
    parser.add_argument("--no-web-sync", action="store_true")
    parser.add_argument("--use-filtered", action="store_true", help="Build vocab from filtered chunks instead of raw")
    parser.add_argument("--use-claims", action="store_true", help="Build vocab from rewritten claim chunks")
    parser.add_argument("--skip-filter", action="store_true", help="Skip filtering in 'all' command")
    parser.add_argument("--rewrite-cmd", help="Command for claim rewriting; defaults to CLAIM_REWRITE_CMD")
    parser.add_argument("--rewrite-limit", type=int, help="Only rewrite the first N chunks")
    parser.add_argument("--rewrite-force", action="store_true", help="Rewrite existing claim chunks")
    args = parser.parse_args()

    filter_config = load_filter_config()

    ensure_inbox()
    if args.command == "init":
        print(f"Created inbox at {INBOX_DIR}")
        return

    if args.command in {"fetch", "all"}:
        fetched = fetch_sources()
        print(f"Fetched/imported {fetched} source(s) into {RAW_DIR}")
        if args.command == "fetch":
            return

    chunks = build_chunks(args.chunk_words)

    if args.command == "filter":
        filter_chunks(filter_config)
        return

    if args.command == "rewrite":
        cmd = [sys.executable, str(ROOT / "data_gen" / "rewrite_claims.py")]
        if args.rewrite_cmd:
            cmd.extend(["--cmd", args.rewrite_cmd])
        if args.rewrite_limit:
            cmd.extend(["--limit", str(args.rewrite_limit)])
        if args.rewrite_force:
            cmd.append("--force")
        subprocess.run(cmd, check=True, cwd=ROOT)
        return

    if args.command == "all" and not args.skip_filter:
        filter_chunks(filter_config)

    if args.use_claims:
        claim_files = sorted(CLAIM_CHUNK_DIR.glob("chunk_*.txt"))
        if not claim_files:
            raise RuntimeError("No claim chunks found. Run `npm run corpus:rewrite` first.")
        chunks = [
            path.read_text(encoding="utf-8", errors="replace").strip()
            for path in claim_files
        ]
        chunks = [c for c in chunks if c]
        CORPUS_PATH.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
        print(f"Using {len(chunks)} rewritten claim chunks for vocab build")
    elif args.use_filtered:
        filtered_files = sorted(FILTERED_CHUNK_DIR.glob("chunk_*.txt"))
        if filtered_files:
            chunks = [
                path.read_text(encoding="utf-8", errors="replace").strip()
                for path in filtered_files
            ]
            chunks = [c for c in chunks if c]
            print(f"Using {len(chunks)} filtered chunks for vocab build")

    counts, vocab, report = build_vocab(chunks, filter_config)
    report.update(
        {
            "raw_files": len(list(RAW_DIR.glob("*.txt"))),
            "chunks": len(chunks),
            "chunk_words": args.chunk_words,
            "corpus_path": str(CORPUS_PATH.relative_to(ROOT)),
            "vocab_path": str(VOCAB_JSON_PATH.relative_to(ROOT)),
            "chunk_dir": str(CHUNK_DIR.relative_to(ROOT)),
            "used_filtered": args.use_filtered,
            "used_claims": args.use_claims,
        }
    )

    if not args.no_web_sync:
        sync_web_vocab(vocab)
        report["web_vocab_synced"] = True

    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Raw files: {report['raw_files']}")
    print(f"Chunks: {report['chunks']}")
    print(f"Tokens: {report['total_tokens']:,}")
    print(f"Unique words: {report['unique_words']:,}")
    print(f"Vocab size: {report['vocab_size']}")
    print(f"Excluded vocab words: {len(report.get('excluded_vocab_words', []))}")
    print(f"UNK rate: {report['unk_rate']:.2f}%")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
