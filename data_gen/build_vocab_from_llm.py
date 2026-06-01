#!/usr/bin/env python3
"""
Build new vocabulary from LLM-refined corpus.
1. Run LLM refinement on all chunks
2. Analyze word frequencies in refined text
3. Build new 1750-word vocab from most common words
4. Re-check UNK rate with new vocab
"""

import glob
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Configuration
BASE_URL = "https://forge.hidden-layer.io/v1"
API_KEY = "sk-82101236-d9f1-41aa-a42a-f27d88785d11"
MODEL_NAME = "current"

INPUT_DIR = "rewritten_pipeline"  # Word-mapped chunks
OUTPUT_DIR = "llm_refined"
NEW_VOCAB_FILE = "new_vocab.txt"
NEW_VOCAB_SIZE = 1750

# Special tokens to always include
SPECIAL_TOKENS = ["<UNK>"]

# Reserved domain-critical words (from your original vocab)
RESERVED_WORDS = {
    "apex",
    "predator",
    "scarcity",
    "scarce",
    "sovereign",
    "sovereignty",
    "monetary",
    "energy",
    "property",
    "satoshi",
    "nakamoto",
    "microstrategy",
    "hyperbitcoinization",
    "indestructible",
    "collateralize",
    "collateral",
    "treasury",
    "custodian",
    "custodial",
    "inflationary",
    "fiat",
    "debasement",
    "laser",
    "maxi",
    "maximalist",
    "bitcoin",
    "btc",
    "million",
    "billion",
    "the",
    "and",
    "you",
    "a",
    "to",
    "of",
    "i",
    "is",
    "that",
    "in",
    "it",
    "if",
    "so",
    "or",
    "we",
    "have",
    "for",
    "they",
    "like",
    "on",
    "your",
    "know",
    "what",
    "this",
    "but",
    "be",
    "are",
    "was",
    "were",
    "been",
    "my",
    "me",
    "he",
    "she",
    "him",
    "her",
    "his",
    "their",
    "them",
    "not",
    "no",
    "yes",
    "can",
    "will",
    "would",
    "could",
    "should",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "only",
    "own",
    "same",
    "than",
    "too",
    "very",
}

# LLM refinement settings
REFINE_PROMPT = """Improve the flow and readability of this text. Keep the meaning and conversational style. Fix awkward phrasing. Make it sound natural.

ORIGINAL:
{text}

IMPROVED:
"""

MAX_UNK_RATE = 30.0  # Allow higher UNK for quality
MAX_RETRIES = 2
NUM_WORKERS = 4
DEBUG_MODE = False

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def refine_chunk(chunk_path, chunk_name, worker_id):
    """Refine a chunk with LLM."""
    with open(chunk_path, "r", encoding="utf-8") as f:
        original_text = f.read()

    for retry in range(MAX_RETRIES):
        try:
            prompt = REFINE_PROMPT.format(text=original_text[:2500])

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=8192,
                reasoning_effort="none",
            )

            refined_text = response.choices[0].message.content
            if not refined_text or len(refined_text.strip()) < 50:
                print(
                    f"[Worker {worker_id}] {chunk_name} retry {retry + 1}: Empty response"
                )
                continue

            # Success
            output_path = os.path.join(OUTPUT_DIR, chunk_name)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(refined_text)

            print(f"[Worker {worker_id}] {chunk_name}: OK ({len(refined_text)} chars)")
            return True, chunk_name, len(refined_text)

        except Exception as e:
            print(f"[Worker {worker_id}] {chunk_name} retry {retry + 1}: ERROR - {e}")
            continue

    # Failed - keep original
    output_path = os.path.join(OUTPUT_DIR, chunk_name)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(original_text)

    print(f"[Worker {worker_id}] {chunk_name}: FAILED (kept original)")
    return False, chunk_name, len(original_text)


def analyze_corpus_for_vocab():
    """Analyze refined corpus and build new vocab."""
    print("\n" + "=" * 60)
    print("ANALYZING REFINED CORPUS FOR NEW VOCAB")
    print("=" * 60)

    # Collect all words
    all_words = []
    chunk_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.txt")))

    print(f"Processing {len(chunk_files)} chunks...")
    for chunk_path in chunk_files:
        with open(chunk_path, "r", encoding="utf-8") as f:
            text = f.read()
        # Extract words
        words = re.findall(r"\b\w+\b", text.lower())
        all_words.extend(words)

    print(f"Total words: {len(all_words):,}")

    # Count frequencies
    word_counts = Counter(all_words)

    print(f"\nUnique words: {len(word_counts)}")
    print("\nTop 50 most common words:")
    for word, count in word_counts.most_common(50):
        print(f"  {word}: {count:,}")

    # Build vocab
    # 1. Start with special tokens
    vocab = list(SPECIAL_TOKENS)

    # 2. Add reserved words that exist in corpus
    reserved_in_corpus = [w for w in RESERVED_WORDS if w in word_counts]
    vocab.extend(reserved_in_corpus)
    print(f"\nReserved words kept: {len(reserved_in_corpus)}")

    # 3. Fill rest with most common words, excluding reserved
    remaining_slots = NEW_VOCAB_SIZE - len(vocab)
    top_words = [
        w
        for w, _ in word_counts.most_common()
        if w not in RESERVED_WORDS and w not in vocab
    ][:remaining_slots]
    vocab.extend(top_words)

    print(f"Top words added: {len(top_words)}")
    print(f"\nFinal vocab size: {len(vocab)}")

    # Save vocab
    with open(NEW_VOCAB_FILE, "w", encoding="utf-8") as f:
        for word in vocab:
            f.write(word + "\n")

    print(f"\nVocab saved to {NEW_VOCAB_FILE}")

    # Show vocab stats
    print("\nVocab composition:")
    print(f"  Special tokens: {len(SPECIAL_TOKENS)}")
    print(f"  Reserved words: {len(reserved_in_corpus)}")
    print(f"  Top frequency words: {len(top_words)}")

    # Show some sample words from each category
    print("\nSample reserved words:", reserved_in_corpus[:10])
    print("Sample top words:", top_words[:20])

    return word_counts, vocab


def check_new_unk_rate(word_counts, vocab):
    """Check what the UNK rate would be with new vocab."""
    print("\n" + "=" * 60)
    print("CHECKING NEW UNK RATE")
    print("=" * 60)

    vocab_set = set(vocab)
    total_words = 0
    unk_words = 0

    for word, count in word_counts.items():
        total_words += count
        if word not in vocab_set:
            unk_words += count

    unk_rate = unk_words / total_words * 100

    print(f"Total words: {total_words:,}")
    print(f"Total unique: {len(word_counts)}")
    print(f"Vocab size: {len(vocab)}")
    print(f"UNK words (unique): {len(word_counts) - len(vocab_set)}")
    print(f"UNK tokens: {unk_words:,}")
    print(f"UNK rate: {unk_rate:.2f}%")

    # Show top UNK words
    unk_word_counts = [
        (w, c) for w, c in word_counts.most_common() if w not in vocab_set
    ]
    print(f"\nTop 20 UNK words:")
    for word, count in unk_word_counts[:20]:
        print(f"  {word}: {count:,} ({count / total_words * 100:.3f}%)")

    return unk_rate


def main():
    """Main pipeline."""
    print("=" * 60)
    print("LLM REFINEMENT + NEW VOCAB BUILD")
    print("=" * 60)

    # Step 1: Run LLM refinement
    print("\n" + "=" * 60)
    print("STEP 1: LLM REFINEMENT")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chunk_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))
    print(f"Found {len(chunk_files)} chunks to refine")

    if DEBUG_MODE:
        # Process first 3 chunks
        for i, chunk_path in enumerate(chunk_files[:3]):
            chunk_name = os.path.basename(chunk_path)
            success, name, length = refine_chunk(chunk_path, chunk_name, 0)
            print(f"Result: {'Success' if success else 'Failed'} ({length} chars)")
    else:
        # Process all chunks
        successful = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {}
            for i, chunk_path in enumerate(chunk_files):
                chunk_name = os.path.basename(chunk_path)
                future = executor.submit(
                    refine_chunk, chunk_path, chunk_name, i % NUM_WORKERS
                )
                futures[future] = chunk_name

            for future in as_completed(futures):
                success, chunk_name, length = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1

        print(f"\nRefinement complete: {successful} successful, {failed} failed")

    # Step 2: Analyze corpus and build new vocab
    word_counts, vocab = analyze_corpus_for_vocab()

    # Step 3: Check new UNK rate
    unk_rate = check_new_unk_rate(word_counts, vocab)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print("="*60)
    print(f"New vocab: {NEW_VOCAB_FILE}")
    print(f"Refined chunks: {OUTPUT_DIR}/")
    print(f"Expected UNK rate: {unk_rate:.2f}%")

    return unk_rate


if __name__ == "__main__":
    main()
