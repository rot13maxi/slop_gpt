#!/bin/bash
# inscribe.sh - Automates the two-step Bitcoin inscription process
# 
# This script inscribes the web-ui/inscription.html as a parent,
# then inscribes pleb.slop as an application/octet-stream child on top of it.
#
# Usage: ./inscribe.sh [--fee-rate RATE]
#
# Requirements:
# - ord wallet configured and unlocked
# - bitcoind running (for regtest mining)
# - Both inscription.html and pleb.slop files present

set -e

# Default fee rate (optimized for regtest)
FEE_RATE=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fee-rate)
            FEE_RATE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--fee-rate RATE]"
            exit 1
            ;;
    esac
done

# Configuration - adjust these for your environment
ORD_HOST="${ORD_HOST:-localhost}"
ORD_PORT="${ORD_PORT:-9001}"
BITCOIND_HOST="${BITCOIND_HOST:-localhost}"
BITCOIND_PORT="${BITCOIND_PORT:-9000}"
ORD_DATA_DIR="${ORD_DATA_DIR:-}"

ORD_CMD=(ord)
if [[ -n "$ORD_DATA_DIR" ]]; then
    ORD_CMD+=(--datadir "$ORD_DATA_DIR")
fi

# File paths (relative to project root)
PARENT_FILE="web-ui/inscription.html"
WEIGHTS_FILE="pleb.slop"
CHILD_FILE=""
CHILD_DIR=""

cleanup() {
    if [[ -n "$CHILD_DIR" && -d "$CHILD_DIR" ]]; then
        rm -rf "$CHILD_DIR"
    fi
}
trap cleanup EXIT

echo "=============================================="
echo "  Bitcoin Pleb Slop Inscription"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Fee rate: ${FEE_RATE} sat/vB"
echo "  Parent file: ${PARENT_FILE}"
echo "  Weights file: ${WEIGHTS_FILE}"
if [[ -n "$ORD_DATA_DIR" ]]; then
    echo "  ord data dir: ${ORD_DATA_DIR}"
fi
echo ""

# Check required files exist
if [[ ! -f "$PARENT_FILE" ]]; then
    echo "ERROR: Parent file not found: $PARENT_FILE"
    exit 1
fi

if [[ ! -f "$WEIGHTS_FILE" ]]; then
    echo "ERROR: Weights file not found: $WEIGHTS_FILE"
    exit 1
fi

# ord maps .bin to application/octet-stream. Keep the tracked .slop artifact name
# in the repo, but inscribe a byte-identical .bin copy so public gateways do not
# treat the model weights as text.
CHILD_DIR=$(mktemp -d "${TMPDIR:-/tmp}/pleb-weights.XXXXXX")
CHILD_FILE="$CHILD_DIR/pleb-weights.bin"
cp "$WEIGHTS_FILE" "$CHILD_FILE"

echo "  Child file: ${CHILD_FILE} (byte-identical .bin copy)"
echo ""

# Step 1: Inscribe the parent (inscription.html)
echo "=============================================="
echo "  Step 1: Inscribe parent (inscription.html)"
echo "=============================================="
echo ""

PARENT_OUTPUT=$("${ORD_CMD[@]}" wallet inscribe --fee-rate "$FEE_RATE" --file "$PARENT_FILE")
echo "$PARENT_OUTPUT"
echo ""

# Extract parent inscription ID (format: <TXID>i0)
PARENT_ID=$(echo "$PARENT_OUTPUT" | grep -Eo '[a-fA-F0-9]{64}i[0-9]+' | head -1)

if [[ -z "$PARENT_ID" ]]; then
    echo "ERROR: Could not extract parent inscription ID from output"
    echo "Output was: $PARENT_OUTPUT"
    exit 1
fi

echo "✓ Parent inscription ID: $PARENT_ID"
echo "$PARENT_ID" > .parent_inscription_id
echo ""

# Step 2: Mine a block to confirm the parent
echo "=============================================="
echo "  Step 2: Mine block to confirm parent"
echo "=============================================="
echo ""

# Try ord generate when running against an ord data dir, otherwise try bitcoin-cli
# and fall back to ord generate.
if [[ -n "$ORD_DATA_DIR" ]]; then
    MINING_ADDRESS=$("${ORD_CMD[@]}" wallet receive | sed -nE 's/.*"([^"]+)".*/\1/p' | tail -1)
    bitcoin-cli -datadir="$ORD_DATA_DIR" generatetoaddress 1 "$MINING_ADDRESS" | head -5
elif command -v bitcoin-cli &> /dev/null; then
    echo "Using bitcoin-cli to mine block..."
    bitcoin-cli -regtest generateblock 1 | head -5
else
    echo "bitcoin-cli not found, using 'ord generate'..."
    "${ORD_CMD[@]}" generate 1
fi

echo ""
echo "✓ Block mined, waiting for confirmation..."
sleep 2

# Step 3: Inscribe the child weights as application/octet-stream
echo "=============================================="
echo "  Step 3: Inscribe child weights (.bin)"
echo "=============================================="
echo ""

echo "Parent ID: $PARENT_ID"
echo ""

CHILD_OUTPUT=$("${ORD_CMD[@]}" wallet inscribe --fee-rate "$FEE_RATE" --parent "$PARENT_ID" --file "$CHILD_FILE")
echo "$CHILD_OUTPUT"
echo ""

# Extract child inscription ID
CHILD_ID=$(echo "$CHILD_OUTPUT" | grep -Eo '[a-fA-F0-9]{64}i[0-9]+' | head -1)

if [[ -z "$CHILD_ID" ]]; then
    echo "ERROR: Could not extract child inscription ID from output"
    echo "Output was: $CHILD_OUTPUT"
    exit 1
fi

echo "✓ Child inscription ID: $CHILD_ID"
echo "$CHILD_ID" > .child_inscription_id
echo ""

echo "=============================================="
echo "  Step 4: Mine block to confirm child"
echo "=============================================="
echo ""

if [[ -n "$ORD_DATA_DIR" ]]; then
    MINING_ADDRESS=$("${ORD_CMD[@]}" wallet receive | sed -nE 's/.*"([^"]+)".*/\1/p' | tail -1)
    bitcoin-cli -datadir="$ORD_DATA_DIR" generatetoaddress 1 "$MINING_ADDRESS" | head -5
elif command -v bitcoin-cli &> /dev/null; then
    echo "Using bitcoin-cli to mine block..."
    bitcoin-cli -regtest generateblock 1 | head -5
else
    echo "Skipping child confirmation block: bitcoin-cli not found"
fi

echo ""
echo "✓ Child confirmation block mined."
echo ""

# Summary
echo "=============================================="
echo "  INScriptions Complete!"
echo "=============================================="
echo ""
echo "Parent (inscription.html):"
echo "  Inscription ID: $PARENT_ID"
echo ""
echo "Child weights (application/octet-stream .bin copy of pleb.slop):"
echo "  Inscription ID: $CHILD_ID"
echo ""
echo "View your inscriptions:"
echo "  Parent: https://ordinals.com/inscription/$PARENT_ID"
echo "  Child:  https://ordinals.com/inscription/$CHILD_ID"
echo ""
echo "Note: The child inscription is 'inscribed on' the parent,"
echo "      creating a hierarchical relationship on Bitcoin."
echo "=============================================="
