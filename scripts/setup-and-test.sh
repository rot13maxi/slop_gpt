#!/bin/bash
# setup-and-test.sh - Self-contained setup and test script for pleb_slop
#
# This script:
# 1. Clones and builds ord from source (if not already built)
# 2. Clones and builds bitcoind from source (if not already built)
# 3. Sets up PATH with both binaries
# 4. Runs 'ord env ./test-env' to start regtest environment
# 5. Waits for ord server to be ready
# 6. Runs scripts/inscribe.sh to inscribe UI and weights
# 7. Exports inscription IDs and runs npm test
# 8. Cleans up on exit
#
# Usage: ./scripts/setup-and-test.sh
#
# Idempotent: Skips clone/build steps if binaries already exist

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Configuration
BUILD_DIR="$PROJECT_ROOT/build"
ORD_DIR="$BUILD_DIR/ord"
BITCOIN_DIR="$BUILD_DIR/bitcoin"
ORD_BINARY="$ORD_DIR/target/release/ord"
BITCOIND_BINARY="$BITCOIN_DIR/build/bin/bitcoind"
BITCOIN_CLI_BINARY="$BITCOIN_DIR/build/bin/bitcoin-cli"
TEST_ENV_DIR="$PROJECT_ROOT/test-env"
ORD_PID_FILE="$PROJECT_ROOT/.ord-env.pid"

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    # Kill ord env process if running
    if [ -f "$ORD_PID_FILE" ]; then
        ORD_PID=$(cat "$ORD_PID_FILE")
        if kill -0 "$ORD_PID" 2>/dev/null; then
            log_info "Stopping ord env (PID: $ORD_PID)..."
            kill "$ORD_PID" 2>/dev/null || true
        fi
        rm -f "$ORD_PID_FILE"
    fi
    
    # Also try to kill any ord server on port 9001
    pkill -f "ord.*server.*9001" 2>/dev/null || true
    
    log_info "Cleanup complete"
}

# Set trap for cleanup
trap cleanup EXIT

# Step 1: Clone and build ord
log_step "Step 1: Building ord..."

if [ ! -d "$ORD_DIR" ]; then
    log_info "Cloning ord repository..."
    mkdir -p "$BUILD_DIR"
    git clone https://github.com/ordinals/ord "$ORD_DIR"
else
    log_info "ord directory already exists, skipping clone"
fi

if [ ! -f "$ORD_BINARY" ]; then
    log_info "Building ord with cargo..."
    cd "$ORD_DIR"
    cargo build --release
    cd "$PROJECT_ROOT"
else
    log_info "ord binary already exists at $ORD_BINARY, skipping build"
fi

# Step 2: Clone and build bitcoind
log_step "Step 2: Building bitcoind..."

if [ ! -d "$BITCOIN_DIR" ]; then
    log_info "Cloning bitcoin repository..."
    mkdir -p "$BUILD_DIR"
    git clone https://github.com/bitcoin/bitcoin "$BITCOIN_DIR"
else
    log_info "bitcoin directory already exists, skipping clone"
fi

if [ ! -f "$BITCOIND_BINARY" ] || [ ! -f "$BITCOIN_CLI_BINARY" ]; then
    log_info "Building bitcoind (CMake)..."
    cd "$BITCOIN_DIR"

    # Bitcoin Core v28+ uses CMake (autogen.sh/configure no longer exist)
    cmake -B build -DBUILD_TESTS=OFF -DBUILD_BENCH=OFF -DWITH_GUI=OFF -DENABLE_IPC=OFF
    cmake --build build -j$(nproc)

    cd "$PROJECT_ROOT"
else
    log_info "bitcoind binaries already exist, skipping build"
fi

# Step 3: Add binaries to PATH
log_step "Step 3: Setting up PATH..."

export PATH="$ORD_DIR/target/release:$BITCOIN_DIR/build/bin:$PATH"

log_info "ord binary: $(which ord)"
log_info "bitcoind binary: $(which bitcoind)"
log_info "bitcoin-cli binary: $(which bitcoin-cli)"

# Step 4: Create test environment and start ord env
log_step "Step 4: Starting regtest environment with 'ord env'..."

mkdir -p "$TEST_ENV_DIR"

# Start ord env in background
# This starts both bitcoind (regtest) and ord server on ports 9000/9001
log_info "Running 'ord env ./test-env'..."
ord env ./test-env &
ORD_ENV_PID=$!
echo $ORD_ENV_PID > "$ORD_PID_FILE"

log_info "ord env started (PID: $ORD_ENV_PID)"

# Step 5: Wait for ord server to be ready
log_step "Step 5: Waiting for ord server to be ready..."

MAX_RETRIES=60
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:9001 2>/dev/null || true)
    if [ -n "$HTTP_CODE" ] && [ "$HTTP_CODE" -ge 200 ] 2>/dev/null && [ "$HTTP_CODE" -lt 500 ] 2>/dev/null; then
        log_info "ord server is ready on http://127.0.0.1:9001 (HTTP $HTTP_CODE)"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    log_info "Waiting for server... (attempt $RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    log_error "ord server failed to start within timeout"
    exit 1
fi

log_step "Step 5b: Funding ord wallet..."
WALLET_ADDRESS=$(ord --datadir "$TEST_ENV_DIR" wallet receive | sed -nE 's/.*"([^"]+)".*/\1/p' | tail -1)
log_info "Wallet address: $WALLET_ADDRESS"
bitcoin-cli -datadir="$TEST_ENV_DIR" generatetoaddress 110 "$WALLET_ADDRESS" >/dev/null
ord --datadir "$TEST_ENV_DIR" wallet --no-sync balance >/dev/null || true

# Step 6: Run inscribe.sh
log_step "Step 6: Running inscribe.sh..."

chmod +x "$SCRIPT_DIR/inscribe.sh"
export PATH  # Ensure PATH is exported for inscribe.sh
export ORD_DATA_DIR="$TEST_ENV_DIR"

# Capture the output to extract inscription IDs
set +e
INScribe_OUTPUT=$(bash "$SCRIPT_DIR/inscribe.sh" 2>&1)
INSCRIBE_EXIT=$?
set -e
echo "$INScribe_OUTPUT"

if [ $INSCRIBE_EXIT -ne 0 ]; then
    log_error "inscribe.sh failed with exit code: $INSCRIBE_EXIT"
    exit $INSCRIBE_EXIT
fi

# Extract inscription IDs from inscribe.sh output
# The script outputs lines like:
# "✓ Parent inscription ID: <TXID>i0"
# "✓ Child inscription ID: <TXID>i0"
PARENT_INSCRIPTION_ID=$(echo "$INScribe_OUTPUT" | sed -nE 's/.*Parent inscription ID: ([a-fA-F0-9]+i[0-9]+).*/\1/p' | head -1)
CHILD_INSCRIPTION_ID=$(echo "$INScribe_OUTPUT" | sed -nE 's/.*Child inscription ID: ([a-fA-F0-9]+i[0-9]+).*/\1/p' | head -1)

if [ -z "$PARENT_INSCRIPTION_ID" ]; then
    log_error "Could not extract parent inscription ID"
    exit 1
fi

if [ -z "$CHILD_INSCRIPTION_ID" ]; then
    log_error "Could not extract child inscription ID"
    exit 1
fi

log_info "Parent inscription ID: $PARENT_INSCRIPTION_ID"
log_info "Child inscription ID: $CHILD_INSCRIPTION_ID"

# Step 7: Export env vars and run npm test
log_step "Step 7: Running Playwright tests..."

export REGTEST_URL="http://127.0.0.1:9001"
export PARENT_INSCRIPTION_ID="$PARENT_INSCRIPTION_ID"
export CHILD_INSCRIPTION_ID="$CHILD_INSCRIPTION_ID"

log_info "Environment variables:"
log_info "  REGTEST_URL: $REGTEST_URL"
log_info "  PARENT_INSCRIPTION_ID: $PARENT_INSCRIPTION_ID"
log_info "  CHILD_INSCRIPTION_ID: $CHILD_INSCRIPTION_ID"

cd "$PROJECT_ROOT/web-ui"

# Run the Playwright tests
if [ -d "node_modules" ]; then
    log_info "Running regtest Playwright tests..."
    npx playwright test --grep "e2e regtest test|children endpoint"
else
    log_warn "node_modules not found, installing dependencies first..."
    cd "$PROJECT_ROOT"
    npm install
    cd web-ui
    npx playwright test --grep "e2e regtest test|children endpoint"
fi

TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    log_info ""
    log_info "=============================================="
    log_info "  All tests passed!"
    log_info "=============================================="
    log_info ""
    log_info "Summary:"
    log_info "  - ord built from source: $ORD_BINARY"
    log_info "  - bitcoind built from source: $BITCOIND_BINARY"
    log_info "  - Regtest environment running on ports 9000/9001"
    log_info "  - Parent (UI) inscribed: $PARENT_INSCRIPTION_ID"
    log_info "  - Child (weights) inscribed: $CHILD_INSCRIPTION_ID"
    log_info "  - Playwright tests: PASSED"
    log_info ""
else
    log_error "Playwright tests failed with exit code: $TEST_EXIT_CODE"
    exit $TEST_EXIT_CODE
fi
