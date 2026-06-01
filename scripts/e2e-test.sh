#!/bin/bash
# e2e-test.sh - End-to-end regression test for pleb_slop inscription
# This script spins up a Bitcoin regtest environment, inscribes UI and weights,
# and verifies the full stack works in the browser
#
# PREREQUISITES:
# - ord: https://docs.ordinals.com/ordit
# - bitcoind: https://bitcoin.org/en/download
# - node.js with playwright installed
#
# USAGE:
#   # Auto-setup mode (requires ord and bitcoind):
#   ./scripts/e2e-test.sh
#
#   # Manual mode (with existing inscriptions):
#   REGTEST_URL=http://localhost:9001 \
#   PARENT_INSCRIPTION_ID=<ui-id> \
#   CHILD_INSCRIPTION_ID=<weights-id> \
#   ./scripts/e2e-test.sh

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

cleanup() {
    log_info "Cleaning up..."
    # Kill any running ord server
    if [ -f ".ord.pid" ]; then
        kill $(cat .ord.pid) 2>/dev/null || true
        rm -f .ord.pid
    fi
    # Kill bitcoind if running
    if [ -f ".bitcoind.pid" ]; then
        kill $(cat .bitcoind.pid) 2>/dev/null || true
        rm -f .bitcoind.pid
    fi
    # Clean up test environment
    if [ -d "./test-env" ]; then
        rm -rf ./test-env
    fi
    # Remove inscription ID files
    rm -f .parent_inscription_id .child_inscription_id
}

# Set trap for cleanup
trap cleanup EXIT

# Check if running in manual mode (environment variables set)
if [ -n "$REGTEST_URL" ] && [ -n "$PARENT_INSCRIPTION_ID" ] && [ -n "$CHILD_INSCRIPTION_ID" ]; then
    log_step "Running in manual mode with existing inscriptions"
    log_info "REGTEST_URL: $REGTEST_URL"
    log_info "PARENT_INSCRIPTION_ID (UI): $PARENT_INSCRIPTION_ID"
    log_info "CHILD_INSCRIPTION_ID (weights): $CHILD_INSCRIPTION_ID"
    
    # Run the Playwright e2e tests
    log_step "Running Playwright e2e tests..."
    cd web-ui
    REGTEST_URL="$REGTEST_URL" \
    PARENT_INSCRIPTION_ID="$PARENT_INSCRIPTION_ID" \
    CHILD_INSCRIPTION_ID="$CHILD_INSCRIPTION_ID" \
    npx playwright test --grep "e2e regtest test|children endpoint"
    
    if [ $? -eq 0 ]; then
        log_info ""
        log_info "=== E2E Test Complete ==="
        log_info "All tests passed!"
        exit 0
    else
        log_error "Playwright tests failed"
        exit 1
    fi
fi

# Auto-setup mode - requires ord and bitcoind
log_step "Checking prerequisites for auto-setup mode..."

# Check if ord is available
if ! command -v ord &> /dev/null; then
    log_error "ord command not found"
    log_error ""
    log_error "To run the full e2e test, you need:"
    log_error "  1. Install ord: https://docs.ordinals.com/ordit"
    log_error "  2. Install bitcoind: https://bitcoin.org/en/download"
    log_error ""
    log_error "Alternatively, run in manual mode with existing inscriptions:"
    log_error "  REGTEST_URL=http://localhost:9001 \\"
    log_error "  PARENT_INSCRIPTION_ID=<ui-id> \\"
    log_error "  CHILD_INSCRIPTION_ID=<weights-id> \\"
    log_error "  ./scripts/e2e-test.sh"
    exit 1
fi

# Check if bitcoind is available
if ! command -v bitcoind &> /dev/null; then
    log_error "bitcoind command not found"
    log_error ""
    log_error "To run the full e2e test, you need:"
    log_error "  1. Install bitcoind: https://bitcoin.org/en/download"
    log_error "  2. Install ord: https://docs.ordinals.com/ordit"
    exit 1
fi

# Check if bitcoin-cli is available
if ! command -v bitcoin-cli &> /dev/null; then
    log_error "bitcoin-cli command not found"
    exit 1
fi

log_info "All prerequisites found. Starting auto-setup..."

# Check if we have node and playwright
if [ ! -d "node_modules" ]; then
    log_info "Installing npm dependencies..."
    npm install
fi

# Create test environment
log_step "Creating test environment..."
mkdir -p test-env

# Initialize ord wallet if needed
log_step "Setting up ord wallet..."
if ! ord --regtest wallet create 2>/dev/null; then
    log_info "Wallet already exists"
fi

# Start bitcoind regtest
log_step "Starting bitcoind regtest..."
bitcoind -regtest -datadir=./test-env -daemon=1 \
    -rpcuser=test -rpcpassword=test -rpcport=9000 \
    -server=1 -txindex=1 -fallbackfee=0.0001

# Save bitcoind PID for cleanup
echo $! > .bitcoind.pid

# Wait for bitcoind to be ready
log_info "Waiting for bitcoind to start..."
sleep 5

# Create ord wallet
log_step "Creating ord wallet..."
if ! ord --regtest wallet create 2>/dev/null; then
    log_info "Ord wallet already exists"
fi

# Get wallet address
log_step "Getting wallet address..."
ADDRESS=$(ord --regtest wallet address)
log_info "Wallet address: $ADDRESS"

# Mine blocks to the wallet
log_step "Mining 110 blocks to wallet..."
bitcoin-cli -regtest -rpcuser=test -rpcpassword=test -rpcport=9000 generatetoaddress 110 "$ADDRESS"

# Wait for confirmations
sleep 3

# Check balance
BALANCE=$(ord --regtest wallet balance 2>/dev/null | grep -oP 'confirmed\s+\K[0-9.]+' || echo "0")
log_info "Wallet balance: $BALANCE BTC"

if [ "$BALANCE" = "0" ] || [ -z "$BALANCE" ]; then
    log_error "Wallet has no balance. Check bitcoind logs."
    exit 1
fi

# Start ord server
log_step "Starting ord server on port 9001..."
ord --regtest server --http-port 9001 --data-dir ./test-env &
ORD_PID=$!
echo $ORD_PID > .ord.pid

# Wait for ord server to start
log_info "Waiting for ord server to start..."
sleep 10

# Check if server is running
if ! kill -0 $ORD_PID 2>/dev/null; then
    log_error "Ord server failed to start"
    exit 1
fi

log_step "Running inscription script..."
chmod +x scripts/inscribe.sh
bash scripts/inscribe.sh

# Wait for indexing
log_step "Waiting for indexing..."
sleep 10

# Read inscription IDs
if [ ! -f ".parent_inscription_id" ] || [ ! -f ".child_inscription_id" ]; then
    log_error "Inscription IDs not found"
    exit 1
fi

PARENT_ID=$(cat .parent_inscription_id)
CHILD_ID=$(cat .child_inscription_id)

log_info ""
log_info "=== Inscription Complete ==="
log_info "Parent (UI) ID: $PARENT_ID"
log_info "Child (weights) ID: $CHILD_ID"
log_info ""

# Run Playwright tests
log_step "Running Playwright e2e tests..."
cd web-ui
REGTEST_URL="http://localhost:9001" \
PARENT_INSCRIPTION_ID="$PARENT_ID" \
CHILD_INSCRIPTION_ID="$CHILD_ID" \
npx playwright test --grep "e2e regtest test|children endpoint"

PLAYWRIGHT_EXIT=$?

if [ $PLAYWRIGHT_EXIT -eq 0 ]; then
    log_info ""
    log_info "=== E2E Test Complete ==="
    log_info "All tests passed!"
    log_info ""
    log_info "Summary:"
    log_info "  - Ord server running on port 9001"
    log_info "  - Bitcoind regtest running on port 9000"
    log_info "  - Parent (UI) inscribed: $PARENT_ID"
    log_info "  - Child (weights) inscribed: $CHILD_ID"
    log_info "  - Children endpoint: OK"
    log_info "  - Browser tests: PASSED"
    log_info ""
    exit 0
else
    log_error "Playwright tests failed"
    exit 1
fi
