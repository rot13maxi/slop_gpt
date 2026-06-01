# Scripts

This directory contains utility scripts for the Bitcoin Pleb Slop project.

## inscribe.sh

Automates the two-step Bitcoin inscription process for inscribing content onto the Bitcoin blockchain using the Ordinals protocol.

### Two-Inscription Architecture

The inscription process uses a **parent-child relationship**:

1. **Parent Inscription**: The `web-ui/inscription.html` file is inscribed first as the "parent" inscription. This creates the foundational inscription on Bitcoin.

2. **Block Confirmation**: After the parent is inscribed, a new block must be mined to confirm the transaction. This is required before a child inscription can reference the parent.

3. **Child Inscription**: A byte-identical temporary `.bin` copy of `pleb.slop` is then inscribed as a "child" of the parent, creating a hierarchical relationship. The `.bin` extension makes `ord` classify the weights as `application/octet-stream`, which keeps public gateways from treating the model bytes as text.

This architecture allows for:
- **Hierarchical content**: Related inscriptions can be linked together
- **Metadata preservation**: The child can reference and build upon the parent
- **Ordinal relationships**: Bitcoin ordinals support this parent-child binding

### Usage

```bash
# Basic usage (uses default fee rate of 1 sat/vB for regtest)
./scripts/inscribe.sh

# Custom fee rate
./scripts/inscribe.sh --fee-rate 5
```

### Requirements

- `ord` wallet configured and unlocked
- `bitcoind` running (for regtest mining) or `ord` environment available
- Both `web-ui/inscription.html` and `pleb.slop` files present in the project root

### Environment Variables

- `ORD_HOST`: Ord daemon host (default: localhost)
- `ORD_PORT`: Ord daemon port (default: 9001)
- `BITCOIND_HOST`: Bitcoin daemon host (default: localhost)
- `BITCOIND_PORT`: Bitcoin daemon port (default: 9000)

### Output

The script will display:
- Both inscription IDs (parent and child)
- Links to view the inscriptions on ordinals.com
