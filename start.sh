#!/bin/bash
# ============================================
# Monty - Start Here
# ============================================
# Just run: ./start.sh
# Monty will ask you what to build.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default configuration
export MONTY_AGENT="${MONTY_AGENT:-claude}"

# Launch Monty
exec "$SCRIPT_DIR/scripts/monty.sh" "$@"
