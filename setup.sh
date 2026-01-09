#!/bin/bash
# ============================================
# Monty Setup Wizard
# ============================================
# Launches a web UI to configure your project,
# then starts Monty to build it.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WIZARD_DIR="$SCRIPT_DIR/wizard"
PORT=${MONTY_PORT:-3456}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Banner
echo -e "${YELLOW}"
cat << 'EOF'
  __  __             _
 |  \/  | ___  _ __ | |_ _   _
 | |\/| |/ _ \| '_ \| __| | | |
 | |  | | (_) | | | | |_| |_| |
 |_|  |_|\___/|_| |_|\__|\__, |
                         |___/
EOF
echo -e "${NC}"
echo -e "${MAGENTA}  Setup Wizard${NC}"
echo ""

# Check for Python
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}Error: Python 3 is required${NC}"
    exit 1
fi

# Check/install dependencies
echo -e "${CYAN}Checking dependencies...${NC}"
if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    echo -e "${YELLOW}Installing wizard dependencies...${NC}"
    pip3 install -q fastapi uvicorn pydantic
fi

# Check for Claude CLI
if ! command -v claude &>/dev/null; then
    echo -e "${RED}Error: Claude CLI not found${NC}"
    echo ""
    echo "Install with: npm install -g @anthropic-ai/claude-code"
    exit 1
fi

# Cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down wizard...${NC}"
    kill $SERVER_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start the wizard server
echo -e "${GREEN}Starting setup wizard on http://localhost:$PORT${NC}"
echo ""

cd "$WIZARD_DIR"
python3 server.py &
SERVER_PID=$!

# Wait for server to start
sleep 1

# Open browser (macOS/Linux)
if command -v open &>/dev/null; then
    open "http://localhost:$PORT"
elif command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:$PORT"
else
    echo -e "${CYAN}Open your browser to: http://localhost:$PORT${NC}"
fi

echo -e "${CYAN}Press Ctrl+C to stop the wizard${NC}"
echo ""

# Wait for server
wait $SERVER_PID
