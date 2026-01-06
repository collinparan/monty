#!/bin/bash
# ============================================
# Monty - Autonomous AI Development Loop
# "Excellent..." - C. Montgomery Burns
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="$SCRIPT_DIR/monty.log"
AGENT="${MONTY_AGENT:-claude}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Banner
show_banner() {
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
    echo -e "${MAGENTA}  \"Excellent...\" - C. Montgomery Burns${NC}"
    echo ""
}

# Check for Claude CLI
check_claude() {
    if ! command -v claude &>/dev/null; then
        echo -e "${RED}Error: Claude CLI not found${NC}"
        echo ""
        echo "Install with: npm install -g @anthropic-ai/claude-code"
        exit 1
    fi
}

# Cleanup handler
cleanup() {
    echo ""
    echo -e "${YELLOW}Interrupted. Goodbye!${NC}"
    exit 130
}

trap cleanup SIGINT SIGTERM

# ============================================
# Main
# ============================================

show_banner
check_claude

echo -e "${CYAN}Starting Monty...${NC}"
echo -e "${CYAN}Tell me what you want to build.${NC}"
echo ""

# Launch Claude in interactive mode with the prompt context
cd "$PROJECT_ROOT"
exec claude --dangerously-skip-permissions --append-system-prompt "$(cat "$SCRIPT_DIR/prompt.md")

---

You are Monty. Greet the user and ask them what they want to build today.

Once you understand their requirements:
1. Create/update prd.json with user stories
2. Create/update progress.txt with relevant patterns
3. Start implementing stories one by one
4. Commit after each completed story

Keep the conversation going. Ask clarifying questions. Be helpful and direct."
