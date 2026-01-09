#!/bin/bash
# ============================================
# Monty - Autonomous AI Development Loop
# "Excellent..." - C. Montgomery Burns
# ============================================
# Usage: ./monty.sh [max_iterations]

set -e

MAX_ITERATIONS=${1:-20}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="$SCRIPT_DIR/monty.log"
PRD_FILE="$SCRIPT_DIR/prd.json"
PROGRESS_FILE="$SCRIPT_DIR/progress.txt"

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

# Initialize progress file if it doesn't exist
init_progress() {
    if [ ! -f "$PROGRESS_FILE" ]; then
        echo "# Monty Progress Log" > "$PROGRESS_FILE"
        echo "Started: $(date)" >> "$PROGRESS_FILE"
        echo "---" >> "$PROGRESS_FILE"
    fi
}

# ============================================
# Main
# ============================================

show_banner
check_claude
init_progress

cd "$PROJECT_ROOT"

# Check if launched from wizard with PRD (autonomous loop mode)
if [ -n "$MONTY_WIZARD" ] && [ -f "$PRD_FILE" ]; then
    PROJECT_NAME=$(jq -r '.projectName // "Project"' "$PRD_FILE")
    OUTPUT_DIR=$(jq -r '.outputDir // "./output"' "$PRD_FILE")

    echo -e "${CYAN}Starting Monty in autonomous mode...${NC}"
    echo -e "${GREEN}Project: $PROJECT_NAME${NC}"
    echo -e "${GREEN}Output: $OUTPUT_DIR${NC}"
    echo -e "${GREEN}Max iterations: $MAX_ITERATIONS${NC}"
    echo ""

    # The famous loop!
    for i in $(seq 1 $MAX_ITERATIONS); do
        echo ""
        echo -e "${YELLOW}═══════════════════════════════════════════════════════${NC}"
        echo -e "${YELLOW}  Monty Iteration $i of $MAX_ITERATIONS${NC}"
        echo -e "${YELLOW}═══════════════════════════════════════════════════════${NC}"
        echo ""

        # Run Claude with the prompt, capture output
        OUTPUT=$(claude --dangerously-skip-permissions -p "$(cat "$SCRIPT_DIR/prompt.md")

---

Read scripts/prd.json for project requirements and scripts/progress.txt for context.

IMPORTANT OUTPUT DIRECTORY: All code must be created in: $OUTPUT_DIR
Create a self-contained Docker Compose stack. Run 'docker-compose up -d' from that directory.

This is iteration $i of $MAX_ITERATIONS. Check prd.json for pending stories.

For each story you work on:
1. Echo '[US-XXX] Starting: title' when you begin
2. Implement following acceptance criteria exactly
3. Echo '[US-XXX] Complete' when done
4. Update prd.json to mark status as 'complete'
5. Commit your changes

When ALL stories are complete, output exactly: <monty>COMPLETE</monty>
If you're blocked and need help, output: <monty>BLOCKED</monty> with explanation.
If there's more work to do, just continue working." 2>&1 | tee -a "$LOG_FILE") || true

        # Check for completion signal
        if echo "$OUTPUT" | grep -q "<monty>COMPLETE</monty>"; then
            echo ""
            echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
            echo -e "${GREEN}  Monty completed all tasks!${NC}"
            echo -e "${GREEN}  Finished at iteration $i of $MAX_ITERATIONS${NC}"
            echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
            echo ""
            echo -e "${CYAN}Your project is ready in: $OUTPUT_DIR${NC}"
            echo -e "${CYAN}Start it with: cd $OUTPUT_DIR && docker-compose up -d${NC}"
            exit 0
        fi

        # Check for blocked signal
        if echo "$OUTPUT" | grep -q "<monty>BLOCKED</monty>"; then
            echo ""
            echo -e "${RED}═══════════════════════════════════════════════════════${NC}"
            echo -e "${RED}  Monty is blocked and needs help${NC}"
            echo -e "${RED}═══════════════════════════════════════════════════════${NC}"
            echo ""
            echo "Check $LOG_FILE for details"
            exit 1
        fi

        echo -e "${CYAN}Iteration $i complete. Continuing...${NC}"
        sleep 2
    done

    echo ""
    echo -e "${YELLOW}═══════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}  Monty reached max iterations ($MAX_ITERATIONS)${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Check $PROGRESS_FILE for status."
    echo "Run again to continue: ./start.sh"
    exit 1

else
    # Interactive mode - let user describe what they want
    echo -e "${CYAN}Starting Monty in interactive mode...${NC}"
    echo -e "${CYAN}Tell me what you want to build.${NC}"
    echo ""

    exec claude --dangerously-skip-permissions --append-system-prompt "$(cat "$SCRIPT_DIR/prompt.md")

---

You are Monty, an autonomous AI developer. Greet the user and ask what they want to build.

Once you understand their requirements:
1. Create scripts/prd.json with user stories (include outputDir field)
2. Initialize scripts/progress.txt with patterns
3. Tell the user to run './start.sh' again to begin autonomous building

Be helpful, ask clarifying questions, and create a solid plan before they start the build."
fi
