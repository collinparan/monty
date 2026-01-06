#!/bin/bash
set -e

# ============================================
# Monty - Autonomous Python/FastAPI Dev Loop
# "Excellent..." - C. Montgomery Burns
# ============================================

# Configuration
MAX_ITERATIONS=${1:-15}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="$SCRIPT_DIR/monty.log"
AGENT=${MONTY_AGENT:-"claude"}  # claude, amp, or cursor

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Logging
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "${BLUE}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

log_error() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1"
    echo -e "${RED}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

log_success() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "${GREEN}${msg}${NC}"
    echo "$msg" >> "$LOG_FILE"
}

# Agent command mapping
get_agent_cmd() {
    case $AGENT in
        claude) echo "claude --dangerously-skip-permissions" ;;
        amp)    echo "amp --dangerously-allow-all" ;;
        cursor) echo "cursor --agent" ;;
        *)      
            log_error "Unknown agent: $AGENT. Using claude."
            echo "claude --dangerously-skip-permissions" 
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    local missing=()
    
    # Check for agent CLI
    case $AGENT in
        claude) command -v claude &>/dev/null || missing+=("claude CLI") ;;
        amp)    command -v amp &>/dev/null || missing+=("amp CLI") ;;
        cursor) command -v cursor &>/dev/null || missing+=("cursor CLI") ;;
    esac
    
    # Check for required files
    [[ -f "$SCRIPT_DIR/prompt.md" ]] || missing+=("prompt.md")
    [[ -f "$SCRIPT_DIR/prd.json" ]] || missing+=("prd.json")
    [[ -f "$SCRIPT_DIR/progress.txt" ]] || missing+=("progress.txt")
    
    # Check for jq (optional but useful)
    command -v jq &>/dev/null || log "${YELLOW}⚠ jq not installed (optional, for monitoring)${NC}"
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing prerequisites: ${missing[*]}"
        exit 1
    fi
}

# Check Docker services
check_docker_services() {
    if ! command -v docker &>/dev/null; then
        log "${YELLOW}⚠ Docker not found${NC}"
        return
    fi
    
    local services=("postgres" "neo4j")
    local running=()
    local not_running=()
    
    for svc in "${services[@]}"; do
        if docker ps --format '{{.Names}}' 2>/dev/null | grep -qi "$svc"; then
            running+=("$svc")
        else
            not_running+=("$svc")
        fi
    done
    
    if [[ ${#running[@]} -gt 0 ]]; then
        log_success "✓ Docker services running: ${running[*]}"
    fi
    
    if [[ ${#not_running[@]} -gt 0 ]]; then
        log "${YELLOW}⚠ Docker services not detected: ${not_running[*]}${NC}"
        log "${YELLOW}  Run: docker-compose -f docker/docker-compose.yml up -d${NC}"
    fi
    
    # Check for GPU services (Chatterbox)
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -qi "chatterbox"; then
        log_success "✓ Chatterbox TTS service running"
    fi
}

# Check Python environment
check_python_env() {
    if [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        log_success "✓ Python project detected (pyproject.toml)"
    elif [[ -f "$PROJECT_ROOT/requirements.txt" ]]; then
        log_success "✓ Python project detected (requirements.txt)"
    else
        log "${YELLOW}⚠ No Python project files found${NC}"
    fi
    
    # Check for virtual environment
    if [[ -n "$VIRTUAL_ENV" ]]; then
        log_success "✓ Virtual environment active: $VIRTUAL_ENV"
    elif [[ -d "$PROJECT_ROOT/.venv" ]]; then
        log "${YELLOW}⚠ .venv exists but not activated${NC}"
    fi
}

# Get pending story count
get_pending_count() {
    if command -v jq &>/dev/null; then
        jq '[.userStories[] | select(.status == "pending")] | length' "$SCRIPT_DIR/prd.json" 2>/dev/null || echo "?"
    else
        grep -c '"status": "pending"' "$SCRIPT_DIR/prd.json" 2>/dev/null || echo "?"
    fi
}

# Get completed story count
get_completed_count() {
    if command -v jq &>/dev/null; then
        jq '[.userStories[] | select(.status == "complete")] | length' "$SCRIPT_DIR/prd.json" 2>/dev/null || echo "?"
    else
        grep -c '"status": "complete"' "$SCRIPT_DIR/prd.json" 2>/dev/null || echo "?"
    fi
}

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
    echo -e "${CYAN}  🐍 Python/FastAPI + PGVector + Graphiti${NC}"
    echo -e "${CYAN}  Transform AI Development Loop${NC}"
    echo -e "${MAGENTA}  \"Excellent...\" - C. Montgomery Burns${NC}"
    echo ""
}

# Summary
show_summary() {
    local pending=$(get_pending_count)
    local completed=$(get_completed_count)
    
    echo -e "${BOLD}Configuration:${NC}"
    echo -e "  Agent:          ${CYAN}$AGENT${NC}"
    echo -e "  Max iterations: ${CYAN}$MAX_ITERATIONS${NC}"
    echo -e "  Project root:   ${CYAN}$PROJECT_ROOT${NC}"
    echo -e "  Log file:       ${CYAN}$LOG_FILE${NC}"
    echo ""
    echo -e "${BOLD}Stories:${NC}"
    echo -e "  Pending:        ${YELLOW}$pending${NC}"
    echo -e "  Completed:      ${GREEN}$completed${NC}"
    echo ""
}

# Main loop
run_loop() {
    local start_time=$(date +%s)
    
    for i in $(seq 1 $MAX_ITERATIONS); do
        echo ""
        echo -e "${YELLOW}═══════════════════════════════════════════════════════${NC}"
        echo -e "${YELLOW}  Iteration $i of $MAX_ITERATIONS │ Pending: $(get_pending_count) │ Done: $(get_completed_count)${NC}"
        echo -e "${YELLOW}═══════════════════════════════════════════════════════${NC}"
        
        local iter_start=$(date +%s)
        
        # Pipe prompt to agent
        OUTPUT=$(cat "$SCRIPT_DIR/prompt.md" \
            | $(get_agent_cmd) 2>&1 \
            | tee -a "$LOG_FILE") || true
        
        local iter_end=$(date +%s)
        local duration=$((iter_end - iter_start))
        
        log "Iteration $i completed in ${duration}s"
        
        # Check for completion signal
        if echo "$OUTPUT" | grep -q "<monty>COMPLETE</monty>"; then
            local total_time=$(( $(date +%s) - start_time ))
            echo ""
            echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
            echo -e "${GREEN}║                                                       ║${NC}"
            echo -e "${GREEN}║   ✅ Excellent! All stories complete!                 ║${NC}"
            echo -e "${GREEN}║                                                       ║${NC}"
            echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
            echo ""
            log_success "Total iterations: $i"
            log_success "Total time: $((total_time / 60))m $((total_time % 60))s"
            log_success "Log file: $LOG_FILE"
            return 0
        fi
        
        # Check for blocking error
        if echo "$OUTPUT" | grep -q "<monty>BLOCKED</monty>"; then
            echo ""
            echo -e "${RED}╔═══════════════════════════════════════════════════════╗${NC}"
            echo -e "${RED}║   🚫 Monty is blocked and needs human intervention    ║${NC}"
            echo -e "${RED}╚═══════════════════════════════════════════════════════╝${NC}"
            echo ""
            log_error "BLOCKED at iteration $i - check progress.txt for details"
            
            # Try to extract reason
            local reason=$(echo "$OUTPUT" | grep -A1 "<monty>BLOCKED</monty>" | tail -1)
            if [[ -n "$reason" ]]; then
                log_error "Reason: $reason"
            fi
            
            return 2
        fi
        
        # Brief pause between iterations
        sleep 3
    done
    
    echo ""
    echo -e "${YELLOW}╔═══════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║   ⚠️  Max iterations ($MAX_ITERATIONS) reached                       ║${NC}"
    echo -e "${YELLOW}╚═══════════════════════════════════════════════════════╝${NC}"
    echo ""
    log "Max iterations reached without completion"
    log "Pending stories: $(get_pending_count)"
    return 1
}

# Cleanup handler
cleanup() {
    echo ""
    log "Interrupted by user"
    exit 130
}

trap cleanup SIGINT SIGTERM

# ============================================
# Main
# ============================================

show_banner
check_prerequisites
check_docker_services
check_python_env
echo ""
show_summary

# Confirm before starting
read -p "Start Monty? [Y/n] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Nn]$ ]]; then
    log "Aborted by user"
    exit 0
fi

echo ""
log "Starting Monty..."
echo "════════════════════════════════════════════════════════"

run_loop
exit $?
