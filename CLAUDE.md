# CLAUDE.md - Monty Project Guide

This file provides context for Claude Code when working on the Monty project.

## What is Monty?

Monty is an autonomous AI development agent that builds complete Docker Compose applications. It's named after C. Montgomery Burns ("Excellent...").

**Key concept:** Monty runs Claude Code in an iterative loop, implementing user stories one at a time until the project is complete.

## Project Structure

```
monty/
├── start.sh              # Entry point - launches Monty
├── setup.sh              # Web wizard launcher
├── wizard/               # Setup wizard (FastAPI + vanilla JS)
│   ├── index.html        # Web UI for project configuration
│   ├── server.py         # Backend: PRD generation, connection testing, WebSocket progress
│   └── requirements.txt  # Python deps (fastapi, uvicorn, asyncpg, aiohttp)
├── scripts/
│   ├── monty.sh          # Main loop script - runs Claude in iterations
│   ├── prompt.md         # System prompt for Monty agent
│   ├── prd.json          # Product requirements (generated)
│   └── progress.txt      # Build progress log
└── output/               # Where generated projects go (gitignored)
```

## How Monty Works

### Two Modes

1. **Interactive Mode** (`./start.sh` without wizard)
   - User chats with Claude to describe project
   - Claude creates `prd.json` with user stories
   - User runs `./start.sh` again to start building

2. **Wizard Mode** (`./setup.sh`)
   - Web UI collects project requirements
   - Generates `prd.json` automatically
   - Launches Monty in autonomous loop mode
   - Streams progress via WebSocket

### The Loop (from `scripts/monty.sh`)

```bash
for i in $(seq 1 $MAX_ITERATIONS); do
    OUTPUT=$(claude --dangerously-skip-permissions -p "...")

    if echo "$OUTPUT" | grep -q "<monty>COMPLETE</monty>"; then
        exit 0  # All done!
    fi

    if echo "$OUTPUT" | grep -q "<monty>BLOCKED</monty>"; then
        exit 1  # Need human help
    fi
done
```

Each iteration:
1. Reads `prd.json` for pending stories
2. Implements the next story
3. Commits changes
4. Updates story status
5. Outputs completion signal or continues

### Signal Protocol

- `<monty>COMPLETE</monty>` - All stories done, exit successfully
- `<monty>BLOCKED</monty>` - Need human intervention
- `[US-XXX] Starting: title` - Beginning a story
- `[US-XXX] Complete` - Finished a story

## Key Files

### `scripts/prompt.md`

The system prompt that defines Monty's behavior. Includes:
- Core workflow (read context → select story → implement → verify → commit)
- Tech stack patterns (FastAPI, PostgreSQL, Redis, etc.)
- Code patterns (async, WebSocket, SSE, etc.)
- Verification requirements (ruff, pyright, pytest)

### `wizard/server.py`

FastAPI backend that:
- Serves the setup wizard HTML
- Generates `prd.json` from user input
- Tests database/API connections
- Discovers schema (tables, columns, endpoints)
- Launches Monty and streams progress via WebSocket

Key endpoints:
- `GET /` - Serve wizard HTML
- `POST /api/project` - Create project and start Monty
- `POST /api/test-connection` - Test DB/API and discover schema
- `WS /ws/{session_id}` - Stream build progress

### `wizard/index.html`

Vanilla JS web UI with:
- Project configuration form
- Data source selection (legacy DB, REST API, files)
- Connection testing with schema discovery
- Real-time progress display with iteration counter

## Data Source Integration

The wizard can connect to existing data sources:

### Database Discovery

For PostgreSQL, discovers:
- Table names
- Column names and types
- Approximate row counts

Creates individual user stories for each selected table.

### REST API Discovery

For REST APIs, attempts to find:
- OpenAPI/Swagger spec at common paths
- Endpoint methods, paths, descriptions

Creates user stories for API client and each endpoint.

## Multi-Agent Architecture (CrewAI)

When "Multi-Agent (CrewAI)" is selected, Monty generates a 3-agent crew:

### The Agents

| Agent | Role | Tools | Key Behaviors |
|-------|------|-------|---------------|
| **Manager** | Orchestrator | None (delegates) | Plans tasks, ensures alignment, synthesizes results |
| **Data Engineer** | Query Specialist | `sql_query`, `cypher_query`, `schema_inspector` | SQL/Cypher queries, data extraction, validation |
| **Data Scientist** | ML Specialist | `train_model`, `predict`, `explain_model` | Explainable ML with InterpretML |

### Why This Crew?

1. **Manager prevents drift** - Without a manager, specialists work in silos and lose sight of the goal
2. **Data Engineer handles complexity** - SQL and Cypher require specialized knowledge
3. **Data Scientist explains** - Using InterpretML ensures predictions are trustworthy

### Explainable ML Models

The Data Scientist uses [InterpretML](https://interpret.ml/) for glass-box models:

- **EBM (Explainable Boosting Machine)** - Best accuracy while remaining interpretable
- **Decision Tree** - Simple, visual, inherently interpretable
- **Random Forest** - Feature importance, ensemble predictions

### Crew Execution Flow

```
1. User Request → Manager (planning_task)
2. Manager delegates → Data Engineer (data_extraction_task)
3. Data ready → Data Scientist (modeling_task)
4. Results ready → Manager (synthesis_task)
5. Final response with explanations
```

### Files Generated for Multi-Agent

```
app/
├── agents/
│   ├── __init__.py
│   ├── crew.py           # Crew definition
│   ├── agents.py         # Manager, Data Engineer, Data Scientist
│   ├── tasks.py          # Task definitions
│   └── tools/
│       ├── __init__.py
│       ├── sql_tools.py      # SQLQueryTool, SchemaInspectorTool
│       ├── cypher_tools.py   # CypherQueryTool
│       └── ml_tools.py       # TrainModelTool, PredictTool, ExplainModelTool
├── routers/
│   └── crew.py           # POST /crew/start, GET /crew/status/{id}
```

### Key Dependencies for Multi-Agent

```
crewai>=0.28.0
langchain-anthropic>=0.1.0
interpret>=0.4.0
neo4j>=5.0.0
```

## Common Tasks

### Adding a new data source type

1. Add checkbox in `wizard/index.html` under "Data Sources"
2. Add detail form (like `legacy-db-details`)
3. Update `updateTestConnectionVisibility()` in JS
4. Add connection test logic in `wizard/server.py`
5. Add story generation in `generate_user_stories()`

### Modifying the agent prompt

Edit `scripts/prompt.md`. Key sections:
- Core Workflow - the step-by-step process
- Tech Stack - available technologies
- Agentic AI Patterns - code examples
- Verification Requirements - what must pass before commit

### Adding wizard features

1. HTML form elements in `wizard/index.html`
2. JS handlers in the `<script>` section
3. Pydantic models in `wizard/server.py`
4. Update `ProjectConfig`, `generate_user_stories()`, `generate_prd()`

## Development Tips

### Testing the wizard locally

```bash
cd wizard
pip install -r requirements.txt
python server.py
# Open http://localhost:3456
```

### Testing Monty without full runs

```bash
# Just test Claude CLI is working
claude --version

# Test with a simple prompt
claude -p "Say hello"

# Run one iteration
MONTY_WIZARD=1 ./scripts/monty.sh 1
```

### Debugging

- Check `scripts/monty.log` for full Claude output
- Check `scripts/progress.txt` for story progress
- Check `scripts/prd.json` for generated requirements

## Environment Variables

- `MONTY_WIZARD=1` - Enables autonomous loop mode
- `MONTY_AGENT=claude` - Which agent to use (default: claude)
- `MONTY_PORT=3456` - Wizard server port

## Output Convention

All generated projects go in `./output/` directory. This is:
- Gitignored (except `.gitkeep`)
- Separate from Monty's own code
- Self-contained Docker Compose stacks
