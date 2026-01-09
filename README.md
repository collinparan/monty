# Monty

**"Excellent..."** - C. Montgomery Burns

Monty is an autonomous AI development agent that builds complete, production-ready applications from a simple description. Tell Monty what you want to build, and it will create a self-contained Docker Compose stack with all the code, configuration, and documentation you need.

## Quick Start

### Option 1: Setup Wizard (Recommended)

```bash
./setup.sh
```

This opens a web UI where you can:
- Describe what you want to build
- Select your tech stack preferences
- Connect to existing databases or APIs
- Watch Monty build your project in real-time

### Option 2: Interactive Mode

```bash
./start.sh
```

This starts Monty in interactive mode where you can chat directly with it about what you want to build.

## How It Works

1. **You describe your project** - Either through the web wizard or by chatting with Monty
2. **Monty creates a plan** - Generates `prd.json` with user stories and requirements
3. **Monty builds iteratively** - Runs in a loop, implementing one story at a time
4. **You get a complete stack** - Everything goes into `./output/` as a Docker Compose project

### The Famous Loop

Inspired by Ralph Wiggum, Monty runs in an autonomous loop:

```
═══════════════════════════════════════════════════════
  Monty Iteration 1 of 20
═══════════════════════════════════════════════════════

[US-001] Starting: Docker Compose infrastructure
... implementing ...
[US-001] Complete

═══════════════════════════════════════════════════════
  Monty Iteration 2 of 20
═══════════════════════════════════════════════════════

[US-002] Starting: FastAPI project scaffolding
... implementing ...
```

When all stories are complete, Monty outputs `<monty>COMPLETE</monty>` and exits.

## Features

### Setup Wizard

The web-based setup wizard (`./setup.sh`) provides:

- **Project configuration** - Name, description, output directory
- **Tech stack selection** - Backend framework, databases, features
- **Data source integration** - Connect to existing databases or APIs
- **Schema discovery** - Automatically discovers tables/endpoints from your data sources
- **Real-time progress** - Watch Monty work with live updates

### Data Source Support

Monty can connect to your existing data:

| Source | Discovery | What Monty Creates |
|--------|-----------|-------------------|
| PostgreSQL | Tables, columns, row counts | Models, repositories, CRUD operations |
| MySQL | Tables, columns | Models, repositories |
| REST APIs | OpenAPI/Swagger endpoints | HTTP client, typed methods |
| GraphQL | Schema introspection | Client, queries, mutations |
| CSV/S3 | File structure | Import services, validation |

### AI Agent Support

| Type | Framework | Use Case |
|------|-----------|----------|
| Single Agent | Claude Tools / LangGraph | One agent with tool use |
| Multi-Agent | CrewAI | Multiple specialized agents collaborating |

**Why CrewAI for multi-agent?** CrewAI provides explainable agent interactions - each agent has a clear role, goal, and backstory. You can see exactly why each agent made its decisions.

#### Default Crew: Manager + Data Engineer + Data Scientist

When you select "Multi-Agent (CrewAI)", Monty creates a 3-agent crew:

| Agent | Role | Responsibilities |
|-------|------|------------------|
| **Manager** | Orchestrator | Breaks down requests, delegates tasks, ensures goal alignment, synthesizes final results |
| **Data Engineer** | Query Specialist | Writes SQL/Cypher queries, extracts data, validates quality, prepares datasets |
| **Data Scientist** | ML Specialist | Trains explainable models (EBM, Decision Tree, Random Forest), provides predictions with explanations |

**Explainable ML with InterpretML:**

The Data Scientist uses [InterpretML](https://interpret.ml/) for interpretable models:

- **[Explainable Boosting Machine (EBM)](https://interpret.ml/docs/ebm.html)** - Glass-box model with accuracy comparable to gradient boosting
- **[Decision Trees](https://interpret.ml/docs/dt.html)** - Inherently interpretable, easy to visualize
- **[Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)** - Feature importance and ensemble predictions

Every prediction includes an explanation of *why* - which features contributed and by how much.

**Example Crew Flow:**

```
User: "Why did sales drop last quarter?"

Manager → Plans: "Need sales data + analysis"
    │
    ├── Data Engineer → Queries database, returns clean dataset
    │
    ├── Data Scientist → Trains EBM, identifies top factors:
    │                    - Marketing spend: -40%
    │                    - Seasonality: -30%
    │                    - Competitor pricing: -20%
    │
    └── Manager → Synthesizes: "Sales dropped 15% primarily due to
                               reduced marketing spend (40% impact)..."
```

### Output Structure

Every project Monty creates includes:

```
output/
├── docker-compose.yml    # All services configured
├── .env.example          # Environment variables template
├── README.md             # Setup and usage instructions
├── app/                  # Application code
│   ├── main.py
│   ├── config.py
│   ├── models/
│   ├── routers/
│   └── services/
├── tests/                # Test suite
└── Dockerfile           # Container configuration
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Required for Monty to work
ANTHROPIC_API_KEY=your-api-key

# Optional: for database schema discovery
DATABASE_URL=postgresql://user:pass@host:5432/db
```

### Max Iterations

By default, Monty runs up to 20 iterations. Change this:

```bash
./start.sh 50  # Run up to 50 iterations
```

## Project Structure

```
monty/
├── start.sh              # Simple launcher
├── setup.sh              # Web wizard launcher
├── wizard/               # Setup wizard UI
│   ├── index.html        # Web interface
│   ├── server.py         # FastAPI backend
│   └── requirements.txt  # Python dependencies
├── scripts/
│   ├── monty.sh          # Main Monty script
│   ├── prompt.md         # AI agent instructions
│   ├── prd.json          # Generated requirements
│   └── progress.txt      # Build progress log
└── output/               # Generated projects go here
```

## Requirements

- **Claude Code CLI** - Install with `npm install -g @anthropic-ai/claude-code`
- **Python 3.9+** - For the setup wizard
- **Docker** - To run the generated projects

## Examples

### Build a REST API

```
./setup.sh
→ Project: "User Management API"
→ Description: "REST API for managing users with authentication"
→ Backend: FastAPI
→ Database: PostgreSQL
→ Features: Authentication, REST API
→ [Start Building]
```

### Connect to Existing Database

```
./setup.sh
→ Project: "Analytics Dashboard"
→ Data Source: Legacy Database → PostgreSQL
→ Connection: postgresql://readonly:pass@db.company.com/analytics
→ [Test Connection] → Discovers 15 tables
→ Select: users, orders, products
→ [Start Building]
```

### Build from Command Line

```bash
./start.sh
> I want to build a realtime chat application with WebSocket support,
> user authentication, and message history stored in PostgreSQL.
```

## Troubleshooting

### Monty exits immediately

Make sure Claude Code CLI is installed:
```bash
npm install -g @anthropic-ai/claude-code
```

### Connection test fails

For database connections, ensure:
- The database is accessible from your machine
- Credentials are correct
- Required Python packages are installed: `pip install asyncpg aiohttp`

### Monty gets stuck

Check the logs:
```bash
cat scripts/monty.log
cat scripts/progress.txt
```

If blocked, Monty outputs `<monty>BLOCKED</monty>` with an explanation.

## License

MIT
