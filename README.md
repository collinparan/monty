# Monty

> "Excellent..." - C. Montgomery Burns

An autonomous AI coding loop for building Dockerized Python/FastAPI middlewares with PostgreSQL/PGVector and Neo4j/Graphiti. Based on Geoffrey Huntley's [Ralph](https://x.com/ryancarson) concept, renamed in honor of Monty Python and Montgomery Burns.

---

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [User Stories](#user-stories)
- [Frontend Options](#frontend-options)
- [How It Works](#how-it-works)
- [File Reference](#file-reference)
- [Docker Services](#docker-services)
- [Monitoring](#monitoring)
- [Writing Good Stories](#writing-good-stories)
- [When NOT to Use Monty](#when-not-to-use-monty)
- [Troubleshooting](#troubleshooting)

---

## Overview

Monty runs your AI coding agent (Claude Code, Amp, or Cursor) in a loop, completing user stories one at a time while you sleep. Each iteration:

1. Reads the task list from `prd.json`
2. Picks the highest priority pending story
3. Implements it
4. Runs verification (ruff, pyright, pytest)
5. Commits if passing
6. Marks story complete
7. Logs learnings to `progress.txt`
8. Repeats until done

Memory persists between iterations via:
- **Git commits** - Code changes
- **prd.json** - Task status
- **progress.txt** - Learnings and patterns

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **API Framework** | FastAPI | Async Python web framework |
| **Vector Database** | PostgreSQL + PGVector | Relational data + embeddings |
| **Graph Database** | Neo4j + Graphiti | Temporal knowledge graph |
| **Orchestration** | Docker Compose | Container management |
| **Reverse Proxy** | Nginx | External entrypoints, WebSocket support |
| **Voice AI** | ElevenLabs or Chatterbox | Text-to-speech (optional) |
| **Chat UI** | Static HTML/JS | SSE, WebSocket, REST clients (optional) |

---

## Architecture

```
                         ┌──────────────────────┐
                         │       Nginx          │
                         │     :80 / :443       │
                         └──────────┬───────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
         ▼                          ▼                          ▼
    /webhooks                  /ws/*  /stream/*              /chat
    (ElevenLabs)               (Real-time)                (Static UI)
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    │
                         ┌──────────▼───────────┐
                         │      FastAPI         │
                         │       :8000          │
                         └─────┬─────────┬──────┘
                               │         │
              ┌────────────────▼──┐  ┌───▼────────────────┐
              │    PostgreSQL     │  │      Neo4j         │
              │    + PGVector     │  │    + Graphiti      │
              │      :5432        │  │   :7474 / :7687    │
              └───────────────────┘  └────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │     Chatterbox      │  (optional, GPU)
                    │   Self-hosted TTS   │
                    │       :8001         │
                    └─────────────────────┘
```

---

## Quick Start

1. Clone this repo and cd into it:
   ```bash
   git clone git@github.com:collinparan/monty.git
   cd monty
   ```

2. Launch Claude Code:
   ```bash
   claude --dangerously-skip-permissions
   ```

3. Tell Claude what you want to build:
   ```
   I want to build a realtime chat API with WebSocket support
   ```

Claude reads `scripts/prompt.md` for patterns, creates user stories in `scripts/prd.json`, and starts implementing. You just have a conversation about what you're building.

---

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# PostgreSQL
POSTGRES_DB=ai_shs_voice
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/ai_shs_voice

# Neo4j
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_URI=bolt://neo4j:7687

# OpenAI (for embeddings)
OPENAI_API_KEY=sk-...

# ElevenLabs (if using hosted voice AI)
ELEVENLABS_API_KEY=...
ELEVENLABS_WEBHOOK_SECRET=...

# Chatterbox (if using self-hosted TTS)
CHATTERBOX_URL=http://chatterbox:8001

# Application
LOG_LEVEL=INFO
```

### Agent Selection

| Agent | Command | Environment Variable |
|-------|---------|---------------------|
| Claude Code | `claude --dangerously-skip-permissions` | `MONTY_AGENT=claude` |
| Amp | `amp --dangerously-allow-all` | `MONTY_AGENT=amp` |
| Cursor | `cursor --agent` | `MONTY_AGENT=cursor` |

---

## User Stories

The `prd.json` file contains user stories organized by priority. Stories are grouped into:

### Core Backend (US-001 to US-012)
Always included. Builds the FastAPI + PGVector + Graphiti foundation.

| ID | Title |
|----|-------|
| US-001 | Docker Compose infrastructure setup |
| US-002 | Project scaffolding with FastAPI |
| US-003 | PostgreSQL + PGVector database setup |
| US-004 | Neo4j + Graphiti service setup |
| US-005 | Transcript model with embeddings |
| US-006 | Embedding service |
| US-007 | Semantic search endpoint |
| US-008 | Knowledge graph episode ingestion |
| US-009 | Knowledge graph search |
| US-010 | SSE streaming endpoint |
| US-011 | WebSocket endpoint |
| US-012 | Hybrid RAG service |

### Optional: ElevenLabs Voice (VF-001 to VF-002)
For hosted voice AI with webhook integration.

| ID | Title |
|----|-------|
| VF-001 | ElevenLabs webhook receiver |
| VF-002 | ElevenLabs outbound call API |

### Optional: Chatterbox TTS (CB-001 to CB-002)
For self-hosted text-to-speech with GPU.

| ID | Title |
|----|-------|
| CB-001 | Chatterbox TTS service container |
| CB-002 | TTS proxy endpoint in main app |

### Optional: Static Chatbot UI (CF-001 to CF-003)
For traditional text chat interface.

| ID | Title |
|----|-------|
| CF-001 | Static frontend scaffolding |
| CF-002 | Nginx static hosting config |
| CF-003 | Chat conversation API |

**To customize:** Delete story sections you don't need from `prd.json`.

---

## Frontend Options

### Option 1: Voice AI with ElevenLabs (Hosted)

Webhook-driven integration with ElevenLabs conversational AI.

**Features:**
- Receives conversation events via webhooks
- Signature verification for security
- Stores transcripts with embeddings
- Triggers knowledge graph ingestion

**Stories to include:** VF-001, VF-002

---

### Option 2: Voice AI with Chatterbox (Self-Hosted)

Open-source TTS from Resemble AI. Requires NVIDIA GPU.

**Features:**
- 350M parameter Turbo model
- Zero-shot voice cloning
- Paralinguistic tags: `[laugh]`, `[cough]`, `[chuckle]`, `[sigh]`
- Sub-200ms latency

**Stories to include:** CB-001, CB-002

**Start with GPU:**
```bash
docker-compose -f docker/docker-compose.yml --profile gpu up -d
```

**Usage in code:**
```python
# Supports expressive tags
text = "Hi there! [chuckle] How can I help you today?"
audio = await tts_service.synthesize(text, voice_ref="customer_service.wav")
```

---

### Option 3: Static Chatbot UI

Plain HTML/JavaScript frontend with multiple transport options.

**Transport Options:**

| Transport | Direction | Use Case |
|-----------|-----------|----------|
| WebSocket | Bidirectional | Real-time chat with typing indicators |
| SSE | Server → Client | Streaming responses |
| REST | Request/Response | Simple polling, widest compatibility |

**Stories to include:** CF-001, CF-002, CF-003

**Files created:**
```
frontend/
├── index.html      # Chat interface
├── css/
│   └── styles.css  # Dark mode UI
└── js/
    ├── api.js      # REST client
    ├── websocket.js # WebSocket with reconnection
    ├── sse.js      # Server-Sent Events
    └── app.js      # Main application logic
```

---

### Option 4: No Frontend

Backend API only for integration with existing systems.

**Action:** Remove all VF-*, CB-*, and CF-* stories from `prd.json`.

---

## How It Works

### Memory Between Iterations

Monty has no memory between iterations. Context persists through:

| File | Purpose |
|------|---------|
| `prd.json` | Task list with status (pending/complete) |
| `progress.txt` | Learnings, patterns, gotchas |
| Git history | Code changes and commits |

### The Codebase Patterns Section

The top of `progress.txt` contains a **Codebase Patterns** section. Monty reads this first each iteration, so learnings compound:

```markdown
## Codebase Patterns
- Database names: Always prefix with `ai_`
- PGVector: Use HNSW index for cosine similarity
- Graphiti: Episodes need reference_time for temporal queries
- FastAPI: Use BackgroundTasks for webhook processing
```

By story 10, Monty knows patterns from stories 1-9.

### Verification Checks

Before committing, Monty runs:

```bash
ruff check . --fix    # Linting
ruff format .         # Formatting
pyright               # Type checking
pytest -x -q          # Tests (stop on first failure)
python -c "from app.main import app"  # App imports cleanly
```

### Stop Conditions

- `<monty>COMPLETE</monty>` - All stories done, exit 0
- `<monty>BLOCKED</monty>` - Needs human help, exit 2
- Max iterations reached - exit 1

---

## File Reference

### scripts/monty/

| File | Purpose |
|------|---------|
| `monty.sh` | Main bash loop |
| `prompt.md` | Agent instructions with all patterns |
| `prd.json` | User stories and status |
| `progress.txt` | Learnings and patterns |
| `monty.log` | Execution log |

### docker/

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Container orchestration |
| `Dockerfile` | FastAPI app container |
| `nginx/nginx.conf` | Reverse proxy config |
| `chatterbox/Dockerfile` | TTS container (optional) |
| `chatterbox/tts_server.py` | TTS FastAPI wrapper (optional) |

### app/

| File | Purpose |
|------|---------|
| `main.py` | FastAPI application entry |
| `config.py` | Pydantic Settings |
| `database.py` | SQLAlchemy async + PGVector |
| `models/` | SQLAlchemy models |
| `schemas/` | Pydantic schemas |
| `routers/` | API route modules |
| `services/embeddings.py` | OpenAI embedding generation |
| `services/graph.py` | Graphiti operations |
| `services/rag.py` | Hybrid RAG (vector + graph) |
| `services/tts.py` | Chatterbox wrapper (optional) |

---

## Docker Services

### Start Services

```bash
# Core services
docker-compose -f docker/docker-compose.yml up -d

# With GPU (Chatterbox)
docker-compose -f docker/docker-compose.yml --profile gpu up -d

# Rebuild after changes
docker-compose -f docker/docker-compose.yml up -d --build app
```

### Service Ports

| Service | Port | Purpose |
|---------|------|---------|
| Nginx | 80, 443 | External access |
| FastAPI | 8000 | API (internal) |
| PostgreSQL | 5432 | Database |
| Neo4j Browser | 7474 | Graph visualization |
| Neo4j Bolt | 7687 | Graph queries |
| Chatterbox | 8001 | TTS (optional) |

### Useful Commands

```bash
# View logs
docker-compose -f docker/docker-compose.yml logs -f app

# Shell into app container
docker-compose -f docker/docker-compose.yml exec app bash

# Run migrations
docker-compose -f docker/docker-compose.yml exec app alembic upgrade head

# Access Neo4j browser
open http://localhost:7474
```

---

## Monitoring

### Watch Story Progress

```bash
# Requires jq
watch -n 5 "jq '.userStories[] | {id, title, status}' scripts/monty/prd.json"
```

### Tail Monty Log

```bash
tail -f scripts/monty/monty.log
```

### Check Git Commits

```bash
git log --oneline -15
```

### View Learnings

```bash
cat scripts/monty/progress.txt
```

---

## Writing Good Stories

### ✅ Right Size (One Context Window)

```json
{
  "id": "US-005",
  "title": "Transcript model with embeddings",
  "acceptanceCriteria": [
    "ConversationTranscript model with Vector(1536)",
    "Fields: id, conversation_id, speaker, content, embedding, timestamp",
    "Alembic migration with HNSW index",
    "Model exports from app/models/__init__.py"
  ],
  "priority": 5,
  "status": "pending"
}
```

### ❌ Too Big

```json
{
  "title": "Build complete RAG system with semantic search and knowledge graph"
}
```

### Key Principles

1. **Explicit acceptance criteria** - Monty needs clear pass/fail conditions
2. **One concern per story** - Don't mix API + database + frontend
3. **Include test requirements** - "pytest passes" is implicit
4. **Small enough for one context window** - If you can't describe it in 5-7 bullet points, split it

---

## When NOT to Use Monty

| Scenario | Why |
|----------|-----|
| Exploratory/research work | No clear acceptance criteria |
| Major architectural refactors | Too large for single stories |
| Security-critical code | Needs human review |
| Complex schema design | Requires human judgment |
| Production deployment configs | High risk of errors |
| Work requiring external credentials you haven't set up | Will block immediately |

---

## Troubleshooting

### Monty is stuck on the same story

1. Check `progress.txt` for error patterns
2. Look at `monty.log` for agent output
3. Run verification manually:
   ```bash
   ruff check . && pyright && pytest
   ```
4. Story might be too large - split it

### Docker services not detected

```bash
# Check running containers
docker ps

# Start services
docker-compose -f docker/docker-compose.yml up -d

# Check logs
docker-compose -f docker/docker-compose.yml logs postgres neo4j
```

### Agent command not found

```bash
# Claude Code
npm install -g @anthropic-ai/claude-code

# Amp
npm install -g @anthropic-ai/amp

# Verify
which claude  # or amp, cursor
```

### PGVector extension error

```sql
-- Connect to database and run:
CREATE EXTENSION IF NOT EXISTS vector;
```

### Neo4j connection refused

```bash
# Check Neo4j is healthy
docker-compose -f docker/docker-compose.yml logs neo4j

# Wait for startup (can take 30-60s)
docker-compose -f docker/docker-compose.yml exec neo4j cypher-shell -u neo4j -p password "RETURN 1"
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All stories complete |
| 1 | Max iterations reached |
| 2 | Blocked (needs human intervention) |
| 130 | Interrupted by user (Ctrl+C) |

---

## Credits

- Based on [Ralph](https://github.com/snarktank/ai-dev-tasks) by Geoffrey Huntley
- Named after Monty Python and C. Montgomery Burns 
- Built for AI initiatives
