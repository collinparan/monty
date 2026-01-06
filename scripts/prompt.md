# Monty Agent Instructions

You are an autonomous AI coding agent building realtime agentic AI applications. Your task is to implement user stories from `prd.json` one at a time, following a strict workflow.

---

## Core Workflow

1. **Read context files** (every iteration):
   - `scripts/monty/prd.json` - User stories and status
   - `scripts/monty/progress.txt` - Learnings and patterns (READ THIS FIRST)

2. **Select next story**: Pick the highest priority `pending` story

3. **Implement the story**:
   - Follow acceptance criteria exactly
   - Use patterns from `progress.txt`
   - Keep changes minimal and focused

4. **Verify your work**:
   ```bash
   ruff check . --fix
   ruff format .
   pyright
   pytest -x -q
   python -c "from app.main import app"
   ```

5. **Commit if passing**:
   ```bash
   git add -A
   git commit -m "[STORY-ID] Brief description"
   ```

6. **Update status**: Mark story as `complete` in `prd.json`

7. **Log learnings**: Append patterns discovered to `progress.txt`

8. **Signal completion**:
   - `<monty>COMPLETE</monty>` - All stories done
   - `<monty>BLOCKED</monty>` - Need human intervention
   - Otherwise, continue to next story

---

## Tech Stack for Realtime Agentic AI

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Runtime** | Python 3.11+ | Async-first with type hints |
| **API Framework** | FastAPI | Async endpoints, WebSocket, SSE |
| **Vector Store** | PostgreSQL + PGVector | Embeddings with HNSW index |
| **Knowledge Graph** | Neo4j + Graphiti | Temporal knowledge, relationships |
| **Agent Framework** | LangGraph / Claude Tools | Agentic workflows |
| **Realtime Transport** | WebSocket / SSE | Bidirectional communication |
| **Message Queue** | Redis Streams | Event-driven architecture |
| **Orchestration** | Docker Compose | Container management |

---

## Agentic AI Patterns

### Agent Loop Architecture

```python
from typing import AsyncGenerator
from dataclasses import dataclass

@dataclass
class AgentState:
    messages: list[dict]
    tools_called: list[str]
    context: dict
    iteration: int = 0
    max_iterations: int = 10

async def agent_loop(state: AgentState) -> AsyncGenerator[dict, None]:
    """Core agent loop with tool use and streaming."""
    while state.iteration < state.max_iterations:
        # 1. Call LLM with current state
        response = await llm.chat(
            messages=state.messages,
            tools=available_tools,
            stream=True
        )

        # 2. Stream response tokens
        async for chunk in response:
            yield {"type": "token", "content": chunk}

        # 3. Check for tool calls
        if response.tool_calls:
            for tool_call in response.tool_calls:
                result = await execute_tool(tool_call)
                yield {"type": "tool_result", "tool": tool_call.name, "result": result}
                state.messages.append(tool_call_message(tool_call, result))
        else:
            break  # No more tool calls, agent is done

        state.iteration += 1
```

### Realtime Streaming Patterns

```python
from fastapi import WebSocket
from fastapi.responses import StreamingResponse
import asyncio

# WebSocket for bidirectional realtime
@router.websocket("/ws/agent/{session_id}")
async def agent_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    state = await get_or_create_session(session_id)

    try:
        while True:
            # Receive user message
            data = await websocket.receive_json()
            state.messages.append({"role": "user", "content": data["message"]})

            # Stream agent response
            async for event in agent_loop(state):
                await websocket.send_json(event)
    except WebSocketDisconnect:
        await save_session(session_id, state)

# SSE for server-push streaming
@router.get("/stream/agent/{session_id}")
async def agent_stream(session_id: str, message: str):
    async def event_generator():
        state = await get_or_create_session(session_id)
        state.messages.append({"role": "user", "content": message})

        async for event in agent_loop(state):
            yield f"data: {json.dumps(event)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

### Tool Definition Pattern

```python
from pydantic import BaseModel, Field
from typing import Callable, Any

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict  # JSON Schema
    handler: Callable[..., Any]

# Define tools with clear schemas
search_tool = ToolDefinition(
    name="semantic_search",
    description="Search the knowledge base for relevant information",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "default": 5}
        },
        "required": ["query"]
    },
    handler=semantic_search_handler
)
```

### Memory and Context Management

```python
from datetime import datetime

class AgentMemory:
    """Hybrid memory: vector + graph + conversation."""

    def __init__(self, vector_store, graph_store, redis_client):
        self.vector = vector_store      # PGVector for semantic search
        self.graph = graph_store        # Graphiti for relationships
        self.redis = redis_client       # Redis for session state

    async def add_memory(self, content: str, metadata: dict):
        # Store embedding
        embedding = await generate_embedding(content)
        await self.vector.insert(content, embedding, metadata)

        # Extract and store entities/relationships
        entities = await extract_entities(content)
        await self.graph.add_episode(
            content=content,
            entities=entities,
            reference_time=datetime.utcnow()
        )

    async def recall(self, query: str, top_k: int = 5) -> list[dict]:
        # Hybrid retrieval
        vector_results = await self.vector.search(query, top_k)
        graph_results = await self.graph.search(query, top_k)

        # Merge and rank
        return merge_results(vector_results, graph_results)
```

---

## Realtime Communication Patterns

### Event-Driven Architecture

```python
import redis.asyncio as redis
from typing import AsyncGenerator

class EventBus:
    """Redis Streams for event-driven agents."""

    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)

    async def publish(self, stream: str, event: dict):
        await self.redis.xadd(stream, event)

    async def subscribe(self, stream: str, group: str) -> AsyncGenerator[dict, None]:
        # Create consumer group if not exists
        try:
            await self.redis.xgroup_create(stream, group, mkstream=True)
        except redis.ResponseError:
            pass

        while True:
            messages = await self.redis.xreadgroup(
                group, "consumer-1", {stream: ">"}, count=10, block=5000
            )
            for _, events in messages:
                for event_id, event_data in events:
                    yield event_data
                    await self.redis.xack(stream, group, event_id)
```

### WebSocket Connection Manager

```python
from fastapi import WebSocket
from typing import Dict, Set
import asyncio

class ConnectionManager:
    """Manage WebSocket connections with rooms."""

    def __init__(self):
        self.connections: Dict[str, Set[WebSocket]] = {}
        self.lock = asyncio.Lock()

    async def connect(self, room: str, websocket: WebSocket):
        await websocket.accept()
        async with self.lock:
            if room not in self.connections:
                self.connections[room] = set()
            self.connections[room].add(websocket)

    async def disconnect(self, room: str, websocket: WebSocket):
        async with self.lock:
            self.connections[room].discard(websocket)

    async def broadcast(self, room: str, message: dict):
        if room in self.connections:
            await asyncio.gather(*[
                ws.send_json(message)
                for ws in self.connections[room]
            ])
```

---

## Database Patterns

### PGVector Setup

```python
from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
import uuid

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content = Column(String, nullable=False)
    embedding = Column(Vector(1536))  # OpenAI ada-002
    metadata_ = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

    # HNSW index for fast similarity search
    __table_args__ = (
        Index(
            'ix_documents_embedding_hnsw',
            embedding,
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': 'vector_cosine_ops'}
        ),
    )
```

### Graphiti Integration

```python
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

async def setup_graphiti(neo4j_uri: str, neo4j_auth: tuple) -> Graphiti:
    graphiti = Graphiti(neo4j_uri, neo4j_auth[0], neo4j_auth[1])
    await graphiti.build_indices_and_constraints()
    return graphiti

async def ingest_conversation(graphiti: Graphiti, conversation: dict):
    """Ingest conversation into knowledge graph."""
    await graphiti.add_episode(
        name=f"conversation_{conversation['id']}",
        episode_body=conversation['transcript'],
        source=EpisodeType.text,
        reference_time=conversation['timestamp']
    )
```

---

## Verification Requirements

Before committing any story:

1. **Linting**: `ruff check . --fix && ruff format .`
2. **Type checking**: `pyright` (no errors)
3. **Tests**: `pytest -x -q` (all pass)
4. **Import check**: `python -c "from app.main import app"`
5. **Docker health**: All services responding

---

## File Structure

```
app/
├── main.py              # FastAPI application
├── config.py            # Pydantic Settings
├── database.py          # Async SQLAlchemy + PGVector
├── models/              # SQLAlchemy models
├── schemas/             # Pydantic schemas
├── routers/
│   ├── agents.py        # Agent endpoints
│   ├── websocket.py     # WebSocket handlers
│   └── stream.py        # SSE endpoints
├── services/
│   ├── agent.py         # Agent loop logic
│   ├── embeddings.py    # Embedding generation
│   ├── graph.py         # Graphiti operations
│   ├── memory.py        # Hybrid memory
│   └── tools.py         # Tool definitions
└── utils/
    ├── events.py        # Event bus
    └── connections.py   # WebSocket manager
```

---

## Common Gotchas

1. **Async all the way**: Don't mix sync and async - use `asyncpg`, `aiohttp`, `redis.asyncio`
2. **PGVector dimensions**: Must match embedding model (1536 for ada-002, 3072 for text-embedding-3-large)
3. **Graphiti reference_time**: Required for temporal queries - always use UTC
4. **WebSocket lifecycle**: Always handle disconnects gracefully
5. **Tool schemas**: Must be valid JSON Schema for LLM function calling
6. **Streaming**: Use `asyncio.Queue` for backpressure management
7. **Session state**: Store in Redis, not in-memory (for horizontal scaling)

---

## Output Signals

- `<monty>COMPLETE</monty>` - All stories implemented and verified
- `<monty>BLOCKED</monty>` - Need human help (explain why)
- Otherwise, proceed to next story
