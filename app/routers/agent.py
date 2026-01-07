"""REST and streaming endpoints for AI agent interaction."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Optional
from uuid import uuid4

import redis.asyncio as redis
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.main import get_redis
from app.services.agent import AgentEvent, AgentState, create_agent_service
from app.services.agent_tools import create_technician_analytics_tools

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/v1/agent", tags=["Agent"])


# Session storage TTL (1 hour)
SESSION_TTL_SECONDS = 3600


# Request/Response schemas


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""

    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    session_id: Optional[str] = Field(
        default=None, description="Session ID for conversation continuity"
    )


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""

    response: str = Field(..., description="Agent response text")
    session_id: str = Field(..., description="Session ID for follow-up messages")
    tools_called: list[str] = Field(default_factory=list, description="Tools invoked")
    iterations: int = Field(..., description="Agent iterations used")


class StreamRequest(BaseModel):
    """Request schema for stream endpoint (via query params)."""

    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None


class ExamplePrompt(BaseModel):
    """Example prompt for the agent."""

    title: str
    prompt: str
    category: str


# Example prompts for common queries
EXAMPLE_PROMPTS = [
    ExamplePrompt(
        title="High-risk technicians",
        prompt="Show me all high-risk technicians in the US-WEST region",
        category="Risk Analysis",
    ),
    ExamplePrompt(
        title="Retention prediction",
        prompt="What is the retention prediction for technician TECH-001?",
        category="Predictions",
    ),
    ExamplePrompt(
        title="Forecast headcount",
        prompt="What is the 90-day headcount forecast for all regions?",
        category="Forecasting",
    ),
    ExamplePrompt(
        title="ROI analysis",
        prompt="Calculate the ROI if we invest $500 per technician to improve retention by 20%",
        category="ROI",
    ),
    ExamplePrompt(
        title="Feature importance",
        prompt="What are the top factors affecting technician retention?",
        category="Insights",
    ),
    ExamplePrompt(
        title="Regional summary",
        prompt="Give me a summary of technician metrics by region",
        category="Regional",
    ),
]


# Helper functions


async def get_or_create_session(
    session_id: Optional[str], redis_client: redis.Redis
) -> tuple[str, AgentState]:
    """Get existing session state or create new one.

    Args:
        session_id: Optional session ID
        redis_client: Redis client

    Returns:
        Tuple of (session_id, state)
    """
    if session_id:
        # Try to load existing session
        session_key = f"agent:session:{session_id}"
        session_data = await redis_client.get(session_key)

        if session_data:
            try:
                data = json.loads(session_data)
                state = AgentState(
                    messages=data.get("messages", []),
                    tools_called=data.get("tools_called", []),
                    context=data.get("context", {}),
                    iteration=0,  # Reset iteration for new message
                    max_iterations=settings.agent_max_iterations,
                )
                return session_id, state
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load session {session_id}: {e}")

    # Create new session
    new_session_id = str(uuid4())
    state = AgentState(max_iterations=settings.agent_max_iterations)
    return new_session_id, state


async def save_session(session_id: str, state: AgentState, redis_client: redis.Redis) -> None:
    """Save session state to Redis.

    Args:
        session_id: Session ID
        state: Agent state to save
        redis_client: Redis client
    """
    session_key = f"agent:session:{session_id}"
    session_data = {
        "messages": state.messages,
        "tools_called": state.tools_called,
        "context": state.context,
    }
    await redis_client.setex(session_key, SESSION_TTL_SECONDS, json.dumps(session_data))


def get_db_session_factory(session: AsyncSession):
    """Create a database session factory for the agent tools.

    Args:
        session: The SQLAlchemy async session

    Returns:
        Async context manager that yields the session
    """

    @asynccontextmanager
    async def session_factory():
        yield session

    return session_factory


# Endpoints


@router.post("/chat", response_model=ChatResponse)
async def agent_chat(
    request: ChatRequest,
    session: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis),
) -> ChatResponse:
    """Synchronous chat endpoint for agent interaction.

    Sends a message to the agent and waits for the complete response.
    Useful for simple queries where streaming isn't needed.
    """
    # Get or create session
    session_id, state = await get_or_create_session(request.session_id, redis_client)

    # Create tool registry with database access
    db_factory = get_db_session_factory(session)
    tool_registry = create_technician_analytics_tools(db_factory)

    # Create agent service
    agent_service = create_agent_service(tool_registry)

    try:
        # Run agent
        response_text, final_state = await agent_service.chat(request.message, state)

        # Save session
        await save_session(session_id, final_state, redis_client)

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            tools_called=final_state.tools_called,
            iterations=final_state.iteration,
        )

    except Exception as e:
        logger.exception(f"Agent chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent error: {e!s}",
        ) from e


@router.get("/stream")
async def agent_stream(
    message: str = Query(..., min_length=1, max_length=10000, description="User message"),
    session_id: Optional[str] = Query(default=None, description="Session ID"),
    session: AsyncSession = Depends(get_db),
    redis_client: redis.Redis = Depends(get_redis),
) -> StreamingResponse:
    """Server-Sent Events streaming endpoint for agent interaction.

    Streams agent responses as they are generated, including:
    - Token events for incremental text
    - Tool call events when agent uses tools
    - Tool result events with tool outputs
    - Done event when complete
    """
    # Get or create session
    session_id, state = await get_or_create_session(session_id, redis_client)

    # Create tool registry with database access
    db_factory = get_db_session_factory(session)
    tool_registry = create_technician_analytics_tools(db_factory)

    # Create agent service
    agent_service = create_agent_service(tool_registry)

    async def event_generator():
        """Generate SSE events from agent run."""
        try:
            state.add_user_message(message)

            # Send session ID first
            yield AgentEvent(
                type="session",
                data={"session_id": session_id},
            ).to_sse()

            # Run agent and stream events
            async for event in agent_service.run(state):
                yield event.to_sse()

            # Save session after completion
            await save_session(session_id, state, redis_client)

            # Send final DONE marker
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.exception(f"Agent stream error: {e}")
            yield AgentEvent(type="error", data=str(e)).to_sse()
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    redis_client: redis.Redis = Depends(get_redis),
) -> dict:
    """Get session information and conversation history.

    Useful for debugging or resuming conversations.
    """
    session_key = f"agent:session:{session_id}"
    session_data = await redis_client.get(session_key)

    if not session_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    try:
        data = json.loads(session_data)
        return {
            "session_id": session_id,
            "message_count": len(data.get("messages", [])),
            "tools_called": data.get("tools_called", []),
            "messages": data.get("messages", []),
        }
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to parse session data",
        )


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    redis_client: redis.Redis = Depends(get_redis),
) -> dict:
    """Delete a session and its conversation history."""
    session_key = f"agent:session:{session_id}"
    deleted = await redis_client.delete(session_key)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    return {"status": "deleted", "session_id": session_id}


@router.get("/examples", response_model=list[ExamplePrompt])
async def get_example_prompts() -> list[ExamplePrompt]:
    """Get example prompts for common agent queries.

    These can be displayed in the UI as quick-start suggestions.
    """
    return EXAMPLE_PROMPTS
