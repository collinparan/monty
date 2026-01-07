"""Core agent loop with tool use and streaming for technician analytics.

This module implements the AI agent that can use tools to query data,
make predictions, and provide insights about technician analytics.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

import anthropic

from app.config import get_settings
from app.logging_config import get_logger, metrics
from app.services.tools import ToolRegistry, ToolResult

logger = get_logger(__name__)
settings = get_settings()


# Domain-specific system prompt for technician analytics
SYSTEM_PROMPT = """You are an AI assistant specialized in technician analytics for Sears Home Services.
You help operations managers understand technician recruitment, retention, and performance metrics.

Your capabilities include:
- Querying technician data by region, status, and risk level
- Getting retention/recruitment predictions with explanations
- Viewing time-series forecasts for headcount and job demand
- Calculating ROI for retention interventions
- Analyzing feature importance from ML models
- Summarizing regional performance

When providing insights:
- Always be data-driven and cite specific numbers when available
- Explain predictions in plain language (e.g., "A tenure of 30 days contributes 0.3 to the risk score...")
- Proactively highlight high-risk areas that need attention
- Suggest actionable recommendations based on the data
- If data is missing or a model isn't trained, explain what would be needed

When using tools:
- Use query_technicians to find technicians matching specific criteria
- Use get_prediction for individual technician risk assessments
- Use get_forecast for future trend projections
- Use calculate_roi to evaluate intervention strategies
- Use get_feature_importance to understand key drivers
- Use get_regional_summary for geographic analysis

Be concise but thorough. Focus on actionable insights."""


@dataclass
class AgentState:
    """State maintained across agent iterations.

    Attributes:
        messages: Conversation history
        tools_called: List of tools called in this conversation
        context: Additional context data
        iteration: Current iteration count
        max_iterations: Maximum allowed iterations
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0
    max_iterations: int = 10

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation."""
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_use(self, tool_id: str, tool_name: str, tool_input: dict) -> None:
        """Record a tool use in the conversation."""
        self.messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_name,
                        "input": tool_input,
                    }
                ],
            }
        )
        self.tools_called.append(tool_name)

    def add_tool_result(self, tool_id: str, result: str, is_error: bool = False) -> None:
        """Add a tool result to the conversation."""
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result,
                        "is_error": is_error,
                    }
                ],
            }
        )


@dataclass
class AgentEvent:
    """Event emitted by the agent loop.

    Attributes:
        type: Event type (token, tool_call, tool_result, done, error)
        data: Event data
        timestamp: Event timestamp
    """

    type: str  # token, tool_call, tool_result, done, error
    data: Any
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    def to_sse(self) -> str:
        """Format event for Server-Sent Events."""
        return f"data: {json.dumps(self.to_dict())}\n\n"


class AgentService:
    """Core agent service with tool use and streaming.

    The agent uses Claude API to process user queries and can invoke
    tools to query data, make predictions, and generate insights.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_iterations: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ):
        """Initialize the agent service.

        Args:
            tool_registry: Registry of available tools
            api_key: Anthropic API key (defaults to settings)
            model: Model to use (defaults to settings)
            max_tokens: Max tokens in response (defaults to settings)
            max_iterations: Max tool use iterations (defaults to settings)
            system_prompt: Custom system prompt (defaults to SYSTEM_PROMPT)
        """
        self.tool_registry = tool_registry
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.anthropic_model
        self.max_tokens = max_tokens or settings.anthropic_max_tokens
        self.max_iterations = max_iterations or settings.agent_max_iterations
        self.system_prompt = system_prompt or SYSTEM_PROMPT

        # Initialize Anthropic client
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key) if self.api_key else None

    async def chat(
        self,
        message: str,
        state: Optional[AgentState] = None,
    ) -> tuple[str, AgentState]:
        """Process a user message and return response (non-streaming).

        Args:
            message: User's message
            state: Existing conversation state (optional)

        Returns:
            Tuple of (response text, updated state)
        """
        if not self.client:
            return "Agent not configured: missing API key", state or AgentState()

        state = state or AgentState(max_iterations=self.max_iterations)
        state.add_user_message(message)

        response_text = ""

        async for event in self.run(state):
            if event.type == "token":
                response_text += event.data
            elif event.type == "done":
                break
            elif event.type == "error":
                response_text = f"Error: {event.data}"
                break

        return response_text, state

    async def run(
        self,
        state: AgentState,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Run the agent loop, yielding events as they occur.

        This is the core agent loop that:
        1. Calls Claude with the current state
        2. Yields token events as they stream in
        3. Detects and executes tool calls
        4. Continues until no more tool calls or max iterations

        Args:
            state: Current agent state

        Yields:
            AgentEvent objects for each significant event
        """
        if not self.client:
            yield AgentEvent(type="error", data="Agent not configured: missing API key")
            return

        # Get tool schemas for Claude
        tools = self.tool_registry.export_for_llm()

        agent_start_time = time.time()

        while state.iteration < state.max_iterations:
            state.iteration += 1

            try:
                # Call Claude API with streaming
                collected_content: list[dict] = []
                current_text = ""
                tool_uses: list[dict] = []

                async with self.client.messages.stream(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=self.system_prompt,
                    messages=state.messages,
                    tools=tools if tools else anthropic.NOT_GIVEN,
                ) as stream:
                    async for event in stream:
                        if event.type == "content_block_start":
                            if event.content_block.type == "text":
                                current_text = ""
                            elif event.content_block.type == "tool_use":
                                tool_uses.append(
                                    {
                                        "id": event.content_block.id,
                                        "name": event.content_block.name,
                                        "input": {},
                                    }
                                )

                        elif event.type == "content_block_delta":
                            if hasattr(event.delta, "text"):
                                current_text += event.delta.text
                                yield AgentEvent(type="token", data=event.delta.text)
                            elif hasattr(event.delta, "partial_json"):
                                # Accumulate tool input JSON
                                if tool_uses:
                                    # We'll get the full input after the block ends
                                    pass

                        elif event.type == "content_block_stop":
                            if current_text:
                                collected_content.append(
                                    {
                                        "type": "text",
                                        "text": current_text,
                                    }
                                )
                                current_text = ""

                    # Get the final message for tool inputs
                    final_message = await stream.get_final_message()

                # Extract tool uses from final message
                for content_block in final_message.content:
                    if content_block.type == "tool_use":
                        tool_uses_processed = False
                        for tu in tool_uses:
                            if tu["id"] == content_block.id:
                                tu["input"] = content_block.input
                                tool_uses_processed = True
                                break
                        if not tool_uses_processed:
                            tool_uses.append(
                                {
                                    "id": content_block.id,
                                    "name": content_block.name,
                                    "input": content_block.input,
                                }
                            )

                # If no tool calls, we're done
                if not tool_uses:
                    # Add final assistant message
                    if collected_content:
                        text_content = " ".join(
                            c["text"] for c in collected_content if c["type"] == "text"
                        )
                        state.add_assistant_message(text_content)
                    total_time_ms = (time.time() - agent_start_time) * 1000
                    metrics.increment("agent.requests")
                    metrics.record_timing("agent.total_time_ms", total_time_ms)
                    logger.info(
                        "agent_completed",
                        iterations=state.iteration,
                        total_time_ms=round(total_time_ms, 2),
                        tools_called=len(state.tools_called),
                    )
                    yield AgentEvent(type="done", data={"iterations": state.iteration})
                    return

                # Process tool calls
                for tool_use in tool_uses:
                    tool_name = tool_use["name"]
                    tool_input = tool_use["input"]
                    tool_id = tool_use["id"]

                    yield AgentEvent(
                        type="tool_call",
                        data={
                            "tool_id": tool_id,
                            "tool_name": tool_name,
                            "tool_input": tool_input,
                        },
                    )

                    # Execute the tool
                    result = await self.tool_registry.execute(tool_name, tool_input)

                    # Track tool metrics
                    metrics.increment(f"tool.{tool_name}.calls")
                    if result.execution_time_ms is not None:
                        metrics.record_timing(f"tool.{tool_name}.time_ms", result.execution_time_ms)

                    if result.success:
                        logger.info(
                            "tool_executed",
                            tool_name=tool_name,
                            execution_time_ms=result.execution_time_ms,
                        )
                    else:
                        metrics.increment(f"tool.{tool_name}.errors")
                        logger.warning(
                            "tool_failed",
                            tool_name=tool_name,
                            error=result.error,
                            execution_time_ms=result.execution_time_ms,
                        )

                    yield AgentEvent(
                        type="tool_result",
                        data={
                            "tool_id": tool_id,
                            "tool_name": tool_name,
                            "success": result.success,
                            "result": result.result if result.success else None,
                            "error": result.error if not result.success else None,
                            "execution_time_ms": result.execution_time_ms,
                        },
                    )

                    # Record in state
                    state.add_tool_use(tool_id, tool_name, tool_input)
                    state.add_tool_result(
                        tool_id,
                        json.dumps(result.result) if result.success else str(result.error),
                        is_error=not result.success,
                    )

            except anthropic.APIError as e:
                logger.exception(f"Anthropic API error: {e}")
                yield AgentEvent(type="error", data=f"API error: {e}")
                return
            except Exception as e:
                logger.exception(f"Agent error: {e}")
                yield AgentEvent(type="error", data=str(e))
                return

        # Max iterations reached
        yield AgentEvent(
            type="done",
            data={
                "iterations": state.iteration,
                "max_iterations_reached": True,
            },
        )

    async def run_sync(self, message: str, state: Optional[AgentState] = None) -> ToolResult:
        """Run agent synchronously and return final result.

        Useful for simple requests where streaming isn't needed.

        Args:
            message: User message
            state: Optional existing state

        Returns:
            ToolResult with the agent's response
        """
        start_time = time.time()
        response, final_state = await self.chat(message, state)
        execution_time = (time.time() - start_time) * 1000

        return ToolResult(
            tool_name="agent",
            success=True,
            result={
                "response": response,
                "tools_called": final_state.tools_called,
                "iterations": final_state.iteration,
            },
            execution_time_ms=round(execution_time, 2),
        )


def create_agent_service(tool_registry: ToolRegistry) -> AgentService:
    """Factory function to create an agent service with default settings.

    Args:
        tool_registry: Registry of available tools

    Returns:
        Configured AgentService instance
    """
    return AgentService(
        tool_registry=tool_registry,
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_model,
        max_tokens=settings.anthropic_max_tokens,
        max_iterations=settings.agent_max_iterations,
    )
