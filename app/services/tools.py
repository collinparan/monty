"""Tool definition framework for AI agent.

Provides a registry pattern for defining and executing agent tools with JSON Schema
parameter validation and async execution support.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolParameter(BaseModel):
    """Individual parameter definition for a tool."""

    name: str
    type: str  # JSON Schema type: string, number, integer, boolean, array, object
    description: str
    required: bool = False
    default: Optional[Any] = None
    enum: Optional[list[Any]] = None


class ToolDefinition(BaseModel):
    """Definition of an agent tool with JSON Schema parameters.

    Tools are callable functions that the AI agent can invoke to perform
    specific operations like querying data, making predictions, etc.
    """

    name: str = Field(..., description="Unique tool name")
    description: str = Field(..., description="Human-readable description for the LLM")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for parameters"
    )
    handler: Optional[Callable[..., Any]] = Field(
        default=None, exclude=True, description="Async function to execute"
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def to_llm_schema(self) -> dict[str, Any]:
        """Export tool definition in LLM function calling format.

        Returns format compatible with Claude/OpenAI function calling.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def to_openai_schema(self) -> dict[str, Any]:
        """Export tool definition in OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolError(Exception):
    """Error raised during tool execution."""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        self.message = message
        super().__init__(f"Tool '{tool_name}' error: {message}")


class ToolResult(BaseModel):
    """Result of tool execution."""

    tool_name: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None


class ToolRegistry:
    """Registry for managing and executing agent tools.

    The registry provides:
    - Tool registration with validation
    - Tool lookup by name
    - Async tool execution with error handling
    - Schema export for LLM function calling
    """

    def __init__(self):
        """Initialize empty tool registry."""
        self._tools: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, Callable[..., Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Any],
    ) -> ToolDefinition:
        """Register a new tool.

        Args:
            name: Unique tool identifier
            description: Human-readable description for the LLM
            parameters: JSON Schema for tool parameters
            handler: Async function to execute

        Returns:
            The registered ToolDefinition

        Raises:
            ValueError: If tool name already registered
        """
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered")

        tool = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
        )

        self._tools[name] = tool
        self._handlers[name] = handler

        logger.info(f"Registered tool: {name}")
        return tool

    def register_tool(self, tool: ToolDefinition, handler: Callable[..., Any]) -> None:
        """Register a pre-defined ToolDefinition with its handler.

        Args:
            tool: The tool definition
            handler: Async function to execute
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")

        self._tools[tool.name] = tool
        self._handlers[tool.name] = handler

        logger.info(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> bool:
        """Unregister a tool by name.

        Args:
            name: Tool name to unregister

        Returns:
            True if tool was unregistered, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            del self._handlers[name]
            logger.info(f"Unregistered tool: {name}")
            return True
        return False

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name.

        Args:
            name: Tool name

        Returns:
            ToolDefinition if found, None otherwise
        """
        return self._tools.get(name)

    def get_handler(self, name: str) -> Optional[Callable[..., Any]]:
        """Get a tool handler by name.

        Args:
            name: Tool name

        Returns:
            Handler function if found, None otherwise
        """
        return self._handlers.get(name)

    def list_tools(self) -> list[str]:
        """Get list of all registered tool names."""
        return list(self._tools.keys())

    def get_all_tools(self) -> list[ToolDefinition]:
        """Get all registered tool definitions."""
        return list(self._tools.values())

    async def execute(
        self,
        name: str,
        arguments: Optional[dict[str, Any]] = None,
    ) -> ToolResult:
        """Execute a tool by name with given arguments.

        Args:
            name: Tool name to execute
            arguments: Dictionary of arguments to pass to handler

        Returns:
            ToolResult with success status and result or error
        """
        import time

        start_time = time.perf_counter()

        if name not in self._tools:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Tool '{name}' not found",
            )

        handler = self._handlers.get(name)
        if not handler:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"No handler registered for tool '{name}'",
            )

        arguments = arguments or {}

        try:
            # Execute handler (supports both sync and async)
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**arguments)
            else:
                # Run sync handler in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: handler(**arguments))

            execution_time = (time.perf_counter() - start_time) * 1000

            return ToolResult(
                tool_name=name,
                success=True,
                result=result,
                execution_time_ms=round(execution_time, 2),
            )

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            logger.exception(f"Tool '{name}' execution failed: {e}")

            return ToolResult(
                tool_name=name,
                success=False,
                error=str(e),
                execution_time_ms=round(execution_time, 2),
            )

    def export_for_llm(self) -> list[dict[str, Any]]:
        """Export all tools in LLM function calling format.

        Returns format compatible with Claude API tool use.
        """
        return [tool.to_llm_schema() for tool in self._tools.values()]

    def export_for_openai(self) -> list[dict[str, Any]]:
        """Export all tools in OpenAI function calling format."""
        return [tool.to_openai_schema() for tool in self._tools.values()]


def tool(
    name: str,
    description: str,
    parameters: Optional[dict[str, Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for marking a function as a tool.

    Usage:
        @tool("my_tool", "Description of my tool", {...parameters...})
        async def my_tool(arg1: str, arg2: int) -> dict:
            ...

    Args:
        name: Unique tool identifier
        description: Human-readable description
        parameters: JSON Schema for parameters (optional)

    Returns:
        Decorated function with tool metadata
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func._tool_name = name  # type: ignore[attr-defined]
        func._tool_description = description  # type: ignore[attr-defined]
        func._tool_parameters = parameters or {  # type: ignore[attr-defined]
            "type": "object",
            "properties": {},
            "required": [],
        }
        return func

    return decorator


def create_json_schema(
    properties: dict[str, dict[str, Any]],
    required: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Helper to create JSON Schema for tool parameters.

    Args:
        properties: Dictionary of property name to schema
        required: List of required property names

    Returns:
        JSON Schema object

    Example:
        schema = create_json_schema(
            properties={
                "region": {"type": "string", "description": "Region code"},
                "limit": {"type": "integer", "description": "Max results", "default": 10}
            },
            required=["region"]
        )
    """
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
    }


# Global registry instance (can be used as singleton or create new instances)
_global_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get or create the global tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry
