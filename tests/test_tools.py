"""Tests for the tool definition framework."""

from __future__ import annotations

import pytest

from app.services.tools import (
    ToolDefinition,
    ToolRegistry,
    ToolResult,
    create_json_schema,
    get_tool_registry,
    tool,
)


class TestToolDefinition:
    """Tests for ToolDefinition model."""

    def test_create_tool_definition(self):
        """Test creating a basic tool definition."""
        tool_def = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )

        assert tool_def.name == "test_tool"
        assert tool_def.description == "A test tool"
        assert tool_def.parameters["type"] == "object"
        assert "query" in tool_def.parameters["properties"]

    def test_to_llm_schema(self):
        """Test export to LLM function calling format."""
        tool_def = ToolDefinition(
            name="search",
            description="Search for data",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        )

        schema = tool_def.to_llm_schema()

        assert schema["name"] == "search"
        assert schema["description"] == "Search for data"
        assert schema["input_schema"]["type"] == "object"

    def test_to_openai_schema(self):
        """Test export to OpenAI function calling format."""
        tool_def = ToolDefinition(
            name="calculate",
            description="Perform calculations",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
            },
        )

        schema = tool_def.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "calculate"
        assert schema["function"]["description"] == "Perform calculations"


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()

        async def handler(x: int) -> int:
            return x * 2

        tool_def = registry.register(
            name="double",
            description="Double a number",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            handler=handler,
        )

        assert tool_def.name == "double"
        assert "double" in registry.list_tools()

    def test_register_duplicate_raises(self):
        """Test that registering duplicate tool name raises error."""
        registry = ToolRegistry()

        async def handler1():
            pass

        async def handler2():
            pass

        registry.register("tool1", "First tool", {}, handler1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("tool1", "Second tool", {}, handler2)

    def test_get_tool(self):
        """Test retrieving a registered tool."""
        registry = ToolRegistry()

        async def handler():
            return "result"

        registry.register("my_tool", "My tool description", {}, handler)

        tool_def = registry.get("my_tool")
        assert tool_def is not None
        assert tool_def.name == "my_tool"

        # Non-existent tool returns None
        assert registry.get("nonexistent") is None

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()

        async def handler():
            pass

        registry.register("temp_tool", "Temporary", {}, handler)
        assert "temp_tool" in registry.list_tools()

        result = registry.unregister("temp_tool")
        assert result is True
        assert "temp_tool" not in registry.list_tools()

        # Unregistering non-existent returns False
        result = registry.unregister("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_execute_async_tool(self):
        """Test executing an async tool."""
        registry = ToolRegistry()

        async def add_numbers(a: int, b: int) -> int:
            return a + b

        registry.register(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
            handler=add_numbers,
        )

        result = await registry.execute("add", {"a": 5, "b": 3})

        assert result.success is True
        assert result.result == 8
        assert result.tool_name == "add"
        assert result.execution_time_ms is not None
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_sync_tool(self):
        """Test executing a sync tool (runs in executor)."""
        registry = ToolRegistry()

        def multiply(a: int, b: int) -> int:
            return a * b

        registry.register(
            name="multiply",
            description="Multiply two numbers",
            parameters={},
            handler=multiply,
        )

        result = await registry.execute("multiply", {"a": 4, "b": 5})

        assert result.success is True
        assert result.result == 20

    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self):
        """Test executing a non-existent tool."""
        registry = ToolRegistry()

        result = await registry.execute("nonexistent", {})

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_with_error(self):
        """Test tool execution error handling."""
        registry = ToolRegistry()

        async def failing_tool():
            raise ValueError("Something went wrong")

        registry.register("fail", "A failing tool", {}, failing_tool)

        result = await registry.execute("fail", {})

        assert result.success is False
        assert "Something went wrong" in result.error

    def test_export_for_llm(self):
        """Test exporting all tools for LLM function calling."""
        registry = ToolRegistry()

        async def handler1():
            pass

        async def handler2():
            pass

        registry.register("tool1", "First tool", {"type": "object"}, handler1)
        registry.register("tool2", "Second tool", {"type": "object"}, handler2)

        schemas = registry.export_for_llm()

        assert len(schemas) == 2
        names = [s["name"] for s in schemas]
        assert "tool1" in names
        assert "tool2" in names

    def test_export_for_openai(self):
        """Test exporting all tools for OpenAI function calling."""
        registry = ToolRegistry()

        async def handler():
            pass

        registry.register("openai_tool", "An OpenAI tool", {}, handler)

        schemas = registry.export_for_openai()

        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "openai_tool"


class TestToolDecorator:
    """Tests for the @tool decorator."""

    def test_tool_decorator(self):
        """Test the tool decorator adds metadata."""

        @tool(
            "decorated_tool",
            "A decorated tool",
            {"type": "object", "properties": {"x": {"type": "integer"}}},
        )
        async def my_tool(x: int) -> int:
            return x

        assert hasattr(my_tool, "_tool_name")
        assert my_tool._tool_name == "decorated_tool"
        assert my_tool._tool_description == "A decorated tool"
        assert my_tool._tool_parameters["type"] == "object"

    def test_tool_decorator_default_params(self):
        """Test decorator with default parameters."""

        @tool("simple", "Simple tool")
        async def simple_tool():
            pass

        assert simple_tool._tool_parameters == {
            "type": "object",
            "properties": {},
            "required": [],
        }


class TestHelpers:
    """Tests for helper functions."""

    def test_create_json_schema(self):
        """Test JSON Schema creation helper."""
        schema = create_json_schema(
            properties={
                "region": {"type": "string", "description": "Region code"},
                "limit": {"type": "integer", "default": 10},
            },
            required=["region"],
        )

        assert schema["type"] == "object"
        assert "region" in schema["properties"]
        assert schema["required"] == ["region"]

    def test_create_json_schema_no_required(self):
        """Test JSON Schema with no required fields."""
        schema = create_json_schema(
            properties={"optional": {"type": "string"}},
        )

        assert schema["required"] == []

    def test_get_tool_registry_singleton(self):
        """Test global registry singleton."""
        registry1 = get_tool_registry()
        registry2 = get_tool_registry()

        assert registry1 is registry2


class TestToolResult:
    """Tests for ToolResult model."""

    def test_success_result(self):
        """Test successful tool result."""
        result = ToolResult(
            tool_name="test",
            success=True,
            result={"data": "value"},
            execution_time_ms=15.5,
        )

        assert result.success is True
        assert result.result == {"data": "value"}
        assert result.error is None

    def test_error_result(self):
        """Test error tool result."""
        result = ToolResult(
            tool_name="test",
            success=False,
            error="Something failed",
        )

        assert result.success is False
        assert result.result is None
        assert result.error == "Something failed"
