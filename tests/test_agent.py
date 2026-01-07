"""Tests for the agent service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from app.services.agent import (
    SYSTEM_PROMPT,
    AgentEvent,
    AgentService,
    AgentState,
    create_agent_service,
)
from app.services.tools import ToolRegistry


class TestAgentState:
    """Tests for AgentState dataclass."""

    def test_initial_state(self):
        """Test initial state values."""
        state = AgentState()

        assert state.messages == []
        assert state.tools_called == []
        assert state.context == {}
        assert state.iteration == 0
        assert state.max_iterations == 10

    def test_add_user_message(self):
        """Test adding user messages."""
        state = AgentState()
        state.add_user_message("Hello")

        assert len(state.messages) == 1
        assert state.messages[0]["role"] == "user"
        assert state.messages[0]["content"] == "Hello"

    def test_add_assistant_message(self):
        """Test adding assistant messages."""
        state = AgentState()
        state.add_assistant_message("Hi there!")

        assert len(state.messages) == 1
        assert state.messages[0]["role"] == "assistant"
        assert state.messages[0]["content"] == "Hi there!"

    def test_add_tool_use(self):
        """Test recording tool use."""
        state = AgentState()
        state.add_tool_use("tool_123", "query_technicians", {"region": "US-WEST"})

        assert len(state.messages) == 1
        assert state.messages[0]["role"] == "assistant"
        assert state.messages[0]["content"][0]["type"] == "tool_use"
        assert state.messages[0]["content"][0]["name"] == "query_technicians"
        assert "query_technicians" in state.tools_called

    def test_add_tool_result(self):
        """Test adding tool results."""
        state = AgentState()
        state.add_tool_result("tool_123", '{"count": 5}', is_error=False)

        assert len(state.messages) == 1
        assert state.messages[0]["role"] == "user"
        assert state.messages[0]["content"][0]["type"] == "tool_result"
        assert state.messages[0]["content"][0]["is_error"] is False


class TestAgentEvent:
    """Tests for AgentEvent dataclass."""

    def test_event_creation(self):
        """Test creating events."""
        event = AgentEvent(type="token", data="Hello")

        assert event.type == "token"
        assert event.data == "Hello"
        assert event.timestamp > 0

    def test_to_dict(self):
        """Test converting event to dictionary."""
        event = AgentEvent(type="tool_call", data={"name": "test_tool"})
        d = event.to_dict()

        assert d["type"] == "tool_call"
        assert d["data"]["name"] == "test_tool"
        assert "timestamp" in d

    def test_to_sse(self):
        """Test formatting event for SSE."""
        event = AgentEvent(type="token", data="test")
        sse = event.to_sse()

        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")
        parsed = json.loads(sse[6:-2])
        assert parsed["type"] == "token"


class TestAgentService:
    """Tests for AgentService class."""

    def test_init_with_registry(self):
        """Test initializing service with tool registry."""
        registry = ToolRegistry()
        service = AgentService(
            tool_registry=registry,
            api_key="test-key",
            model="claude-3-sonnet",
        )

        assert service.tool_registry is registry
        assert service.api_key == "test-key"
        assert service.model == "claude-3-sonnet"

    def test_init_without_api_key(self):
        """Test service without API key."""
        registry = ToolRegistry()
        service = AgentService(tool_registry=registry, api_key="")

        assert service.client is None

    @pytest.mark.asyncio
    async def test_chat_without_client(self):
        """Test chat returns error when client not configured."""
        registry = ToolRegistry()
        service = AgentService(tool_registry=registry, api_key="")

        response, _state = await service.chat("Hello")

        assert "not configured" in response.lower()

    def test_system_prompt_exists(self):
        """Test that system prompt is defined."""
        assert SYSTEM_PROMPT is not None
        assert len(SYSTEM_PROMPT) > 100
        assert "technician" in SYSTEM_PROMPT.lower()


class TestAgentServiceWithMocks:
    """Tests for AgentService with mocked Anthropic client."""

    @pytest.fixture
    def mock_stream(self):
        """Create mock stream context manager."""

        @dataclass
        class MockContentBlock:
            type: str
            id: str = "test-id"
            name: str = ""
            text: str = ""
            input: dict = None

            def __post_init__(self):
                if self.input is None:
                    self.input = {}

        @dataclass
        class MockDelta:
            text: str = ""
            partial_json: str = ""

        @dataclass
        class MockEvent:
            type: str
            content_block: MockContentBlock = None
            delta: MockDelta = None

        @dataclass
        class MockMessage:
            content: list

        class MockStream:
            def __init__(self, events, final_content):
                self.events = events
                self.final_content = final_content
                self._index = 0

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._index >= len(self.events):
                    raise StopAsyncIteration
                event = self.events[self._index]
                self._index += 1
                return event

            async def get_final_message(self):
                return MockMessage(content=self.final_content)

        return MockStream, MockEvent, MockContentBlock, MockDelta

    @pytest.mark.asyncio
    async def test_run_simple_text_response(self, mock_stream):
        """Test agent run with simple text response (no tool use)."""
        MockStream, MockEvent, MockContentBlock, MockDelta = mock_stream

        # Create mock events for a simple text response
        events = [
            MockEvent(
                type="content_block_start",
                content_block=MockContentBlock(type="text"),
            ),
            MockEvent(
                type="content_block_delta",
                delta=MockDelta(text="Hello! "),
            ),
            MockEvent(
                type="content_block_delta",
                delta=MockDelta(text="I can help."),
            ),
            MockEvent(type="content_block_stop"),
        ]

        final_content = [MockContentBlock(type="text", text="Hello! I can help.")]
        stream = MockStream(events, final_content)

        # Create service with mocked client
        registry = ToolRegistry()
        service = AgentService(
            tool_registry=registry,
            api_key="test-key",
        )

        # Mock the client
        service.client = MagicMock()
        service.client.messages = MagicMock()
        service.client.messages.stream = MagicMock(return_value=stream)

        # Run agent
        state = AgentState()
        state.add_user_message("Hello")

        collected_events = []
        async for event in service.run(state):
            collected_events.append(event)

        # Verify events
        token_events = [e for e in collected_events if e.type == "token"]
        done_events = [e for e in collected_events if e.type == "done"]

        assert len(token_events) == 2
        assert token_events[0].data == "Hello! "
        assert token_events[1].data == "I can help."
        assert len(done_events) == 1

    @pytest.mark.asyncio
    async def test_run_with_tool_call(self, mock_stream):
        """Test agent run with tool call."""
        MockStream, MockEvent, MockContentBlock, MockDelta = mock_stream

        # First response - tool use
        tool_events = [
            MockEvent(
                type="content_block_start",
                content_block=MockContentBlock(
                    type="tool_use", id="tool_1", name="query_technicians"
                ),
            ),
            MockEvent(type="content_block_stop"),
        ]

        tool_content = [
            MockContentBlock(
                type="tool_use",
                id="tool_1",
                name="query_technicians",
                input={"region": "US-WEST"},
            )
        ]
        tool_stream = MockStream(tool_events, tool_content)

        # Second response - text after tool result
        text_events = [
            MockEvent(
                type="content_block_start",
                content_block=MockContentBlock(type="text"),
            ),
            MockEvent(
                type="content_block_delta",
                delta=MockDelta(text="Found 5 technicians."),
            ),
            MockEvent(type="content_block_stop"),
        ]

        text_content = [MockContentBlock(type="text", text="Found 5 technicians.")]
        text_stream = MockStream(text_events, text_content)

        # Create service with mocked client
        registry = ToolRegistry()

        # Register a mock tool
        async def mock_query(**kwargs):
            return {"count": 5, "technicians": []}

        registry.register(
            "query_technicians",
            "Query technicians",
            {"type": "object"},
            mock_query,
        )

        service = AgentService(
            tool_registry=registry,
            api_key="test-key",
        )

        # Mock the client to return different streams
        call_count = 0

        def mock_stream_factory(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tool_stream
            return text_stream

        service.client = MagicMock()
        service.client.messages = MagicMock()
        service.client.messages.stream = mock_stream_factory

        # Run agent
        state = AgentState()
        state.add_user_message("Find technicians in US-WEST")

        collected_events = []
        async for event in service.run(state):
            collected_events.append(event)

        # Verify events
        tool_call_events = [e for e in collected_events if e.type == "tool_call"]
        tool_result_events = [e for e in collected_events if e.type == "tool_result"]

        assert len(tool_call_events) == 1
        assert tool_call_events[0].data["tool_name"] == "query_technicians"
        assert len(tool_result_events) == 1
        assert tool_result_events[0].data["success"] is True


class TestCreateAgentService:
    """Tests for create_agent_service factory function."""

    def test_create_with_registry(self):
        """Test factory creates service with registry."""
        registry = ToolRegistry()

        with patch("app.services.agent.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.anthropic_model = "claude-3-sonnet"
            mock_settings.anthropic_max_tokens = 4096
            mock_settings.agent_max_iterations = 10

            service = create_agent_service(registry)

            assert service.tool_registry is registry
            assert service.api_key == "test-key"
            assert service.model == "claude-3-sonnet"


class TestAgentIntegration:
    """Integration-style tests for agent behavior."""

    @pytest.mark.asyncio
    async def test_max_iterations_limit(self):
        """Test that agent respects max iterations."""
        registry = ToolRegistry()
        service = AgentService(
            tool_registry=registry,
            api_key="test-key",
            max_iterations=2,
        )

        # Create a state that's already at max iterations
        state = AgentState(max_iterations=2)
        state.iteration = 2  # Already at max

        # The loop should not execute
        events = []
        async for event in service.run(state):
            events.append(event)

        # Should get a done event immediately
        assert len(events) == 1
        assert events[0].type == "done"
        assert events[0].data.get("max_iterations_reached") is True

    def test_conversation_state_persistence(self):
        """Test that state persists across messages."""
        state = AgentState()

        # Simulate a conversation
        state.add_user_message("First message")
        state.add_assistant_message("First response")
        state.add_user_message("Second message")
        state.add_assistant_message("Second response")

        assert len(state.messages) == 4
        assert state.messages[0]["content"] == "First message"
        assert state.messages[-1]["content"] == "Second response"
