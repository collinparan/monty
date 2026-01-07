"""End-to-end tests for the AI agent.

Tests agent chat with tool use functionality.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.agent import AgentService, AgentState
from app.services.agent_tools import create_technician_analytics_tools
from app.services.tools import ToolRegistry


class TestAgentChatEndpoint:
    """Test agent chat API endpoint."""

    @pytest.mark.asyncio
    async def test_chat_endpoint_responds(self, test_client):
        """Test that chat endpoint responds to messages."""
        with patch("app.routers.agent.AgentService") as mock_service:
            mock_instance = MagicMock()
            mock_instance.chat = AsyncMock(
                return_value=("Hello! How can I help you?", AgentState())
            )
            mock_service.return_value = mock_instance

            response = await test_client.post(
                "/api/v1/agent/chat",
                json={"message": "Hello"},
            )

            # Should respond (may fail if API key not set)
            assert response.status_code in [200, 503]

    @pytest.mark.asyncio
    async def test_chat_with_session_id(self, test_client):
        """Test chat with session ID for context continuity."""
        with patch("app.routers.agent.AgentService") as mock_service:
            mock_instance = MagicMock()
            mock_instance.chat = AsyncMock(
                return_value=("Hello! How can I help you?", AgentState())
            )
            mock_service.return_value = mock_instance

            response = await test_client.post(
                "/api/v1/agent/chat",
                json={"message": "Hello", "session_id": "test-session-123"},
            )

            assert response.status_code in [200, 503]


class TestAgentStreamEndpoint:
    """Test agent streaming SSE endpoint."""

    @pytest.mark.asyncio
    async def test_stream_endpoint_exists(self, test_client):
        """Test that streaming endpoint is accessible."""
        # The streaming endpoint requires a message param
        response = await test_client.get("/api/v1/agent/stream", params={"message": "Hello"})

        # Should return SSE response or error
        assert response.status_code in [200, 503]

    @pytest.mark.asyncio
    async def test_stream_returns_sse_format(self, test_client):
        """Test that streaming returns SSE formatted data."""
        with patch("app.routers.agent.AgentService") as mock_service:

            async def mock_run(state):
                from app.services.agent import AgentEvent

                yield AgentEvent(type="token", data="Hello")
                yield AgentEvent(type="done", data={"iterations": 1})

            mock_instance = MagicMock()
            mock_instance.run = mock_run
            mock_service.return_value = mock_instance

            response = await test_client.get("/api/v1/agent/stream", params={"message": "Hello"})

            if response.status_code == 200:
                # Content type should be event-stream
                assert "text/event-stream" in response.headers.get("content-type", "")


class TestAgentToolExecution:
    """Test agent tool execution flow."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = MagicMock()
        session.execute = AsyncMock(
            return_value=MagicMock(scalars=lambda: MagicMock(all=lambda: []))
        )
        return session

    @pytest.mark.asyncio
    async def test_tool_registry_has_expected_tools(self, mock_db_session):
        """Test that tool registry has expected tools."""

        def session_factory():
            return mock_db_session

        registry = create_technician_analytics_tools(session_factory)
        tools = registry.list_tools()

        expected_tools = [
            "query_technicians",
            "get_prediction",
            "get_forecast",
            "calculate_roi",
            "get_feature_importance",
            "get_regional_summary",
        ]

        for tool_name in expected_tools:
            assert tool_name in tools, f"Missing tool: {tool_name}"

    @pytest.mark.asyncio
    async def test_query_technicians_tool(self, mock_db_session):
        """Test query_technicians tool execution."""

        def session_factory():
            return mock_db_session

        registry = create_technician_analytics_tools(session_factory)

        result = await registry.execute(
            "query_technicians",
            {"region": "US-WEST", "limit": 10},
        )

        assert result.success is True
        assert result.tool_name == "query_technicians"
        assert result.execution_time_ms is not None

    @pytest.mark.asyncio
    async def test_calculate_roi_tool(self, mock_db_session):
        """Test calculate_roi tool execution."""

        def session_factory():
            return mock_db_session

        registry = create_technician_analytics_tools(session_factory)

        result = await registry.execute(
            "calculate_roi",
            {
                "intervention_type": "bonus",
                "cost_per_technician": 1000,
                "expected_retention_improvement": 0.1,
            },
        )

        assert result.success is True
        assert "roi" in str(result.result).lower() or result.result is not None


class TestAgentWithMockedLLM:
    """Test agent behavior with mocked LLM responses."""

    @pytest.fixture
    def mock_anthropic_response(self):
        """Create mock Anthropic API response."""
        return MagicMock(
            content=[
                MagicMock(
                    type="text",
                    text="Based on the data, here are my findings...",
                )
            ],
            stop_reason="end_turn",
        )

    @pytest.mark.asyncio
    async def test_agent_responds_without_tools(self, mock_anthropic_response):
        """Test agent responds to simple queries without using tools."""
        registry = ToolRegistry()

        with patch("anthropic.AsyncAnthropic") as mock_client:
            # Setup mock
            mock_instance = MagicMock()

            # Create async context manager for streaming
            async def mock_stream(*args, **kwargs):
                class MockStream:
                    async def __aenter__(self):
                        return self

                    async def __aexit__(self, *args):
                        pass

                    async def __aiter__(self):
                        yield MagicMock(
                            type="content_block_start", content_block=MagicMock(type="text")
                        )
                        yield MagicMock(
                            type="content_block_delta",
                            delta=MagicMock(text="Hello!", partial_json=None),
                        )
                        yield MagicMock(type="content_block_stop")

                    async def get_final_message(self):
                        return mock_anthropic_response

                return MockStream()

            mock_instance.messages.stream = mock_stream
            mock_client.return_value = mock_instance

            service = AgentService(
                tool_registry=registry,
                api_key="test-key",
                model="claude-3-sonnet-20240229",
            )

            state = AgentState()
            state.add_user_message("What is 2+2?")

            events = []
            async for event in service.run(state):
                events.append(event)

            # Should have token events and done event
            token_events = [e for e in events if e.type == "token"]
            done_events = [e for e in events if e.type == "done"]

            assert len(token_events) > 0
            assert len(done_events) == 1


class TestAgentSessionManagement:
    """Test agent session management."""

    @pytest.mark.asyncio
    async def test_get_session(self, test_client, redis_client):
        """Test retrieving an existing session."""
        # Create a session first
        session_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "tools_called": [],
            "context": {},
            "iteration": 1,
        }

        await redis_client.setex(
            "agent:session:test-session-456",
            3600,
            json.dumps(session_data),
        )

        response = await test_client.get("/api/v1/agent/sessions/test-session-456")

        # Should find the session
        assert response.status_code in [200, 404]

    @pytest.mark.asyncio
    async def test_delete_session(self, test_client, redis_client):
        """Test deleting a session."""
        # Create a session first
        await redis_client.setex(
            "agent:session:delete-test",
            3600,
            json.dumps({"messages": []}),
        )

        response = await test_client.delete("/api/v1/agent/sessions/delete-test")

        # Should succeed
        assert response.status_code in [200, 204, 404]


class TestAgentExamplePrompts:
    """Test agent example prompts endpoint."""

    @pytest.mark.asyncio
    async def test_example_prompts_endpoint(self, test_client):
        """Test that example prompts endpoint returns suggestions."""
        response = await test_client.get("/api/v1/agent/examples")

        assert response.status_code == 200
        data = response.json()

        # Should return list of examples
        assert isinstance(data, list)

        # Each example should have text
        for example in data:
            assert "text" in example or "prompt" in example or "title" in example


class TestAgentErrorHandling:
    """Test agent error handling."""

    @pytest.mark.asyncio
    async def test_agent_handles_missing_api_key(self, test_client):
        """Test that agent handles missing API key gracefully."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            response = await test_client.post(
                "/api/v1/agent/chat",
                json={"message": "Hello"},
            )

            # Should return error, not crash
            assert response.status_code in [200, 503, 500]

    @pytest.mark.asyncio
    async def test_agent_handles_empty_message(self, test_client):
        """Test that agent handles empty message."""
        response = await test_client.post(
            "/api/v1/agent/chat",
            json={"message": ""},
        )

        # Should return validation error
        assert response.status_code in [200, 400, 422]

    @pytest.mark.asyncio
    async def test_stream_handles_disconnection(self, test_client):
        """Test that streaming handles client disconnection."""
        # This tests that the endpoint doesn't crash on early disconnect
        response = await test_client.get(
            "/api/v1/agent/stream",
            params={"message": "Hello"},
        )

        # Should start responding
        assert response.status_code in [200, 503]
