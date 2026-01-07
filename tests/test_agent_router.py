"""Tests for agent router endpoints."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from app.routers.agent import (
    EXAMPLE_PROMPTS,
    ChatRequest,
    ChatResponse,
    ExamplePrompt,
    get_or_create_session,
    save_session,
)
from app.services.agent import AgentState


class TestChatSchemas:
    """Tests for chat request/response schemas."""

    def test_chat_request_valid(self):
        """Test valid chat request."""
        request = ChatRequest(message="Hello", session_id="test-123")

        assert request.message == "Hello"
        assert request.session_id == "test-123"

    def test_chat_request_without_session(self):
        """Test chat request without session ID."""
        request = ChatRequest(message="Hello")

        assert request.message == "Hello"
        assert request.session_id is None

    def test_chat_response_valid(self):
        """Test valid chat response."""
        response = ChatResponse(
            response="Hi there!",
            session_id="session-123",
            tools_called=["query_technicians"],
            iterations=2,
        )

        assert response.response == "Hi there!"
        assert response.session_id == "session-123"
        assert "query_technicians" in response.tools_called


class TestExamplePrompts:
    """Tests for example prompts."""

    def test_example_prompts_exist(self):
        """Test that example prompts are defined."""
        assert len(EXAMPLE_PROMPTS) > 0

    def test_example_prompts_have_required_fields(self):
        """Test that example prompts have all required fields."""
        for prompt in EXAMPLE_PROMPTS:
            assert isinstance(prompt, ExamplePrompt)
            assert len(prompt.title) > 0
            assert len(prompt.prompt) > 0
            assert len(prompt.category) > 0

    def test_example_prompts_categories(self):
        """Test that example prompts cover different categories."""
        categories = {p.category for p in EXAMPLE_PROMPTS}
        assert len(categories) >= 3  # At least 3 different categories


class TestSessionManagement:
    """Tests for session management functions."""

    @pytest.mark.asyncio
    async def test_get_or_create_session_new(self):
        """Test creating a new session."""
        mock_redis = MagicMock()
        mock_redis.get = MagicMock(return_value=None)

        async def mock_get(key):
            return None

        mock_redis.get = mock_get

        session_id, state = await get_or_create_session(None, mock_redis)

        assert session_id is not None
        assert len(session_id) == 36  # UUID format
        assert isinstance(state, AgentState)
        assert state.messages == []

    @pytest.mark.asyncio
    async def test_get_or_create_session_existing(self):
        """Test loading an existing session."""
        session_data = json.dumps(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "tools_called": ["query_technicians"],
                "context": {"test": "value"},
            }
        )

        mock_redis = MagicMock()

        async def mock_get(key):
            return session_data

        mock_redis.get = mock_get

        session_id, state = await get_or_create_session("existing-id", mock_redis)

        assert session_id == "existing-id"
        assert len(state.messages) == 1
        assert state.messages[0]["content"] == "Hello"
        assert "query_technicians" in state.tools_called

    @pytest.mark.asyncio
    async def test_get_or_create_session_invalid_data(self):
        """Test handling invalid session data."""
        mock_redis = MagicMock()

        async def mock_get(key):
            return "invalid json"

        mock_redis.get = mock_get

        session_id, state = await get_or_create_session("bad-id", mock_redis)

        # Should create new session on invalid data
        assert session_id != "bad-id"
        assert state.messages == []

    @pytest.mark.asyncio
    async def test_save_session(self):
        """Test saving session to Redis."""
        state = AgentState()
        state.add_user_message("Hello")
        state.tools_called.append("test_tool")

        mock_redis = MagicMock()
        saved_data = {}

        async def mock_setex(key, ttl, data):
            saved_data["key"] = key
            saved_data["ttl"] = ttl
            saved_data["data"] = json.loads(data)

        mock_redis.setex = mock_setex

        await save_session("test-session", state, mock_redis)

        assert "agent:session:test-session" in saved_data["key"]
        assert saved_data["ttl"] == 3600
        assert saved_data["data"]["messages"][0]["content"] == "Hello"
        assert "test_tool" in saved_data["data"]["tools_called"]


class TestAgentEndpoints:
    """Integration-style tests for agent endpoints."""

    @pytest.mark.asyncio
    async def test_chat_endpoint_structure(self):
        """Test that chat endpoint exists and has correct structure."""
        from app.routers.agent import router

        routes = [r for r in router.routes if hasattr(r, "path")]
        paths = [r.path for r in routes]

        assert "/chat" in paths
        assert "/stream" in paths
        assert "/sessions/{session_id}" in paths
        assert "/examples" in paths

    @pytest.mark.asyncio
    async def test_stream_endpoint_response_type(self):
        """Test stream endpoint returns StreamingResponse."""
        from app.routers.agent import agent_stream

        # The function should return a StreamingResponse
        # when called with proper dependencies
        # This is a structural test - full integration needs the app context
        assert callable(agent_stream)

    def test_example_prompts_endpoint(self):
        """Test example prompts structure."""
        from app.routers.agent import get_example_prompts

        # Should be async
        assert callable(get_example_prompts)


class TestDBSessionFactory:
    """Tests for database session factory."""

    @pytest.mark.asyncio
    async def test_session_factory_yields_session(self):
        """Test that session factory yields the provided session."""
        from app.routers.agent import get_db_session_factory

        mock_session = MagicMock()
        factory = get_db_session_factory(mock_session)

        async with factory() as session:
            assert session is mock_session


class TestAgentRouterWithMocks:
    """Tests with mocked dependencies."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies."""
        mock_session = MagicMock()
        mock_redis = MagicMock()

        # Mock Redis methods
        async def mock_get(key):
            return None

        async def mock_setex(key, ttl, data):
            pass

        async def mock_delete(key):
            return 0

        mock_redis.get = mock_get
        mock_redis.setex = mock_setex
        mock_redis.delete = mock_delete

        return mock_session, mock_redis

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, mock_dependencies):
        """Test getting non-existent session raises 404."""
        from fastapi import HTTPException

        from app.routers.agent import get_session

        _mock_session, mock_redis = mock_dependencies

        with pytest.raises(HTTPException) as exc_info:
            await get_session("nonexistent-id", mock_redis)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_session_not_found(self, mock_dependencies):
        """Test deleting non-existent session raises 404."""
        from fastapi import HTTPException

        from app.routers.agent import delete_session

        _mock_session, mock_redis = mock_dependencies

        with pytest.raises(HTTPException) as exc_info:
            await delete_session("nonexistent-id", mock_redis)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_session_success(self, mock_dependencies):
        """Test successful session deletion."""
        from app.routers.agent import delete_session

        _mock_session, mock_redis = mock_dependencies

        # Mock successful deletion
        async def mock_delete(key):
            return 1

        mock_redis.delete = mock_delete

        result = await delete_session("test-id", mock_redis)

        assert result["status"] == "deleted"
        assert result["session_id"] == "test-id"
