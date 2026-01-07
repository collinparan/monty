"""Tests for Embedding service."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.embeddings import EmbeddingService


@pytest.fixture
def mock_embedding() -> list[float]:
    """Generate a mock embedding vector."""
    return [0.1] * 1536


@pytest.fixture
def embedding_service() -> EmbeddingService:
    """Create embedding service instance."""
    return EmbeddingService(api_key="test-key")


class TestEmbeddingService:
    """Tests for EmbeddingService."""

    def test_build_technician_profile_text_minimal(self, embedding_service: EmbeddingService):
        """Test profile text generation with minimal data."""
        text = embedding_service.build_technician_profile_text(
            region="Chicago",
            status="ACTIVE",
        )

        assert "Region: Chicago" in text
        assert "Status: ACTIVE" in text

    def test_build_technician_profile_text_full(self, embedding_service: EmbeddingService):
        """Test profile text generation with all fields."""
        text = embedding_service.build_technician_profile_text(
            region="Dallas",
            status="ACTIVE",
            tenure_days=400,
            skills=["HVAC", "Plumbing", "Electrical"],
            certifications=["EPA 608", "NATE"],
            metrics={
                "avg_jobs_per_week": 12.5,
                "avg_rating": 4.8,
                "completion_rate": 0.95,
                "jobs_last_30d": 45,
            },
        )

        assert "Region: Dallas" in text
        assert "Status: ACTIVE" in text
        assert "1 year(s)" in text
        assert "HVAC" in text
        assert "Plumbing" in text
        assert "EPA 608" in text
        assert "rating: 4.8" in text
        assert "completion: 95%" in text

    def test_build_technician_profile_text_short_tenure(self, embedding_service: EmbeddingService):
        """Test profile text with tenure less than a year."""
        text = embedding_service.build_technician_profile_text(
            region="Houston",
            status="ACTIVE",
            tenure_days=45,
        )

        assert "Tenure: 45 days" in text

    @pytest.mark.asyncio
    async def test_generate_embedding(
        self, embedding_service: EmbeddingService, mock_embedding: list[float]
    ):
        """Test single embedding generation."""
        mock_response = {
            "data": [{"embedding": mock_embedding, "index": 0}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.return_value = AsyncMock(
                json=lambda: mock_response,
                raise_for_status=lambda: None,
            )

            result = await embedding_service.generate_embedding("Test text")

            assert len(result) == 1536
            assert result == mock_embedding

    @pytest.mark.asyncio
    async def test_generate_embedding_empty_text(self, embedding_service: EmbeddingService):
        """Test embedding generation with empty text returns zeros."""
        result = await embedding_service.generate_embedding("")

        assert len(result) == 1536
        assert all(v == 0.0 for v in result)

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(
        self, embedding_service: EmbeddingService, mock_embedding: list[float]
    ):
        """Test batch embedding generation."""
        mock_response = {
            "data": [
                {"embedding": mock_embedding, "index": 0},
                {"embedding": [0.2] * 1536, "index": 1},
            ],
            "model": "text-embedding-3-small",
        }

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.return_value = AsyncMock(
                json=lambda: mock_response,
                raise_for_status=lambda: None,
            )

            result = await embedding_service.generate_embeddings(["Text 1", "Text 2"])

            assert len(result) == 2
            assert len(result[0]) == 1536
            assert len(result[1]) == 1536

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_empty(
        self, embedding_service: EmbeddingService, mock_embedding: list[float]
    ):
        """Test batch embedding with some empty texts."""
        mock_response = {
            "data": [{"embedding": mock_embedding, "index": 0}],
            "model": "text-embedding-3-small",
        }

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.return_value = AsyncMock(
                json=lambda: mock_response,
                raise_for_status=lambda: None,
            )

            result = await embedding_service.generate_embeddings(["Valid text", "", ""])

            assert len(result) == 3
            # First should have embedding
            assert result[0] == mock_embedding
            # Others should be zeros
            assert all(v == 0.0 for v in result[1])
            assert all(v == 0.0 for v in result[2])

    @pytest.mark.asyncio
    async def test_embed_technician_profile(
        self, embedding_service: EmbeddingService, mock_embedding: list[float]
    ):
        """Test technician profile embedding."""
        mock_response = {
            "data": [{"embedding": mock_embedding, "index": 0}],
            "model": "text-embedding-3-small",
        }

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.return_value = AsyncMock(
                json=lambda: mock_response,
                raise_for_status=lambda: None,
            )

            profile_text, embedding = await embedding_service.embed_technician_profile(
                region="Chicago",
                status="ACTIVE",
                tenure_days=180,
                skills=["HVAC"],
                metrics={"avg_rating": 4.5},
            )

            assert "Region: Chicago" in profile_text
            assert "HVAC" in profile_text
            assert len(embedding) == 1536

    @pytest.mark.asyncio
    async def test_embed_technician_profiles_batch(
        self, embedding_service: EmbeddingService, mock_embedding: list[float]
    ):
        """Test batch technician profile embedding."""
        mock_response = {
            "data": [
                {"embedding": mock_embedding, "index": 0},
                {"embedding": [0.2] * 1536, "index": 1},
            ],
            "model": "text-embedding-3-small",
        }

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.return_value = AsyncMock(
                json=lambda: mock_response,
                raise_for_status=lambda: None,
            )

            profiles = [
                {"region": "Chicago", "status": "ACTIVE", "tenure_days": 100},
                {"region": "Dallas", "status": "CHURNED", "skills": ["Plumbing"]},
            ]

            result = await embedding_service.embed_technician_profiles_batch(profiles)

            assert len(result) == 2
            assert "Chicago" in result[0][0]
            assert "Dallas" in result[1][0]
            assert len(result[0][1]) == 1536
            assert len(result[1][1]) == 1536

    @pytest.mark.asyncio
    async def test_semantic_search(
        self, embedding_service: EmbeddingService, mock_embedding: list[float]
    ):
        """Test semantic search functionality."""
        # Create some test embeddings
        embeddings = [
            [1.0, 0.0, 0.0] + [0.0] * 1533,  # Different direction
            mock_embedding,  # Similar to query
            [0.0, 1.0, 0.0] + [0.0] * 1533,  # Different direction
        ]

        mock_response = {
            "data": [{"embedding": mock_embedding, "index": 0}],
            "model": "text-embedding-3-small",
        }

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.return_value = AsyncMock(
                json=lambda: mock_response,
                raise_for_status=lambda: None,
            )

            results = await embedding_service.semantic_search(
                query="test query",
                embeddings=embeddings,
                top_k=2,
            )

            assert len(results) == 2
            # Index 1 should be most similar (same embedding)
            assert results[0][0] == 1
            assert results[0][1] > 0.9  # High similarity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
