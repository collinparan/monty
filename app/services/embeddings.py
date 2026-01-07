"""Embedding service for generating text embeddings."""

from __future__ import annotations

from typing import Any, Optional

import httpx

from app.config import get_settings

settings = get_settings()


class EmbeddingService:
    """Service for generating text embeddings using OpenAI API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        """Initialize the embedding service.

        Args:
            api_key: OpenAI API key. Defaults to settings.
            model: Embedding model name. Defaults to settings.
            dimensions: Embedding dimensions. Defaults to settings.
        """
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_embedding_model
        self.dimensions = dimensions or settings.openai_embedding_dimensions
        self.base_url = "https://api.openai.com/v1"

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            return [0.0] * self.dimensions

        embeddings = await self.generate_embeddings([text])
        return embeddings[0]

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in a single API call.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter out empty texts and track indices
        valid_indices: list[int] = []
        valid_texts: list[str] = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_indices.append(i)
                valid_texts.append(text.strip())

        if not valid_texts:
            return [[0.0] * self.dimensions for _ in texts]

        # Call OpenAI API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "input": valid_texts,
                    "model": self.model,
                    "dimensions": self.dimensions,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()

        # Extract embeddings from response
        api_embeddings = [item["embedding"] for item in data["data"]]

        # Reconstruct full list with zeros for empty texts
        result: list[list[float]] = [[0.0] * self.dimensions for _ in texts]
        for orig_idx, embedding in zip(valid_indices, api_embeddings):
            result[orig_idx] = embedding

        return result

    def build_technician_profile_text(
        self,
        region: str,
        status: str,
        tenure_days: Optional[int] = None,
        skills: Optional[list[str]] = None,
        certifications: Optional[list[str]] = None,
        metrics: Optional[dict[str, Any]] = None,
    ) -> str:
        """Build profile text for a technician for embedding.

        Args:
            region: Technician's region
            status: Current status (ACTIVE, CHURNED, etc.)
            tenure_days: Days with company
            skills: List of skills
            certifications: List of certifications
            metrics: Performance metrics dict

        Returns:
            Formatted profile text for embedding
        """
        parts = [
            f"Region: {region}",
            f"Status: {status}",
        ]

        if tenure_days is not None:
            years = tenure_days // 365
            months = (tenure_days % 365) // 30
            if years > 0:
                parts.append(f"Tenure: {years} year(s), {months} month(s)")
            else:
                parts.append(f"Tenure: {tenure_days} days")

        if skills:
            parts.append(f"Skills: {', '.join(skills)}")

        if certifications:
            parts.append(f"Certifications: {', '.join(certifications)}")

        if metrics:
            metric_parts = []
            if "avg_jobs_per_week" in metrics:
                metric_parts.append(f"avg jobs/week: {metrics['avg_jobs_per_week']:.1f}")
            if "avg_rating" in metrics:
                metric_parts.append(f"rating: {metrics['avg_rating']:.1f}")
            if "completion_rate" in metrics:
                metric_parts.append(f"completion: {metrics['completion_rate']:.0%}")
            if "jobs_last_30d" in metrics:
                metric_parts.append(f"jobs last 30d: {metrics['jobs_last_30d']}")
            if metric_parts:
                parts.append(f"Performance: {', '.join(metric_parts)}")

        return ". ".join(parts)

    async def embed_technician_profile(
        self,
        region: str,
        status: str,
        tenure_days: Optional[int] = None,
        skills: Optional[list[str]] = None,
        certifications: Optional[list[str]] = None,
        metrics: Optional[dict[str, Any]] = None,
    ) -> tuple[str, list[float]]:
        """Generate embedding for a technician profile.

        Args:
            region: Technician's region
            status: Current status
            tenure_days: Days with company
            skills: List of skills
            certifications: List of certifications
            metrics: Performance metrics

        Returns:
            Tuple of (profile_text, embedding_vector)
        """
        profile_text = self.build_technician_profile_text(
            region=region,
            status=status,
            tenure_days=tenure_days,
            skills=skills,
            certifications=certifications,
            metrics=metrics,
        )

        embedding = await self.generate_embedding(profile_text)
        return profile_text, embedding

    async def embed_technician_profiles_batch(
        self, profiles: list[dict[str, Any]]
    ) -> list[tuple[str, list[float]]]:
        """Generate embeddings for multiple technician profiles.

        Args:
            profiles: List of profile dicts with keys:
                region, status, tenure_days, skills, certifications, metrics

        Returns:
            List of (profile_text, embedding_vector) tuples
        """
        # Build all profile texts
        profile_texts = [
            self.build_technician_profile_text(
                region=p.get("region", "Unknown"),
                status=p.get("status", "UNKNOWN"),
                tenure_days=p.get("tenure_days"),
                skills=p.get("skills"),
                certifications=p.get("certifications"),
                metrics=p.get("metrics"),
            )
            for p in profiles
        ]

        # Generate embeddings in batch
        embeddings = await self.generate_embeddings(profile_texts)

        return list(zip(profile_texts, embeddings))

    async def semantic_search(
        self,
        query: str,
        embeddings: list[list[float]],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Perform semantic search using cosine similarity.

        Args:
            query: Search query text
            embeddings: List of embedding vectors to search
            top_k: Number of top results to return

        Returns:
            List of (index, similarity_score) tuples sorted by score descending
        """
        import numpy as np

        if not embeddings:
            return []

        # Generate query embedding
        query_embedding = await self.generate_embedding(query)
        query_vec = np.array(query_embedding)

        # Calculate cosine similarity with all embeddings
        similarities = []
        for i, emb in enumerate(embeddings):
            emb_vec = np.array(emb)
            # Cosine similarity
            dot_product = np.dot(query_vec, emb_vec)
            norm_product = np.linalg.norm(query_vec) * np.linalg.norm(emb_vec)
            if norm_product > 0:
                similarity = float(dot_product / norm_product)
            else:
                similarity = 0.0
            similarities.append((i, similarity))

        # Sort by similarity descending and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
