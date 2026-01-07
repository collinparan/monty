"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "shs-technician-analytics"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    secret_key: str = Field(default="change-me-in-production")

    # PostgreSQL + PGVector
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "monty"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    database_url: str = "postgresql+asyncpg://postgres:postgres@postgres:5432/monty"

    # Database pool
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout: int = 30

    # Redis
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_password: str = ""
    redis_url: str = "redis://redis:6379/0"
    redis_max_connections: int = 10

    # Neo4j
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Snowflake
    snowflake_account: str = ""
    snowflake_user: str = ""
    snowflake_database: str = ""
    snowflake_schema: str = ""
    snowflake_warehouse: str = ""
    snowflake_role: str = ""
    snowflake_authenticator: str = "SNOWFLAKE_JWT"
    snowflake_private_key_file: str = "/keys/sf_rsa_key.p8"
    snowflake_private_key_file_pwd: str = ""
    snowflake_cache_ttl_seconds: int = 300

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_max_tokens: int = 4096

    # OpenAI
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536

    # ML Models
    models_dir: Path = Path("/models")
    model_version_prefix: str = "v"
    interpret_n_jobs: int = -1
    interpret_random_state: int = 42
    prophet_yearly_seasonality: bool = True
    prophet_weekly_seasonality: bool = True
    prophet_daily_seasonality: bool = False

    # Agent
    agent_max_iterations: int = 10
    agent_timeout_seconds: int = 300
    agent_stream_enabled: bool = True

    # CORS
    cors_origins: str = "http://localhost:3000,http://localhost:8000"

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
