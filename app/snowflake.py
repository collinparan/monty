"""Snowflake connector with key-pair authentication."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import snowflake.connector
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

from app.config import get_settings

settings = get_settings()

# Thread pool for running sync Snowflake operations
_executor = ThreadPoolExecutor(max_workers=4)


def _load_private_key() -> bytes:
    """Load and decrypt the private key file."""
    key_path = Path(settings.snowflake_private_key_file)

    if not key_path.exists():
        raise FileNotFoundError(f"Snowflake private key not found: {key_path}")

    with key_path.open("rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=settings.snowflake_private_key_file_pwd.encode()
            if settings.snowflake_private_key_file_pwd
            else None,
            backend=default_backend(),
        )

    # Convert to DER format for Snowflake
    private_key_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    return private_key_bytes


@contextmanager
def get_snowflake_connection():
    """Get a Snowflake connection using key-pair authentication."""
    private_key_bytes = _load_private_key()

    conn = snowflake.connector.connect(
        account=settings.snowflake_account,
        user=settings.snowflake_user,
        private_key=private_key_bytes,
        database=settings.snowflake_database,
        schema=settings.snowflake_schema,
        warehouse=settings.snowflake_warehouse,
        role=settings.snowflake_role,
    )

    try:
        yield conn
    finally:
        conn.close()


def _execute_query_sync(query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Execute a Snowflake query synchronously."""
    with get_snowflake_connection() as conn:
        cursor = conn.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            columns = [col[0] for col in cursor.description] if cursor.description else []
            rows = cursor.fetchall()

            return [dict(zip(columns, row)) for row in rows]
        finally:
            cursor.close()


async def execute_query(query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Execute a Snowflake query asynchronously."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _execute_query_sync, query, params)


async def test_connection() -> bool:
    """Test Snowflake connection."""
    try:
        result = await execute_query("SELECT CURRENT_TIMESTAMP() AS ts")
        return len(result) > 0
    except Exception:
        return False
