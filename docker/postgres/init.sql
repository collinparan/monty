-- Initialize PostgreSQL with PGVector extension
-- This script runs automatically when the container is first created

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Verify extensions are installed
SELECT
    extname,
    extversion
FROM
    pg_extension
WHERE
    extname IN ('vector', 'uuid-ossp');
