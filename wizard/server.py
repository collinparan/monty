#!/usr/bin/env python3
"""
Monty Setup Wizard - Backend Server

A simple FastAPI server that:
1. Serves the setup wizard HTML
2. Generates prd.json from user input
3. Launches Monty and streams progress via WebSocket
"""

import asyncio
import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

app = FastAPI(title="Monty Setup Wizard")

# Store active sessions
sessions: dict[str, dict] = {}


class TechStack(BaseModel):
    backend: list[str] = ["fastapi"]
    database: list[str] = ["postgresql"]
    features: list[str] = []


class LegacyDatabase(BaseModel):
    type: str = ""
    connectionInfo: str = ""


class RestApi(BaseModel):
    baseUrl: str = ""
    authType: str = ""
    endpoints: str = ""


class FileSource(BaseModel):
    location: str = ""


class DataSourceDetails(BaseModel):
    legacyDatabase: Optional[LegacyDatabase] = None
    restApi: Optional[RestApi] = None
    files: Optional[FileSource] = None


class DiscoveredSchema(BaseModel):
    tables: Optional[list[dict]] = None
    endpoints: Optional[list[dict]] = None


class ProjectConfig(BaseModel):
    projectName: str
    description: str
    outputDir: str = "./output"
    techStack: TechStack = TechStack()
    dataSources: list[str] = []
    dataSourceDetails: DataSourceDetails = DataSourceDetails()
    discoveredSchema: Optional[DiscoveredSchema] = None


def generate_user_stories(config: ProjectConfig) -> list[dict]:
    """Generate user stories based on project config."""
    stories = []
    priority = 1

    # Always start with Docker Compose setup
    stories.append({
        "id": "US-001",
        "title": "Docker Compose infrastructure",
        "description": f"Set up Docker Compose with services for {config.projectName}",
        "acceptanceCriteria": [
            "docker-compose.yml with all required services",
            "Health checks for all services",
            "Persistent volumes for data",
            ".env.example with required variables",
            "All services start with `docker-compose up -d`"
        ],
        "priority": priority,
        "status": "pending"
    })
    priority += 1

    # Backend scaffolding based on choice
    backend = config.techStack.backend[0] if config.techStack.backend else "fastapi"
    if backend == "fastapi":
        stories.append({
            "id": f"US-{priority:03d}",
            "title": "FastAPI project scaffolding",
            "description": "Create the base FastAPI application structure",
            "acceptanceCriteria": [
                "app/main.py with FastAPI app, CORS, lifespan",
                "app/config.py with Pydantic Settings",
                "Dockerfile for the application",
                "pyproject.toml with dependencies",
                "/health endpoint returns 200"
            ],
            "priority": priority,
            "status": "pending"
        })
    elif backend == "express":
        stories.append({
            "id": f"US-{priority:03d}",
            "title": "Express.js project scaffolding",
            "description": "Create the base Express application structure",
            "acceptanceCriteria": [
                "src/index.js with Express app setup",
                "Environment config with dotenv",
                "Dockerfile for the application",
                "package.json with dependencies",
                "/health endpoint returns 200"
            ],
            "priority": priority,
            "status": "pending"
        })
    elif backend == "flask":
        stories.append({
            "id": f"US-{priority:03d}",
            "title": "Flask project scaffolding",
            "description": "Create the base Flask application structure",
            "acceptanceCriteria": [
                "app/__init__.py with Flask app factory",
                "app/config.py with configuration classes",
                "Dockerfile for the application",
                "requirements.txt with dependencies",
                "/health endpoint returns 200"
            ],
            "priority": priority,
            "status": "pending"
        })
    priority += 1

    # Database setup
    for db in config.techStack.database:
        if db == "postgresql":
            stories.append({
                "id": f"US-{priority:03d}",
                "title": "PostgreSQL database setup",
                "description": "Configure PostgreSQL with migrations",
                "acceptanceCriteria": [
                    "Database service in docker-compose.yml",
                    "Database connection configuration",
                    "Migration setup (Alembic for Python, Knex for Node)",
                    "Initial migration with base tables",
                    "Connection pool configuration"
                ],
                "priority": priority,
                "status": "pending"
            })
            priority += 1
        elif db == "mongodb":
            stories.append({
                "id": f"US-{priority:03d}",
                "title": "MongoDB database setup",
                "description": "Configure MongoDB connection",
                "acceptanceCriteria": [
                    "MongoDB service in docker-compose.yml",
                    "Database connection configuration",
                    "Index setup for common queries",
                    "Connection health check"
                ],
                "priority": priority,
                "status": "pending"
            })
            priority += 1
        elif db == "redis":
            stories.append({
                "id": f"US-{priority:03d}",
                "title": "Redis cache setup",
                "description": "Configure Redis for caching and sessions",
                "acceptanceCriteria": [
                    "Redis service in docker-compose.yml",
                    "Redis client configuration",
                    "Cache helper functions",
                    "Session storage (if auth enabled)"
                ],
                "priority": priority,
                "status": "pending"
            })
            priority += 1

    # Feature-based stories
    if "auth" in config.techStack.features:
        stories.append({
            "id": f"US-{priority:03d}",
            "title": "User authentication",
            "description": "Implement user registration and login",
            "acceptanceCriteria": [
                "User model with email, password hash",
                "POST /auth/register endpoint",
                "POST /auth/login endpoint with JWT",
                "Password hashing with bcrypt",
                "Protected route middleware",
                "Unit tests for auth flow"
            ],
            "priority": priority,
            "status": "pending"
        })
        priority += 1

    if "websocket" in config.techStack.features:
        stories.append({
            "id": f"US-{priority:03d}",
            "title": "WebSocket realtime support",
            "description": "Implement WebSocket for realtime communication",
            "acceptanceCriteria": [
                "WebSocket endpoint setup",
                "Connection manager for multiple clients",
                "JSON message protocol",
                "Graceful disconnect handling",
                "Integration test with WS client"
            ],
            "priority": priority,
            "status": "pending"
        })
        priority += 1

    if "ai" in config.techStack.features:
        stories.append({
            "id": f"US-{priority:03d}",
            "title": "AI/LLM integration",
            "description": "Integrate AI capabilities with Claude or OpenAI",
            "acceptanceCriteria": [
                "LLM client service with async support",
                "Chat completion endpoint",
                "Streaming response support",
                "Token usage tracking",
                "Error handling for rate limits"
            ],
            "priority": priority,
            "status": "pending"
        })
        priority += 1

    if "api" in config.techStack.features:
        stories.append({
            "id": f"US-{priority:03d}",
            "title": "REST API CRUD endpoints",
            "description": "Implement core CRUD API endpoints",
            "acceptanceCriteria": [
                "Resource model based on project needs",
                "GET /api/resources - list with pagination",
                "POST /api/resources - create",
                "GET /api/resources/:id - read",
                "PUT /api/resources/:id - update",
                "DELETE /api/resources/:id - delete",
                "OpenAPI documentation"
            ],
            "priority": priority,
            "status": "pending"
        })
        priority += 1

    if "multi-agent" in config.techStack.features:
        # Story 1: Base CrewAI setup with 3-agent crew
        stories.append({
            "id": f"US-{priority:03d}",
            "title": "CrewAI 3-agent crew setup",
            "description": "Set up CrewAI with Manager + Data Engineer + Data Scientist agents",
            "acceptanceCriteria": [
                "crewai, langchain-anthropic, interpret in dependencies",
                "app/agents/ directory structure",
                "Manager Agent: orchestrates crew, ensures goal alignment, synthesizes results",
                "Data Engineer Agent: SQL/Cypher queries, data extraction, schema inspection",
                "Data Scientist Agent: explainable ML with InterpretML (EBM, Decision Tree, Random Forest)",
                "All agents with clear role, goal, backstory for explainability",
                "Verbose mode enabled on all agents"
            ],
            "priority": priority,
            "status": "pending"
        })
        priority += 1

        # Story 2: Data Engineer tools
        stories.append({
            "id": f"US-{priority:03d}",
            "title": "Data Engineer agent tools",
            "description": "Implement tools for SQL, Cypher, and schema inspection",
            "acceptanceCriteria": [
                "SQLQueryTool: execute SQL against PostgreSQL/MySQL",
                "CypherQueryTool: execute Cypher against Neo4j",
                "SchemaInspectorTool: inspect tables, columns, relationships",
                "Tools return formatted, readable output",
                "Connection pooling and error handling",
                "Environment variables for database credentials"
            ],
            "priority": priority,
            "status": "pending"
        })
        priority += 1

        # Story 3: Data Scientist tools with InterpretML
        stories.append({
            "id": f"US-{priority:03d}",
            "title": "Data Scientist agent tools (InterpretML)",
            "description": "Implement explainable ML tools using InterpretML",
            "acceptanceCriteria": [
                "TrainModelTool: train EBM, Decision Tree, or Random Forest",
                "PredictTool: make predictions WITH explanations",
                "ExplainModelTool: global and local feature explanations",
                "Support for ExplainableBoostingRegressor/Classifier",
                "Support for RegressionTree/ClassificationTree",
                "Support for RandomForestRegressor/Classifier with feature importance",
                "Model persistence for reuse"
            ],
            "priority": priority,
            "status": "pending"
        })
        priority += 1

        # Story 4: Tasks and crew orchestration
        stories.append({
            "id": f"US-{priority:03d}",
            "title": "CrewAI tasks and crew orchestration",
            "description": "Define tasks and crew execution flow",
            "acceptanceCriteria": [
                "PlanningTask: Manager breaks down user request",
                "DataExtractionTask: Data Engineer queries and prepares data",
                "ModelingTask: Data Scientist trains model and explains",
                "SynthesisTask: Manager synthesizes final response",
                "Task dependencies using context parameter",
                "Sequential process for predictable execution"
            ],
            "priority": priority,
            "status": "pending"
        })
        priority += 1

        # Story 5: API endpoints
        stories.append({
            "id": f"US-{priority:03d}",
            "title": "CrewAI API endpoints",
            "description": "FastAPI endpoints for crew execution",
            "acceptanceCriteria": [
                "POST /crew/start - start crew with user request",
                "GET /crew/status/{task_id} - check execution status",
                "GET /crew/result/{task_id} - get final result with explanations",
                "Background task execution for long-running crews",
                "WebSocket endpoint for real-time progress (optional)",
                "Usage metrics returned for explainability"
            ],
            "priority": priority,
            "status": "pending"
        })
        priority += 1

    # Data source integration stories
    if "legacy-db" in config.dataSources and config.dataSourceDetails.legacyDatabase:
        db_info = config.dataSourceDetails.legacyDatabase
        db_type = db_info.type or "legacy database"

        # Base connection story
        stories.append({
            "id": f"US-{priority:03d}",
            "title": f"Legacy {db_type} connection",
            "description": f"Connect to existing {db_type} database",
            "acceptanceCriteria": [
                f"Database connector for {db_type}",
                "Connection pool configuration",
                "Environment variables for credentials (never hardcode)",
                "Health check for database connectivity",
                "Data access layer/repository pattern",
                "Query helpers for common operations",
                f"Connection info: {db_info.connectionInfo[:100]}..." if len(db_info.connectionInfo) > 100 else f"Connection info: {db_info.connectionInfo}"
            ],
            "priority": priority,
            "status": "pending"
        })
        priority += 1

        # If we have discovered schema, create stories for each table
        if config.discoveredSchema and config.discoveredSchema.tables:
            for table in config.discoveredSchema.tables:
                table_name = table.get("name", "unknown")
                columns = table.get("columns", [])
                row_count = table.get("rowCount", 0)

                # Create model/repository for this table
                column_list = ", ".join([f"{c['name']} ({c['type']})" for c in columns[:5]])
                if len(columns) > 5:
                    column_list += f", ... (+{len(columns) - 5} more)"

                stories.append({
                    "id": f"US-{priority:03d}",
                    "title": f"Data model for {table_name}",
                    "description": f"Create model and repository for {table_name} table (~{row_count:,} rows)",
                    "acceptanceCriteria": [
                        f"Pydantic/dataclass model for {table_name}",
                        f"Columns: {column_list}",
                        f"Repository with CRUD operations",
                        "Async query methods",
                        "Pagination support for list queries",
                        f"Unit tests for {table_name} repository"
                    ],
                    "priority": priority,
                    "status": "pending"
                })
                priority += 1

    if "rest-api" in config.dataSources and config.dataSourceDetails.restApi:
        api_info = config.dataSourceDetails.restApi

        # Base API client story
        stories.append({
            "id": f"US-{priority:03d}",
            "title": "External REST API client",
            "description": f"HTTP client for {api_info.baseUrl or 'external API'}",
            "acceptanceCriteria": [
                "HTTP client service (aiohttp/httpx for Python, axios for Node)",
                f"Authentication: {api_info.authType or 'as specified'}",
                "Request/response logging",
                "Retry logic with exponential backoff",
                "Rate limiting handling",
                "Error handling and circuit breaker pattern"
            ],
            "priority": priority,
            "status": "pending"
        })
        priority += 1

        # If we have discovered endpoints, create stories for key ones
        if config.discoveredSchema and config.discoveredSchema.endpoints:
            # Group endpoints by resource
            for endpoint in config.discoveredSchema.endpoints[:10]:  # Limit to first 10
                method = endpoint.get("method", "GET")
                path = endpoint.get("path", "/")
                desc = endpoint.get("description", "")

                stories.append({
                    "id": f"US-{priority:03d}",
                    "title": f"Integrate {method} {path}",
                    "description": desc or f"Implement client method for {method} {path}",
                    "acceptanceCriteria": [
                        f"Client method for {method} {path}",
                        "Request/response type definitions",
                        "Error handling",
                        "Unit test with mocked response"
                    ],
                    "priority": priority,
                    "status": "pending"
                })
                priority += 1
        elif api_info.endpoints:
            # Fall back to manually specified endpoints
            stories.append({
                "id": f"US-{priority:03d}",
                "title": "API endpoint integrations",
                "description": "Implement client methods for specified endpoints",
                "acceptanceCriteria": [
                    f"Endpoints: {api_info.endpoints[:200]}..." if len(api_info.endpoints) > 200 else f"Endpoints: {api_info.endpoints}",
                    "Request/response type definitions for each",
                    "Error handling",
                    "Unit tests"
                ],
                "priority": priority,
                "status": "pending"
            })
            priority += 1

    if "graphql" in config.dataSources:
        stories.append({
            "id": f"US-{priority:03d}",
            "title": "GraphQL API integration",
            "description": "Connect to external GraphQL API",
            "acceptanceCriteria": [
                "GraphQL client setup (gql/graphql-request)",
                "Query and mutation helpers",
                "Authentication handling",
                "Error handling",
                "Response type definitions"
            ],
            "priority": priority,
            "status": "pending"
        })
        priority += 1

    if ("csv-files" in config.dataSources or "s3" in config.dataSources) and config.dataSourceDetails.files:
        file_info = config.dataSourceDetails.files
        stories.append({
            "id": f"US-{priority:03d}",
            "title": "File/S3 data ingestion",
            "description": "Import data from files or cloud storage",
            "acceptanceCriteria": [
                "File reader service (pandas for Python, papaparse for Node)",
                "S3 client if cloud storage (boto3/aws-sdk)",
                "Data validation and schema enforcement",
                "Batch import functionality",
                "Progress tracking for large files",
                "Error handling for malformed data",
                f"Location: {file_info.location}" if file_info.location else "Configurable file location"
            ],
            "priority": priority,
            "status": "pending"
        })
        priority += 1

    # Always end with integration tests
    stories.append({
        "id": f"US-{priority:03d}",
        "title": "Integration tests and documentation",
        "description": "Add tests and README documentation",
        "acceptanceCriteria": [
            "Test setup with pytest/jest",
            "Integration tests for main endpoints",
            "README.md with setup instructions",
            "API documentation",
            "All tests pass"
        ],
        "priority": priority,
        "status": "pending"
    })

    return stories


def generate_prd(config: ProjectConfig, stories: list[dict]) -> dict:
    """Generate the full PRD document."""
    backend = config.techStack.backend[0] if config.techStack.backend else "fastapi"

    prd = {
        "projectName": config.projectName,
        "description": config.description,
        "outputDir": config.outputDir,
        "techStack": {
            "backend": backend,
            "databases": config.techStack.database,
            "features": config.techStack.features,
            "orchestration": "Docker Compose"
        },
        "requirements": [
            f"All code must be in a self-contained Docker Compose stack in {config.outputDir}",
            "docker-compose up -d must start all services",
            "Include .env.example with all required environment variables",
            "Include README.md with setup and usage instructions",
            "All services must have health checks"
        ],
        "userStories": stories
    }

    # Add data sources if specified
    if config.dataSources and "none" not in config.dataSources:
        prd["dataSources"] = {
            "types": config.dataSources,
            "details": {}
        }

        if config.dataSourceDetails.legacyDatabase:
            db = config.dataSourceDetails.legacyDatabase
            prd["dataSources"]["details"]["legacyDatabase"] = {
                "type": db.type,
                "connectionInfo": db.connectionInfo
            }
            prd["requirements"].append(
                f"Connect to existing {db.type} database - credentials via environment variables"
            )

        if config.dataSourceDetails.restApi:
            api = config.dataSourceDetails.restApi
            prd["dataSources"]["details"]["restApi"] = {
                "baseUrl": api.baseUrl,
                "authType": api.authType,
                "endpoints": api.endpoints
            }
            prd["requirements"].append(
                f"Integrate with external REST API at {api.baseUrl}"
            )

        if config.dataSourceDetails.files:
            files = config.dataSourceDetails.files
            prd["dataSources"]["details"]["files"] = {
                "location": files.location
            }
            prd["requirements"].append(
                f"Import data from files at {files.location}"
            )

        # Include discovered schema if available
        if config.discoveredSchema:
            prd["dataSources"]["discoveredSchema"] = {}

            if config.discoveredSchema.tables:
                prd["dataSources"]["discoveredSchema"]["tables"] = config.discoveredSchema.tables
                table_names = [t.get("name") for t in config.discoveredSchema.tables]
                prd["requirements"].append(
                    f"Create data models for tables: {', '.join(table_names)}"
                )

            if config.discoveredSchema.endpoints:
                prd["dataSources"]["discoveredSchema"]["endpoints"] = config.discoveredSchema.endpoints
                endpoint_count = len(config.discoveredSchema.endpoints)
                prd["requirements"].append(
                    f"Implement client methods for {endpoint_count} discovered API endpoints"
                )

    return prd


@app.get("/")
async def serve_index():
    """Serve the setup wizard HTML."""
    return FileResponse(SCRIPT_DIR / "index.html")


class ConnectionTestRequest(BaseModel):
    type: str  # 'database' or 'rest-api'
    dbType: Optional[str] = None
    connectionInfo: Optional[str] = None
    baseUrl: Optional[str] = None
    authType: Optional[str] = None


@app.post("/api/test-connection")
async def test_connection(request: ConnectionTestRequest):
    """Test connection to a data source and discover its schema."""
    try:
        if request.type == "database":
            return await test_database_connection(request)
        elif request.type == "rest-api":
            return await test_api_connection(request)
        else:
            return JSONResponse({"success": False, "error": "Unknown connection type"})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


async def test_database_connection(request: ConnectionTestRequest):
    """Test database connection and discover schema."""
    import asyncio

    # Parse connection info to extract connection details
    # For now, we'll try to parse common formats and test connectivity
    conn_info = request.connectionInfo or ""
    db_type = request.dbType or ""

    # Try to extract host/database from connection info
    # Common patterns: "host:port/database", "host database_name", connection string
    schema = {"tables": []}

    if db_type == "postgresql":
        try:
            import asyncpg
            # Try to parse connection string or components
            # Look for patterns like: host=X dbname=Y or postgresql://...
            conn_string = None

            if "postgresql://" in conn_info or "postgres://" in conn_info:
                conn_string = conn_info.strip()
            else:
                # Try to build connection from description
                # This is a simplified parser - in production you'd want more robust parsing
                parts = {}
                for part in conn_info.replace(",", " ").split():
                    if "=" in part:
                        k, v = part.split("=", 1)
                        parts[k.lower()] = v
                    elif "." in part or part.replace("-", "").isalnum():
                        if "host" not in parts:
                            parts["host"] = part

                if parts:
                    host = parts.get("host", "localhost")
                    port = parts.get("port", "5432")
                    database = parts.get("database", parts.get("dbname", "postgres"))
                    user = parts.get("user", parts.get("username", "postgres"))
                    password = parts.get("password", "")

                    conn_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"

            if conn_string:
                conn = await asyncio.wait_for(
                    asyncpg.connect(conn_string),
                    timeout=10.0
                )

                # Get table list with row counts
                tables = await conn.fetch("""
                    SELECT
                        t.table_name,
                        (SELECT count(*) FROM information_schema.columns c
                         WHERE c.table_name = t.table_name AND c.table_schema = 'public') as column_count
                    FROM information_schema.tables t
                    WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE'
                    ORDER BY t.table_name
                """)

                for table in tables:
                    # Get columns for each table
                    columns = await conn.fetch("""
                        SELECT column_name, data_type, is_nullable
                        FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = $1
                        ORDER BY ordinal_position
                    """, table["table_name"])

                    # Get approximate row count
                    try:
                        row_count = await conn.fetchval(
                            f"SELECT reltuples::bigint FROM pg_class WHERE relname = $1",
                            table["table_name"]
                        )
                    except Exception:
                        row_count = 0

                    schema["tables"].append({
                        "name": table["table_name"],
                        "columns": [
                            {"name": c["column_name"], "type": c["data_type"], "nullable": c["is_nullable"] == "YES"}
                            for c in columns
                        ],
                        "rowCount": row_count or 0
                    })

                await conn.close()
                return JSONResponse({"success": True, "schema": schema})

        except asyncio.TimeoutError:
            return JSONResponse({"success": False, "error": "Connection timed out"})
        except ImportError:
            return JSONResponse({
                "success": False,
                "error": "asyncpg not installed. Install with: pip install asyncpg"
            })
        except Exception as e:
            return JSONResponse({"success": False, "error": f"PostgreSQL connection failed: {str(e)}"})

    elif db_type == "mysql":
        try:
            import aiomysql
            # Similar parsing logic for MySQL
            return JSONResponse({
                "success": False,
                "error": "MySQL schema discovery not yet implemented. Please describe your tables manually."
            })
        except ImportError:
            return JSONResponse({
                "success": False,
                "error": "aiomysql not installed. Install with: pip install aiomysql"
            })

    # For other database types, return a helpful message
    return JSONResponse({
        "success": True,
        "schema": {"tables": []},
        "message": f"Connection info recorded for {db_type}. Schema discovery for this database type is manual - please describe your tables in the connection info field."
    })


async def test_api_connection(request: ConnectionTestRequest):
    """Test REST API connection and try to discover endpoints."""
    import aiohttp

    base_url = (request.baseUrl or "").rstrip("/")
    if not base_url:
        return JSONResponse({"success": False, "error": "Base URL is required"})

    schema = {"endpoints": []}

    try:
        async with aiohttp.ClientSession() as session:
            # Try common discovery endpoints
            discovery_paths = [
                "/openapi.json",
                "/swagger.json",
                "/api/swagger.json",
                "/api/openapi.json",
                "/api-docs",
                "/v1/openapi.json",
                "/docs/openapi.json",
            ]

            # First, try to reach the base URL
            try:
                async with session.get(base_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status >= 400:
                        # Try with /health or /api
                        for path in ["/health", "/api", "/api/health"]:
                            try:
                                async with session.get(f"{base_url}{path}", timeout=aiohttp.ClientTimeout(total=5)) as r:
                                    if r.status < 400:
                                        break
                            except Exception:
                                continue
            except Exception as e:
                return JSONResponse({"success": False, "error": f"Cannot reach {base_url}: {str(e)}"})

            # Try to find OpenAPI/Swagger spec
            for path in discovery_paths:
                try:
                    async with session.get(f"{base_url}{path}", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            try:
                                spec = await resp.json()
                                # Parse OpenAPI spec
                                if "paths" in spec:
                                    for path_name, methods in spec["paths"].items():
                                        for method, details in methods.items():
                                            if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                                                schema["endpoints"].append({
                                                    "method": method.upper(),
                                                    "path": path_name,
                                                    "description": details.get("summary", details.get("description", ""))[:100]
                                                })
                                    return JSONResponse({"success": True, "schema": schema})
                            except Exception:
                                continue
                except Exception:
                    continue

            # No OpenAPI spec found - return success but note manual entry needed
            return JSONResponse({
                "success": True,
                "schema": {"endpoints": []},
                "message": "Connected successfully. No OpenAPI spec found - please list endpoints manually."
            })

    except ImportError:
        return JSONResponse({
            "success": False,
            "error": "aiohttp not installed. Install with: pip install aiohttp"
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


@app.post("/api/project")
async def create_project(config: ProjectConfig):
    """Create a new project and start Monty."""
    session_id = str(uuid.uuid4())

    # Generate stories and PRD
    stories = generate_user_stories(config)
    prd = generate_prd(config, stories)

    # Save PRD to scripts directory
    prd_path = SCRIPTS_DIR / "prd.json"
    with open(prd_path, "w") as f:
        json.dump(prd, f, indent=2)

    # Initialize progress file
    progress_path = SCRIPTS_DIR / "progress.txt"
    with open(progress_path, "w") as f:
        f.write(f"# Monty Progress Log\n\n")
        f.write(f"## Project: {config.projectName}\n\n")
        f.write(f"Started: Building {config.description}\n\n")
        f.write(f"Output directory: {config.outputDir}\n\n")
        f.write("---\n\n")
        f.write("## Story Progress\n\n")

    # Store session
    sessions[session_id] = {
        "config": config.model_dump(),
        "stories": stories,
        "prd": prd,
        "process": None,
        "websockets": []
    }

    return JSONResponse({
        "sessionId": session_id,
        "stories": stories,
        "prdPath": str(prd_path)
    })


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming Monty progress."""
    await websocket.accept()

    if session_id not in sessions:
        await websocket.send_json({"type": "error", "message": "Invalid session"})
        await websocket.close()
        return

    session = sessions[session_id]
    session["websockets"].append(websocket)

    try:
        # Start Monty if not already running
        if session["process"] is None:
            await start_monty(session_id, websocket)

        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                # Handle any client messages if needed
            except asyncio.TimeoutError:
                # Check if Monty is still running
                if session["process"] and session["process"].poll() is not None:
                    break
            except WebSocketDisconnect:
                break

    finally:
        if websocket in session["websockets"]:
            session["websockets"].remove(websocket)


async def start_monty(session_id: str, websocket: WebSocket):
    """Start Monty and stream its output."""
    session = sessions[session_id]

    await websocket.send_json({
        "type": "log",
        "message": "Starting Monty...",
        "level": "info"
    })

    # Build the command - run Claude with the Monty prompt
    monty_script = PROJECT_ROOT / "start.sh"

    # Start Monty process
    process = subprocess.Popen(
        [str(monty_script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "MONTY_WIZARD": "1"}
    )

    session["process"] = process

    # Stream output
    current_story = None

    async def read_output():
        nonlocal current_story

        while True:
            line = await asyncio.get_event_loop().run_in_executor(
                None, process.stdout.readline
            )

            if not line:
                if process.poll() is not None:
                    break
                continue

            line = line.strip()
            if not line:
                continue

            # Parse Monty events from output
            event = parse_monty_output(line, current_story, session["stories"])

            if event:
                if event.get("type") == "story_start":
                    current_story = event.get("storyId")
                elif event.get("type") == "story_complete":
                    # Update story status
                    for story in session["stories"]:
                        if story["id"] == event.get("storyId"):
                            story["status"] = "complete"
                            break

                # Send to all connected websockets
                for ws in session["websockets"]:
                    try:
                        await ws.send_json(event)
                    except Exception:
                        pass

        # Check final status
        exit_code = process.poll()
        final_event = {
            "type": "complete" if exit_code == 0 else "error",
            "message": "Monty finished" if exit_code == 0 else f"Monty exited with code {exit_code}"
        }

        for ws in session["websockets"]:
            try:
                await ws.send_json(final_event)
            except Exception:
                pass

    asyncio.create_task(read_output())


def parse_monty_output(line: str, current_story: Optional[str], stories: list[dict]) -> Optional[dict]:
    """Parse Monty's output and convert to events."""

    # Check for iteration markers (the famous loop!)
    if match := re.search(r'Monty Iteration (\d+) of (\d+)', line):
        return {
            "type": "iteration",
            "current": int(match.group(1)),
            "total": int(match.group(2)),
            "message": line
        }

    # Check for story start markers [US-XXX] Starting:
    if match := re.search(r'\[(US-\d+)\]\s*Starting:\s*(.+)', line):
        story_id = match.group(1)
        title = match.group(2)
        return {
            "type": "story_start",
            "storyId": story_id,
            "title": title
        }

    # Check for story complete markers [US-XXX] Complete
    if match := re.search(r'\[(US-\d+)\]\s*Complete', line):
        story_id = match.group(1)
        return {
            "type": "story_complete",
            "storyId": story_id
        }

    # Check for completion markers
    if "<monty>COMPLETE</monty>" in line:
        return {"type": "complete", "message": "All stories completed!"}

    if "<monty>BLOCKED</monty>" in line:
        return {"type": "blocked", "message": line}

    # Check for tool calls (Claude Code outputs these)
    if "Tool:" in line or "Bash:" in line or "Edit:" in line or "Write:" in line:
        return {
            "type": "tool_call",
            "tool": line.split(":")[0] if ":" in line else "Tool",
            "description": line
        }

    # Check for git commits (backup story completion indicator)
    if "git commit" in line.lower() and current_story:
        return {
            "type": "story_complete",
            "storyId": current_story
        }

    # Default: just log the line
    level = "info"
    if "error" in line.lower():
        level = "error"
    elif "success" in line.lower() or "complete" in line.lower():
        level = "success"

    return {
        "type": "log",
        "message": line,
        "level": level
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3456)
