"""
FastAPI Application Entry Point — Member 5 (Backend Gateway Layer).

Responsibilities:
  - Create and configure the FastAPI app instance.
  - Register CORS middleware.
  - Mount the REST API router (/api/*) and the WebSocket router (/ws/*).
  - Expose the /health liveness endpoint.
  - Initialise the database (create tables) on startup.
  - Optionally seed Member 4's ChromaDB knowledge base if available.

Collaboration boundaries:
  - Does NOT import backend.core.graph (owned by Member 1).
  - Does NOT import backend.agents.* at module level.
  - Does NOT import backend.core.file_manager (owned by Member 2).
  - Any imports from other members are done lazily/optionally in startup.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import api_router
from backend.api.websocket import ws_router
from backend.config import config
from backend.schemas.models import _db_backend, init_db

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup:
      1. Log active database backend.
      2. Create all ORM tables.
      3. Optionally seed ChromaDB (if Member 4's vector_store is available).

    Shutdown:
      - Logs a clean shutdown message (DB connections close automatically
        because SQLAlchemy pool_pre_ping handles stale connections).
    """
    # --- Database initialisation ---
    logger.info("Database backend: %s", _db_backend.upper())
    logger.info("Database URL: %s", _safe_db_url(config.DATABASE_URL or ""))
    try:
        init_db()
        logger.info("Database tables initialised successfully.")
    except Exception as exc:
        logger.error("Database initialisation failed: %s", exc)
        # Do NOT raise — let the server start so /health still responds.

    # --- Optional: seed ChromaDB knowledge base (Member 4) ---
    _try_seed_chroma()

    # --- Log graph availability ---
    _log_graph_status()

    logger.info(
        "AutoDev backend started — host=%s  port=%s",
        config.HOST,
        config.PORT,
    )
    yield

    # Shutdown
    logger.info("AutoDev backend shutting down.")


def _safe_db_url(url: str) -> str:
    """Redact the password from the DB URL for safe logging."""
    if "@" in url:
        scheme, rest = url.split("://", 1)
        credentials, host_part = rest.rsplit("@", 1)
        user = credentials.split(":")[0]
        return f"{scheme}://{user}:***@{host_part}"
    return url


def _try_seed_chroma() -> None:
    """
    Attempt to seed Member 4's ChromaDB knowledge base.
    Fails silently if Member 4's branch has not yet been merged.
    """
    try:
        from backend.core.vector_store import seed_knowledge_base  # type: ignore
        seed_knowledge_base()
        logger.info("ChromaDB knowledge base seeded (Member 4).")
    except ModuleNotFoundError:
        logger.info(
            "ChromaDB seed skipped — backend/core/vector_store.py not yet merged."
        )
    except Exception as exc:
        logger.warning("ChromaDB seed failed: %s", exc)


def _log_graph_status() -> None:
    """Log whether the LangGraph orchestrator is currently available."""
    try:
        import importlib
        mod = importlib.import_module("backend.core.graph")
        if hasattr(mod, "app_graph"):
            logger.info(
                "LangGraph orchestrator: AVAILABLE (backend.core.graph.app_graph)"
            )
        else:
            logger.warning(
                "LangGraph orchestrator: backend/core/graph.py found but "
                "'app_graph' attribute is missing. "
                "POST /api/generate will return 503."
            )
    except ModuleNotFoundError:
        logger.warning(
            "LangGraph orchestrator: NOT YET AVAILABLE "
            "(backend/core/graph.py not merged). "
            "POST /api/generate will return HTTP 503 until Member 1's PR is merged."
        )
    except Exception as exc:
        logger.warning("LangGraph graph check failed: %s", exc)


# ---------------------------------------------------------------------------
# CORS origins
# ---------------------------------------------------------------------------

def _get_cors_origins() -> list[str]:
    """
    Read allowed CORS origins from CORS_ORIGINS env var (comma-separated).
    Falls back to safe development defaults if not set.
    """
    env_origins = os.getenv("CORS_ORIGINS", "")
    if env_origins.strip():
        return [o.strip() for o in env_origins.split(",") if o.strip()]
    # Safe development defaults (Vite default port + common alternatives)
    return [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AutoDev — AI DevTeam Backend",
    description=(
        "Multi-agent autonomous software engineering backend. "
        "Orchestrated by LangGraph (Member 1). "
        "Gateway layer implemented by Member 5."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(api_router, prefix="/api")
app.include_router(ws_router)           # WebSocket routes have their own /ws/* prefix


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    tags=["Health"],
    summary="Liveness check",
    response_description="Server is alive and database is reachable.",
)
def health_check() -> dict:
    """
    Returns 200 OK if the server is running and the database layer is
    operational.  Also reports whether the LangGraph orchestrator is
    available so the team can verify integration status without reading logs.
    """
    # Quick DB probe
    db_ok = True
    try:
        from backend.schemas.models import engine
        with engine.connect():
            pass
    except Exception:
        db_ok = False

    # Graph probe (re-uses the same lazy check as the adapter)
    graph_available = False
    try:
        import importlib
        mod = importlib.import_module("backend.core.graph")
        graph_available = hasattr(mod, "app_graph")
    except Exception:
        pass

    return {
        "status": "ok",
        "database": "connected" if db_ok else "unavailable",
        "database_backend": _db_backend,
        "orchestration_graph": "available" if graph_available else "not_yet_merged",
        "version": "1.0.0",
    }


# ---------------------------------------------------------------------------
# Uvicorn entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        log_level="info",
    )

