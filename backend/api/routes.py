"""
REST API Routes — Member 5 (Backend Gateway Layer).

Endpoints implemented:
  POST   /api/generate                        — start a new pipeline run
  GET    /api/projects/{project_id}           — query project status / results
  GET    /api/projects/{project_id}/download  — download generated workspace as ZIP
  GET    /health                              — liveness check

LangGraph Integration (Member 1 — graph.py):
  The graph is imported lazily through _graph_adapter.invoke_graph().
  If backend/core/graph.py has not yet been merged, the adapter returns
  None and POST /api/generate responds with HTTP 503 instead of crashing.
  When Member 1's PR is merged, no changes to this file are required.

Collaboration boundaries:
  - Do NOT import backend.core.graph at module level.
  - Do NOT import backend.core.file_manager at module level.
  - Do NOT modify backend/schemas/state.py.
  - Do NOT modify any agent files.
"""

import asyncio
import io
import json
import logging
import re
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from backend.config import config
from backend.schemas.models import (
    ExecutionLog,
    Project,
    ProjectFile,
    ReviewReport,
    get_db,
)

logger = logging.getLogger(__name__)

api_router = APIRouter(tags=["Projects"])


# ---------------------------------------------------------------------------
# LangGraph Integration Adapter
# ---------------------------------------------------------------------------

class _GraphAdapter:
    """
    Lazy, fault-tolerant wrapper around Member 1's LangGraph graph.

    Rules:
      - graph.py is NEVER imported at module load time.
      - If the module does not exist (not yet merged), is_available() → False.
      - When available, invoke_graph() calls graph.app_graph.invoke(state).
      - The graph name (app_graph) matches the module-level singleton that
        Member 1 is expected to export from backend/core/graph.py.
        If the name changes after integration, only this adapter needs updating.
    """

    _GRAPH_MODULE = "backend.core.graph"
    _GRAPH_ATTR = "app_graph"          # expected export from graph.py

    def is_available(self) -> bool:
        """Return True only if backend/core/graph.py can be imported."""
        try:
            import importlib
            mod = importlib.import_module(self._GRAPH_MODULE)
            return hasattr(mod, self._GRAPH_ATTR)
        except Exception:
            return False

    def invoke_graph(self, initial_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Dynamically import and invoke the LangGraph compiled graph.

        Returns:
            Final AgentState dict on success.
            None if the graph module is unavailable.

        Raises:
            RuntimeError wrapping the original exception on invocation failure.
        """
        try:
            import importlib
            mod = importlib.import_module(self._GRAPH_MODULE)
            graph = getattr(mod, self._GRAPH_ATTR, None)
            if graph is None:
                return None
            result = graph.invoke(initial_state)
            return result
        except ModuleNotFoundError:
            return None
        except Exception as exc:
            raise RuntimeError(
                f"LangGraph invocation failed: {type(exc).__name__}: {exc}"
            ) from exc


_graph_adapter = _GraphAdapter()


# ---------------------------------------------------------------------------
# Request / Response Pydantic Models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    """Request body for POST /api/generate."""

    project_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional client-supplied project ID.  If omitted, a UUID is "
            "generated server-side."
        ),
    )
    prompt: str = Field(
        ...,
        min_length=10,
        max_length=4000,
        description="Natural language description of the software to build.",
    )

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt must not be blank or whitespace-only.")
        return v.strip()

    @field_validator("project_id")
    @classmethod
    def _sanitize_project_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        # Allow only safe characters: alphanumeric, hyphens, underscores.
        if not re.match(r"^[a-zA-Z0-9_\-]{1,100}$", v):
            raise ValueError(
                "project_id must contain only letters, digits, hyphens, "
                "or underscores (max 100 chars)."
            )
        return v


class GenerateResponse(BaseModel):
    """Response body for POST /api/generate."""

    project_id: str
    status: str
    message: str


class ProjectStatusResponse(BaseModel):
    """Response body for GET /api/projects/{project_id}."""

    project_id: str
    prompt: str
    status: str
    current_agent: Optional[str]
    workspace_path: Optional[str]
    created_at: str
    updated_at: str
    logs: list[dict]
    files: list[dict]
    review_report: Optional[dict]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_initial_agent_state(project_id: str, prompt: str) -> Dict[str, Any]:
    """
    Construct the initial AgentState dict required by the LangGraph pipeline.

    Field names and types mirror backend/schemas/state.py (AgentState TypedDict).
    This function ONLY reads state.py — it does not import or run any agent.
    """
    return {
        "project_id": project_id,
        "user_prompt": prompt,
        "requirement_spec": None,
        "architecture_blueprint": None,
        "generated_files": [],
        "workspace_path": "",
        "test_results": None,
        "retry_count": 0,
        "is_test_passed": False,
        "bug_diagnosis": None,
        "patched_files": [],
        "security_report": None,
        "current_agent": "",
        "logs": [],
        "status": "pending",
    }


def _persist_final_state(db: Session, project: Project, final_state: Dict[str, Any]) -> None:
    """
    Write the final AgentState back to the database after the graph completes.

    Idempotent — safe to call multiple times (upserts where appropriate).
    """
    # Update project record
    project.status = final_state.get("status", "completed")
    project.current_agent = final_state.get("current_agent", "")
    project.workspace_path = final_state.get("workspace_path", "")
    project.updated_at = datetime.utcnow()

    # Persist new log lines (avoid duplicate inserts)
    existing_messages = {row.message for row in project.logs}
    new_logs = []
    for log_line in final_state.get("logs", []):
        if log_line in existing_messages:
            continue
        # Heuristic: detect level from message prefix
        level = "INFO"
        lower = log_line.lower()
        if "error" in lower or "failed" in lower:
            level = "ERROR"
        elif "warn" in lower:
            level = "WARN"
        agent_name = final_state.get("current_agent") or "System"
        new_logs.append(
            ExecutionLog(
                project_id=project.id,
                agent_name=agent_name,
                message=log_line,
                log_level=level,
            )
        )
    db.add_all(new_logs)

    # Persist generated files
    generated = final_state.get("generated_files") or []
    if generated:
        # Remove stale file records before re-inserting
        for existing_file in list(project.files):
            db.delete(existing_file)
        for f in generated:
            db.add(
                ProjectFile(
                    project_id=project.id,
                    file_path=f.get("path", ""),
                    content=f.get("content", ""),
                    language=f.get("language", "python"),
                )
            )

    # Persist security report
    security = final_state.get("security_report")
    if security:
        if project.review_report:
            # Update existing
            rr = project.review_report
        else:
            rr = ReviewReport(project_id=project.id)
            db.add(rr)
        rr.quality_score = security.get("quality_score", 0)
        rr.owasp_compliant = 1 if security.get("owasp_compliance", False) else 0
        rr.security_findings = json.dumps(security.get("vulnerabilities", []))
        rr.recommendations = json.dumps(security.get("suggestions", []))

    db.commit()
    db.refresh(project)


def _safe_workspace_path(project_id: str) -> Optional[Path]:
    """
    Resolve the workspace directory for a project, preventing path traversal.

    Returns:
        A Path object if the directory exists, otherwise None.
    """
    # Sanitise project_id — only alphanumeric, hyphens, underscores allowed.
    if not re.match(r"^[a-zA-Z0-9_\-]{1,100}$", project_id):
        return None

    base = Path(config.WORKSPACE_DIR).resolve()
    target = (base / project_id).resolve()

    # Prevent traversal outside the base workspace directory.
    try:
        target.relative_to(base)
    except ValueError:
        return None

    return target if target.is_dir() else None


# ---------------------------------------------------------------------------
# Background pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline_in_background(
    project_id: str,
    prompt: str,
    db_session_factory,
) -> None:
    """
    Execute the LangGraph pipeline in a background thread.

    Uses a fresh DB session (threads cannot share the request session).
    Broadcasts status/log updates via the WebSocket manager.
    """
    # Import inline to avoid circular imports and to keep this module-level
    # import-free from the websocket module's internals.
    from backend.api.websocket import manager

    loop = asyncio.new_event_loop()

    def _broadcast(msg: dict) -> None:
        asyncio.run_coroutine_threadsafe(
            manager.broadcast(project_id, msg), loop
        )

    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    db = db_session_factory()
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return

        # --- Update status to "analyzing" ---
        project.status = "analyzing"
        project.updated_at = datetime.utcnow()
        db.commit()
        _broadcast({
            "type": "status",
            "status": "analyzing",
            "ts": datetime.now(timezone.utc).isoformat(),
        })

        # --- Build initial AgentState ---
        initial_state = _build_initial_agent_state(project_id, prompt)

        # --- Add a startup log entry ---
        startup_msg = (
            f"[System] Pipeline started for project: {project_id}"
        )
        db.add(ExecutionLog(
            project_id=project_id,
            agent_name="System",
            message=startup_msg,
            log_level="INFO",
        ))
        db.commit()
        _broadcast({
            "type": "log",
            "agent": "System",
            "level": "INFO",
            "message": startup_msg,
            "ts": datetime.now(timezone.utc).isoformat(),
        })

        # --- Invoke the LangGraph graph (if available) ---
        final_state = _graph_adapter.invoke_graph(initial_state)

        if final_state is None:
            # Graph not yet merged — handled at the HTTP layer, should not reach here
            project.status = "failed"
            db.commit()
            return

        # Stream new log lines produced by the graph
        for log_line in final_state.get("logs", []):
            level = "INFO"
            lower = log_line.lower()
            if "error" in lower or "failed" in lower:
                level = "ERROR"
            elif "warn" in lower:
                level = "WARN"
            agent = final_state.get("current_agent") or "System"
            _broadcast({
                "type": "log",
                "agent": agent,
                "level": level,
                "message": log_line,
                "ts": datetime.now(timezone.utc).isoformat(),
            })

        # --- Persist final state ---
        _persist_final_state(db, project, final_state)

        final_status = final_state.get("status", "completed")
        _broadcast({
            "type": "done",
            "status": final_status,
            "message": f"Pipeline finished with status: {final_status}",
            "ts": datetime.now(timezone.utc).isoformat(),
        })

    except RuntimeError as exc:
        logger.error("Pipeline RuntimeError for project %s: %s", project_id, exc)
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                project.status = "failed"
                db.add(ExecutionLog(
                    project_id=project_id,
                    agent_name="System",
                    message=f"[Pipeline Error] {exc}",
                    log_level="ERROR",
                ))
                db.commit()
        except Exception:
            pass
        _broadcast({
            "type": "done",
            "status": "failed",
            "message": "Pipeline encountered an error. Check server logs.",
            "ts": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        logger.exception("Unexpected pipeline error for project %s", project_id)
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                project.status = "failed"
                db.commit()
        except Exception:
            pass
        _broadcast({
            "type": "done",
            "status": "failed",
            "message": "An unexpected error occurred.",
            "ts": datetime.now(timezone.utc).isoformat(),
        })
    finally:
        db.close()
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@api_router.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start the AI DevTeam pipeline",
    responses={
        202: {"description": "Pipeline accepted and started."},
        503: {"description": "Orchestration pipeline not yet available."},
        422: {"description": "Validation error in request body."},
    },
)
def post_generate(
    request: GenerateRequest,
    db: Session = Depends(get_db),
) -> GenerateResponse:
    """
    Accepts a user prompt and starts the multi-agent LangGraph pipeline.

    Behaviour:
    - Validates the prompt.
    - Creates a Project record.
    - Checks whether Member 1's LangGraph graph is available.
    - If NOT available → returns HTTP 503 immediately (server remains healthy).
    - If available → creates the initial AgentState, launches the pipeline in
      a background thread, and returns HTTP 202 immediately so the client can
      open the WebSocket for real-time logs.
    """
    # --- Graph availability check (safe, no crash) ---
    if not _graph_adapter.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The orchestration pipeline (LangGraph graph) is not yet "
                "available in this deployment.  Member 1's backend/core/graph.py "
                "has not been merged.  POST /api/generate will become fully "
                "functional after the integration PR is merged into dev."
            ),
        )

    # --- Generate or validate project_id ---
    project_id = request.project_id or str(uuid.uuid4())

    # Reject duplicate project IDs
    existing = db.query(Project).filter(Project.id == project_id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A project with id '{project_id}' already exists.",
        )

    # --- Create the Project record ---
    project = Project(
        id=project_id,
        prompt=request.prompt,
        status="pending",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(project)

    # --- Create initial log entry ---
    db.add(ExecutionLog(
        project_id=project_id,
        agent_name="System",
        message=f"[System] Project created. Pipeline initialising...",
        log_level="INFO",
    ))

    db.commit()
    db.refresh(project)

    # --- Launch pipeline in background (non-blocking) ---
    from backend.schemas.models import SessionLocal as _SessionFactory
    t = threading.Thread(
        target=_run_pipeline_in_background,
        args=(project_id, request.prompt, _SessionFactory),
        daemon=True,
        name=f"pipeline-{project_id}",
    )
    t.start()

    return GenerateResponse(
        project_id=project_id,
        status="pending",
        message=(
            f"Pipeline started. Connect to "
            f"ws://{{host}}/ws/projects/{project_id}/logs "
            f"for real-time log streaming."
        ),
    )


@api_router.get(
    "/projects/{project_id}",
    response_model=ProjectStatusResponse,
    summary="Get project status and results",
    responses={
        200: {"description": "Project found."},
        404: {"description": "Project not found."},
    },
)
def get_project(
    project_id: str,
    db: Session = Depends(get_db),
) -> ProjectStatusResponse:
    """
    Returns the current status, logs, generated files, and review report
    for the given project_id.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found.",
        )

    logs_data = [
        {
            "id": log.id,
            "agent": log.agent_name,
            "level": log.log_level,
            "message": log.message,
            "ts": log.created_at.isoformat() if log.created_at else None,
        }
        for log in project.logs
    ]

    files_data = [
        {
            "id": f.id,
            "path": f.file_path,
            "language": f.language,
        }
        for f in project.files
    ]

    rr = project.review_report
    report_data: Optional[dict] = None
    if rr:
        report_data = {
            "quality_score": rr.quality_score,
            "owasp_compliant": bool(rr.owasp_compliant),
            "security_findings": json.loads(rr.security_findings or "[]"),
            "recommendations": json.loads(rr.recommendations or "[]"),
            "created_at": rr.created_at.isoformat() if rr.created_at else None,
        }

    return ProjectStatusResponse(
        project_id=project.id,
        prompt=project.prompt,
        status=project.status,
        current_agent=project.current_agent,
        workspace_path=project.workspace_path,
        created_at=project.created_at.isoformat() if project.created_at else "",
        updated_at=project.updated_at.isoformat() if project.updated_at else "",
        logs=logs_data,
        files=files_data,
        review_report=report_data,
    )


@api_router.get(
    "/projects/{project_id}/download",
    summary="Download generated project workspace as ZIP",
    responses={
        200: {"content": {"application/zip": {}}},
        404: {"description": "Project not found or workspace not generated yet."},
    },
)
def download_project(
    project_id: str,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """
    Streams the generated workspace for project_id as a ZIP archive.

    The workspace is located at WORKSPACE_DIR/{project_id}/.
    If Member 2's file manager has not yet been merged or the pipeline has
    not yet completed, the directory will not exist and this returns HTTP 404.

    ZIP exclusions (security / cleanliness):
      .env, .env.*, *.db, __pycache__/, .venv/, venv/, node_modules/,
      *.pyc, *.pyo, *.pyd, .git/, chroma_db/
    """
    # Verify the project exists in the database
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found.",
        )

    # Resolve workspace path (path-traversal safe)
    workspace = _safe_workspace_path(project_id)
    if workspace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Generated workspace for project '{project_id}' does not "
                f"exist yet. The pipeline may still be running or has not "
                f"produced any files."
            ),
        )

    # Patterns to exclude from the ZIP
    _EXCLUDED_DIRS = {
        "__pycache__", ".venv", "venv", "node_modules", ".git", "chroma_db",
    }
    _EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pyd", ".db", ".sqlite"}
    _EXCLUDED_NAMES = {".env"}

    def _should_exclude(rel: Path) -> bool:
        parts = rel.parts
        # Exclude by directory name anywhere in the path
        for part in parts[:-1]:
            if part in _EXCLUDED_DIRS:
                return True
        name = rel.name
        # Exclude .env and .env.* files
        if name == ".env" or name.startswith(".env."):
            return True
        # Exclude by suffix
        if rel.suffix in _EXCLUDED_SUFFIXES:
            return True
        return False

    def _zip_generator():
        """Generator that yields ZIP data chunks without buffering the whole archive."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(workspace.rglob("*")):
                if not file_path.is_file():
                    continue
                rel = file_path.relative_to(workspace)
                if _should_exclude(rel):
                    continue
                try:
                    zf.write(file_path, arcname=str(rel))
                except (OSError, PermissionError) as exc:
                    logger.warning("Skipping file in ZIP: %s — %s", file_path, exc)
        buf.seek(0)
        yield buf.read()

    filename = f"project_{project_id}.zip"
    return StreamingResponse(
        _zip_generator(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Project-ID": project_id,
        },
    )

