"""
SQLAlchemy ORM Models — Member 5 (Backend Gateway Layer).

Defines the persistence schema for:
  - Project          : top-level pipeline run record
  - ProjectFile      : generated source files linked to a project
  - ExecutionLog     : agent log entries streamed over WebSocket
  - ReviewReport     : structured security / quality report from Member 4

Database: PostgreSQL (primary) with automatic SQLite fallback for local dev.
Engine is created from config.DATABASE_URL; SQLite is used when that URL is
absent or the PostgreSQL server is unreachable.

Do NOT modify: backend/schemas/state.py  (shared contract — read-only)
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, relationship, sessionmaker

from backend.config import config


# ---------------------------------------------------------------------------
# Engine & session factory
# ---------------------------------------------------------------------------

def _build_engine():
    """
    Build the SQLAlchemy engine.

    Priority:
      1. Use DATABASE_URL from config (expected: PostgreSQL).
      2. If the URL is missing or connection fails during startup, fall back
         to a local SQLite file (./autodev_local.db) so the server still
         starts cleanly for development.
    """
    db_url = config.DATABASE_URL or ""

    # --- PostgreSQL path ---
    if db_url.startswith("postgresql"):
        try:
            engine = create_engine(
                db_url,
                pool_pre_ping=True,     # cheap "SELECT 1" check on borrow
                pool_size=5,
                max_overflow=10,
            )
            # Probe the connection eagerly so we can fall back if Postgres
            # is not running (common in local dev without Docker).
            with engine.connect():
                pass
            return engine, "postgresql"
        except Exception:
            # Fall through to SQLite fallback.
            pass

    # --- SQLite fallback ---
    sqlite_url = "sqlite:///./autodev_local.db"
    engine = create_engine(
        sqlite_url,
        connect_args={"check_same_thread": False},
    )

    # Enable WAL mode and foreign-key enforcement for SQLite.
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine, "sqlite"


engine, _db_backend = _build_engine()

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    __allow_unmapped__ = True


# ---------------------------------------------------------------------------
# Helper: generate a compact unique ID
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class Project(Base):
    """
    Top-level record for one pipeline execution run.

    Lifecycle of `status` mirrors AgentState["status"]:
      pending → analyzing → designing → coding →
      testing → debugging → reviewing → completed | failed
    """

    __tablename__ = "projects"

    id = Column(
        String(36),
        primary_key=True,
        default=_new_uuid,
        comment="UUID primary key (project_id used throughout the system)",
    )
    prompt = Column(
        Text,
        nullable=False,
        comment="Raw user prompt submitted to POST /api/generate",
    )
    status = Column(
        String(50),
        nullable=False,
        default="pending",
        index=True,
        comment="Current pipeline status (mirrors AgentState.status)",
    )
    current_agent = Column(
        String(100),
        nullable=True,
        comment="Name of the agent currently executing",
    )
    workspace_path = Column(
        String(500),
        nullable=True,
        comment="Absolute/relative path to generated workspace on disk",
    )
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    files: Mapped[list["ProjectFile"]] = relationship(
        "ProjectFile",
        back_populates="project",
        cascade="all, delete-orphan",
        lazy="select",
    )
    logs: Mapped[list["ExecutionLog"]] = relationship(
        "ExecutionLog",
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="ExecutionLog.created_at",
        lazy="select",
    )
    review_report: Mapped[Optional["ReviewReport"]] = relationship(
        "ReviewReport",
        back_populates="project",
        cascade="all, delete-orphan",
        uselist=False,
        lazy="select",
    )

    def __repr__(self) -> str:
        return f"<Project id={self.id!r} status={self.status!r}>"


class ProjectFile(Base):
    """
    One generated source file linked to a Project.

    Populated by POST /api/generate after the LangGraph workflow writes
    generated_files into AgentState.  Each entry mirrors one element of
    AgentState["generated_files"] (a dict with keys: path, content, language).
    """

    __tablename__ = "project_files"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    project_id = Column(
        String(36),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    file_path = Column(
        String(500),
        nullable=False,
        comment="Relative path of the file within the workspace",
    )
    content = Column(
        Text,
        nullable=False,
        default="",
        comment="Full source code content of the file",
    )
    language = Column(
        String(50),
        nullable=True,
        default="python",
        comment="Programming language (from AgentState GeneratedFile.language)",
    )
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    # Relationship
    project: Mapped["Project"] = relationship("Project", back_populates="files")

    def __repr__(self) -> str:
        return f"<ProjectFile project_id={self.project_id!r} path={self.file_path!r}>"


class ExecutionLog(Base):
    """
    One log line emitted by an agent during a pipeline run.

    Rows are inserted as agents update AgentState["logs"].
    They are also broadcast in real-time over the WebSocket endpoint
    /ws/projects/{project_id}/logs.

    log_level: "INFO" | "WARN" | "ERROR" | "DEBUG"
    """

    __tablename__ = "execution_logs"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    project_id = Column(
        String(36),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    agent_name = Column(
        String(100),
        nullable=False,
        default="System",
        comment="Name of the agent that emitted this log line",
    )
    message = Column(
        Text,
        nullable=False,
        comment="Log message text",
    )
    log_level = Column(
        String(10),
        nullable=False,
        default="INFO",
        comment="Severity: INFO | WARN | ERROR | DEBUG",
    )
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )

    # Relationship
    project: Mapped["Project"] = relationship("Project", back_populates="logs")

    def __repr__(self) -> str:
        return (
            f"<ExecutionLog project_id={self.project_id!r} "
            f"agent={self.agent_name!r} level={self.log_level!r}>"
        )


class ReviewReport(Base):
    """
    Structured security / quality report produced by Member 4's review agent.

    Maps to AgentState["security_report"] (a SecurityReviewReport.model_dump()).
    One-to-one with Project.

    security_findings and recommendations are stored as JSON text
    (comma-joined lists) for maximum portability across PostgreSQL and SQLite.
    """

    __tablename__ = "review_reports"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    project_id = Column(
        String(36),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    quality_score = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Quality score 0–100 from SecurityReviewReport.quality_score",
    )
    owasp_compliant = Column(
        Integer,          # SQLite-compatible boolean (0/1)
        nullable=False,
        default=0,
        comment="1 if OWASP compliance passed",
    )
    security_findings = Column(
        Text,
        nullable=True,
        comment="JSON array text: list of vulnerability strings",
    )
    recommendations = Column(
        Text,
        nullable=True,
        comment="JSON array text: list of suggestion strings",
    )
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    # Relationship
    project: Mapped["Project"] = relationship("Project", back_populates="review_report")

    def __repr__(self) -> str:
        return (
            f"<ReviewReport project_id={self.project_id!r} "
            f"score={self.quality_score}>"
        )


# ---------------------------------------------------------------------------
# DB initialiser — called by backend/main.py at startup
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables that do not yet exist. Safe to call repeatedly."""
    Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# FastAPI dependency — yields a scoped DB session
# ---------------------------------------------------------------------------

def get_db():
    """
    Yield a SQLAlchemy Session for use as a FastAPI dependency.

    Usage:
        from backend.schemas.models import get_db
        from sqlalchemy.orm import Session

        @router.get("/example")
        def example(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

