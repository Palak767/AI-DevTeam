"""
WebSocket Connection Manager — Member 5 (Backend Gateway Layer).

Endpoint: /ws/projects/{project_id}/logs

Responsibilities:
  - Accept multiple simultaneous clients per project.
  - Broadcast real-time JSON log messages to all connected clients.
  - Handle client disconnections gracefully (never crash other clients).
  - Expose a broadcast helper that LangGraph agents / route handlers can call
    after Member 1's graph is merged.

The WebSocket handler itself does NOT own or run the LangGraph pipeline.
It is a pure transport layer for log streaming.

Message schema (JSON):
  { "type": "log",   "agent": str, "level": str, "message": str, "ts": ISO-8601 }
  { "type": "status","status": str }
  { "type": "done",  "status": str, "message": str }
  { "type": "error", "message": str }
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

ws_router = APIRouter(tags=["WebSocket"])


# ---------------------------------------------------------------------------
# Connection Manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    """
    Manages a registry of active WebSocket connections grouped by project_id.

    Thread-safety note:
      FastAPI / Starlette's WebSocket handling is single-threaded within the
      asyncio event loop, so a plain dict is safe here.  If background threads
      (e.g., the LangGraph executor thread) call broadcast(), they must do so
      via asyncio.run_coroutine_threadsafe().
    """

    def __init__(self) -> None:
        # project_id → list of connected WebSocket clients
        self._connections: Dict[str, List[WebSocket]] = {}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self, project_id: str, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        if project_id not in self._connections:
            self._connections[project_id] = []
        self._connections[project_id].append(websocket)
        logger.info(
            "WebSocket connected: project=%s  total_clients=%d",
            project_id,
            len(self._connections[project_id]),
        )

    def disconnect(self, project_id: str, websocket: WebSocket) -> None:
        """Remove a disconnected client from the registry."""
        project_connections = self._connections.get(project_id, [])
        if websocket in project_connections:
            project_connections.remove(websocket)
        if not project_connections:
            self._connections.pop(project_id, None)
        logger.info(
            "WebSocket disconnected: project=%s  remaining_clients=%d",
            project_id,
            len(self._connections.get(project_id, [])),
        )

    # ------------------------------------------------------------------
    # Broadcasting helpers
    # ------------------------------------------------------------------

    async def broadcast(self, project_id: str, message: dict) -> None:
        """
        Send a JSON message to all clients watching project_id.

        Clients that have already disconnected are silently removed.
        A failure on one client never affects others.
        """
        dead: List[WebSocket] = []
        for ws in list(self._connections.get(project_id, [])):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(project_id, ws)

    async def broadcast_log(
        self,
        project_id: str,
        message: str,
        agent: str = "System",
        level: str = "INFO",
    ) -> None:
        """Convenience wrapper — formats a standard log message and broadcasts."""
        await self.broadcast(
            project_id,
            {
                "type": "log",
                "agent": agent,
                "level": level,
                "message": message,
                "ts": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def broadcast_status(self, project_id: str, status_value: str) -> None:
        """Notify clients that the pipeline status has changed."""
        await self.broadcast(
            project_id,
            {
                "type": "status",
                "status": status_value,
                "ts": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def broadcast_done(
        self, project_id: str, final_status: str, message: str = ""
    ) -> None:
        """
        Signal that the pipeline has finished (success or failure).
        Clients should close the WebSocket after receiving this.
        """
        await self.broadcast(
            project_id,
            {
                "type": "done",
                "status": final_status,
                "message": message,
                "ts": datetime.now(timezone.utc).isoformat(),
            },
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def active_project_ids(self) -> List[str]:
        """Return the list of project IDs that currently have live clients."""
        return list(self._connections.keys())

    def client_count(self, project_id: str) -> int:
        """Return the number of live clients for a project."""
        return len(self._connections.get(project_id, []))


# ---------------------------------------------------------------------------
# Module-level singleton — imported by routes.py and main.py
# ---------------------------------------------------------------------------

manager = ConnectionManager()


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@ws_router.websocket("/ws/projects/{project_id}/logs")
async def websocket_logs(project_id: str, websocket: WebSocket) -> None:
    """
    Real-time log streaming endpoint for a pipeline execution.

    Clients connect here immediately after calling POST /api/generate.
    The endpoint:
      1. Accepts the connection.
      2. Sends a "connected" acknowledgement.
      3. Keeps the connection alive, waiting for broadcasted messages.
      4. Handles disconnection gracefully.

    The LangGraph pipeline (when available) calls manager.broadcast_log()
    from route handlers or background tasks to push log lines to all
    connected clients in real-time.

    Clients should close after receiving a {"type": "done"} message.
    """
    await manager.connect(project_id, websocket)
    try:
        # Acknowledge the connection
        await websocket.send_json(
            {
                "type": "log",
                "agent": "System",
                "level": "INFO",
                "message": (
                    f"[WebSocket] Connected to log stream for project: {project_id}"
                ),
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Keep-alive loop — the server pushes via broadcast(), clients just
        # listen.  We read from the client only to detect disconnect.
        while True:
            try:
                # recv_text with a generous timeout; we don't process client
                # messages but we need this to detect disconnects.
                data = await asyncio.wait_for(
                    websocket.receive_text(), timeout=30.0
                )
                # Accept a voluntary close message from the client.
                if data == "close":
                    break
            except asyncio.TimeoutError:
                # Heartbeat — send a ping to keep the connection alive
                # and check the client is still there.
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break  # client gone

    except WebSocketDisconnect:
        pass  # Normal client close — handled below
    except Exception as exc:
        logger.warning("WebSocket error for project %s: %s", project_id, exc)
        try:
            await websocket.send_json(
                {"type": "error", "message": "Internal WebSocket error."}
            )
        except Exception:
            pass
    finally:
        manager.disconnect(project_id, websocket)

