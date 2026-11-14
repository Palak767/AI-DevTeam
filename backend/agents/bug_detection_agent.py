"""
Bug Detection Agent entrypoint for Multi-Agent AI DevTeam / AutoDev.
Re-exports run_bug_agent from Member 3's backend.testing_debugging subsystem.
"""

from backend.schemas.state import AgentState
from backend.testing_debugging.bug_detection_agent import run_bug_agent, BugDetectionAgent

__all__ = ["run_bug_agent", "BugDetectionAgent"]

