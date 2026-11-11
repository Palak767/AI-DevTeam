"""
Testing Agent entrypoint for Multi-Agent AI DevTeam / AutoDev.
Re-exports run_testing_agent from Member 3's backend.testing_debugging subsystem.
"""

from backend.schemas.state import AgentState
from backend.testing_debugging.testing_agent import run_testing_agent, TestingAgent

__all__ = ["run_testing_agent", "TestingAgent"]

