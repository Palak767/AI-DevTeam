"""
Unit tests for Member 3 Workflow Nodes, Conditional Routing, and Subgraph Builder.
Tests verify LangGraph integration without requiring Docker daemon or external APIs.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from backend.schemas.state import AgentState, TestResult
from backend.testing_debugging.docker_sandbox import DockerDaemonUnavailableError
from backend.testing_debugging.schemas import BugCategoryEnum
from backend.testing_debugging.workflow_nodes import (
    NODE_BUG_DETECTION,
    NODE_REPORT_GENERATION,
    NODE_TESTING,
    ROUTE_BUG_DETECTION,
    ROUTE_CODE_REVIEW,
    ROUTE_GENERATE_REPORT,
    bug_detection_node,
    build_testing_subgraph,
    report_generation_node,
    route_after_debugging,
    route_after_testing,
    testing_node,
)


# Prevent pytest collection of non-test callables and classes
TestResult.__test__ = False
testing_node.__test__ = False


@pytest.fixture
def base_state(tmp_path: Path) -> AgentState:
    """Provides a valid baseline AgentState for workflow node tests."""
    return {
        "project_id": "proj_langgraph_test",
        "user_prompt": "Create an authenticated API endpoint",
        "requirement_spec": {
            "project_title": "Auth API",
            "functional_requirements": ["JWT login"],
            "non_functional_requirements": ["Low latency"],
            "tech_stack": ["FastAPI", "Pytest"],
            "acceptance_criteria": ["Tests pass"],
        },
        "architecture_blueprint": None,
        "generated_files": [
            {"path": "app/auth.py", "content": "def login():\n    return True\n", "language": "python"}
        ],
        "workspace_path": str(tmp_path),
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


# ============================================================================
# 1. Testing Node Tests
# ============================================================================

def test_testing_node_success(base_state: AgentState):
    """Verifies testing_node successfully executes and updates state when tests pass."""
    mock_agent = MagicMock()
    mock_agent.run.side_effect = lambda s: {
        **s,
        "is_test_passed": True,
        "status": "reviewing",
        "test_results": TestResult(
            passed=True, total_tests=3, passed_tests=3, failed_tests=0
        ).model_dump(),
        "logs": s.get("logs", []) + ["[Testing Agent] All 3 tests passed."],
    }

    result = testing_node(base_state, testing_agent=mock_agent)

    assert result["current_agent"] == "Testing Agent"
    assert result["is_test_passed"] is True
    assert result["status"] == "reviewing"
    assert any("Initiating Testing Agent execution" in log for log in result["logs"])
    mock_agent.run.assert_called_once()


def test_testing_node_failure(base_state: AgentState):
    """Verifies testing_node correctly records failed tests."""
    mock_agent = MagicMock()
    mock_agent.run.side_effect = lambda s: {
        **s,
        "is_test_passed": False,
        "status": "debugging",
        "test_results": TestResult(
            passed=False, total_tests=3, passed_tests=2, failed_tests=1, error_summary="AssertionError"
        ).model_dump(),
        "logs": s.get("logs", []) + ["[Testing Agent] 1 failed."],
    }

    result = testing_node(base_state, testing_agent=mock_agent)

    assert result["current_agent"] == "Testing Agent"
    assert result["is_test_passed"] is False
    assert result["status"] == "debugging"
    assert result["test_results"]["failed_tests"] == 1


def test_testing_node_docker_infrastructure_failure(base_state: AgentState):
    """Verifies testing_node safely handles DockerDaemonUnavailableError without crashing."""
    mock_agent = MagicMock()
    mock_agent.run.side_effect = DockerDaemonUnavailableError("Docker daemon is stopped")

    result = testing_node(base_state, testing_agent=mock_agent)

    assert result["current_agent"] == "Testing Agent"
    assert result["is_test_passed"] is False
    assert result["status"] == "failed"
    assert result["test_results"] is not None
    assert result["test_results"]["passed"] is False
    assert any("Testing node execution halted unexpectedly" in log for log in result["logs"])


def test_testing_node_preserves_unrelated_state(base_state: AgentState):
    """Verifies testing_node preserves custom or unrelated keys in state."""
    base_state["custom_metadata"] = {"team": "backend", "priority": "high"}
    mock_agent = MagicMock()
    mock_agent.run.side_effect = lambda s: s

    result = testing_node(base_state, testing_agent=mock_agent)

    assert result["custom_metadata"] == {"team": "backend", "priority": "high"}
    assert result["project_id"] == "proj_langgraph_test"


# ============================================================================
# 2. Routing After Testing Tests
# ============================================================================

def test_route_after_testing_passed(base_state: AgentState):
    """Tests pass -> routes to code_review."""
    base_state["is_test_passed"] = True
    base_state["test_results"] = {"passed": True, "total_tests": 4, "failed_tests": 0}

    assert route_after_testing(base_state) == ROUTE_CODE_REVIEW


def test_route_after_testing_failed_with_retry_budget(base_state: AgentState):
    """Tests fail and retry budget remains -> routes to bug_detection."""
    base_state["is_test_passed"] = False
    base_state["retry_count"] = 0
    base_state["test_results"] = {"passed": False, "total_tests": 3, "failed_tests": 1}

    assert route_after_testing(base_state, max_retries=3) == ROUTE_BUG_DETECTION


def test_route_after_testing_retry_budget_exhausted(base_state: AgentState):
    """Tests fail and retries exhausted -> routes to generate_report."""
    base_state["is_test_passed"] = False
    base_state["retry_count"] = 3
    base_state["test_results"] = {"passed": False, "total_tests": 3, "failed_tests": 1}

    assert route_after_testing(base_state, max_retries=3) == ROUTE_GENERATE_REPORT


def test_route_after_testing_docker_infrastructure_failure(base_state: AgentState):
    """Docker failure with 0 tests executed -> routes directly to generate_report."""
    base_state["is_test_passed"] = False
    base_state["retry_count"] = 0
    base_state["status"] = "failed"
    base_state["test_results"] = {
        "passed": False,
        "total_tests": 0,
        "error_summary": "Docker daemon unavailable on host pipe",
    }
    base_state["logs"] = ["[Testing Agent Error] Docker daemon unavailable"]

    assert route_after_testing(base_state) == ROUTE_GENERATE_REPORT


# ============================================================================
# 3. Bug Detection Node Tests
# ============================================================================

def test_bug_detection_node_skips_when_already_passing(base_state: AgentState):
    """When tests are already passing, bug_detection_node skips repair and sets reviewing."""
    base_state["is_test_passed"] = True
    mock_bug_agent = MagicMock()

    result = bug_detection_node(base_state, bug_agent=mock_bug_agent)

    assert result["status"] == "reviewing"
    assert any("Tests already passing" in log for log in result["logs"])
    mock_bug_agent.self_correct.assert_not_called()


def test_bug_detection_node_halts_when_retries_exhausted(base_state: AgentState):
    """When retry budget is exhausted, bug_detection_node immediately halts with status 'failed'."""
    base_state["is_test_passed"] = False
    base_state["retry_count"] = 3
    mock_bug_agent = MagicMock()

    result = bug_detection_node(base_state, bug_agent=mock_bug_agent)

    assert result["status"] == "failed"
    assert any("Retry budget exhausted" in log for log in result["logs"])
    mock_bug_agent.self_correct.assert_not_called()


def test_bug_detection_node_single_step_execution(base_state: AgentState):
    """Verifies bug_detection_node executes a repair step and increments retry count."""
    base_state["is_test_passed"] = False
    base_state["retry_count"] = 0
    base_state["test_results"] = {"passed": False, "total_tests": 2, "failed_tests": 1}

    mock_bug_agent = MagicMock()
    mock_bug_agent.self_correct.side_effect = lambda s, single_step=True: {
        **s,
        "retry_count": s.get("retry_count", 0) + 1,
        "is_test_passed": True,
        "status": "reviewing",
        "patched_files": [{"path": "app/auth.py", "content": "fixed"}],
    }

    result = bug_detection_node(base_state, bug_agent=mock_bug_agent, single_step=True)

    assert result["current_agent"] == "Bug Detection Agent"
    assert result["retry_count"] == 1
    assert result["is_test_passed"] is True
    assert result["status"] == "reviewing"
    assert mock_bug_agent.self_correct.call_count == 1
    call_kwargs = mock_bug_agent.self_correct.call_args.kwargs
    assert call_kwargs.get("single_step") is True


def test_bug_detection_node_error_handling(base_state: AgentState):
    """Verifies bug_detection_node catches unexpected exceptions safely."""
    base_state["is_test_passed"] = False
    mock_bug_agent = MagicMock()
    mock_bug_agent.self_correct.side_effect = RuntimeError("AST parse crash")

    result = bug_detection_node(base_state, bug_agent=mock_bug_agent)

    assert result["status"] == "failed"
    assert any("Bug detection node encountered error" in log for log in result["logs"])


# ============================================================================
# 4. Routing After Debugging Tests
# ============================================================================

def test_route_after_debugging_passed(base_state: AgentState):
    """Tests passed after repair -> routes to code_review."""
    base_state["is_test_passed"] = True

    assert route_after_debugging(base_state) == ROUTE_CODE_REVIEW


def test_route_after_debugging_failed_with_retry_budget(base_state: AgentState):
    """Tests still fail and retry budget remains -> routes back to bug_detection."""
    base_state["is_test_passed"] = False
    base_state["retry_count"] = 1
    base_state["status"] = "debugging"
    base_state["bug_diagnosis"] = "[logic_error] missing return value"

    assert route_after_debugging(base_state, max_retries=3) == ROUTE_BUG_DETECTION


def test_route_after_debugging_retry_budget_exhausted(base_state: AgentState):
    """Retry budget reached -> routes to generate_report."""
    base_state["is_test_passed"] = False
    base_state["retry_count"] = 3

    assert route_after_debugging(base_state, max_retries=3) == ROUTE_GENERATE_REPORT


def test_route_after_debugging_inconclusive_diagnosis(base_state: AgentState):
    """Inconclusive diagnosis halts repair -> routes to generate_report."""
    base_state["is_test_passed"] = False
    base_state["retry_count"] = 1
    base_state["bug_diagnosis"] = f"[{BugCategoryEnum.INCONCLUSIVE.value}] Cannot determine root cause"

    assert route_after_debugging(base_state) == ROUTE_GENERATE_REPORT


def test_route_after_debugging_patch_rejected(base_state: AgentState):
    """Patch rejection halts repair -> routes to generate_report."""
    base_state["is_test_passed"] = False
    base_state["retry_count"] = 1
    base_state["status"] = "failed"
    base_state["logs"] = ["[Bug Agent] Proposed patch rejected: Syntax error in replacement."]

    assert route_after_debugging(base_state) == ROUTE_GENERATE_REPORT


# ============================================================================
# 5. Report Generation Node Tests
# ============================================================================

def test_report_generation_node_execution(base_state: AgentState, tmp_path: Path):
    """Verifies report_generation_node executes and writes report files into workspace."""
    base_state["is_test_passed"] = True
    base_state["test_results"] = {
        "passed": True,
        "total_tests": 5,
        "passed_tests": 5,
        "failed_tests": 0,
        "raw_output": "5 passed in 0.2s",
    }
    base_state["workspace_path"] = str(tmp_path)

    result = report_generation_node(base_state)

    assert result["current_agent"] == "Report Generator"
    assert result["status"] == "reviewing"
    assert any("Successfully produced test reports" in log for log in result["logs"])

    # Verify reports directory and files on disk
    reports_dir = tmp_path / "reports"
    assert reports_dir.is_dir()
    json_files = list(reports_dir.glob("*.json"))
    md_files = list(reports_dir.glob("*.md"))
    assert len(json_files) == 1
    assert len(md_files) == 1


def test_report_generation_node_error_handling(base_state: AgentState):
    """Verifies report_generation_node catches disk/filesystem errors gracefully."""
    mock_generator = MagicMock()
    mock_generator.generate_reports.side_effect = OSError("Read-only filesystem")

    result = report_generation_node(base_state, report_generator=mock_generator)

    assert result["current_agent"] == "Report Generator"
    assert any("Report generation failed" in log for log in result["logs"])


# ============================================================================
# 6. LangGraph Subgraph Compilation and Execution Tests
# ============================================================================

def test_build_testing_subgraph_compilation():
    """Verifies that build_testing_subgraph compiles successfully into a LangGraph graph."""
    subgraph = build_testing_subgraph(compile_graph=True)
    assert subgraph is not None

    # Check uncompiled graph structure
    raw_graph = build_testing_subgraph(compile_graph=False)
    assert NODE_TESTING in raw_graph.nodes
    assert NODE_BUG_DETECTION in raw_graph.nodes
    assert NODE_REPORT_GENERATION in raw_graph.nodes


def test_subgraph_execution_all_tests_pass(base_state: AgentState, tmp_path: Path):
    """End-to-end subgraph execution where tests pass on initial run."""
    base_state["workspace_path"] = str(tmp_path)

    with patch("backend.testing_debugging.workflow_nodes.TestingAgent") as MockTesting:
        instance = MockTesting.return_value
        instance.run.side_effect = lambda s: {
            **s,
            "is_test_passed": True,
            "status": "reviewing",
            "test_results": TestResult(
                passed=True, total_tests=2, passed_tests=2, failed_tests=0
            ).model_dump(),
            "logs": s.get("logs", []) + ["[Testing Agent] 2 tests passed."],
        }

        compiled_subgraph = build_testing_subgraph(compile_graph=True)
        final_state = compiled_subgraph.invoke(base_state)

        assert final_state["is_test_passed"] is True
        assert final_state["status"] == "reviewing"
        assert final_state["current_agent"] == "Testing Agent"


def test_subgraph_execution_repair_and_pass(base_state: AgentState, tmp_path: Path):
    """
    End-to-end subgraph execution where tests fail initially,
    route to bug_detection_node, pass after patch, and exit with is_test_passed=True.
    """
    base_state["workspace_path"] = str(tmp_path)

    with patch("backend.testing_debugging.workflow_nodes.TestingAgent") as MockTesting, \
         patch("backend.testing_debugging.workflow_nodes.BugDetectionAgent") as MockBug:

        testing_instance = MockTesting.return_value
        testing_instance.run.side_effect = lambda s: {
            **s,
            "is_test_passed": False,
            "status": "debugging",
            "test_results": TestResult(
                passed=False, total_tests=2, passed_tests=1, failed_tests=1
            ).model_dump(),
            "logs": s.get("logs", []) + ["[Testing Agent] Test failed."],
        }

        bug_instance = MockBug.return_value
        bug_instance.self_correct.side_effect = lambda s, single_step=True: {
            **s,
            "retry_count": s.get("retry_count", 0) + 1,
            "is_test_passed": True,
            "status": "reviewing",
            "test_results": TestResult(
                passed=True, total_tests=2, passed_tests=2, failed_tests=0
            ).model_dump(),
            "logs": s.get("logs", []) + ["[Bug Agent] Patch applied successfully."],
        }

        compiled_subgraph = build_testing_subgraph(compile_graph=True)
        final_state = compiled_subgraph.invoke(base_state)

        assert final_state["is_test_passed"] is True
        assert final_state["retry_count"] == 1
        assert final_state["status"] == "reviewing"


def test_subgraph_execution_exhausted_retries_generates_report(base_state: AgentState, tmp_path: Path):
    """
    End-to-end subgraph execution where tests fail, retries are exhausted,
    and report_generation_node is invoked before terminating.
    """
    base_state["workspace_path"] = str(tmp_path)

    with patch("backend.testing_debugging.workflow_nodes.TestingAgent") as MockTesting, \
         patch("backend.testing_debugging.workflow_nodes.BugDetectionAgent") as MockBug:

        testing_instance = MockTesting.return_value
        testing_instance.run.side_effect = lambda s: {
            **s,
            "is_test_passed": False,
            "status": "debugging",
            "retry_count": 0,
            "test_results": TestResult(
                passed=False, total_tests=1, passed_tests=0, failed_tests=1
            ).model_dump(),
        }

        # Simulate BugDetectionAgent running 1 step and failing, incrementing retry count
        def bug_step(s, single_step=True):
            new_retries = s.get("retry_count", 0) + 1
            return {
                **s,
                "retry_count": new_retries,
                "is_test_passed": False,
                "status": "debugging" if new_retries < 3 else "failed",
                "test_results": TestResult(
                    passed=False, total_tests=1, passed_tests=0, failed_tests=1
                ).model_dump(),
            }

        bug_instance = MockBug.return_value
        bug_instance.self_correct.side_effect = bug_step

        compiled_subgraph = build_testing_subgraph(compile_graph=True)
        final_state = compiled_subgraph.invoke(base_state)

        assert final_state["is_test_passed"] is False
        assert final_state["retry_count"] >= 3
        assert final_state["status"] == "failed"
        assert final_state["current_agent"] == "Report Generator"
