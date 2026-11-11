"""
Workflow Nodes and Conditional Routing for Member 3: Testing & Debugging Subsystem.

Provides callable LangGraph nodes, deterministic routers, and subgraph constructor
connecting Docker Sandbox, Pytest Execution Engine, AI Testing Agent, Bug Detection
& Self-Correction Agent, and Test Report Generator into a unified pipeline.
"""

from copy import deepcopy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    StateGraph = None
    END = "__end__"
    LANGGRAPH_AVAILABLE = False

from backend.config import config
from backend.schemas.state import AgentState, TestResult
from backend.testing_debugging.bug_detection_agent import BugDetectionAgent
from backend.testing_debugging.report_generator import TestReportGenerator
from backend.testing_debugging.schemas import BugCategoryEnum
from backend.testing_debugging.testing_agent import TestingAgent

logger = logging.getLogger(__name__)

# ============================================================================
# Routing Identifiers and Node Names
# ============================================================================

ROUTE_CODE_REVIEW = "code_review"
ROUTE_BUG_DETECTION = "bug_detection"
ROUTE_GENERATE_REPORT = "generate_report"

NODE_TESTING = "testing_node"
NODE_BUG_DETECTION = "bug_detection_node"
NODE_REPORT_GENERATION = "report_generation_node"


# ============================================================================
# LangGraph Callable Workflow Nodes
# ============================================================================

def testing_node(
    state: AgentState,
    testing_agent: Optional[TestingAgent] = None,
) -> AgentState:
    """
    LangGraph node that synthesizes tests and executes them inside Docker Sandbox.

    Args:
        state: Shared AgentState dictionary.
        testing_agent: Optional TestingAgent instance for dependency injection.

    Returns:
        AgentState: Updated state containing test results, logs, and updated status.
    """
    updated_state: AgentState = deepcopy(state) if isinstance(state, dict) else state

    if "logs" not in updated_state or updated_state["logs"] is None:
        updated_state["logs"] = []

    updated_state["current_agent"] = "Testing Agent"
    updated_state["status"] = "testing"
    updated_state["logs"].append("[Workflow Node] Initiating Testing Agent execution.")

    try:
        agent = testing_agent or TestingAgent()
        updated_state = agent.run(updated_state)

    except Exception as exc:
        err_msg = f"[Workflow Node Error] Testing node execution halted unexpectedly: {str(exc)}"
        logger.error(err_msg, exc_info=True)
        updated_state["logs"].append(err_msg)
        updated_state["is_test_passed"] = False
        updated_state["status"] = "failed"

        if updated_state.get("test_results") is None:
            updated_state["test_results"] = TestResult(
                passed=False,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                raw_output=str(exc),
                error_summary=err_msg,
            ).model_dump()

    return updated_state


# Prevent pytest from treating testing_node as a test function
testing_node.__test__ = False


def bug_detection_node(
    state: AgentState,
    bug_agent: Optional[BugDetectionAgent] = None,
    single_step: bool = True,
) -> AgentState:
    """
    LangGraph node that diagnoses test failures and applies verified code patches.

    Args:
        state: Shared AgentState dictionary.
        bug_agent: Optional BugDetectionAgent instance for dependency injection.
        single_step: If True, execute one repair iteration per node invocation.

    Returns:
        AgentState: Updated state with diagnosis, applied patches, and rerun results.
    """
    updated_state: AgentState = deepcopy(state) if isinstance(state, dict) else state

    if "logs" not in updated_state or updated_state["logs"] is None:
        updated_state["logs"] = []

    updated_state["current_agent"] = "Bug Detection Agent"

    # If tests are already passing, skip self-correction
    if updated_state.get("is_test_passed", False):
        updated_state["logs"].append(
            "[Workflow Node] Tests already passing. Skipping bug detection node."
        )
        updated_state["status"] = "reviewing"
        return updated_state

    # Check retry budget before attempting repair
    max_retries = getattr(config, "MAX_RETRY_LIMIT", 3)
    current_retries = updated_state.get("retry_count", 0)
    if current_retries >= max_retries:
        updated_state["logs"].append(
            f"[Workflow Node] Retry budget exhausted ({current_retries}/{max_retries}). Halting repair."
        )
        updated_state["status"] = "failed"
        return updated_state

    updated_state["status"] = "debugging"
    updated_state["logs"].append(
        f"[Workflow Node] Starting repair attempt (Attempt {current_retries + 1}/{max_retries})."
    )

    try:
        agent = bug_agent or BugDetectionAgent(max_retries=max_retries)
        updated_state = agent.self_correct(updated_state, single_step=single_step)

    except Exception as exc:
        err_msg = f"[Workflow Node Error] Bug detection node encountered error: {str(exc)}"
        logger.error(err_msg, exc_info=True)
        updated_state["logs"].append(err_msg)
        updated_state["status"] = "failed"

    return updated_state


def report_generation_node(
    state: AgentState,
    report_generator: Optional[TestReportGenerator] = None,
) -> AgentState:
    """
    LangGraph node that synthesizes JSON and Markdown test reports.

    Args:
        state: Shared AgentState dictionary.
        report_generator: Optional TestReportGenerator for dependency injection.

    Returns:
        AgentState: Updated state with report generation audit log.
    """
    updated_state: AgentState = deepcopy(state) if isinstance(state, dict) else state

    if "logs" not in updated_state or updated_state["logs"] is None:
        updated_state["logs"] = []

    updated_state["current_agent"] = "Report Generator"
    updated_state["logs"].append("[Workflow Node] Generating testing and debugging reports.")

    # Preserve terminal status if already reviewing or failed
    if updated_state.get("is_test_passed", False):
        if updated_state.get("status") not in ("reviewing", "completed"):
            updated_state["status"] = "reviewing"
    else:
        updated_state["status"] = "failed"

    try:
        generator = report_generator or TestReportGenerator()
        generator.generate_reports(updated_state)

    except Exception as exc:
        err_msg = f"[Workflow Node Error] Report generation failed: {str(exc)}"
        logger.error(err_msg, exc_info=True)
        updated_state["logs"].append(err_msg)

    return updated_state


# ============================================================================
# Deterministic Conditional Routing Functions
# ============================================================================

def route_after_testing(
    state: AgentState,
    max_retries: Optional[int] = None,
) -> str:
    """
    Determines next pipeline stage after Pytest execution in Docker Sandbox.

    Rules:
    - If tests pass -> ROUTE_CODE_REVIEW ("code_review")
    - If infrastructure error (Docker unavailable, sandbox failure) -> ROUTE_GENERATE_REPORT ("generate_report")
    - If tests fail and retry budget remains -> ROUTE_BUG_DETECTION ("bug_detection")
    - If retry budget exhausted -> ROUTE_GENERATE_REPORT ("generate_report")

    Args:
        state: Shared AgentState dictionary.
        max_retries: Optional override for MAX_RETRY_LIMIT.

    Returns:
        str: Route identifier ("code_review", "bug_detection", or "generate_report").
    """
    # 1. Successful test pass
    if state.get("is_test_passed", False) is True:
        return ROUTE_CODE_REVIEW

    # 2. Check for unrecoverable infrastructure failure
    test_res = state.get("test_results") or {}
    total_tests = test_res.get("total_tests", 0)
    logs_str = " ".join(state.get("logs") or [])
    has_infra_failure = (
        state.get("status") == "failed"
        and (
            total_tests == 0
            or "Docker daemon unavailable" in logs_str
            or "Sandbox execution failed" in logs_str
            or "Docker daemon" in (test_res.get("error_summary") or "")
        )
    )
    if has_infra_failure:
        return ROUTE_GENERATE_REPORT

    # 3. Check retry budget
    limit = max_retries if max_retries is not None else getattr(config, "MAX_RETRY_LIMIT", 3)
    current_retries = state.get("retry_count", 0)

    if current_retries < limit:
        return ROUTE_BUG_DETECTION

    return ROUTE_GENERATE_REPORT


def route_after_debugging(
    state: AgentState,
    max_retries: Optional[int] = None,
) -> str:
    """
    Determines next pipeline stage following a debugging and repair attempt.

    Rules:
    - If tests pass after repair -> ROUTE_CODE_REVIEW ("code_review")
    - If diagnosis is inconclusive or patch rejected/failed -> ROUTE_GENERATE_REPORT ("generate_report")
    - If tests fail and retry budget remains -> ROUTE_BUG_DETECTION ("bug_detection")
    - If retry budget is exhausted -> ROUTE_GENERATE_REPORT ("generate_report")

    Args:
        state: Shared AgentState dictionary.
        max_retries: Optional override for MAX_RETRY_LIMIT.

    Returns:
        str: Route identifier ("code_review", "bug_detection", or "generate_report").
    """
    # 1. Successful repair pass
    if state.get("is_test_passed", False) is True:
        return ROUTE_CODE_REVIEW

    # 2. Inconclusive diagnosis or failed patch application halts repair
    diag = state.get("bug_diagnosis") or ""
    status = state.get("status") or ""
    logs_str = " ".join(state.get("logs") or [])

    is_inconclusive = (
        f"[{BugCategoryEnum.INCONCLUSIVE.value}]" in diag
        or "inconclusive" in diag.lower()
        or "Inconclusive diagnosis" in logs_str
    )
    patch_rejected = (
        "Proposed patch rejected" in logs_str
        or "Failed to apply patch" in logs_str
    )

    if is_inconclusive or patch_rejected:
        return ROUTE_GENERATE_REPORT

    # 3. Check retry budget
    limit = max_retries if max_retries is not None else getattr(config, "MAX_RETRY_LIMIT", 3)
    current_retries = state.get("retry_count", 0)

    if current_retries < limit and status != "failed":
        return ROUTE_BUG_DETECTION

    return ROUTE_GENERATE_REPORT


# ============================================================================
# LangGraph Subgraph Builder
# ============================================================================

def build_testing_subgraph(
    compile_graph: bool = True,
    review_target: Optional[str] = None,
):
    """
    Constructs a standalone LangGraph StateGraph connecting Member 3 components.

    Sub-workflow topology:
      [START] -> testing_node
                     |
                     +---> (all passed) ------------> [review_target or END] (Handoff to Member 4)
                     +---> (failed, retries left) --> bug_detection_node
                     +---> (failed, retries done) --> report_generation_node -> [END]
                                                            ^
                     +--------------------------------------+
                     | (inconclusive / patch failed / retries done)
                     |
               bug_detection_node <----+ (failed, retries left)
                     |                 |
                     +-----------------+
                     |
                     +---> (passed) ----------------> [review_target or END] (Handoff to Member 4)

    Args:
        compile_graph: If True, returns a compiled Runnable graph. If False, returns the raw StateGraph.
        review_target: Node name or destination when tests pass (defaults to LangGraph END).

    Returns:
        Union[CompiledStateGraph, StateGraph]: LangGraph testing and debugging workflow.

    Raises:
        ImportError: If langgraph is not installed in the environment.
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError(
            "LangGraph is required to build the testing subgraph. "
            "Install it via 'pip install langgraph'."
        )

    handoff = review_target if review_target is not None else END

    workflow = StateGraph(AgentState)

    # Register workflow nodes
    workflow.add_node(NODE_TESTING, testing_node)
    workflow.add_node(NODE_BUG_DETECTION, bug_detection_node)
    workflow.add_node(NODE_REPORT_GENERATION, report_generation_node)

    # Entry point
    workflow.set_entry_point(NODE_TESTING)

    # Conditional routing from testing_node
    workflow.add_conditional_edges(
        NODE_TESTING,
        route_after_testing,
        {
            ROUTE_CODE_REVIEW: handoff,
            ROUTE_BUG_DETECTION: NODE_BUG_DETECTION,
            ROUTE_GENERATE_REPORT: NODE_REPORT_GENERATION,
        },
    )

    # Conditional routing from bug_detection_node
    workflow.add_conditional_edges(
        NODE_BUG_DETECTION,
        route_after_debugging,
        {
            ROUTE_CODE_REVIEW: handoff,
            ROUTE_BUG_DETECTION: NODE_BUG_DETECTION,
            ROUTE_GENERATE_REPORT: NODE_REPORT_GENERATION,
        },
    )

    # Terminal edge from report_generation_node
    workflow.add_edge(NODE_REPORT_GENERATION, END)

    if compile_graph:
        return workflow.compile()
    return workflow
