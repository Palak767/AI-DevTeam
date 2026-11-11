"""
Unit tests for Member 3: Bug Detection and Self-Correction Agent.

Tests root cause diagnosis, patch validation, atomic file application,
hash verification, bounded self-correction loops, and AgentState integration.
Runs completely offline without live Gemini credentials or Docker.
"""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.schemas.state import AgentState, TestResult
from backend.testing_debugging.bug_detection_agent import (
    BugDetectionAgent,
    SafePatchManager,
    run_bug_agent,
)
from backend.testing_debugging.schemas import (
    BugCategoryEnum,
    BugDiagnosis,
    PatchProposal,
)
from backend.testing_debugging.test_executor import TestExecutor

# Inform pytest not to collect TestResult or BugDetectionAgent as test suites
TestResult.__test__ = False
BugDetectionAgent.__test__ = False


# ============================================================================
# 1. SafePatchManager & Patch Validation Tests
# ============================================================================

def test_patch_validation_success():
    """Verify valid patch proposal validation."""
    mgr = SafePatchManager()
    with tempfile.TemporaryDirectory() as ws_dir:
        ws = Path(ws_dir)
        target = ws / "calc.py"
        target.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

        patch = PatchProposal(
            target_file="calc.py",
            original_code="return a - b",
            replacement_code="return a + b",
            explanation="Fix incorrect subtraction operator in add function",
            expected_effect="Test will pass addition assertions",
        )

        is_valid, err = mgr.validate_patch(ws, patch)
        assert is_valid is True
        assert err is None
        assert patch.validation_status == "valid"


def test_patch_validation_path_traversal():
    """Verify patch targeting a path outside workspace is rejected."""
    mgr = SafePatchManager()
    with tempfile.TemporaryDirectory() as ws_dir:
        patch = PatchProposal(
            target_file="../../outside.py",
            original_code="x = 1",
            replacement_code="x = 2",
            explanation="Path traversal attempt",
            expected_effect="None",
        )
        is_valid, err = mgr.validate_patch(ws_dir, patch)
        assert is_valid is False
        assert "escapes workspace boundary" in err


def test_patch_validation_protected_files():
    """Verify patches targeting protected files (.env, state.py, config.py) are rejected."""
    mgr = SafePatchManager()
    with tempfile.TemporaryDirectory() as ws_dir:
        for protected in [".env", ".env.local", "backend/schemas/state.py", "backend/config.py"]:
            patch = PatchProposal(
                target_file=protected,
                original_code="A = 1",
                replacement_code="A = 2",
                explanation="Modifying protected file",
                expected_effect="None",
            )
            is_valid, err = mgr.validate_patch(ws_dir, patch)
            assert is_valid is False
            assert "protected" in err.lower()


def test_patch_validation_original_content_mismatch():
    """Verify patch is rejected when original_code does not match target file."""
    mgr = SafePatchManager()
    with tempfile.TemporaryDirectory() as ws_dir:
        ws = Path(ws_dir)
        target = ws / "main.py"
        target.write_text("def run():\n    return 42\n", encoding="utf-8")

        patch = PatchProposal(
            target_file="main.py",
            original_code="return 100",  # Does not exist in main.py
            replacement_code="return 200",
            explanation="Mismatch test",
            expected_effect="None",
        )
        is_valid, err = mgr.validate_patch(ws, patch)
        assert is_valid is False
        assert "Original code snippet not found" in err


def test_patch_validation_invalid_resulting_syntax():
    """Verify patch is rejected if replacement causes a Python SyntaxError."""
    mgr = SafePatchManager()
    with tempfile.TemporaryDirectory() as ws_dir:
        ws = Path(ws_dir)
        target = ws / "syntax_test.py"
        target.write_text("def test_ok():\n    assert True\n", encoding="utf-8")

        patch = PatchProposal(
            target_file="syntax_test.py",
            original_code="assert True",
            replacement_code="def broken(:\n    assert",  # Syntax error
            explanation="Syntax error test",
            expected_effect="None",
        )
        is_valid, err = mgr.validate_patch(ws, patch)
        assert is_valid is False
        assert "Python syntax error" in err


def test_patch_validation_security_violation():
    """Verify patch is rejected if it introduces disallowed imports like subprocess."""
    mgr = SafePatchManager()
    with tempfile.TemporaryDirectory() as ws_dir:
        ws = Path(ws_dir)
        target = ws / "service.py"
        target.write_text("def process():\n    pass\n", encoding="utf-8")

        patch = PatchProposal(
            target_file="service.py",
            original_code="pass",
            replacement_code="import subprocess\n    subprocess.run(['ls'])",
            explanation="Malicious patch attempt",
            expected_effect="None",
        )
        is_valid, err = mgr.validate_patch(ws, patch)
        assert is_valid is False
        assert "violates security policies" in err


def test_atomic_patch_application_and_hash_verification():
    """Verify atomic patch application updates file and records before/after SHA-256 hashes."""
    mgr = SafePatchManager()
    with tempfile.TemporaryDirectory() as ws_dir:
        ws = Path(ws_dir)
        target = ws / "math_ops.py"
        initial_content = "def multiply(a, b):\n    return a + b\n"
        target.write_text(initial_content, encoding="utf-8")

        expected_before_hash = hashlib.sha256(initial_content.encode("utf-8")).hexdigest()
        expected_after_content = "def multiply(a, b):\n    return a * b\n"
        expected_after_hash = hashlib.sha256(expected_after_content.encode("utf-8")).hexdigest()

        patch = PatchProposal(
            target_file="math_ops.py",
            original_code="return a + b",
            replacement_code="return a * b",
            explanation="Fix multiplication operator",
            expected_effect="Multiplication tests pass",
        )

        success, audit_record, err = mgr.apply_patch(ws, patch)
        assert success is True
        assert err is None
        assert audit_record is not None
        assert audit_record.before_hash == expected_before_hash
        assert audit_record.after_hash == expected_after_hash
        assert audit_record.applied is True

        # Verify disk content updated
        new_content_on_disk = target.read_text(encoding="utf-8")
        assert new_content_on_disk == expected_after_content
        assert len(mgr.audit_trail) == 1


# ============================================================================
# 2. Gemini Root Cause Diagnosis Tests
# ============================================================================

def test_successful_diagnosis_mocked_gemini():
    """Verify Gemini diagnosis parses category, root cause, and structured patch proposal."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """
{
    "diagnosis": {
        "category": "logic_error",
        "failure_description": "test_discount failed with AssertionError",
        "root_cause": "Discount calculation subtracted fixed amount instead of percentage",
        "relevant_file": "pricing.py",
        "code_location": "apply_discount",
        "suggested_fix": "Change price - discount to price * (1 - discount)",
        "confidence": "high",
        "evidence": "AssertionError: 80 != 90"
    },
    "patch": {
        "target_file": "pricing.py",
        "original_code": "return price - discount",
        "replacement_code": "return price * (1 - discount)",
        "explanation": "Compute percentage discount accurately",
        "expected_effect": "Test discount assertions pass"
    }
}
"""
    mock_model.generate_content.return_value = mock_response
    agent = BugDetectionAgent(model=mock_model)

    test_res = TestResult(
        passed=False,
        total_tests=2,
        passed_tests=1,
        failed_tests=1,
        raw_output="FAILED test_pricing.py::test_discount - AssertionError: 80 != 90",
        error_summary="AssertionError: 80 != 90",
    )
    source_files = [{"path": "pricing.py", "content": "def apply_discount(price, discount):\n    return price - discount\n"}]

    diag, patch_prop = agent.diagnose_failure(test_res, source_files)

    assert diag.category == BugCategoryEnum.LOGIC_ERROR
    assert diag.confidence == "high"
    assert "Discount calculation" in diag.root_cause
    assert patch_prop is not None
    assert patch_prop.target_file == "pricing.py"
    assert patch_prop.original_code == "return price - discount"
    assert patch_prop.replacement_code == "return price * (1 - discount)"


def test_inconclusive_diagnosis_mocked_gemini():
    """Verify inconclusive diagnosis does not produce a patch."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """
{
    "diagnosis": {
        "category": "inconclusive",
        "failure_description": "Ambiguous failure",
        "root_cause": "Traceback truncated and source file missing",
        "relevant_file": null,
        "code_location": null,
        "suggested_fix": "Provide full tracebacks",
        "confidence": "low",
        "evidence": "unknown error"
    },
    "patch": null
}
"""
    mock_model.generate_content.return_value = mock_response
    agent = BugDetectionAgent(model=mock_model)

    test_res = TestResult(passed=False, raw_output="unknown error")
    diag, patch_prop = agent.diagnose_failure(test_res, [])

    assert diag.category == BugCategoryEnum.INCONCLUSIVE
    assert patch_prop is None


def test_invalid_model_output_falls_back_heuristic():
    """Verify invalid model output falls back to heuristic diagnosis without throwing."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "INVALID_NON_JSON_RESPONSE"
    mock_model.generate_content.return_value = mock_response
    agent = BugDetectionAgent(model=mock_model)

    test_res = TestResult(
        passed=False,
        raw_output="FAILED test_foo.py - TypeError: unsupported operand type",
        error_summary="TypeError: unsupported operand type",
    )
    diag, patch_prop = agent.diagnose_failure(test_res, [{"path": "foo.py", "content": "pass"}])

    assert diag.category == BugCategoryEnum.TYPE_ERROR
    assert diag.confidence == "medium"
    assert patch_prop is None  # Never invent a patch on fallback


def test_api_failure_falls_back_heuristic():
    """Verify Gemini API exception triggers heuristic diagnosis gracefully."""
    mock_model = MagicMock()
    mock_model.generate_content.side_effect = Exception("API rate limit reached")
    agent = BugDetectionAgent(model=mock_model)

    test_res = TestResult(
        passed=False,
        raw_output="SyntaxError: invalid syntax in app.py at line 5",
        error_summary="SyntaxError: invalid syntax",
    )
    diag, patch_prop = agent.diagnose_failure(test_res, [])

    assert diag.category == BugCategoryEnum.SYNTAX_ERROR
    assert patch_prop is None


# ============================================================================
# 3. Bounded Self-Correction Coordinator Tests
# ============================================================================

def test_self_correct_already_passing():
    """Verify self-correction takes no action when tests are already passing."""
    agent = BugDetectionAgent(model=MagicMock())
    state: AgentState = {
        "project_id": "test_pass",
        "user_prompt": "App",
        "requirement_spec": None,
        "architecture_blueprint": None,
        "generated_files": [],
        "workspace_path": "/tmp/ws",
        "test_results": {"passed": True},
        "retry_count": 0,
        "is_test_passed": True,
        "bug_diagnosis": None,
        "patched_files": [],
        "security_report": None,
        "current_agent": "Testing Agent",
        "logs": [],
        "status": "testing",
    }

    updated = agent.self_correct(state)
    assert updated["status"] == "reviewing"
    assert any("already passing" in log for log in updated["logs"])


def test_self_correct_successful_on_first_try():
    """Verify end-to-end self-correction when patch fixes the bug on attempt 1."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """
{
    "diagnosis": {
        "category": "logic_error",
        "failure_description": "calculate returned wrong value",
        "root_cause": "Used addition instead of multiplication",
        "relevant_file": "logic.py",
        "code_location": "calculate",
        "suggested_fix": "Use multiplication",
        "confidence": "high",
        "evidence": "AssertionError"
    },
    "patch": {
        "target_file": "logic.py",
        "original_code": "return x + 2",
        "replacement_code": "return x * 2",
        "explanation": "Use multiplication",
        "expected_effect": "Tests pass"
    }
}
"""
    mock_model.generate_content.return_value = mock_response

    mock_executor = MagicMock(spec=TestExecutor)
    # Rerun succeeds!
    mock_executor.run_tests.return_value = TestResult(
        passed=True,
        total_tests=3,
        passed_tests=3,
        failed_tests=0,
        raw_output="=== 3 passed ===",
    )

    with tempfile.TemporaryDirectory() as ws_dir:
        ws = Path(ws_dir)
        (ws / "logic.py").write_text("def calculate(x):\n    return x + 2\n", encoding="utf-8")

        agent = BugDetectionAgent(model=mock_model, executor=mock_executor, max_retries=3)

        state: AgentState = {
            "project_id": "proj_fix",
            "user_prompt": "App",
            "requirement_spec": None,
            "architecture_blueprint": None,
            "generated_files": [{"path": "logic.py", "content": "def calculate(x):\n    return x + 2\n"}],
            "workspace_path": ws_dir,
            "test_results": {"passed": False, "failed_tests": 1, "total_tests": 3, "raw_output": "AssertionError"},
            "retry_count": 0,
            "is_test_passed": False,
            "bug_diagnosis": None,
            "patched_files": [],
            "security_report": None,
            "current_agent": "Testing Agent",
            "logs": [],
            "status": "debugging",
        }

        updated_state = agent.self_correct(state)

        # Assertions
        assert updated_state["is_test_passed"] is True
        assert updated_state["status"] == "reviewing"
        assert updated_state["retry_count"] == 1
        assert len(updated_state["patched_files"]) == 1
        assert updated_state["patched_files"][0]["path"] == "logic.py"
        assert "return x * 2" in updated_state["patched_files"][0]["content"]
        assert "logic_error" in updated_state["bug_diagnosis"]
        assert any("SUCCESSFUL" in log for log in updated_state["logs"])

        # Check disk was updated
        assert "return x * 2" in (ws / "logic.py").read_text(encoding="utf-8")


def test_self_correct_exhausts_retries_without_infinite_loop():
    """Verify that when patches do not resolve failures, retry limit is strictly respected."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """
{
    "diagnosis": {
        "category": "logic_error",
        "failure_description": "Persistent failure",
        "root_cause": "Logic error",
        "relevant_file": "app.py",
        "suggested_fix": "Patch",
        "confidence": "high",
        "evidence": "AssertionError"
    },
    "patch": {
        "target_file": "app.py",
        "original_code": "x = 1",
        "replacement_code": "x = 2",
        "explanation": "Try changing x",
        "expected_effect": "Test pass"
    }
}
"""
    mock_model.generate_content.return_value = mock_response

    mock_executor = MagicMock(spec=TestExecutor)
    # Rerun keeps failing!
    mock_executor.run_tests.return_value = TestResult(
        passed=False,
        total_tests=2,
        failed_tests=1,
        raw_output="AssertionError: still failed",
    )

    with tempfile.TemporaryDirectory() as ws_dir:
        ws = Path(ws_dir)
        (ws / "app.py").write_text("x = 1\n", encoding="utf-8")

        # Set max_retries = 1 so attempt 1 exhausts the limit after test failure
        agent = BugDetectionAgent(model=mock_model, executor=mock_executor, max_retries=1)

        state: AgentState = {
            "project_id": "proj_retry_exhaust",
            "user_prompt": "App",
            "requirement_spec": None,
            "architecture_blueprint": None,
            "generated_files": [{"path": "app.py", "content": "x = 1\n"}],
            "workspace_path": ws_dir,
            "test_results": {"passed": False},
            "retry_count": 0,
            "is_test_passed": False,
            "bug_diagnosis": None,
            "patched_files": [],
            "security_report": None,
            "current_agent": "Testing Agent",
            "logs": [],
            "status": "debugging",
        }

        updated_state = agent.self_correct(state)

        # Loop must terminate at max_retries
        assert updated_state["is_test_passed"] is False
        assert updated_state["status"] == "failed"
        assert updated_state["retry_count"] >= 1
        assert any("Maximum retry limit" in log for log in updated_state["logs"])


def test_self_correct_halts_when_initial_retry_count_at_maximum():
    """Verify self-correction halts immediately if state already reached MAX_RETRY_LIMIT."""
    agent = BugDetectionAgent(model=MagicMock(), max_retries=3)

    state: AgentState = {
        "project_id": "proj_already_max",
        "user_prompt": "App",
        "requirement_spec": None,
        "architecture_blueprint": None,
        "generated_files": [],
        "workspace_path": "/tmp/ws",
        "test_results": {"passed": False},
        "retry_count": 3,  # Already at max
        "is_test_passed": False,
        "bug_diagnosis": None,
        "patched_files": [],
        "security_report": None,
        "current_agent": "Testing Agent",
        "logs": [],
        "status": "debugging",
    }

    updated = agent.self_correct(state)
    assert updated["status"] == "failed"
    assert any("already reached" in log for log in updated["logs"])


def test_self_correct_halts_on_inconclusive_diagnosis():
    """Verify self-correction halts without applying blind patches when diagnosis is inconclusive."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """
{
    "diagnosis": {
        "category": "inconclusive",
        "failure_description": "Unknown",
        "root_cause": "Insufficient info",
        "suggested_fix": "Manual review",
        "confidence": "low",
        "evidence": "none"
    },
    "patch": null
}
"""
    mock_model.generate_content.return_value = mock_response
    agent = BugDetectionAgent(model=mock_model)

    state: AgentState = {
        "project_id": "proj_inconclusive",
        "user_prompt": "App",
        "requirement_spec": None,
        "architecture_blueprint": None,
        "generated_files": [],
        "workspace_path": "/tmp/ws",
        "test_results": {"passed": False},
        "retry_count": 0,
        "is_test_passed": False,
        "bug_diagnosis": None,
        "patched_files": [],
        "security_report": None,
        "current_agent": "Testing Agent",
        "logs": [],
        "status": "debugging",
    }

    updated = agent.self_correct(state)
    assert updated["status"] == "failed"
    assert any("Inconclusive diagnosis" in log for log in updated["logs"])
    assert updated["retry_count"] == 0  # No attempt burned


def test_run_bug_agent_entrypoint():
    """Verify module-level run_bug_agent convenience entrypoint."""
    with patch("backend.testing_debugging.bug_detection_agent.BugDetectionAgent.self_correct") as mock_sc:
        mock_sc.return_value = {"status": "reviewing"}
        state: AgentState = {"project_id": "p1"}  # type: ignore
        res = run_bug_agent(state)
        assert res == {"status": "reviewing"}
        mock_sc.assert_called_once()
