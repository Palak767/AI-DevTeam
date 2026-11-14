"""
Unit tests for Member 3: Test Report Generator.

Tests JSON serialization, Markdown report rendering, status determination,
secret redaction, atomic writing, unique naming, path traversal safety,
and central AgentState integration.
Runs completely offline.
"""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.schemas.state import AgentState, TestResult
from backend.testing_debugging.docker_sandbox import DockerSecurityViolationError
from backend.testing_debugging.report_generator import (
    TestReportGenerator,
    redact_secrets,
)
from backend.testing_debugging.schemas import (
    FinalTestStatusEnum,
    TestReportData,
)

# Inform pytest not to collect TestResult or TestReportGenerator as test suites
TestResult.__test__ = False
TestReportGenerator.__test__ = False


# ============================================================================
# 1. Secret Redaction Tests
# ============================================================================

def test_redact_secrets():
    """Verify sensitive tokens and keys are sanitized from text."""
    # Google API Key
    text_with_google_key = "Error connecting with key AIzaSyD1234567890123456789012345678901 in client"
    sanitized = redact_secrets(text_with_google_key)
    assert "AIzaSy" not in sanitized
    assert "[REDACTED]" in sanitized

    # Key-value assignment
    text_with_env_secret = "Failed at GEMINI_API_KEY='secret_key_value_987654321'"
    sanitized2 = redact_secrets(text_with_env_secret)
    assert "secret_key_value" not in sanitized2
    assert "REDACTED" in sanitized2

    # Bearer token
    text_with_bearer = "Authorization: Bearer my_secret_bearer_token_12345678"
    sanitized3 = redact_secrets(text_with_bearer)
    assert "my_secret_bearer_token" not in sanitized3

    # Clean text unaffected
    clean_text = "All 5 unit tests passed successfully."
    assert redact_secrets(clean_text) == clean_text


# ============================================================================
# 2. Output Path Security & Directory Validation Tests
# ============================================================================

def test_validate_output_directory_success():
    """Verify output directory creation inside workspace."""
    gen = TestReportGenerator(reports_subdir="custom_reports")
    with tempfile.TemporaryDirectory() as ws_dir:
        out_dir = gen.validate_output_directory(ws_dir)
        assert out_dir.exists()
        assert out_dir.name == "custom_reports"
        assert out_dir.is_dir()


def test_validate_output_directory_path_traversal():
    """Verify path traversal outside workspace is blocked with DockerSecurityViolationError."""
    gen = TestReportGenerator(reports_subdir="../../escaped_reports")
    with tempfile.TemporaryDirectory() as ws_dir:
        with pytest.raises(DockerSecurityViolationError) as exc_info:
            gen.validate_output_directory(ws_dir)
        assert "escapes workspace boundary" in str(exc_info.value)


def test_validate_output_directory_invalid_chars():
    """Verify injection characters in subdirectory path are rejected."""
    gen = TestReportGenerator(reports_subdir="reports; rm -rf /")
    with tempfile.TemporaryDirectory() as ws_dir:
        with pytest.raises(ValueError):
            gen.validate_output_directory(ws_dir)


# ============================================================================
# 3. Final Status Calculation Tests
# ============================================================================

def test_determine_final_status_passed():
    """Verify PASSED status calculation."""
    gen = TestReportGenerator()
    results = {"total_tests": 4, "passed_tests": 4, "failed_tests": 0, "passed": True}
    assert gen.determine_final_status(results, is_test_passed=True) == FinalTestStatusEnum.PASSED


def test_determine_final_status_failed():
    """Verify FAILED status calculation."""
    gen = TestReportGenerator()
    results = {"total_tests": 4, "passed_tests": 0, "failed_tests": 4, "passed": False}
    assert gen.determine_final_status(results, is_test_passed=False) == FinalTestStatusEnum.FAILED


def test_determine_final_status_partially_passed():
    """Verify PARTIALLY_PASSED status calculation when some pass and some fail."""
    gen = TestReportGenerator()
    results = {"total_tests": 5, "passed_tests": 3, "failed_tests": 2, "passed": False}
    assert gen.determine_final_status(results, is_test_passed=False) == FinalTestStatusEnum.PARTIALLY_PASSED


def test_determine_final_status_not_executed():
    """Verify NOT_EXECUTED status when no telemetry is present."""
    gen = TestReportGenerator()
    assert gen.determine_final_status(None, is_test_passed=False) == FinalTestStatusEnum.NOT_EXECUTED
    assert gen.determine_final_status({"total_tests": 0}, is_test_passed=False) == FinalTestStatusEnum.NOT_EXECUTED


# ============================================================================
# 4. JSON & Markdown Report Generation Tests
# ============================================================================

def test_successful_report_generation():
    """Verify complete report generation for passing test execution."""
    gen = TestReportGenerator()

    state: AgentState = {
        "project_id": "proj_success",
        "user_prompt": "Build REST API",
        "requirement_spec": {"project_title": "Order Management API"},
        "architecture_blueprint": None,
        "generated_files": [{"path": "orders.py", "content": "pass"}],
        "workspace_path": "",
        "test_results": {
            "passed": True,
            "total_tests": 6,
            "passed_tests": 6,
            "failed_tests": 0,
            "skipped_tests": 0,
            "error_count": 0,
            "duration_seconds": 0.45,
            "raw_output": "=== 6 passed in 0.45s ===",
        },
        "retry_count": 0,
        "is_test_passed": True,
        "bug_diagnosis": None,
        "patched_files": [],
        "security_report": None,
        "current_agent": "Testing Agent",
        "logs": [],
        "status": "completed",
    }

    with tempfile.TemporaryDirectory() as ws_dir:
        report_data, json_path, md_path = gen.generate_reports(state, workspace_path=ws_dir)

        # File presence
        assert json_path.exists()
        assert md_path.exists()

        # Data model assertions
        assert report_data.final_status == FinalTestStatusEnum.PASSED
        assert report_data.summary.total_tests == 6
        assert report_data.summary.passed_tests == 6
        assert report_data.project_info.project_title == "Order Management API"
        assert len(report_data.recommendations) >= 2

        # JSON validity
        json_content = json_path.read_text(encoding="utf-8")
        parsed_json = json.loads(json_content)
        assert parsed_json["final_status"] == "PASSED"
        assert parsed_json["summary"]["total_tests"] == 6

        # Markdown readability
        md_content = md_path.read_text(encoding="utf-8")
        assert "# Test Execution & Quality Report" in md_content
        assert "**PASSED**" in md_content
        assert "100.0%" in md_content
        assert "Order Management API" in md_content


def test_failed_report_generation():
    """Verify failure details and tracebacks are captured in reports."""
    gen = TestReportGenerator()

    raw_out = """
=========================== short test summary info ============================
FAILED tests/test_calc.py::test_division - ZeroDivisionError: division by zero
========================= 1 failed, 2 passed in 0.08s ==========================
"""

    state: AgentState = {
        "project_id": "proj_fail",
        "user_prompt": "Calculator",
        "requirement_spec": {"project_title": "Calculator Service"},
        "architecture_blueprint": None,
        "generated_files": [],
        "workspace_path": "",
        "test_results": {
            "passed": False,
            "total_tests": 3,
            "passed_tests": 2,
            "failed_tests": 1,
            "error_summary": "ZeroDivisionError: division by zero",
            "raw_output": raw_out,
        },
        "retry_count": 1,
        "is_test_passed": False,
        "bug_diagnosis": "[boundary_error] Missing check for zero denominator in division function.",
        "patched_files": [{"path": "calc.py", "content": "def divide(a, b): return a / b if b else 0\n"}],
        "security_report": None,
        "current_agent": "Bug Detection Agent",
        "logs": [],
        "status": "debugging",
    }

    with tempfile.TemporaryDirectory() as ws_dir:
        report_data, json_path, md_path = gen.generate_reports(state, workspace_path=ws_dir)

        assert report_data.final_status == FinalTestStatusEnum.PARTIALLY_PASSED
        assert len(report_data.failures.failed_test_names) >= 1
        assert "tests/test_calc.py::test_division" in report_data.failures.failed_test_names[0]
        assert "ZeroDivisionError" in report_data.failures.error_messages[0]
        assert report_data.diagnosis.bug_category == "boundary_error"
        assert "Missing check for zero" in report_data.diagnosis.root_cause

        md_content = md_path.read_text(encoding="utf-8")
        assert "PARTIALLY PASSED" in md_content
        assert "ZeroDivisionError" in md_content
        assert "boundary_error" in md_content


def test_no_tests_executed_report():
    """Verify report formatting when no test suite was executed."""
    gen = TestReportGenerator()

    state: AgentState = {
        "project_id": "proj_untested",
        "user_prompt": "New service",
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
        "current_agent": "Coding Agent",
        "logs": [],
        "status": "coding",
    }

    with tempfile.TemporaryDirectory() as ws_dir:
        report_data, json_path, md_path = gen.generate_reports(state, workspace_path=ws_dir)

        assert report_data.final_status == FinalTestStatusEnum.NOT_EXECUTED
        assert report_data.summary.total_tests == 0
        assert "NOT EXECUTED" in md_path.read_text(encoding="utf-8")


def test_unique_filenames():
    """Verify successive report generations create unique timestamped files."""
    gen = TestReportGenerator()
    state: AgentState = {
        "project_id": "p_unique",
        "user_prompt": "App",
        "requirement_spec": None,
        "architecture_blueprint": None,
        "generated_files": [],
        "workspace_path": "",
        "test_results": {"passed": True, "total_tests": 1, "passed_tests": 1},
        "retry_count": 0,
        "is_test_passed": True,
        "bug_diagnosis": None,
        "patched_files": [],
        "security_report": None,
        "current_agent": "Testing Agent",
        "logs": [],
        "status": "completed",
    }

    with tempfile.TemporaryDirectory() as ws_dir:
        _, json1, md1 = gen.generate_reports(state, workspace_path=ws_dir)
        _, json2, md2 = gen.generate_reports(state, workspace_path=ws_dir)

        assert json1 != json2
        assert md1 != md2
        assert json1.exists()
        assert json2.exists()


def test_secret_redaction_in_report_output():
    """Verify that credentials in test outputs or state are redacted from both JSON and Markdown reports."""
    gen = TestReportGenerator()

    secret_key = "AIzaSyD9876543210987654321098765432109"
    state: AgentState = {
        "project_id": "proj_secrets",
        "user_prompt": "App",
        "requirement_spec": None,
        "architecture_blueprint": None,
        "generated_files": [],
        "workspace_path": "",
        "test_results": {
            "passed": False,
            "total_tests": 1,
            "failed_tests": 1,
            "error_summary": f"Failed with key {secret_key}",
            "raw_output": f"Traceback: GEMINI_API_KEY='{secret_key}' is invalid.",
        },
        "retry_count": 0,
        "is_test_passed": False,
        "bug_diagnosis": f"Invalid key {secret_key}",
        "patched_files": [{"path": "auth.py", "content": f"KEY = '{secret_key}'"}],
        "security_report": None,
        "current_agent": "Bug Detection Agent",
        "logs": [],
        "status": "debugging",
    }

    with tempfile.TemporaryDirectory() as ws_dir:
        _, json_path, md_path = gen.generate_reports(state, workspace_path=ws_dir)

        json_text = json_path.read_text(encoding="utf-8")
        md_text = md_path.read_text(encoding="utf-8")

        # Crucial assertion: the real API key must never appear in either report file
        assert secret_key not in json_text
        assert secret_key not in md_text
        assert "[REDACTED]" in json_text or "REDACTED" in json_text
        assert "[REDACTED]" in md_text or "REDACTED" in md_text


def test_agentstate_compatibility_and_logging():
    """Verify report generation logs success into AgentState without mutating other fields."""
    gen = TestReportGenerator()

    state: AgentState = {
        "project_id": "proj_state_compat",
        "user_prompt": "Verify state compatibility",
        "requirement_spec": {"project_title": "State Test"},
        "architecture_blueprint": {"files_to_generate": ["main.py"]},
        "generated_files": [{"path": "main.py", "content": "pass"}],
        "workspace_path": "",
        "test_results": {"passed": True, "total_tests": 2, "passed_tests": 2},
        "retry_count": 1,
        "is_test_passed": True,
        "bug_diagnosis": "None",
        "patched_files": [],
        "security_report": {"score": 98},
        "current_agent": "Testing Agent",
        "logs": ["Initial log entry"],
        "status": "completed",
    }

    with tempfile.TemporaryDirectory() as ws_dir:
        report_data, json_path, md_path = gen.generate_reports(state, workspace_path=ws_dir)

        # Check logs updated
        assert any("[Report Generator]" in log for log in state["logs"])
        assert "Initial log entry" in state["logs"]

        # Check unrelated fields preserved intact
        assert state["security_report"] == {"score": 98}
        assert state["architecture_blueprint"] == {"files_to_generate": ["main.py"]}
        assert state["user_prompt"] == "Verify state compatibility"

