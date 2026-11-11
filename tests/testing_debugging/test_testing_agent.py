"""
Unit tests for Member 3: AI Testing Agent.

Tests Gemini integration, test case synthesis, static AST code validation,
file saving security, TestExecutor invocation, and AgentState integration.
Runs completely offline without an API key and without Docker running.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.schemas.state import AgentState, TestResult
from backend.testing_debugging.docker_sandbox import DockerSecurityViolationError
from backend.testing_debugging.schemas import (
    GeneratedTestSuite,
    TestCaseSpec,
)
from backend.testing_debugging.test_executor import TestExecutor
from backend.testing_debugging.testing_agent import (
    TestCodeSecurityValidator,
    TestingAgent,
    get_gemini_model,
    run_testing_agent,
)

# Inform pytest not to collect TestResult or TestingAgent as test suites
TestResult.__test__ = False
TestingAgent.__test__ = False


# ============================================================================
# 1. Static AST Security Validator Tests
# ============================================================================

def test_validator_accepts_valid_safe_pytest_code():
    """Verify validator accepts standard, safe pytest code covering multiple scenarios."""
    code = """
import pytest
from unittest.mock import MagicMock

def test_addition_positive():
    assert 2 + 2 == 4

def test_division_negative():
    with pytest.raises(ZeroDivisionError):
        _ = 10 / 0

def test_boundary_values():
    assert len([]) == 0
    assert len([1]) == 1

def test_edge_case_unicode():
    text = "Hello 🌍"
    assert text.endswith("🌍")
"""
    is_valid, violations = TestCodeSecurityValidator.validate(code)
    assert is_valid is True
    assert violations == []


def test_validator_rejects_syntax_errors():
    """Verify validator detects and rejects invalid Python syntax."""
    code = "def test_broken(:\n    assert 1 == 1"
    is_valid, violations = TestCodeSecurityValidator.validate(code)
    assert is_valid is False
    assert any("Invalid Python syntax" in v for v in violations)


def test_validator_rejects_empty_code():
    """Verify validator rejects empty or whitespace-only code."""
    is_valid, violations = TestCodeSecurityValidator.validate("")
    assert is_valid is False
    assert "empty" in violations[0].lower()

    is_valid, violations = TestCodeSecurityValidator.validate("   \n\t  ")
    assert is_valid is False


def test_validator_rejects_no_functions():
    """Verify validator rejects code with no functions defined."""
    code = "x = 42\ny = 'no tests here'\n"
    is_valid, violations = TestCodeSecurityValidator.validate(code)
    assert is_valid is False
    assert "No functions defined" in violations[0]


def test_validator_rejects_subprocess_import():
    """Verify validator rejects dangerous subprocess module import."""
    code1 = "import subprocess\ndef test_sub():\n    pass"
    is_valid, violations = TestCodeSecurityValidator.validate(code1)
    assert is_valid is False
    assert any("subprocess" in v for v in violations)

    code2 = "from subprocess import Popen\ndef test_sub2():\n    pass"
    is_valid, violations = TestCodeSecurityValidator.validate(code2)
    assert is_valid is False
    assert any("subprocess" in v for v in violations)


def test_validator_rejects_socket_import():
    """Verify validator rejects low-level socket networking import."""
    code = "import socket\ndef test_sock():\n    pass"
    is_valid, violations = TestCodeSecurityValidator.validate(code)
    assert is_valid is False
    assert any("socket" in v for v in violations)


def test_validator_rejects_eval_and_exec():
    """Verify validator rejects dynamic code execution calls."""
    code = "def test_evil():\n    eval('2 + 2')\n"
    is_valid, violations = TestCodeSecurityValidator.validate(code)
    assert is_valid is False
    assert any("eval" in v for v in violations)

    code2 = "def test_evil_exec():\n    exec('pass')\n"
    is_valid, violations = TestCodeSecurityValidator.validate(code2)
    assert is_valid is False
    assert any("exec" in v for v in violations)


def test_validator_rejects_os_system_calls():
    """Verify validator rejects system and shell execution calls."""
    code = "import os\ndef test_shell():\n    os.system('ls')\n"
    is_valid, violations = TestCodeSecurityValidator.validate(code)
    assert is_valid is False
    assert any("os.system" in v for v in violations)


def test_validator_rejects_file_deletion_calls():
    """Verify validator rejects file and directory deletion calls."""
    code = "import os\ndef test_del():\n    os.remove('/important/file')\n"
    is_valid, violations = TestCodeSecurityValidator.validate(code)
    assert is_valid is False
    assert any("os.remove" in v for v in violations)


def test_validator_rejects_forbidden_string_tokens():
    """Verify validator rejects string literals referencing Docker sockets or system roots."""
    code = 'def test_socket_escape():\n    path = "/var/run/docker.sock"\n'
    is_valid, violations = TestCodeSecurityValidator.validate(code)
    assert is_valid is False
    assert any("docker.sock" in v for v in violations)


# ============================================================================
# 2. Gemini Model Configuration Tests
# ============================================================================

def test_get_gemini_model_missing_api_key():
    """Verify get_gemini_model raises ValueError when GEMINI_API_KEY is not configured."""
    with patch("backend.config.config.GEMINI_API_KEY", ""):
        with pytest.raises(ValueError) as exc_info:
            get_gemini_model()
        assert "GEMINI_API_KEY is not configured" in str(exc_info.value)


def test_get_gemini_model_configured():
    """Verify get_gemini_model configures genai and returns a GenerativeModel."""
    with patch("backend.config.config.GEMINI_API_KEY", "test_mock_key_123"):
        mock_genai = MagicMock()
        with patch("backend.testing_debugging.testing_agent.genai", mock_genai):
            model = get_gemini_model("gemini-test")
            mock_genai.configure.assert_called_once_with(api_key="test_mock_key_123")
            mock_genai.GenerativeModel.assert_called_once()
            assert model is not None


# ============================================================================
# 3. Test Generation with Mocked Gemini Responses
# ============================================================================

def test_successful_test_generation_mocked_gemini():
    """Verify complete test generation from requirements and code using mocked Gemini."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """
{
    "test_file_path": "tests/test_calculator.py",
    "scenarios_summary": "Comprehensive test suite covering basic operations and edge cases",
    "test_cases": [
        {
            "name": "test_add_positive",
            "category": "positive",
            "description": "Verify addition of two positive integers",
            "expected_outcome": "Returns correct sum"
        },
        {
            "name": "test_divide_negative",
            "category": "negative",
            "description": "Verify ZeroDivisionError raised on zero division",
            "expected_outcome": "ZeroDivisionError is raised"
        },
        {
            "name": "test_boundary_zero",
            "category": "boundary",
            "description": "Verify addition with zero boundary",
            "expected_outcome": "Returns unchanged operand"
        }
    ],
    "test_code": "import pytest\\n\\ndef test_add_positive():\\n    assert 2 + 3 == 5\\n\\ndef test_divide_negative():\\n    with pytest.raises(ZeroDivisionError):\\n        _ = 10 / 0\\n\\ndef test_boundary_zero():\\n    assert 0 + 7 == 7\\n"
}
"""
    mock_model.generate_content.return_value = mock_response

    agent = TestingAgent(model=mock_model)
    reqs = {
        "project_title": "Calculator App",
        "functional_requirements": ["Add numbers", "Divide numbers"],
        "acceptance_criteria": ["Division by zero raises ZeroDivisionError"],
    }
    files = [{"path": "calc.py", "content": "def add(a, b): return a + b"}]

    suite = agent.generate_tests(requirement_spec=reqs, generated_files=files)

    assert suite.test_file_path == "tests/test_calculator.py"
    assert len(suite.test_cases) == 3
    assert suite.test_cases[0].category == "positive"
    assert suite.test_cases[1].category == "negative"
    assert suite.test_cases[2].category == "boundary"
    assert "def test_add_positive():" in suite.test_code
    mock_model.generate_content.assert_called_once()


def test_generate_tests_handles_markdown_wrapped_json():
    """Verify model output wrapped in markdown code fences is cleanly parsed."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """```json
{
    "test_file_path": "tests/test_sample.py",
    "scenarios_summary": "Sample suite",
    "test_cases": [
        {
            "name": "test_sample",
            "category": "positive",
            "description": "Sample test",
            "expected_outcome": "Pass"
        }
    ],
    "test_code": "def test_sample():\\n    assert True\\n"
}
```"""
    mock_model.generate_content.return_value = mock_response

    agent = TestingAgent(model=mock_model)
    suite = agent.generate_tests()
    assert suite.test_file_path == "tests/test_sample.py"
    assert len(suite.test_cases) == 1


def test_generate_tests_invalid_json_triggers_safe_fallback():
    """Verify invalid JSON from model falls back cleanly to baseline test suite."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is plain text, not JSON."
    mock_model.generate_content.return_value = mock_response

    agent = TestingAgent(model=mock_model)
    suite = agent.generate_tests(generated_files=[{"path": "app.py", "content": "x = 1"}])

    assert suite is not None
    assert suite.test_file_path == "tests/test_sanity.py"
    assert "Fallback baseline test suite" in suite.scenarios_summary
    assert "def test_environment_sanity():" in suite.test_code


def test_generate_tests_empty_response_triggers_fallback():
    """Verify empty response from model falls back cleanly to baseline test suite."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = ""
    mock_model.generate_content.return_value = mock_response

    agent = TestingAgent(model=mock_model)
    suite = agent.generate_tests()

    assert suite is not None
    assert suite.test_file_path == "tests/test_sanity.py"


def test_generate_tests_api_failure_triggers_fallback():
    """Verify Gemini API exception triggers fallback without crashing."""
    mock_model = MagicMock()
    mock_model.generate_content.side_effect = Exception("API rate limit exceeded")

    agent = TestingAgent(model=mock_model)
    suite = agent.generate_tests()

    assert suite is not None
    assert suite.test_file_path == "tests/test_sanity.py"
    assert "Fallback" in suite.scenarios_summary


def test_generate_tests_unsafe_code_rejection_triggers_fallback():
    """Verify that if Gemini generates unsafe code, it is rejected and fallback is returned."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """
{
    "test_file_path": "tests/test_exploit.py",
    "scenarios_summary": "Unsafe test",
    "test_cases": [],
    "test_code": "import subprocess\\ndef test_run():\\n    subprocess.run(['rm', '-rf'])\\n"
}
"""
    mock_model.generate_content.return_value = mock_response

    agent = TestingAgent(model=mock_model)
    suite = agent.generate_tests()

    # The unsafe code must be rejected and safe fallback returned
    assert "subprocess" not in suite.test_code
    assert suite.test_file_path == "tests/test_sanity.py"


# ============================================================================
# 4. Workspace Staging & File Saving Security Tests
# ============================================================================

def test_save_test_file_valid():
    """Verify saving a validated test suite into a workspace directory."""
    agent = TestingAgent()
    suite = GeneratedTestSuite(
        test_file_path="tests/test_math.py",
        scenarios_summary="Math tests",
        test_cases=[],
        test_code="def test_add(): assert 1 + 1 == 2\n",
    )

    with tempfile.TemporaryDirectory() as ws_dir:
        written_path = agent.save_test_file(ws_dir, suite)
        assert written_path.exists()
        assert written_path.name == "test_math.py"
        assert written_path.read_text(encoding="utf-8") == suite.test_code


def test_save_test_file_path_traversal():
    """Verify path traversal in test_file_path is blocked with DockerSecurityViolationError."""
    agent = TestingAgent()
    suite = GeneratedTestSuite(
        test_file_path="../../outside_test.py",
        scenarios_summary="Traversal attempt",
        test_cases=[],
        test_code="def test_evil(): pass\n",
    )

    with tempfile.TemporaryDirectory() as ws_dir:
        with pytest.raises(DockerSecurityViolationError):
            agent.save_test_file(ws_dir, suite)


# ============================================================================
# 5. TestingAgent End-to-End Orchestration & AgentState Tests
# ============================================================================

def test_testing_agent_run_success():
    """Verify TestingAgent.run updates AgentState correctly on passing tests."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """
{
    "test_file_path": "tests/test_unit.py",
    "scenarios_summary": "Passing unit test suite",
    "test_cases": [
        {"name": "test_pass", "category": "positive", "description": "Pass test", "expected_outcome": "Pass"}
    ],
    "test_code": "def test_pass(): assert True\\n"
}
"""
    mock_model.generate_content.return_value = mock_response

    mock_executor = MagicMock(spec=TestExecutor)
    mock_executor.run_tests.return_value = TestResult(
        passed=True,
        total_tests=3,
        passed_tests=3,
        failed_tests=0,
        raw_output="=== 3 passed ===",
        error_summary=None,
    )
    # Let update_agent_state perform real update logic
    real_executor = TestExecutor(sandbox=MagicMock())
    mock_executor.update_agent_state.side_effect = real_executor.update_agent_state

    agent = TestingAgent(model=mock_model, executor=mock_executor)

    with tempfile.TemporaryDirectory() as ws_dir:
        state: AgentState = {
            "project_id": "proj_101",
            "user_prompt": "Build REST API",
            "requirement_spec": {
                "project_title": "REST API",
                "functional_requirements": ["CRUD endpoints"],
                "acceptance_criteria": ["200 on success"],
            },
            "architecture_blueprint": None,
            "generated_files": [{"path": "main.py", "content": "print('hello')"}],
            "workspace_path": ws_dir,
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

        updated_state = agent.run(state)

        assert updated_state["current_agent"] == "Testing Agent"
        assert updated_state["is_test_passed"] is True
        assert updated_state["status"] == "reviewing"
        assert updated_state["test_results"]["passed"] is True
        assert updated_state["test_results"]["total_tests"] == 3
        assert any("[Testing Agent]" in log for log in updated_state["logs"])
        mock_executor.run_tests.assert_called_once()


def test_testing_agent_run_failure():
    """Verify TestingAgent.run updates AgentState on test failures (increments retries, sets status=debugging)."""
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = """
{
    "test_file_path": "tests/test_unit.py",
    "scenarios_summary": "Failing test suite",
    "test_cases": [
        {"name": "test_fail", "category": "negative", "description": "Failure test", "expected_outcome": "Fail"}
    ],
    "test_code": "def test_fail(): assert False\\n"
}
"""
    mock_model.generate_content.return_value = mock_response

    mock_executor = MagicMock(spec=TestExecutor)
    mock_executor.run_tests.return_value = TestResult(
        passed=False,
        total_tests=3,
        passed_tests=2,
        failed_tests=1,
        raw_output="=== 1 failed, 2 passed ===",
        error_summary="AssertionError: assert False",
    )
    real_executor = TestExecutor(sandbox=MagicMock())
    mock_executor.update_agent_state.side_effect = real_executor.update_agent_state

    agent = TestingAgent(model=mock_model, executor=mock_executor)

    with tempfile.TemporaryDirectory() as ws_dir:
        state: AgentState = {
            "project_id": "proj_102",
            "user_prompt": "Build REST API",
            "requirement_spec": None,
            "architecture_blueprint": None,
            "generated_files": [],
            "workspace_path": ws_dir,
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

        updated_state = agent.run(state)

        assert updated_state["current_agent"] == "Testing Agent"
        assert updated_state["is_test_passed"] is False
        assert updated_state["status"] == "debugging"
        assert updated_state["retry_count"] == 1
        assert updated_state["test_results"]["failed_tests"] == 1
        assert any("FAILED: 1/3" in log for log in updated_state["logs"])


def test_run_testing_agent_entrypoint():
    """Verify run_testing_agent convenience function works as expected."""
    with patch("backend.testing_debugging.testing_agent.TestingAgent.run") as mock_run:
        mock_run.return_value = {"status": "completed"}
        state: AgentState = {"project_id": "test"}  # type: ignore
        res = run_testing_agent(state)
        assert res == {"status": "completed"}
        mock_run.assert_called_once()
