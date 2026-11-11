"""
Unit tests for Member 3: Pytest Execution Engine.

Tests test execution, workspace staging, sensitive file exclusion,
output parsing, and shared state compatibility using mocked Docker sandbox objects.
Runs completely offline without Docker Desktop.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.schemas.state import AgentState, TestResult

# Inform pytest not to collect TestResult as a test suite
TestResult.__test__ = False
from backend.testing_debugging.docker_sandbox import (
    DockerDaemonUnavailableError,
    DockerSandbox,
    DockerSandboxError,
    DockerSecurityViolationError,
)
from backend.testing_debugging.schemas import (
    ContainerStateEnum,
    ContainerStatus,
    DockerSandboxConfig,
    ExecutionResult,
    PytestExecutionDetails,
)
from backend.testing_debugging.test_executor import (
    PytestOutputParser,
    TestExecutor,
)


# ============================================================================
# 1. PytestOutputParser Unit Tests
# ============================================================================

def test_parse_all_passed():
    """Verify parsing when all tests pass."""
    stdout = """
============================= test session starts ==============================
collected 3 items

tests/test_api.py ...                                                    [100%]

============================== 3 passed in 0.04s ===============================
"""
    parsed = PytestOutputParser.parse(stdout=stdout, stderr="", exit_code=0, duration_seconds=0.04)
    assert parsed.passed is True
    assert parsed.total_tests == 3
    assert parsed.passed_tests == 3
    assert parsed.failed_tests == 0
    assert parsed.skipped_tests == 0
    assert parsed.error_count == 0
    assert parsed.exit_code == 0
    assert parsed.error_summary is None


def test_parse_some_failed():
    """Verify parsing when tests fail and extraction of failure summary."""
    stdout = """
============================= test session starts ==============================
collected 4 items

test_calc.py ..F.                                                        [100%]

=================================== FAILURES ===================================
__________________________________ test_divide _________________________________
def test_divide():
>       assert divide(10, 2) == 4
E       assert 5 == 4
test_calc.py:12: AssertionError
=========================== short test summary info ============================
FAILED test_calc.py::test_divide - assert 5 == 4
========================= 1 failed, 3 passed in 0.08s ==========================
"""
    parsed = PytestOutputParser.parse(stdout=stdout, stderr="", exit_code=1, duration_seconds=0.08)
    assert parsed.passed is False
    assert parsed.total_tests == 4
    assert parsed.passed_tests == 3
    assert parsed.failed_tests == 1
    assert parsed.skipped_tests == 0
    assert parsed.exit_code == 1
    assert parsed.error_summary is not None
    assert "FAILED test_calc.py::test_divide - assert 5 == 4" in parsed.error_summary


def test_parse_skipped_tests():
    """Verify parsing output with skipped tests."""
    stdout = """
============================= test session starts ==============================
collected 3 items

test_mod.py .s.                                                          [100%]

========================= 2 passed, 1 skipped in 0.02s =========================
"""
    parsed = PytestOutputParser.parse(stdout=stdout, stderr="", exit_code=0, duration_seconds=0.02)
    assert parsed.passed is True
    assert parsed.total_tests == 3
    assert parsed.passed_tests == 2
    assert parsed.skipped_tests == 1
    assert parsed.failed_tests == 0


def test_parse_collection_error():
    """Verify parsing collection error (syntax error / import failure)."""
    stdout = """
============================= test session starts ==============================
==================================== ERRORS ====================================
_______________________ ERROR collecting tests/test_bad.py _____________________
tests/test_bad.py:1: in <module>
    import non_existent_library
E   ModuleNotFoundError: No module named 'non_existent_library'
=========================== short test summary info ============================
ERROR tests/test_bad.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=============================== 1 error in 0.09s ===============================
"""
    parsed = PytestOutputParser.parse(stdout=stdout, stderr="", exit_code=2, duration_seconds=0.09)
    assert parsed.passed is False
    assert parsed.error_count >= 1
    assert parsed.exit_code == 2
    assert parsed.error_summary is not None
    assert "ERROR" in parsed.error_summary


def test_parse_no_tests_collected():
    """Verify parsing when pytest collects 0 tests (exit code 5)."""
    stdout = """
============================= test session starts ==============================
collected 0 items

============================ no tests ran in 0.00s =============================
"""
    parsed = PytestOutputParser.parse(stdout=stdout, stderr="", exit_code=5, duration_seconds=0.01)
    assert parsed.passed is False
    assert parsed.total_tests == 0
    assert parsed.passed_tests == 0
    assert parsed.failed_tests == 0
    assert parsed.exit_code == 5
    assert "No tests were collected" in parsed.error_summary


def test_parse_timeout():
    """Verify parsing when execution times out."""
    stdout = "test_loop.py .."
    parsed = PytestOutputParser.parse(
        stdout=stdout,
        stderr="[Execution Timed Out after 30 seconds]",
        exit_code=124,
        duration_seconds=30.0,
        timed_out=True,
    )
    assert parsed.passed is False
    assert parsed.timed_out is True
    assert parsed.exit_code == 124
    assert "timed out" in parsed.error_summary.lower()


def test_parse_missing_pytest():
    """Verify parsing when pytest is not installed or command not found."""
    stderr = "sh: pytest: command not found"
    parsed = PytestOutputParser.parse(stdout="", stderr=stderr, exit_code=127)
    assert parsed.passed is False
    assert parsed.exit_code == 127
    assert "Pytest is not installed" in parsed.error_summary


def test_parse_ansi_escape_stripping():
    """Verify ANSI color codes are stripped before parsing."""
    colored_stdout = "\x1b[1m\x1b[32m=== 4 passed in 0.05s ===\x1b[0m"
    clean = PytestOutputParser.strip_ansi(colored_stdout)
    assert "\x1b" not in clean
    parsed = PytestOutputParser.parse(stdout=colored_stdout, stderr="", exit_code=0)
    assert parsed.passed is True
    assert parsed.passed_tests == 4


# ============================================================================
# 2. Workspace Isolation & Sensitive File Exclusion Tests
# ============================================================================

def test_is_sensitive():
    """Verify sensitive file identification."""
    # Should be identified as sensitive
    assert TestExecutor.is_sensitive(".env") is True
    assert TestExecutor.is_sensitive(".env.production") is True
    assert TestExecutor.is_sensitive("id_rsa") is True
    assert TestExecutor.is_sensitive("server.key") is True
    assert TestExecutor.is_sensitive("cert.pem") is True
    assert TestExecutor.is_sensitive("credentials.json") is True
    assert TestExecutor.is_sensitive("auth_token.txt") is True
    assert TestExecutor.is_sensitive(Path(".git") / "HEAD") is True
    assert TestExecutor.is_sensitive(Path(".venv") / "bin" / "python") is True
    assert TestExecutor.is_sensitive(Path("__pycache__") / "mod.pyc") is True
    assert TestExecutor.is_sensitive(Path(".idea") / "workspace.xml") is True
    assert TestExecutor.is_sensitive(Path(".vscode") / "settings.json") is True
    assert TestExecutor.is_sensitive(Path("node_modules") / "pkg") is True

    # Should NOT be identified as sensitive
    assert TestExecutor.is_sensitive("main.py") is False
    assert TestExecutor.is_sensitive("app/routes.py") is False
    assert TestExecutor.is_sensitive("tests/test_api.py") is False
    assert TestExecutor.is_sensitive("README.md") is False
    assert TestExecutor.is_sensitive("requirements.txt") is False


def test_prepare_isolated_workspace_copies_allowed_and_excludes_sensitive():
    """Verify that staging creates an isolated copy with sensitive files removed."""
    sandbox = DockerSandbox(docker_client=MagicMock())
    executor = TestExecutor(sandbox=sandbox)

    with tempfile.TemporaryDirectory() as src_dir:
        src = Path(src_dir)
        # Create standard project files
        (src / "app").mkdir()
        (src / "app" / "main.py").write_text("print('app')")
        (src / "tests").mkdir()
        (src / "tests" / "test_main.py").write_text("def test_ok(): assert True")

        # Create sensitive files that must be excluded
        (src / ".env").write_text("SECRET_KEY=12345")
        (src / ".git").mkdir()
        (src / ".git" / "config").write_text("git config")
        (src / ".venv").mkdir()
        (src / ".venv" / "pyvenv.cfg").write_text("venv config")
        (src / "credentials.json").write_text('{"token": "xyz"}')
        (src / "id_rsa").write_text("PRIVATE KEY")
        (src / "__pycache__").mkdir()
        (src / "__pycache__" / "cached.pyc").write_text("cache")

        staged = executor.prepare_isolated_workspace(src)
        try:
            assert staged.exists()
            assert staged != src

            # Allowed files must exist
            assert (staged / "app" / "main.py").exists()
            assert (staged / "tests" / "test_main.py").exists()

            # Sensitive files must NOT exist
            assert not (staged / ".env").exists()
            assert not (staged / ".git").exists()
            assert not (staged / ".venv").exists()
            assert not (staged / "credentials.json").exists()
            assert not (staged / "id_rsa").exists()
            assert not (staged / "__pycache__").exists()

        finally:
            sandbox.cleanup_isolated_workspace(staged)


# ============================================================================
# 3. Path Validation & Security Tests
# ============================================================================

def test_validate_test_paths_success():
    """Verify valid relative test file paths are accepted and normalized."""
    sandbox = DockerSandbox(docker_client=MagicMock())
    executor = TestExecutor(sandbox=sandbox)

    with tempfile.TemporaryDirectory() as ws_dir:
        ws = Path(ws_dir)
        (ws / "tests").mkdir()
        test_file = ws / "tests" / "test_one.py"
        test_file.write_text("assert True")

        validated = executor.validate_test_paths(ws, ["tests/test_one.py"])
        assert validated == ["tests/test_one.py"]


def test_validate_test_paths_directory_traversal():
    """Verify path traversal outside workspace is blocked with DockerSecurityViolationError."""
    sandbox = DockerSandbox(docker_client=MagicMock())
    executor = TestExecutor(sandbox=sandbox)

    with tempfile.TemporaryDirectory() as ws_dir:
        with pytest.raises(DockerSecurityViolationError) as exc_info:
            executor.validate_test_paths(ws_dir, ["../../etc/passwd"])
        assert "escapes workspace boundary" in str(exc_info.value)


def test_validate_test_paths_injection_characters():
    """Verify test paths with injection characters are rejected."""
    sandbox = DockerSandbox(docker_client=MagicMock())
    executor = TestExecutor(sandbox=sandbox)

    with tempfile.TemporaryDirectory() as ws_dir:
        with pytest.raises(ValueError):
            executor.validate_test_paths(ws_dir, ["tests/test_api.py; rm -rf /"])

        with pytest.raises(ValueError):
            executor.validate_test_paths(ws_dir, [""])


def test_validate_test_paths_nonexistent_file():
    """Verify non-existent test file raises FileNotFoundError."""
    sandbox = DockerSandbox(docker_client=MagicMock())
    executor = TestExecutor(sandbox=sandbox)

    with tempfile.TemporaryDirectory() as ws_dir:
        with pytest.raises(FileNotFoundError):
            executor.validate_test_paths(ws_dir, ["tests/non_existent.py"])


# ============================================================================
# 4. Pytest Execution Lifecycle (Mocked Docker Sandbox)
# ============================================================================

def test_execute_success():
    """Verify end-to-end execution when all tests pass."""
    mock_sandbox = MagicMock(spec=DockerSandbox)
    mock_sandbox.is_daemon_available.return_value = True
    mock_sandbox.validate_mount_security.side_effect = lambda p: Path(p).resolve()

    # Create dummy staged workspace
    staged_dir = tempfile.mkdtemp(prefix="mock_pytest_ws_")
    mock_sandbox.create_isolated_workspace.return_value = Path(staged_dir)

    mock_sandbox.execute_command.return_value = ExecutionResult(
        exit_code=0,
        stdout="=== 5 passed in 0.10s ===",
        stderr="",
        duration_seconds=0.10,
        timed_out=False,
        command=["pytest", "-v", "--tb=short"],
        status=ContainerStatus(state=ContainerStateEnum.COMPLETED, image="python:3.11-slim", exit_code=0),
    )

    executor = TestExecutor(sandbox=mock_sandbox)

    with tempfile.TemporaryDirectory() as ws:
        (Path(ws) / "test_dummy.py").write_text("assert True")
        details = executor.execute(workspace_path=ws)

        assert details.passed is True
        assert details.total_tests == 5
        assert details.passed_tests == 5
        assert details.failed_tests == 0
        assert details.exit_code == 0

        # Verify command arguments
        mock_sandbox.execute_command.assert_called_once()
        _, kwargs = mock_sandbox.execute_command.call_args
        assert kwargs["command"] == ["pytest", "-v", "--tb=short"]

        # Verify staged workspace cleanup was invoked
        mock_sandbox.cleanup_isolated_workspace.assert_called_once()


def test_execute_failed_tests():
    """Verify end-to-end execution when tests fail."""
    mock_sandbox = MagicMock(spec=DockerSandbox)
    mock_sandbox.is_daemon_available.return_value = True
    mock_sandbox.validate_mount_security.side_effect = lambda p: Path(p).resolve()

    staged_dir = tempfile.mkdtemp(prefix="mock_pytest_ws_")
    mock_sandbox.create_isolated_workspace.return_value = Path(staged_dir)

    mock_sandbox.execute_command.return_value = ExecutionResult(
        exit_code=1,
        stdout="""
=========================== short test summary info ============================
FAILED test_mod.py::test_fail - AssertionError
========================= 1 failed, 2 passed in 0.05s ==========================
""",
        stderr="",
        duration_seconds=0.05,
        timed_out=False,
        command=["pytest", "-v", "--tb=short"],
        status=ContainerStatus(state=ContainerStateEnum.FAILED, image="python:3.11-slim", exit_code=1),
    )

    executor = TestExecutor(sandbox=mock_sandbox)

    with tempfile.TemporaryDirectory() as ws:
        (Path(ws) / "test_mod.py").write_text("assert True")
        details = executor.execute(workspace_path=ws)

        assert details.passed is False
        assert details.total_tests == 3
        assert details.passed_tests == 2
        assert details.failed_tests == 1
        assert "FAILED test_mod.py::test_fail" in details.error_summary
        mock_sandbox.cleanup_isolated_workspace.assert_called_once()


def test_execute_timeout():
    """Verify end-to-end execution when container times out."""
    mock_sandbox = MagicMock(spec=DockerSandbox)
    mock_sandbox.is_daemon_available.return_value = True
    mock_sandbox.validate_mount_security.side_effect = lambda p: Path(p).resolve()

    staged_dir = tempfile.mkdtemp(prefix="mock_pytest_ws_")
    mock_sandbox.create_isolated_workspace.return_value = Path(staged_dir)

    mock_sandbox.execute_command.return_value = ExecutionResult(
        exit_code=124,
        stdout="",
        stderr="[Execution Timed Out after 10 seconds]",
        duration_seconds=10.0,
        timed_out=True,
        command=["pytest", "-v", "--tb=short"],
        status=ContainerStatus(state=ContainerStateEnum.TIMED_OUT, image="python:3.11-slim", exit_code=124),
    )

    executor = TestExecutor(sandbox=mock_sandbox)

    with tempfile.TemporaryDirectory() as ws:
        (Path(ws) / "test_loop.py").write_text("while True: pass")
        details = executor.execute(workspace_path=ws, timeout=10)

        assert details.passed is False
        assert details.timed_out is True
        assert details.exit_code == 124
        assert "timed out" in details.error_summary.lower()
        mock_sandbox.cleanup_isolated_workspace.assert_called_once()


def test_execute_daemon_unavailable():
    """Verify graceful handling when Docker daemon is unreachable."""
    mock_sandbox = MagicMock(spec=DockerSandbox)
    mock_sandbox.is_daemon_available.return_value = False

    executor = TestExecutor(sandbox=mock_sandbox)

    with tempfile.TemporaryDirectory() as ws:
        details = executor.execute(workspace_path=ws)
        assert details.passed is False
        assert details.exit_code == -1
        assert "Docker daemon is not running" in details.error_summary
        # Verify no workspace creation was attempted
        mock_sandbox.create_isolated_workspace.assert_not_called()


def test_execute_cleans_temporary_workspace_on_exception():
    """Verify staged temporary workspace is cleaned up even if execution raises an exception."""
    mock_sandbox = MagicMock(spec=DockerSandbox)
    mock_sandbox.is_daemon_available.return_value = True
    mock_sandbox.validate_mount_security.side_effect = lambda p: Path(p).resolve()

    staged_dir = tempfile.mkdtemp(prefix="mock_pytest_ws_")
    mock_sandbox.create_isolated_workspace.return_value = Path(staged_dir)
    mock_sandbox.execute_command.side_effect = DockerSandboxError("Container crash")

    executor = TestExecutor(sandbox=mock_sandbox)

    with tempfile.TemporaryDirectory() as ws:
        (Path(ws) / "test_a.py").write_text("assert True")
        details = executor.execute(workspace_path=ws)
        assert details.passed is False
        assert "Sandbox execution error" in details.error_summary
        mock_sandbox.cleanup_isolated_workspace.assert_called_once()


# ============================================================================
# 5. Shared State & TestResult Compatibility Tests
# ============================================================================

def test_to_test_result_compatibility():
    """Verify PytestExecutionDetails converts cleanly to the shared system TestResult."""
    details = PytestExecutionDetails(
        passed=True,
        total_tests=10,
        passed_tests=10,
        failed_tests=0,
        skipped_tests=0,
        error_count=0,
        exit_code=0,
        raw_output="=== 10 passed in 0.15s ===",
        error_summary=None,
    )

    test_res = details.to_test_result()
    assert isinstance(test_res, TestResult)
    assert test_res.passed is True
    assert test_res.total_tests == 10
    assert test_res.passed_tests == 10
    assert test_res.failed_tests == 0
    assert test_res.raw_output == "=== 10 passed in 0.15s ==="
    assert test_res.error_summary is None


def test_run_tests_returns_shared_test_result():
    """Verify executor.run_tests returns shared TestResult instance."""
    mock_sandbox = MagicMock(spec=DockerSandbox)
    mock_sandbox.is_daemon_available.return_value = True
    mock_sandbox.validate_mount_security.side_effect = lambda p: Path(p).resolve()

    staged_dir = tempfile.mkdtemp(prefix="mock_pytest_ws_")
    mock_sandbox.create_isolated_workspace.return_value = Path(staged_dir)

    mock_sandbox.execute_command.return_value = ExecutionResult(
        exit_code=0,
        stdout="=== 2 passed in 0.02s ===",
        stderr="",
        duration_seconds=0.02,
        timed_out=False,
        command=["pytest"],
        status=ContainerStatus(state=ContainerStateEnum.COMPLETED, image="python:3.11-slim", exit_code=0),
    )

    executor = TestExecutor(sandbox=mock_sandbox)

    with tempfile.TemporaryDirectory() as ws:
        (Path(ws) / "test_main.py").write_text("assert True")
        res = executor.run_tests(workspace_path=ws)

        assert isinstance(res, TestResult)
        assert res.passed is True
        assert res.total_tests == 2
        assert res.passed_tests == 2
        assert res.failed_tests == 0


def test_update_agent_state_passed():
    """Verify update_agent_state updates state correctly when tests pass."""
    executor = TestExecutor(sandbox=MagicMock())

    initial_state: AgentState = {
        "project_id": "proj_001",
        "user_prompt": "Build REST API",
        "requirement_spec": None,
        "architecture_blueprint": None,
        "generated_files": [],
        "workspace_path": "/tmp/ws",
        "test_results": None,
        "retry_count": 0,
        "is_test_passed": False,
        "bug_diagnosis": None,
        "patched_files": [],
        "security_report": None,
        "current_agent": "Coding Agent",
        "logs": ["Initial log"],
        "status": "coding",
    }

    test_res = TestResult(
        passed=True,
        total_tests=5,
        passed_tests=5,
        failed_tests=0,
        raw_output="5 passed",
        error_summary=None,
    )

    updated_state = executor.update_agent_state(initial_state, test_res)

    assert updated_state["is_test_passed"] is True
    assert updated_state["status"] == "reviewing"
    assert updated_state["current_agent"] == "Testing Agent"
    assert updated_state["test_results"]["passed"] is True
    assert updated_state["test_results"]["total_tests"] == 5
    assert updated_state["retry_count"] == 0
    assert any("PASSED: 5/5" in log for log in updated_state["logs"])


def test_update_agent_state_failed():
    """Verify update_agent_state increments retry count and sets status to debugging on failure."""
    executor = TestExecutor(sandbox=MagicMock())

    initial_state: AgentState = {
        "project_id": "proj_002",
        "user_prompt": "Build REST API",
        "requirement_spec": None,
        "architecture_blueprint": None,
        "generated_files": [],
        "workspace_path": "/tmp/ws",
        "test_results": None,
        "retry_count": 1,
        "is_test_passed": False,
        "bug_diagnosis": None,
        "patched_files": [],
        "security_report": None,
        "current_agent": "Coding Agent",
        "logs": [],
        "status": "coding",
    }

    test_res = TestResult(
        passed=False,
        total_tests=5,
        passed_tests=3,
        failed_tests=2,
        raw_output="2 failed",
        error_summary="AssertionError: 5 != 6",
    )

    updated_state = executor.update_agent_state(initial_state, test_res)

    assert updated_state["is_test_passed"] is False
    assert updated_state["status"] == "debugging"
    assert updated_state["current_agent"] == "Testing Agent"
    assert updated_state["retry_count"] == 2
    assert updated_state["test_results"]["failed_tests"] == 2
    assert any("FAILED: 2/5" in log for log in updated_state["logs"])
    assert any("AssertionError" in log for log in updated_state["logs"])
