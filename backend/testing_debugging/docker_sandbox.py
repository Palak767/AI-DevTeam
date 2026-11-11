"""
Docker Sandbox Execution Engine for Member 3: Testing & Autonomous Debugging.

Provides an isolated, secure, and resource-bounded execution environment
for executing untrusted generated code and test suites.

SECURITY ARCHITECTURE & DEFENSE-IN-DEPTH:
Containerization provides strong process, filesystem, and network isolation, but
Docker alone does not guarantee absolute security against kernel-level vulnerabilities.
Therefore, defense-in-depth is strictly enforced via:
1. Complete network isolation (network_mode='none')
2. Non-root user execution (UID:GID 1000:1000)
3. Dropping all Linux kernel capabilities (cap_drop=['ALL'])
4. Enforcing hard memory bounds (default 512MB)
5. Enforcing CPU quota bounds (nano_cpus, default 1.0 CPU)
6. Enforcing process count limits to prevent fork bombs (pids_limit=100)
7. Enforcing execution timeouts to prevent denial of service (infinite loops)
8. Absolute prohibition against mounting host Docker sockets or privileged containers
9. Deterministic container removal and cleanup in all execution branches
10. Isolated ephemeral workspaces on the host with validation against sensitive paths
"""

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import docker
from docker.errors import APIError, DockerException
from docker.models.containers import Container
import requests

from backend.config import config as app_config
from backend.testing_debugging.schemas import (
    ContainerStateEnum,
    ContainerStatus,
    DockerSandboxConfig,
    ExecutionResult,
)


class DockerSandboxError(Exception):
    """Base exception for Docker Sandbox operations."""
    pass


class DockerDaemonUnavailableError(DockerSandboxError):
    """Raised when the Docker daemon is unreachable or not running."""
    pass


class DockerTimeoutError(DockerSandboxError):
    """Raised when a container command exceeds the allocated timeout."""
    pass


class DockerSecurityViolationError(DockerSandboxError):
    """Raised when an operation violates sandbox security policies."""
    pass


# Forbidden mount patterns to prevent Docker-in-Docker socket escapes and system corruption
FORBIDDEN_MOUNT_PATTERNS = [
    "docker.sock",
    "docker_engine",
    "dockerdesktoplinuxengine",
    "pipe",
    "/proc",
    "/sys",
    "/dev",
    "/etc",
    "/root",
    "/var/run",
    "windows",
    "program files",
]


class DockerSandbox:
    """
    Manages ephemeral, secure Docker containers for safe execution of code and tests.
    """

    def __init__(
        self,
        config: Optional[DockerSandboxConfig] = None,
        docker_client: Optional[docker.DockerClient] = None,
    ) -> None:
        """
        Initialize the DockerSandbox.

        Args:
            config: Sandbox configuration specifying resource limits and image.
                    Defaults to team standard configured in backend.config.
            docker_client: Injected DockerClient instance (used for testing and mocking).
        """
        if config is None:
            default_image = getattr(app_config, "DOCKER_IMAGE", "python:3.11-slim")
            self.config = DockerSandboxConfig(image=default_image)
        else:
            self.config = config

        self._injected_client = docker_client
        self._client: Optional[docker.DockerClient] = docker_client

    @property
    def client(self) -> docker.DockerClient:
        """
        Lazily initialize and return the Docker client.
        Raises DockerDaemonUnavailableError if client creation fails.
        """
        if self._client is not None:
            return self._client

        try:
            self._client = docker.from_env()
            return self._client
        except (DockerException, APIError, Exception) as exc:
            raise DockerDaemonUnavailableError(
                f"Failed to connect to Docker daemon: {str(exc)}. "
                "Ensure Docker Desktop is running on the host system."
            ) from exc

    def is_daemon_available(self) -> bool:
        """
        Check whether the Docker daemon is active, responsive, and ready.

        Returns:
            bool: True if daemon responds to ping, False otherwise.
        """
        try:
            return bool(self.client.ping())
        except (
            DockerException,
            APIError,
            requests.exceptions.RequestException,
            DockerSandboxError,
            Exception,
        ):
            return False

    def validate_mount_security(self, host_path: Union[str, Path]) -> Path:
        """
        Validates that a proposed host path mount does not violate security constraints.

        Args:
            host_path: Host filesystem path to be mounted into the container.

        Returns:
            Path: Resolved, validated Path object.

        Raises:
            DockerSecurityViolationError: If path contains sensitive system directories,
                                          sockets, or attempts to mount root drives.
            FileNotFoundError: If path does not exist.
            ValueError: If path is not a directory.
        """
        path_str = str(host_path).lower().replace("\\", "/")

        # Check for forbidden mount tokens (prevent Docker socket escape)
        for forbidden in FORBIDDEN_MOUNT_PATTERNS:
            if forbidden in path_str:
                raise DockerSecurityViolationError(
                    f"Security violation: Mounting '{host_path}' is strictly prohibited "
                    f"because it matches forbidden pattern '{forbidden}'."
                )

        resolved_path = Path(host_path).resolve()

        # Prevent mounting the host root filesystem (e.g. "/" or "C:\")
        if (
            resolved_path == resolved_path.parent
            or str(resolved_path) == resolved_path.anchor
            or str(resolved_path).rstrip("/\\") == resolved_path.anchor.rstrip("/\\")
        ):
            raise DockerSecurityViolationError(
                f"Security violation: Mounting root filesystem '{host_path}' is strictly prohibited."
            )

        if not resolved_path.exists():
            raise FileNotFoundError(f"Host workspace path does not exist: {host_path}")
        if not resolved_path.is_dir():
            raise ValueError(f"Host workspace path must be a directory: {host_path}")

        return resolved_path

    def create_isolated_workspace(self, prefix: str = "autodev_sandbox_") -> Path:
        """
        Creates a dedicated temporary isolated directory on the host for safe workspace execution.

        Args:
            prefix: Directory prefix for identifying ephemeral sandbox workspaces.

        Returns:
            Path: Resolved path to the created temporary workspace.
        """
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        return Path(temp_dir).resolve()

    def cleanup_isolated_workspace(self, workspace_path: Union[str, Path]) -> bool:
        """
        Safely removes a temporary isolated workspace directory from the host.

        Args:
            workspace_path: Path of the directory to clean up.

        Returns:
            bool: True if cleanup succeeded or directory did not exist, False otherwise.
        """
        path = Path(workspace_path).resolve()
        if not path.exists():
            return True
        try:
            shutil.rmtree(path, ignore_errors=True)
            return True
        except Exception:
            return False

    def create_container(
        self,
        command: Union[str, List[str]],
        host_workspace_path: Optional[Union[str, Path]] = None,
        environment: Optional[Dict[str, str]] = None,
    ) -> Container:
        """
        Creates an ephemeral, isolated container enforcing all defense-in-depth policies.

        Args:
            command: Command or argument list to execute.
            host_workspace_path: Optional host path mounted into the container.
            environment: Optional additional environment variables.

        Returns:
            Container: Created Docker container instance (detached, not yet started).

        Raises:
            DockerDaemonUnavailableError: If Docker daemon is unavailable.
            DockerSecurityViolationError: If mount or environment violates security policies.
            DockerSandboxError: If container creation fails.
        """
        if not self.is_daemon_available():
            raise DockerDaemonUnavailableError(
                "Docker daemon is not running or unreachable. Cannot create sandbox container."
            )

        volumes = {}
        if host_workspace_path is not None:
            safe_host_path = self.validate_mount_security(host_workspace_path)
            volumes[str(safe_host_path)] = {
                "bind": self.config.working_dir,
                "mode": "rw",
            }

        # Merge environment safely
        env = dict(self.config.environment)
        if environment:
            # Validate combined environment
            merged = {**env, **environment}
            DockerSandboxConfig.validate_environment(merged)
            env = merged

        try:
            container = self.client.containers.create(
                image=self.config.image,
                command=command,
                working_dir=self.config.working_dir,
                network_mode="none",  # Strictly enforce complete network isolation
                mem_limit=self.config.memory_limit,
                nano_cpus=self.config.nano_cpus,
                pids_limit=self.config.pids_limit,
                privileged=False,     # Strictly non-privileged
                cap_drop=self.config.cap_drop,
                user=self.config.user,
                read_only=self.config.read_only_rootfs,
                environment=env,
                volumes=volumes,
                detach=True,
            )
            return container
        except DockerSecurityViolationError:
            raise
        except (DockerException, APIError) as docker_err:
            raise DockerSandboxError(
                f"Docker runtime error during container creation: {str(docker_err)}"
            ) from docker_err
        except Exception as exc:
            raise DockerSandboxError(
                f"Unexpected sandbox error during container creation: {str(exc)}"
            ) from exc

    def cleanup_container(self, container: Union[Container, str, None]) -> bool:
        """
        Forcefully removes a container to prevent zombie processes or resource leaks.

        Args:
            container: Container instance or container ID string.

        Returns:
            bool: True if cleanup succeeded or container was None, False if an error occurred.
        """
        if container is None:
            return True
        try:
            if isinstance(container, str):
                target = self.client.containers.get(container)
                target.remove(force=True)
            else:
                container.remove(force=True)
            return True
        except Exception:
            return False

    def _decode_stream(self, stream_data: Union[bytes, str, None]) -> str:
        """Safely decodes container output stream into UTF-8 string."""
        if stream_data is None:
            return ""
        if isinstance(stream_data, bytes):
            return stream_data.decode("utf-8", errors="replace")
        return str(stream_data)

    def execute_command(
        self,
        command: Union[str, List[str]],
        host_workspace_path: Optional[Union[str, Path]] = None,
        timeout: Optional[int] = None,
        raise_on_timeout: bool = False,
    ) -> ExecutionResult:
        """
        Executes an approved command inside an ephemeral, resource-bounded container.

        Args:
            command: Command string or list of argument tokens to execute.
            host_workspace_path: Optional host path mounted to config.working_dir.
            timeout: Optional override for execution timeout in seconds.
            raise_on_timeout: If True, raises DockerTimeoutError on timeout instead of
                              returning an ExecutionResult with timed_out=True.

        Returns:
            ExecutionResult: Captured output, exit code, duration, and status.

        Raises:
            DockerDaemonUnavailableError: If Docker daemon is unreachable.
            DockerSecurityViolationError: If workspace mounts violate security rules.
            DockerTimeoutError: If execution times out and raise_on_timeout is True.
            DockerSandboxError: If unexpected container execution errors occur.
        """
        effective_timeout = timeout if timeout is not None else self.config.timeout_seconds

        container: Optional[Container] = None
        start_time = time.perf_counter()
        timed_out = False
        exit_code = -1
        stdout_text = ""
        stderr_text = ""
        container_id: Optional[str] = None
        status_state = ContainerStateEnum.PENDING

        try:
            # Create container using verified isolation controls
            container = self.create_container(
                command=command,
                host_workspace_path=host_workspace_path,
            )

            container_id = container.id if hasattr(container, "id") else None
            status_state = ContainerStateEnum.RUNNING

            # Start container execution
            container.start()

            # Wait for execution with timeout enforcement
            try:
                wait_result = container.wait(timeout=effective_timeout)
                if isinstance(wait_result, dict):
                    exit_code = int(wait_result.get("StatusCode", -1))
                elif isinstance(wait_result, int):
                    exit_code = wait_result
                else:
                    exit_code = int(getattr(wait_result, "status_code", -1))

                status_state = (
                    ContainerStateEnum.COMPLETED if exit_code == 0
                    else ContainerStateEnum.FAILED
                )

            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                TimeoutError,
            ):
                timed_out = True
                status_state = ContainerStateEnum.TIMED_OUT
                exit_code = 124  # POSIX standard timeout exit code
                try:
                    container.kill()
                except Exception:
                    pass

            # Capture logs safely
            try:
                raw_stdout = container.logs(stdout=True, stderr=False)
                stdout_text = self._decode_stream(raw_stdout)
            except Exception:
                stdout_text = ""

            try:
                raw_stderr = container.logs(stdout=False, stderr=True)
                stderr_text = self._decode_stream(raw_stderr)
            except Exception:
                stderr_text = ""

            if timed_out:
                stderr_text += f"\n[Execution Timed Out after {effective_timeout} seconds]"

        except (DockerDaemonUnavailableError, DockerSecurityViolationError):
            raise
        except (DockerException, APIError) as docker_err:
            status_state = ContainerStateEnum.ERROR
            raise DockerSandboxError(
                f"Docker runtime error during execution: {str(docker_err)}"
            ) from docker_err
        except Exception as exc:
            status_state = ContainerStateEnum.ERROR
            raise DockerSandboxError(
                f"Unexpected sandbox error during execution: {str(exc)}"
            ) from exc

        finally:
            # Deterministic cleanup: ensure container is removed in all execution paths
            if container is not None:
                self.cleanup_container(container)

        duration = max(0.0, time.perf_counter() - start_time)

        if timed_out and raise_on_timeout:
            raise DockerTimeoutError(
                f"Command execution timed out after {effective_timeout} seconds."
            )

        container_status = ContainerStatus(
            container_id=container_id,
            state=status_state,
            image=self.config.image,
            exit_code=exit_code,
            error_message=stderr_text if exit_code != 0 and not timed_out else None,
        )

        return ExecutionResult(
            exit_code=exit_code,
            stdout=stdout_text,
            stderr=stderr_text,
            duration_seconds=duration,
            timed_out=timed_out,
            container_id=container_id,
            command=command,
            status=container_status,
        )

    def close(self) -> None:
        """Close client connections if managed internally."""
        if self._client is not None and self._client != self._injected_client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def __enter__(self) -> "DockerSandbox":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
