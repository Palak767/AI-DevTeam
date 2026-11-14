"""
Pytest Execution Engine for Member 3: Testing & Debugging Subsystem.

Responsible for safely executing Pytest test suites within the Docker Sandbox,
isolating workspaces, preventing sensitive file leakage, parsing execution results,
and mapping outcomes to the shared system state contracts.
"""

import fnmatch
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from backend.schemas.state import AgentState, TestResult
from backend.testing_debugging.docker_sandbox import (
    DockerDaemonUnavailableError,
    DockerSandbox,
    DockerSandboxError,
    DockerSecurityViolationError,
)
from backend.testing_debugging.schemas import (
    DockerSandboxConfig,
    ExecutionResult,
    PytestExecutionDetails,
)


# Sensitive patterns that MUST NEVER be copied into the execution container workspace
SENSITIVE_FILE_PATTERNS = [
    # Environment variables & secrets
    ".env*",
    "*.env",
    "*.secret",
    "*.secrets",
    "secrets.*",
    # Cryptographic keys and certificates
    "*.pem",
    "*.key",
    "*.pkcs12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    "id_dsa*",
    # Credentials and tokens
    "*credential*",
    "*secret*",
    "*token*",
    "*auth*.json",
    # Version control metadata
    ".git",
    ".git/**",
    ".gitignore",
    ".gitattributes",
    ".gitmodules",
    # Virtual environments
    ".venv",
    ".venv/**",
    "venv",
    "venv/**",
    "env",
    "env/**",
    "virtualenv",
    "virtualenv/**",
    # Caches and bytecode
    "__pycache__",
    "__pycache__/**",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".pytest_cache",
    ".pytest_cache/**",
    ".mypy_cache",
    ".mypy_cache/**",
    ".ruff_cache",
    ".ruff_cache/**",
    # IDE configurations
    ".idea",
    ".idea/**",
    ".vscode",
    ".vscode/**",
    # Dependency packages
    "node_modules",
    "node_modules/**",
]


class PytestOutputParser:
    """
    Robust parser for pytest console output streams.
    Extracts test counts, exit codes, and diagnostic summaries across diverse failure modes.
    """

    ANSI_REGEX = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    @classmethod
    def strip_ansi(cls, text: str) -> str:
        """Strip ANSI terminal escape sequences from output."""
        if not text:
            return ""
        return cls.ANSI_REGEX.sub("", text)

    @classmethod
    def parse(
        cls,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration_seconds: float = 0.0,
        timed_out: bool = False,
    ) -> PytestExecutionDetails:
        """
        Parses combined output from Pytest execution into structured details.

        Args:
            stdout: Standard output string from the container.
            stderr: Standard error string from the container.
            exit_code: Process termination code.
            duration_seconds: Wall-clock execution time.
            timed_out: Whether execution timed out.

        Returns:
            PytestExecutionDetails: Structured test outcomes and summary.
        """
        clean_out = cls.strip_ansi(stdout or "")
        clean_err = cls.strip_ansi(stderr or "")
        raw_output = f"{clean_out}\n{clean_err}".strip() if clean_err else clean_out.strip()

        # 1. Handle Execution Timeout
        if timed_out or exit_code == 124:
            return PytestExecutionDetails(
                passed=False,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                error_count=0,
                exit_code=124,
                raw_output=raw_output,
                error_summary=f"Pytest execution timed out after {duration_seconds:.1f} seconds.",
                duration_seconds=duration_seconds,
                timed_out=True,
            )

        # 2. Handle Missing Pytest or Binary Not Found
        missing_patterns = [
            "pytest: command not found",
            "pytest: not found",
            "No module named pytest",
            "executable file not found",
        ]
        if exit_code == 127 or any(p in raw_output for p in missing_patterns):
            return PytestExecutionDetails(
                passed=False,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                error_count=1,
                exit_code=exit_code if exit_code != 0 else 127,
                raw_output=raw_output,
                error_summary="Pytest is not installed or not available in the container environment.",
                duration_seconds=duration_seconds,
                timed_out=False,
            )

        # 3. Handle No Tests Collected (pytest exit code 5)
        if exit_code == 5 or "collected 0 items" in clean_out or "no tests ran" in clean_out:
            return PytestExecutionDetails(
                passed=False,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                error_count=0,
                exit_code=5,
                raw_output=raw_output,
                error_summary="No tests were collected. Ensure test files follow standard naming conventions (e.g. test_*.py).",
                duration_seconds=duration_seconds,
                timed_out=False,
            )

        # 4. Extract Counts from Output
        passed_count = 0
        failed_count = 0
        skipped_count = 0
        error_count = 0
        collected_count = 0

        # Scan for 'collected X items'
        collected_match = re.search(r"collected\s+(\d+)\s+item", clean_out)
        if collected_match:
            collected_count = int(collected_match.group(1))

        # Scan for summary line tokens: e.g. '1 failed, 2 passed, 3 skipped, 1 error'
        # Check from bottom lines first for final summary
        lines = [line.strip() for line in clean_out.splitlines() if line.strip()]
        summary_found = False

        for line in reversed(lines[-15:]):
            if ("passed" in line or "failed" in line or "error" in line or "skipped" in line) and (
                line.startswith("=") or line.endswith("=") or "in " in line
            ):
                p_match = re.search(r"(\d+)\s+passed", line)
                f_match = re.search(r"(\d+)\s+failed", line)
                s_match = re.search(r"(\d+)\s+skipped", line)
                e_match = re.search(r"(\d+)\s+error", line)

                if p_match:
                    passed_count = int(p_match.group(1))
                if f_match:
                    failed_count = int(f_match.group(1))
                if s_match:
                    skipped_count = int(s_match.group(1))
                if e_match:
                    error_count = int(e_match.group(1))

                summary_found = True
                break

        # Fallback if summary line was truncated or missing
        if not summary_found:
            passed_matches = re.findall(r"::\w+\s+PASSED", clean_out)
            failed_matches = re.findall(r"::\w+\s+FAILED", clean_out)
            skipped_matches = re.findall(r"::\w+\s+SKIPPED", clean_out)
            error_matches = re.findall(r"::\w+\s+ERROR", clean_out)

            if passed_matches or failed_matches or skipped_matches or error_matches:
                passed_count = len(passed_matches)
                failed_count = len(failed_matches)
                skipped_count = len(skipped_matches)
                error_count = len(error_matches)
            else:
                p_all = re.findall(r"(\d+)\s+passed", clean_out)
                f_all = re.findall(r"(\d+)\s+failed", clean_out)
                s_all = re.findall(r"(\d+)\s+skipped", clean_out)
                e_all = re.findall(r"(\d+)\s+error", clean_out)
                if p_all:
                    passed_count = int(p_all[-1])
                if f_all:
                    failed_count = int(f_all[-1])
                if s_all:
                    skipped_count = int(s_all[-1])
                if e_all:
                    error_count = int(e_all[-1])

        total_tests = passed_count + failed_count + skipped_count + error_count
        if total_tests == 0 and collected_count > 0:
            total_tests = collected_count

        # Check for Collection Errors (pytest exit code 2 or ERRORS block)
        is_collection_error = (
            exit_code == 2
            or "=== ERRORS ===" in clean_out
            or "ERROR collecting" in clean_out
        )
        if is_collection_error and error_count == 0:
            error_count = 1
            if total_tests == 0:
                total_tests = 1

        # 5. Determine Overall Success
        passed = (
            exit_code == 0
            and failed_count == 0
            and error_count == 0
            and total_tests > 0
        )

        # 6. Extract Failure / Error Summary
        error_summary: Optional[str] = None
        if not passed:
            error_summary = cls._extract_failure_summary(clean_out, clean_err, is_collection_error)

        return PytestExecutionDetails(
            passed=passed,
            total_tests=total_tests,
            passed_tests=passed_count,
            failed_tests=failed_count,
            skipped_tests=skipped_count,
            error_count=error_count,
            exit_code=exit_code,
            raw_output=raw_output,
            error_summary=error_summary,
            duration_seconds=duration_seconds,
            timed_out=False,
        )

    @classmethod
    def _extract_failure_summary(
        cls,
        stdout: str,
        stderr: str,
        is_collection_error: bool,
    ) -> str:
        """Extracts targeted diagnostic summary from stdout and stderr."""
        summary_lines: List[str] = []

        # Strategy A: Check 'short test summary info' (most concise)
        if "short test summary info" in stdout:
            parts = stdout.split("short test summary info")
            if len(parts) > 1:
                after_header = parts[1]
                lines_after = after_header.splitlines()
                # Skip remainder of the header line (e.g. '=====\n')
                content_lines = lines_after[1:] if len(lines_after) > 1 else lines_after
                for l in content_lines:
                    sline = l.strip()
                    if sline.startswith("=") and ("passed" in sline or "failed" in sline or "in " in sline):
                        break
                    if sline.startswith("FAILED") or sline.startswith("ERROR"):
                        summary_lines.append(sline)

        # Strategy B: Check '=== FAILURES ===' or '=== ERRORS ==='
        if not summary_lines:
            target_markers = ["=== FAILURES ===", "=== ERRORS ==="]
            for marker in target_markers:
                if marker in stdout:
                    section = stdout.split(marker)[1]
                    section_lines = section.splitlines()
                    for line in section_lines[:20]:
                        sline = line.strip()
                        if sline and not sline.startswith("="):
                            summary_lines.append(sline)
                        if len(summary_lines) >= 8:
                            break
                    break

        # Strategy C: Check for SyntaxError or ImportError
        if not summary_lines:
            combined = stdout + "\n" + stderr
            for line in combined.splitlines():
                if any(err_token in line for err_token in ["SyntaxError:", "ImportError:", "ModuleNotFoundError:", "AssertionError:"]):
                    summary_lines.append(line.strip())

        # Strategy D: Check stderr
        if not summary_lines and stderr.strip():
            summary_lines = [l.strip() for l in stderr.splitlines() if l.strip()][:5]

        # Final Fallback
        if not summary_lines:
            if is_collection_error:
                return "Pytest collection error occurred. Test suite could not be loaded."
            return "Pytest execution failed with non-zero exit code."

        result_summary = "\n".join(summary_lines[:8])
        return result_summary[:2000]


class TestExecutor:
    """
    Executes Python test suites inside an isolated Docker Sandbox.
    Enforces security isolation, temporary workspace staging, sensitive file exclusion,
    and automatic mapping to shared system state models.
    """

    __test__ = False  # Prevent pytest from attempting to collect this class as a test suite

    def __init__(
        self,
        sandbox: Optional[DockerSandbox] = None,
        sandbox_config: Optional[DockerSandboxConfig] = None,
    ) -> None:
        """
        Initialize the TestExecutor.

        Args:
            sandbox: Optional injected DockerSandbox instance.
            sandbox_config: Optional configuration for creating an internal DockerSandbox.
        """
        if sandbox is not None:
            self.sandbox = sandbox
        else:
            self.sandbox = DockerSandbox(config=sandbox_config)

    @classmethod
    def is_sensitive(cls, rel_path: Union[str, Path], custom_patterns: Optional[List[str]] = None) -> bool:
        """
        Determines whether a file or directory path is sensitive and must be excluded.

        Args:
            rel_path: Path relative to the workspace root.
            custom_patterns: Optional custom exclusion patterns.

        Returns:
            bool: True if path should be excluded, False otherwise.
        """
        path_str = str(rel_path).replace("\\", "/")
        path_obj = Path(path_str)
        name = path_obj.name.lower()
        patterns = custom_patterns or SENSITIVE_FILE_PATTERNS

        # Check path components (directories)
        for part in path_obj.parts:
            part_lower = part.lower()
            if part_lower in [
                ".git",
                ".venv",
                "venv",
                "env",
                "virtualenv",
                "__pycache__",
                ".pytest_cache",
                ".mypy_cache",
                ".ruff_cache",
                ".idea",
                ".vscode",
                "node_modules",
            ]:
                return True

        # Check file name and path patterns
        for pattern in patterns:
            pat_lower = pattern.lower()
            if fnmatch.fnmatch(name, pat_lower) or fnmatch.fnmatch(path_str.lower(), pat_lower):
                return True

        # Check sensitive keywords
        sensitive_keywords = ["api_key", "secret", "password", "credential", "auth_token"]
        if any(kw in name for kw in sensitive_keywords):
            return True

        return False

    def validate_test_paths(
        self,
        workspace_path: Union[str, Path],
        test_file_paths: List[str],
    ) -> List[str]:
        """
        Validates that requested test paths remain strictly within the project workspace boundary.

        Args:
            workspace_path: Root project workspace directory.
            test_file_paths: List of relative test file paths to execute.

        Returns:
            List[str]: Validated relative POSIX file paths.

        Raises:
            DockerSecurityViolationError: If a test path escapes the workspace root.
            FileNotFoundError: If a requested test path does not exist.
            ValueError: If a test path is empty or contains injection characters.
        """
        resolved_workspace = Path(workspace_path).resolve()
        validated: List[str] = []

        for path_str in test_file_paths:
            if not path_str or not path_str.strip():
                raise ValueError("Test file path cannot be empty.")

            clean_path = path_str.strip()
            if any(char in clean_path for char in [";", "&", "|", "`", "$", "\n", "\r"]):
                raise ValueError(f"Invalid characters in test path: '{clean_path}'")

            target_path = (resolved_workspace / clean_path).resolve()

            # Ensure path is strictly inside workspace (prevent path traversal / directory escaping)
            try:
                rel_target = target_path.relative_to(resolved_workspace)
            except ValueError as exc:
                raise DockerSecurityViolationError(
                    f"Security violation: Requested test path '{path_str}' escapes workspace boundary."
                ) from exc

            if target_path == resolved_workspace:
                raise DockerSecurityViolationError("Security violation: Test path cannot be the workspace root.")

            if not target_path.exists():
                raise FileNotFoundError(f"Requested test file does not exist: '{clean_path}'")

            validated.append(rel_target.as_posix())

        return validated

    def prepare_isolated_workspace(
        self,
        source_workspace: Union[str, Path],
        custom_excludes: Optional[List[str]] = None,
    ) -> Path:
        """
        Creates a dedicated temporary isolated directory containing only allowed project files.
        Never mounts the original project directory directly.

        Args:
            source_workspace: Original host project directory.
            custom_excludes: Additional exclusion patterns.

        Returns:
            Path: Resolved path to the temporary staged workspace.

        Raises:
            FileNotFoundError: If source workspace does not exist.
            ValueError: If source workspace is not a directory.
            DockerSecurityViolationError: If source workspace violates mount security.
        """
        resolved_source = self.sandbox.validate_mount_security(source_workspace)
        temp_workspace = self.sandbox.create_isolated_workspace(prefix="pytest_ws_")

        try:
            for root, dirs, files in os.walk(resolved_source):
                rel_dir = Path(root).relative_to(resolved_source)

                # Prune sensitive directories in-place to avoid unnecessary traversal
                dirs[:] = [
                    d for d in dirs
                    if not self.is_sensitive(rel_dir / d, custom_excludes)
                ]

                dest_dir = temp_workspace / rel_dir
                dest_dir.mkdir(parents=True, exist_ok=True)

                for file_name in files:
                    rel_file = rel_dir / file_name
                    if not self.is_sensitive(rel_file, custom_excludes):
                        src_file = Path(root) / file_name
                        dst_file = dest_dir / file_name
                        try:
                            shutil.copy2(src_file, dst_file)
                        except Exception:
                            pass

            return temp_workspace

        except Exception as exc:
            # Clean up on failure during staging
            self.sandbox.cleanup_isolated_workspace(temp_workspace)
            raise DockerSandboxError(
                f"Failed to prepare isolated test workspace: {str(exc)}"
            ) from exc

    def execute(
        self,
        workspace_path: Union[str, Path],
        test_file_paths: Optional[List[str]] = None,
        pytest_args: Optional[List[str]] = None,
        timeout: Optional[int] = None,
    ) -> PytestExecutionDetails:
        """
        Executes Pytest inside the Docker Sandbox using an ephemeral staged workspace.

        Args:
            workspace_path: Original project directory path.
            test_file_paths: Optional specific test files to run.
            pytest_args: Optional additional pytest arguments (e.g. ['-k', 'unit']).
            timeout: Optional override for execution timeout in seconds.

        Returns:
            PytestExecutionDetails: Comprehensive parsed results and telemetry.
        """
        # Check daemon availability before preparing workspaces
        if not self.sandbox.is_daemon_available():
            return PytestExecutionDetails(
                passed=False,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                error_count=1,
                exit_code=-1,
                raw_output="Docker daemon is not running or unreachable.",
                error_summary="Docker daemon is not running or unreachable. Cannot execute tests.",
                duration_seconds=0.0,
                timed_out=False,
            )

        # Validate source workspace
        resolved_workspace = self.sandbox.validate_mount_security(workspace_path)

        # Validate specific test paths if provided
        validated_paths: List[str] = []
        if test_file_paths:
            validated_paths = self.validate_test_paths(resolved_workspace, test_file_paths)

        # Prepare isolated staged workspace
        temp_workspace = self.prepare_isolated_workspace(resolved_workspace)

        # Construct Pytest command safely
        cmd: List[str] = ["pytest"]
        if pytest_args:
            cmd.extend(pytest_args)
        else:
            cmd.extend(["-v", "--tb=short"])

        if validated_paths:
            cmd.extend(validated_paths)

        try:
            exec_result: ExecutionResult = self.sandbox.execute_command(
                command=cmd,
                host_workspace_path=temp_workspace,
                timeout=timeout,
                raise_on_timeout=False,
            )

            # Parse container output into structured telemetry
            parsed_details = PytestOutputParser.parse(
                stdout=exec_result.stdout,
                stderr=exec_result.stderr,
                exit_code=exec_result.exit_code,
                duration_seconds=exec_result.duration_seconds,
                timed_out=exec_result.timed_out,
            )
            return parsed_details

        except DockerDaemonUnavailableError:
            return PytestExecutionDetails(
                passed=False,
                exit_code=-1,
                raw_output="Docker daemon is unreachable.",
                error_summary="Docker daemon is not running or unreachable.",
                duration_seconds=0.0,
                timed_out=False,
            )
        except DockerSandboxError as err:
            return PytestExecutionDetails(
                passed=False,
                exit_code=-1,
                raw_output=str(err),
                error_summary=f"Sandbox execution error: {str(err)}",
                duration_seconds=0.0,
                timed_out=False,
            )
        finally:
            # Deterministic cleanup: always remove staged temporary workspace from host
            self.sandbox.cleanup_isolated_workspace(temp_workspace)

    def run_tests(
        self,
        workspace_path: Union[str, Path],
        test_file_paths: Optional[List[str]] = None,
        pytest_args: Optional[List[str]] = None,
        timeout: Optional[int] = None,
    ) -> TestResult:
        """
        Executes tests and maps outcome directly into the shared system TestResult model.

        Args:
            workspace_path: Project directory path.
            test_file_paths: Optional specific test files.
            pytest_args: Optional additional arguments for pytest.
            timeout: Optional execution timeout in seconds.

        Returns:
            TestResult: Validated shared system contract instance.
        """
        details = self.execute(
            workspace_path=workspace_path,
            test_file_paths=test_file_paths,
            pytest_args=pytest_args,
            timeout=timeout,
        )
        return details.to_test_result()

    def update_agent_state(
        self,
        state: AgentState,
        test_result: TestResult,
    ) -> AgentState:
        """
        Applies test execution outcomes to the central LangGraph AgentState.

        Args:
            state: Current pipeline AgentState dictionary.
            test_result: TestResult instance generated by TestExecutor.

        Returns:
            AgentState: Updated AgentState ready for transition to Bug Agent or Reviewer.
        """
        updated_state = dict(state)
        updated_state["test_results"] = test_result.model_dump()
        updated_state["is_test_passed"] = test_result.passed

        if "logs" not in updated_state or updated_state["logs"] is None:
            updated_state["logs"] = []

        updated_state["current_agent"] = "Testing Agent"

        if test_result.passed:
            updated_state["status"] = "reviewing"
            updated_state["logs"].append(
                f"[Testing Agent] Pytest execution PASSED: {test_result.passed_tests}/{test_result.total_tests} tests succeeded."
            )
        else:
            current_retries = updated_state.get("retry_count", 0)
            updated_state["retry_count"] = current_retries + 1
            updated_state["status"] = "debugging"
            updated_state["logs"].append(
                f"[Testing Agent] Pytest execution FAILED: {test_result.failed_tests}/{test_result.total_tests} failed "
                f"(Retry attempt {updated_state['retry_count']})."
            )
            if test_result.error_summary:
                updated_state["logs"].append(
                    f"[Testing Agent Diagnostic] {test_result.error_summary[:300]}"
                )

        return updated_state
