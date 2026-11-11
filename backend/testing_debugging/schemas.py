"""
Pydantic Schemas for Member 3: Docker Sandbox & Execution Subsystem.

Defines configuration models, execution results, container status models,
structured Pytest telemetry, AI Testing Agent specifications,
Bug Detection & Self-Correction contracts, and Test Report Generator data models.
"""

import re
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator

from backend.schemas.state import TestResult


class ContainerStateEnum(str, Enum):
    """Enumeration of container execution lifecycle states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    ERROR = "error"


class ContainerStatus(BaseModel):
    """Detailed status representation of a sandbox execution container."""
    container_id: Optional[str] = Field(
        default=None,
        description="Unique container hash or identifier"
    )
    state: ContainerStateEnum = Field(
        default=ContainerStateEnum.PENDING,
        description="Current execution lifecycle state"
    )
    image: str = Field(
        description="Docker image specified for execution"
    )
    exit_code: Optional[int] = Field(
        default=None,
        description="Exit status code if container terminated"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error details if container execution failed"
    )

    @field_validator("container_id")
    @classmethod
    def validate_container_id(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            if not v:
                return None
            if not re.match(r"^[a-zA-Z0-9_\-\.:]+$", v):
                raise ValueError(f"Invalid characters in container identifier: '{v}'")
        return v

    @field_validator("exit_code")
    @classmethod
    def validate_exit_code(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (-1 <= v <= 255):
            raise ValueError(f"Exit code must be between -1 and 255, got {v}")
        return v


class DockerSandboxConfig(BaseModel):
    """
    Configuration specification for the Docker-based Execution Sandbox.
    Enforces strict security guardrails, resource quotas, and isolation bounds.
    """
    image: str = Field(
        default="python:3.11-slim",
        description="Base Docker image for execution"
    )
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Maximum execution duration in seconds before termination"
    )
    memory_limit: str = Field(
        default="512m",
        description="Hard container memory limit (e.g., '256m', '512m', '1g')"
    )
    nano_cpus: int = Field(
        default=1_000_000_000,
        ge=100_000_000,
        le=4_000_000_000,
        description="CPU quota represented in fractions of a CPU (1e9 = 1.0 CPU)"
    )
    pids_limit: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum number of simultaneous processes to prevent fork bombs"
    )
    working_dir: str = Field(
        default="/workspace",
        description="Working directory inside the sandbox container"
    )
    network_mode: str = Field(
        default="none",
        description="Network isolation mode. Strictly enforced to 'none'"
    )
    user: Optional[str] = Field(
        default="1000:1000",
        description="Non-root user UID:GID for container execution"
    )
    read_only_rootfs: bool = Field(
        default=False,
        description="Mount container root filesystem as read-only"
    )
    cap_drop: List[str] = Field(
        default_factory=lambda: ["ALL"],
        description="Linux capabilities to drop from container"
    )
    environment: Dict[str, str] = Field(
        default_factory=dict,
        description="Safe environment variables passed to the container"
    )

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Docker image cannot be empty.")
        v = v.strip()
        if any(char in v for char in [";", "&", "|", "`", "$", " ", "\n", "\r"]):
            raise ValueError(f"Invalid characters in Docker image name: '{v}'")
        return v

    @field_validator("memory_limit")
    @classmethod
    def validate_memory_limit(cls, v: str) -> str:
        v = v.strip().lower()
        match = re.match(r"^(\d+)([kmg])$", v)
        if not match:
            raise ValueError(
                f"Invalid memory limit format: '{v}'. Must be e.g. '256m', '512m', '1g'."
            )
        val, unit = int(match.group(1)), match.group(2)
        multipliers = {"k": 1024, "m": 1024 * 1024, "g": 1024 * 1024 * 1024}
        total_bytes = val * multipliers[unit]
        if total_bytes < 32 * 1024 * 1024:
            raise ValueError("Memory limit must be at least 32MB.")
        if total_bytes > 4 * 1024 * 1024 * 1024:
            raise ValueError("Memory limit cannot exceed 4GB.")
        return v

    @field_validator("working_dir")
    @classmethod
    def validate_working_dir(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Working directory cannot be empty.")
        v = v.strip()
        if not v.startswith("/"):
            raise ValueError(f"Working directory must be an absolute Unix path: '{v}'")
        if any(char in v for char in [";", "&", "|", "`", "$", "\n", "\r"]):
            raise ValueError(f"Invalid characters in working directory: '{v}'")
        return v

    @field_validator("network_mode")
    @classmethod
    def validate_network_mode(cls, v: str) -> str:
        if v.strip().lower() != "none":
            raise ValueError(
                f"Security violation: Sandbox network_mode must be 'none', got '{v}'."
            )
        return "none"

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: Dict[str, str]) -> Dict[str, str]:
        disallowed_keywords = ["api_key", "secret", "password", "token", "auth", "cred"]
        for key in v:
            key_lower = key.lower()
            if any(forbidden in key_lower for forbidden in disallowed_keywords):
                raise ValueError(
                    f"Security violation: Environment variable '{key}' appears to contain sensitive secrets."
                )
        return v


class ExecutionResult(BaseModel):
    """
    Standard output and telemetry model for containerized command execution.
    """
    exit_code: int = Field(
        description="Command termination code. 0 indicates success, non-zero indicates failure"
    )
    stdout: str = Field(
        default="",
        description="Standard output stream captured from container"
    )
    stderr: str = Field(
        default="",
        description="Standard error stream captured from container"
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock duration of the execution in seconds"
    )
    timed_out: bool = Field(
        default=False,
        description="True if container execution exceeded timeout and was terminated"
    )
    container_id: Optional[str] = Field(
        default=None,
        description="Identifier of the ephemeral container used"
    )
    command: Union[str, List[str]] = Field(
        description="Command or argument list executed inside the sandbox"
    )
    status: ContainerStatus = Field(
        description="Lifecycle status summary of the container"
    )

    @field_validator("exit_code")
    @classmethod
    def validate_exit_code(cls, v: int) -> int:
        if not (-1 <= v <= 255):
            raise ValueError(f"Exit code must be between -1 and 255, got {v}")
        return v

    @field_validator("container_id")
    @classmethod
    def validate_container_id(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            if not v:
                return None
            if not re.match(r"^[a-zA-Z0-9_\-\.:]+$", v):
                raise ValueError(f"Invalid characters in container identifier: '{v}'")
        return v

    @field_validator("command")
    @classmethod
    def validate_command(cls, v: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Command string cannot be empty.")
        elif isinstance(v, list):
            if not v:
                raise ValueError("Command list cannot be empty.")
            if any(not str(item).strip() for item in v):
                raise ValueError("Command list contains empty arguments.")
        else:
            raise ValueError("Command must be a string or list of strings.")
        return v


class PytestExecutionDetails(BaseModel):
    """
    Detailed parsed results and telemetry from Pytest execution.
    Can be converted directly to the shared TestResult schema.
    """
    passed: bool = Field(
        description="True if all tests passed and exit code was 0"
    )
    total_tests: int = Field(
        default=0,
        ge=0,
        description="Total number of tests discovered and executed"
    )
    passed_tests: int = Field(
        default=0,
        ge=0,
        description="Number of passed tests"
    )
    failed_tests: int = Field(
        default=0,
        ge=0,
        description="Number of failed tests"
    )
    skipped_tests: int = Field(
        default=0,
        ge=0,
        description="Number of skipped tests"
    )
    error_count: int = Field(
        default=0,
        ge=0,
        description="Number of test collection or execution errors"
    )
    exit_code: int = Field(
        default=-1,
        description="Process exit code from pytest"
    )
    raw_output: str = Field(
        default="",
        description="Combined stdout and stderr from pytest execution"
    )
    error_summary: Optional[str] = Field(
        default=None,
        description="Extracted failure details or error synopsis"
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Execution duration in seconds"
    )
    timed_out: bool = Field(
        default=False,
        description="True if execution timed out"
    )

    def to_test_result(self) -> TestResult:
        """Map execution details to the shared system TestResult model."""
        return TestResult(
            passed=self.passed,
            total_tests=self.total_tests,
            passed_tests=self.passed_tests,
            failed_tests=self.failed_tests,
            raw_output=self.raw_output,
            error_summary=self.error_summary,
        )


class TestCaseSpec(BaseModel):
    """Specification of an individual generated test case scenario."""
    __test__ = False  # Prevent pytest collection

    name: str = Field(description="Name of the test function, e.g. 'test_valid_user_creation'")
    category: str = Field(
        default="positive",
        description="Category: 'positive', 'negative', 'boundary', 'edge_case', 'validation'"
    )
    description: str = Field(description="Detailed explanation of the scenario tested")
    expected_outcome: str = Field(description="Expected result or assertion target")


class GeneratedTestSuite(BaseModel):
    """Complete generated test suite module containing multiple test cases."""
    test_file_path: str = Field(
        default="tests/test_generated.py",
        description="Relative file path for the test module"
    )
    scenarios_summary: str = Field(
        default="",
        description="High-level description of overall test coverage"
    )
    test_cases: List[TestCaseSpec] = Field(
        default_factory=list,
        description="List of individual test case scenarios"
    )
    test_code: str = Field(
        description="Valid Python pytest source code for the test module"
    )


class TestingAgentResult(BaseModel):
    """Aggregate result from AI Testing Agent generation and execution."""
    success: bool = Field(description="True if test generation and execution finished successfully")
    test_suite: Optional[GeneratedTestSuite] = Field(default=None, description="Generated and approved test suite")
    execution_details: Optional[PytestExecutionDetails] = Field(default=None, description="Execution telemetry from container")
    test_result: Optional[TestResult] = Field(default=None, description="Shared system TestResult")
    validation_errors: List[str] = Field(default_factory=list, description="Static code validation errors or security warnings")
    logs: List[str] = Field(default_factory=list, description="Progress and diagnostic log messages")


# ============================================================================
# Phase 4: Bug Detection & Self-Correction Schemas
# ============================================================================

class BugCategoryEnum(str, Enum):
    """Enumeration of root-cause diagnostic categories."""
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    TYPE_ERROR = "type_error"
    LOGIC_ERROR = "logic_error"
    BOUNDARY_ERROR = "boundary_error"
    VALIDATION_ERROR = "validation_error"
    RUNTIME_ERROR = "runtime_error"
    INCONCLUSIVE = "inconclusive"


class BugDiagnosis(BaseModel):
    """Structured root cause diagnostic report from failure analysis."""
    category: BugCategoryEnum = Field(
        default=BugCategoryEnum.INCONCLUSIVE,
        description="Taxonomy classification of the failure"
    )
    failure_description: str = Field(
        description="Concise description of the observed test failure"
    )
    root_cause: str = Field(
        description="Technical explanation of the underlying bug or defect"
    )
    relevant_file: Optional[str] = Field(
        default=None,
        description="Relative path of the source file containing the bug"
    )
    code_location: Optional[str] = Field(
        default=None,
        description="Function, class, or line range associated with the defect"
    )
    suggested_fix: str = Field(
        description="Actionable guidance on how to resolve the defect"
    )
    confidence: str = Field(
        default="medium",
        description="Confidence level of diagnosis: 'high', 'medium', or 'low'"
    )
    evidence: str = Field(
        default="",
        description="Traceback excerpt, assertion output, or code snippet"
    )


class PatchProposal(BaseModel):
    """Structured code patch proposal with targeted replacement boundaries."""
    target_file: str = Field(
        description="Relative path of file to patch within workspace"
    )
    original_code: str = Field(
        description="Exact literal character sequence in target file to be replaced"
    )
    replacement_code: str = Field(
        description="New replacement code to substitute in place of original_code"
    )
    explanation: str = Field(
        description="Detailed explanation of why this patch resolves the root cause"
    )
    expected_effect: str = Field(
        description="Expected behavioral effect on test suite execution"
    )
    validation_status: str = Field(
        default="pending",
        description="Validation outcome: 'pending', 'valid', or 'rejected'"
    )
    validation_error: Optional[str] = Field(
        default=None,
        description="Error details if patch validation fails"
    )


class PatchAuditRecord(BaseModel):
    """Immutable audit trail entry recording an applied code modification."""
    target_file: str = Field(
        description="Relative path of the modified file"
    )
    before_hash: str = Field(
        description="SHA-256 content hash of the file immediately before patching"
    )
    after_hash: str = Field(
        description="SHA-256 content hash of the file immediately after patching"
    )
    applied: bool = Field(
        description="True if atomic write succeeded"
    )
    timestamp: float = Field(
        description="Unix timestamp when patch was applied"
    )
    explanation: str = Field(
        default="",
        description="Contextual reason for the code change"
    )


class SelfCorrectionResult(BaseModel):
    """Final aggregate outcome of bounded self-correction cycle."""
    success: bool = Field(
        description="True if rerun tests passed after patch application"
    )
    diagnosis: Optional[BugDiagnosis] = Field(
        default=None,
        description="Root cause diagnosis used for repair"
    )
    applied_patches: List[PatchAuditRecord] = Field(
        default_factory=list,
        description="Audit records of all code modifications applied"
    )
    attempts: int = Field(
        default=0,
        description="Total self-correction retry attempts consumed"
    )
    final_test_result: Optional[TestResult] = Field(
        default=None,
        description="Final test execution result following repairs"
    )
    logs: List[str] = Field(
        default_factory=list,
        description="Chronological telemetry logs from the self-correction process"
    )


# ============================================================================
# Phase 5: Test Report Generator Schemas
# ============================================================================

class FinalTestStatusEnum(str, Enum):
    """Enumeration of overall test suite outcome statuses."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    PARTIALLY_PASSED = "PARTIALLY PASSED"
    NOT_EXECUTED = "NOT EXECUTED"


class ReportProjectInfo(BaseModel):
    """Project and metadata context for test reporting."""
    project_id: str = Field(description="Unique project identifier or name")
    project_title: str = Field(default="Software Project", description="Human-readable title")
    timestamp: str = Field(description="ISO-8601 formatted timestamp of report generation")
    agent_name: str = Field(
        default="AI DevTeam - Member 3 (Testing, Debugging & Docker)",
        description="Subsystem generating this report"
    )


class ReportExecutionSummary(BaseModel):
    """High-level statistical breakdown of test suite execution."""
    total_tests: int = Field(default=0, ge=0)
    passed_tests: int = Field(default=0, ge=0)
    failed_tests: int = Field(default=0, ge=0)
    skipped_tests: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    exit_code: int = Field(default=0)
    status: FinalTestStatusEnum = Field(default=FinalTestStatusEnum.NOT_EXECUTED)


class ReportFailureInfo(BaseModel):
    """Structured details of test failures, errors, and tracebacks."""
    failed_test_names: List[str] = Field(default_factory=list)
    error_messages: List[str] = Field(default_factory=list)
    traceback_summary: Optional[str] = Field(default=None)
    failure_categories: List[str] = Field(default_factory=list)


class ReportDiagnosisInfo(BaseModel):
    """Root cause diagnostic details from automated debugging analysis."""
    root_cause: Optional[str] = Field(default=None)
    bug_category: Optional[str] = Field(default=None)
    confidence: Optional[str] = Field(default=None)
    suggested_fix: Optional[str] = Field(default=None)


class ReportSelfCorrectionInfo(BaseModel):
    """Telemetry detailing automated repair and patch application cycles."""
    attempts: int = Field(default=0, ge=0)
    files_patched: List[str] = Field(default_factory=list)
    patch_details: List[Dict[str, Any]] = Field(default_factory=list)
    final_rerun_passed: bool = Field(default=False)


class TestReportData(BaseModel):
    """Complete aggregated test report schema suitable for JSON serialization."""
    __test__ = False  # Prevent pytest collection

    project_info: ReportProjectInfo
    summary: ReportExecutionSummary
    failures: ReportFailureInfo
    diagnosis: ReportDiagnosisInfo
    self_correction: ReportSelfCorrectionInfo
    final_status: FinalTestStatusEnum
    recommendations: List[str] = Field(default_factory=list)
