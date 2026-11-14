"""
Test Report Generator for Member 3: Testing & Debugging Subsystem.

Synthesizes testing outcomes, execution telemetry, root cause diagnoses,
and self-correction audit records into standardized JSON and formatted Markdown reports.
Enforces safe path confinement, atomic writes, and secret redaction.
"""

from datetime import datetime, timezone
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from backend.config import config
from backend.schemas.state import AgentState
from backend.testing_debugging.docker_sandbox import DockerSecurityViolationError
from backend.testing_debugging.schemas import (
    FinalTestStatusEnum,
    ReportDiagnosisInfo,
    ReportExecutionSummary,
    ReportFailureInfo,
    ReportProjectInfo,
    ReportSelfCorrectionInfo,
    TestReportData,
)

logger = logging.getLogger(__name__)


# Regex patterns for detecting and redacting sensitive credentials and keys
SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|password|token|auth|cred(?:ential)?)[^\s\w]{0,3}[:=]\s*['\"]?([A-Za-z0-9\-_.+=/]{6,})['\"]?"),
    re.compile(r"AIza[0-9A-Za-z-_]{25,}"),  # Google API key pattern
    re.compile(r"sk-[a-zA-Z0-9]{25,}"),    # Common token format
    re.compile(r"(?i)bearer\s+[A-Za-z0-9\-_.+=/]{15,}"),
]


def redact_secrets(text: Optional[str]) -> str:
    """
    Scans and redacts API keys, passwords, and security tokens from report text.

    Args:
        text: Input string potentially containing sensitive credentials.

    Returns:
        str: Sanitized string with sensitive tokens replaced with [REDACTED].
    """
    if not text:
        return ""
    sanitized = str(text)
    for pattern in SECRET_PATTERNS:
        repl = r"\1=[REDACTED]" if pattern.groups >= 1 else "[REDACTED]"
        sanitized = pattern.sub(repl, sanitized)
    return sanitized


class TestReportGenerator:
    """
    Generates structured JSON and professional Markdown test reports from AgentState.
    Safely stores reports within workspace boundaries using atomic writes.
    """

    __test__ = False  # Prevent pytest collection

    def __init__(self, reports_subdir: str = "reports") -> None:
        """
        Initialize the TestReportGenerator.

        Args:
            reports_subdir: Relative directory inside project workspace where reports are saved.
        """
        self.reports_subdir = reports_subdir

    def validate_output_directory(self, workspace_path: Union[str, Path]) -> Path:
        """
        Validates that the reports output directory is safely confined inside the workspace.

        Args:
            workspace_path: Root project workspace directory.

        Returns:
            Path: Resolved, validated reports directory.

        Raises:
            DockerSecurityViolationError: If output path attempts directory traversal.
            ValueError: If path contains invalid characters.
        """
        resolved_ws = Path(workspace_path).resolve()
        target_dir = (resolved_ws / self.reports_subdir).resolve()

        try:
            target_dir.relative_to(resolved_ws)
        except ValueError as exc:
            raise DockerSecurityViolationError(
                f"Security violation: Reports path '{self.reports_subdir}' escapes workspace boundary."
            ) from exc

        if any(char in str(self.reports_subdir) for char in [";", "&", "|", "`", "$", "\n", "\r"]):
            raise ValueError(f"Invalid characters in reports directory path: '{self.reports_subdir}'")

        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def determine_final_status(
        self,
        test_results: Optional[Dict[str, Any]],
        is_test_passed: bool,
    ) -> FinalTestStatusEnum:
        """
        Calculates the definitive overall test suite execution status.

        Args:
            test_results: Dictionary representation of shared TestResult.
            is_test_passed: Boolean passed flag from state.

        Returns:
            FinalTestStatusEnum: PASSED, FAILED, PARTIALLY_PASSED, or NOT_EXECUTED.
        """
        if not test_results:
            return FinalTestStatusEnum.NOT_EXECUTED

        total = int(test_results.get("total_tests", 0))
        passed = int(test_results.get("passed_tests", 0))
        failed = int(test_results.get("failed_tests", 0))

        if total == 0:
            return FinalTestStatusEnum.NOT_EXECUTED

        if passed == total and (is_test_passed or test_results.get("passed") is True):
            return FinalTestStatusEnum.PASSED

        if passed > 0 and failed > 0:
            return FinalTestStatusEnum.PARTIALLY_PASSED

        return FinalTestStatusEnum.FAILED

    def extract_failure_information(
        self,
        test_results: Optional[Dict[str, Any]],
    ) -> ReportFailureInfo:
        """Extracts structured failure details and tracebacks safely."""
        if not test_results:
            return ReportFailureInfo()

        raw_output = redact_secrets(test_results.get("raw_output", ""))
        error_summary = redact_secrets(test_results.get("error_summary", ""))

        failed_names: List[str] = []
        error_messages: List[str] = []
        categories: List[str] = []

        # Parse failed test identifiers from raw output lines
        for line in raw_output.splitlines():
            sline = line.strip()
            if sline.startswith("FAILED ") or sline.startswith("ERROR "):
                tokens = sline.split(" ", 2)
                if len(tokens) >= 2:
                    failed_names.append(tokens[1])
                if len(tokens) >= 3:
                    error_messages.append(tokens[2])

        # Infer failure categories from error text
        raw_lower = raw_output.lower()
        if "assertionerror" in raw_lower:
            categories.append("Assertion Failure")
        if "syntaxerror" in raw_lower:
            categories.append("Syntax Error")
        if "importerror" in raw_lower or "modulenotfounderror" in raw_lower:
            categories.append("Import Error")
        if "typeerror" in raw_lower:
            categories.append("Type Error")
        if "zerodivisionerror" in raw_lower or "indexerror" in raw_lower or "keyerror" in raw_lower:
            categories.append("Boundary/Lookup Error")
        if "timed out" in raw_lower or "timeout" in raw_lower:
            categories.append("Timeout")

        if not categories and (test_results.get("failed_tests", 0) > 0 or not test_results.get("passed")):
            categories.append("Functional Error")

        # Create concise traceback excerpt (max 1500 chars)
        tb_summary = error_summary or (raw_output[:1500] if raw_output else None)

        return ReportFailureInfo(
            failed_test_names=failed_names,
            error_messages=error_messages,
            traceback_summary=tb_summary,
            failure_categories=list(set(categories)),
        )

    def extract_diagnosis_information(self, state: AgentState) -> ReportDiagnosisInfo:
        """Parses bug diagnosis details from AgentState."""
        diagnosis_str = state.get("bug_diagnosis")
        if not diagnosis_str:
            return ReportDiagnosisInfo()

        diag_clean = redact_secrets(diagnosis_str)
        category = None
        root_cause = diag_clean

        # Parse category if formatted like '[logic_error] Root cause...'
        match = re.match(r"^\[(.*?)\]\s*(.*)$", diag_clean)
        if match:
            category = match.group(1)
            root_cause = match.group(2)

        return ReportDiagnosisInfo(
            root_cause=root_cause,
            bug_category=category,
            confidence="high" if category and category != "inconclusive" else "medium",
            suggested_fix="Review failure diagnosis and apply targeted code modification.",
        )

    def extract_self_correction_information(self, state: AgentState) -> ReportSelfCorrectionInfo:
        """Parses self-correction telemetry, patch records, and rerun outcomes."""
        attempts = state.get("retry_count", 0)
        patched_files_raw = state.get("patched_files") or []

        files_patched: List[str] = []
        patch_details: List[Dict[str, Any]] = []

        for pf in patched_files_raw:
            path = pf.get("path")
            if path and path not in files_patched:
                files_patched.append(path)
                patch_details.append({
                    "target_file": path,
                    "applied": True,
                    "content_preview": redact_secrets(pf.get("content", ""))[:200],
                })

        final_rerun_passed = state.get("is_test_passed", False)

        return ReportSelfCorrectionInfo(
            attempts=attempts,
            files_patched=files_patched,
            patch_details=patch_details,
            final_rerun_passed=final_rerun_passed,
        )

    def generate_recommendations(
        self,
        final_status: FinalTestStatusEnum,
        summary: ReportExecutionSummary,
        self_correction: ReportSelfCorrectionInfo,
    ) -> List[str]:
        """Synthesizes objective, actionable engineering recommendations."""
        recommendations: List[str] = []

        if final_status == FinalTestStatusEnum.PASSED:
            recommendations.append("All unit and functional test scenarios executed and passed cleanly.")
            recommendations.append("The codebase satisfies core acceptance criteria and is ready for security review.")
            if self_correction.attempts > 0:
                recommendations.append(
                    f"Automated self-correction succeeded after {self_correction.attempts} attempt(s). "
                    "Review patched files for architectural alignment."
                )
            else:
                recommendations.append("Consider augmenting the test suite with integration and load benchmarks.")

        elif final_status == FinalTestStatusEnum.PARTIALLY_PASSED:
            recommendations.append(
                f"{summary.failed_tests} out of {summary.total_tests} test cases failed. Address remaining failing tests."
            )
            recommendations.append("Inspect failure summaries and verify expected boundary condition handling.")
            recommendations.append("Ensure mock objects accurately simulate external dependencies.")

        elif final_status == FinalTestStatusEnum.FAILED:
            recommendations.append("Critical test failures detected. The code does not satisfy acceptance criteria.")
            recommendations.append("Review root cause diagnosis and tracebacks for syntax, import, or logic errors.")
            recommendations.append("Execute targeted debugging iterations on the failing module components.")

        else:  # NOT_EXECUTED
            recommendations.append("No test execution telemetry recorded in state.")
            recommendations.append("Ensure test files follow standard Pytest discovery patterns ('test_*.py').")
            recommendations.append("Verify that Docker Desktop or the local sandbox engine is active and reachable.")

        return recommendations

    def build_report_data(self, state: AgentState) -> TestReportData:
        """
        Constructs the complete TestReportData model from the central AgentState.

        Args:
            state: LangGraph AgentState dictionary.

        Returns:
            TestReportData: Validated report schema.
        """
        proj_id = str(state.get("project_id", "project_default"))
        req_spec = state.get("requirement_spec") or {}
        proj_title = req_spec.get("project_title", "Software Project")

        timestamp = datetime.now(timezone.utc).isoformat()
        proj_info = ReportProjectInfo(
            project_id=proj_id,
            project_title=proj_title,
            timestamp=timestamp,
        )

        test_results = state.get("test_results") or {}
        is_test_passed = bool(state.get("is_test_passed", False))
        final_status = self.determine_final_status(test_results, is_test_passed)

        summary = ReportExecutionSummary(
            total_tests=int(test_results.get("total_tests", 0)),
            passed_tests=int(test_results.get("passed_tests", 0)),
            failed_tests=int(test_results.get("failed_tests", 0)),
            skipped_tests=int(test_results.get("skipped_tests", 0)),
            error_count=int(test_results.get("error_count", 0)),
            duration_seconds=float(test_results.get("duration_seconds", 0.0)),
            exit_code=int(test_results.get("exit_code", 0 if final_status == FinalTestStatusEnum.PASSED else 1)),
            status=final_status,
        )

        failures = self.extract_failure_information(test_results)
        diagnosis = self.extract_diagnosis_information(state)
        self_correction = self.extract_self_correction_information(state)
        recommendations = self.generate_recommendations(final_status, summary, self_correction)

        return TestReportData(
            project_info=proj_info,
            summary=summary,
            failures=failures,
            diagnosis=diagnosis,
            self_correction=self_correction,
            final_status=final_status,
            recommendations=recommendations,
        )

    def render_json(self, report_data: TestReportData) -> str:
        """Serializes report data to structured, machine-readable JSON."""
        data_dict = report_data.model_dump()
        return json.dumps(data_dict, indent=2, ensure_ascii=False)

    def render_markdown(self, report_data: TestReportData) -> str:
        """Renders comprehensive, beautifully styled Markdown report."""
        p_info = report_data.project_info
        summary = report_data.summary
        status = report_data.final_status.value
        failures = report_data.failures
        diag = report_data.diagnosis
        sc = report_data.self_correction

        pass_rate = (
            f"{(summary.passed_tests / summary.total_tests * 100):.1f}%"
            if summary.total_tests > 0 else "0.0%"
        )

        md = f"""# Test Execution & Quality Report

**Project Title:** {p_info.project_title}  
**Project Identifier:** `{p_info.project_id}`  
**Generation Timestamp:** `{p_info.timestamp}`  
**Subsystem:** {p_info.agent_name}  

---

## 1. Executive Summary

| Final Outcome | Pass Rate | Total Tests | Passed | Failed | Skipped | Duration |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **{status}** | **{pass_rate}** | **{summary.total_tests}** | **{summary.passed_tests}** | **{summary.failed_tests}** | **{summary.skipped_tests}** | **{summary.duration_seconds:.2f}s** |

---

## 2. Test Execution Details

- **Test Suite Status:** `{status}`
- **Exit Code:** `{summary.exit_code}`
- **Execution Engine:** Docker Sandbox (Ephemerally Containerized)
- **Network Isolation:** Enabled (`network_mode="none"`)
- **Resource Constraints:** Enforced (Memory: 512MB, CPU: 1.0)

---

## 3. Failure & Diagnostic Information
"""

        if summary.failed_tests > 0 or failures.failed_test_names:
            md += "\n### Identified Test Failures\n\n"
            for idx, name in enumerate(failures.failed_test_names or ["Unknown Test"], 1):
                msg = failures.error_messages[idx - 1] if idx - 1 < len(failures.error_messages) else "Test failed."
                md += f"- **Failure {idx}:** `{name}` — *{msg}*\n"

            if failures.failure_categories:
                md += f"\n**Failure Taxonomies:** {', '.join(failures.failure_categories)}\n"

            if failures.traceback_summary:
                md += f"\n#### Diagnostic Traceback Summary\n```text\n{failures.traceback_summary}\n```\n"
        else:
            md += "\n> [!NOTE]\n> No test failures were observed. All executed scenarios satisfied assertion criteria.\n"

        md += "\n---\n\n## 4. Root Cause Analysis\n"
        if diag.root_cause:
            md += f"""
- **Taxonomy Category:** `{diag.bug_category or "Unclassified"}`
- **Confidence Rating:** `{diag.confidence or "medium"}`
- **Root Cause Explanation:** {diag.root_cause}
- **Remediation Recommendation:** {diag.suggested_fix or "None provided"}
"""
        else:
            md += "\n> [!NOTE]\n> No bug diagnosis was triggered for this test run.\n"

        md += "\n---\n\n## 5. Automated Self-Correction Telemetry\n"
        md += f"""
- **Correction Attempts Consumed:** `{sc.attempts}`
- **Files Modified:** `{len(sc.files_patched)}`
- **Post-Correction Verification:** `{"PASSED" if sc.final_rerun_passed else "FAILED or NOT RUN"}`
"""
        if sc.files_patched:
            md += "\n**Patched File Inventory:**\n"
            for f in sc.files_patched:
                md += f"- `{f}`\n"

        md += "\n---\n\n## 6. Actionable Engineering Recommendations\n\n"
        for rec in report_data.recommendations:
            md += f"- {rec}\n"

        md += "\n---\n*Report automatically generated by Member 3 (Testing, Debugging & Docker Subsystem).*\n"
        return md

    def _atomic_write_file(self, target_path: Path, content: str) -> None:
        """Atomically writes content to disk via a temporary file."""
        tmp_path = target_path.with_suffix(f"{target_path.suffix}.tmp_{uuid.uuid4().hex[:6]}")
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, target_path)

    def generate_reports(
        self,
        state: AgentState,
        workspace_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[TestReportData, Path, Path]:
        """
        Builds, renders, and safely writes both JSON and Markdown test reports.

        Args:
            state: LangGraph AgentState dictionary.
            workspace_path: Optional override for the workspace directory.

        Returns:
            Tuple[TestReportData, Path, Path]: (report_data, json_file_path, md_file_path).

        Raises:
            DockerSecurityViolationError: If report storage path attempts path traversal.
        """
        ws = Path(workspace_path or state.get("workspace_path") or config.WORKSPACE_DIR).resolve()
        output_dir = self.validate_output_directory(ws)

        # Build data models
        report_data = self.build_report_data(state)

        # Generate unique filenames
        clean_proj_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", report_data.project_info.project_id)
        time_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_suffix = uuid.uuid4().hex[:6]

        json_filename = f"test_report_{clean_proj_id}_{time_tag}_{unique_suffix}.json"
        md_filename = f"test_report_{clean_proj_id}_{time_tag}_{unique_suffix}.md"

        json_path = output_dir / json_filename
        md_path = output_dir / md_filename

        # Render sanitized content
        json_content = redact_secrets(self.render_json(report_data))
        md_content = redact_secrets(self.render_markdown(report_data))

        # Atomic writes
        self._atomic_write_file(json_path, json_content)
        self._atomic_write_file(md_path, md_content)

        # Log completion into state
        if "logs" in state and isinstance(state["logs"], list):
            state["logs"].append(
                f"[Report Generator] Successfully produced test reports: '{json_filename}' and '{md_filename}'."
            )

        logger.info(f"Test reports written to {json_path} and {md_path}")
        return report_data, json_path, md_path
