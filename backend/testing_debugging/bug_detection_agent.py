"""
Bug Detection and Self-Correction Agent for Member 3: Testing & Debugging Subsystem.

Diagnoses Pytest failures, performs root cause analysis via Google Gemini,
proposes atomic, validated code patches, and coordinates bounded self-correction cycles
without creating infinite repair loops.
"""

import ast
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from backend.config import config
from backend.schemas.state import AgentState, TestResult
from backend.testing_debugging.docker_sandbox import DockerSecurityViolationError
from backend.testing_debugging.schemas import (
    BugCategoryEnum,
    BugDiagnosis,
    PatchAuditRecord,
    PatchProposal,
    SelfCorrectionResult,
)
from backend.testing_debugging.test_executor import TestExecutor
from backend.testing_debugging.testing_agent import (
    TestCodeSecurityValidator,
    get_gemini_model,
)

logger = logging.getLogger(__name__)


# Files and directories protected from any modification by patches
PROTECTED_TARGET_PATTERNS = [
    ".env*",
    "*.env",
    "*.secret",
    "*.secrets",
    "secrets.*",
    "*.key",
    "*.pem",
    ".git*",
    ".venv*",
    "venv*",
    "backend/schemas/state.py",
    "backend/config.py",
    "requirements.txt",
]


class SafePatchManager:
    """
    Dedicated Member 3 safe patch and file manager adapter.
    Enforces atomic file replacement, path security boundaries, content hashing,
    and post-patch syntax and security verification.
    """

    def __init__(self) -> None:
        self.audit_trail: List[PatchAuditRecord] = []

    @classmethod
    def compute_hash(cls, content: Union[str, bytes]) -> str:
        """Computes SHA-256 hash of content."""
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    @classmethod
    def is_protected(cls, rel_path: Union[str, Path]) -> bool:
        """Determines if a target file is protected from automated patching."""
        p_str = str(rel_path).replace("\\", "/").lower()
        p_obj = Path(p_str)
        name = p_obj.name

        # Protected directories
        for part in p_obj.parts:
            if part in [".git", ".venv", "venv", "env"]:
                return True

        # Sensitive files
        if name.startswith(".env") or name.endswith(".env"):
            return True
        if any(token in name for token in ["secret", "credential", "auth_token"]):
            return True
        if name in ["state.py", "config.py", "requirements.txt"]:
            return True

        return False

    def validate_patch(
        self,
        workspace_path: Union[str, Path],
        patch: PatchProposal,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates a proposed code patch before application.

        Args:
            workspace_path: Workspace root path.
            patch: PatchProposal instance.

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, rejection_reason).
        """
        if not patch.target_file or not patch.target_file.strip():
            return False, "Target file path cannot be empty."

        if not patch.original_code or not patch.original_code.strip():
            return False, "Original code snippet to replace cannot be empty."

        if not patch.replacement_code:
            return False, "Replacement code cannot be None."

        # Check path protection
        if self.is_protected(patch.target_file):
            return False, f"Security violation: Target file '{patch.target_file}' is protected from automated patches."

        resolved_ws = Path(workspace_path).resolve()
        target_path = (resolved_ws / patch.target_file).resolve()

        # Enforce workspace boundary
        try:
            target_path.relative_to(resolved_ws)
        except ValueError:
            return False, f"Security violation: Target path '{patch.target_file}' escapes workspace boundary."

        if target_path == resolved_ws:
            return False, "Security violation: Target path cannot be the workspace root."

        if not target_path.exists():
            return False, f"Target file does not exist: '{patch.target_file}'."

        if not target_path.is_file():
            return False, f"Target path must be a regular file: '{patch.target_file}'."

        # Check literal match in current file
        current_content = target_path.read_text(encoding="utf-8", errors="replace")
        if patch.original_code not in current_content:
            return False, f"Original code snippet not found in target file '{patch.target_file}'. Content may have changed."

        # Dry-run replacement
        tentative_content = current_content.replace(patch.original_code, patch.replacement_code, 1)

        # Check Python syntax if target is a Python file
        if target_path.suffix == ".py":
            try:
                ast.parse(tentative_content)
            except SyntaxError as syn_err:
                return False, f"Patch introduces Python syntax error: {syn_err.msg} at line {syn_err.lineno}."

            # Static AST security verification
            is_safe, sec_violations = TestCodeSecurityValidator.validate(tentative_content, is_test_file=False)
            if not is_safe:
                return False, f"Patch violates security policies: {'; '.join(sec_violations)}."

        patch.validation_status = "valid"
        patch.validation_error = None
        return True, None

    def apply_patch(
        self,
        workspace_path: Union[str, Path],
        patch: PatchProposal,
    ) -> Tuple[bool, Optional[PatchAuditRecord], Optional[str]]:
        """
        Applies a validated patch atomically to the target file.

        Args:
            workspace_path: Workspace root directory.
            patch: PatchProposal instance.

        Returns:
            Tuple[bool, Optional[PatchAuditRecord], Optional[str]]: (success, audit_record, error_message).
        """
        is_valid, err = self.validate_patch(workspace_path, patch)
        if not is_valid:
            patch.validation_status = "rejected"
            patch.validation_error = err
            return False, None, err

        resolved_ws = Path(workspace_path).resolve()
        target_path = (resolved_ws / patch.target_file).resolve()

        try:
            current_content = target_path.read_text(encoding="utf-8", errors="replace")
            before_hash = self.compute_hash(current_content)

            new_content = current_content.replace(patch.original_code, patch.replacement_code, 1)
            after_hash = self.compute_hash(new_content)

            # Atomic file replacement via temporary staged file
            tmp_path = target_path.with_suffix(target_path.suffix + ".patch_tmp")
            tmp_path.write_text(new_content, encoding="utf-8")
            os.replace(tmp_path, target_path)

            audit_record = PatchAuditRecord(
                target_file=patch.target_file,
                before_hash=before_hash,
                after_hash=after_hash,
                applied=True,
                timestamp=time.time(),
                explanation=patch.explanation,
            )
            self.audit_trail.append(audit_record)
            return True, audit_record, None

        except Exception as exc:
            err_msg = f"Failed to atomically apply patch: {str(exc)}"
            return False, None, err_msg


class BugDetectionAgent:
    """
    Diagnoses Pytest failures, performs root cause analysis via Gemini,
    proposes structured code patches, and coordinates bounded self-correction cycles.
    """

    __test__ = False  # Prevent pytest collection

    def __init__(
        self,
        model: Optional[Any] = None,
        executor: Optional[TestExecutor] = None,
        patch_manager: Optional[SafePatchManager] = None,
        model_name: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        """
        Initialize the Bug Detection Agent.

        Args:
            model: Optional injected Gemini model instance.
            executor: Optional injected TestExecutor instance.
            patch_manager: Optional injected SafePatchManager instance.
            model_name: Optional override for Gemini model name.
            max_retries: Optional maximum self-correction retry limit.
        """
        self._injected_model = model
        self.executor = executor or TestExecutor()
        self.patch_manager = patch_manager or SafePatchManager()
        self.model_name = model_name or config.LLM_MODEL
        self.max_retries = max_retries if max_retries is not None else getattr(config, "MAX_RETRY_LIMIT", 3)

    @property
    def model(self) -> Any:
        """Lazily initialize or return the Gemini model."""
        if self._injected_model is not None:
            return self._injected_model
        return get_gemini_model(self.model_name)

    def build_diagnosis_prompt(
        self,
        test_result: TestResult,
        source_files: List[Dict[str, str]],
        requirement_spec: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Constructs prompt for Gemini root cause diagnosis and patch synthesis."""
        code_sections: List[str] = []
        for sf in source_files:
            path = sf.get("path", "file.py")
            content = sf.get("content", "")
            code_sections.append(f"--- File: {path} ---\n{content}")
        code_str = "\n\n".join(code_sections) if code_sections else "# No source files provided."

        req_summary = ""
        if requirement_spec:
            req_summary = (
                f"Title: {requirement_spec.get('project_title', '')}\n"
                f"Requirements: {', '.join(requirement_spec.get('functional_requirements', []))}\n"
                f"Acceptance Criteria: {', '.join(requirement_spec.get('acceptance_criteria', []))}"
            )

        prompt = f"""
You are an expert Senior Software Debugging and Root Cause Analysis Specialist.
Analyze the failed test execution details and source code below to determine the exact root cause of the failure and propose an exact minimal patch to fix the bug.

FAILURE SUMMARY:
{test_result.error_summary or "Test execution failed."}

RAW PYTEST OUTPUT & TRACEBACK:
{test_result.raw_output[:4000]}

SOURCE CODE FILES:
{code_str}

REQUIREMENTS & ACCEPTANCE CRITERIA:
{req_summary or "Standard functional and unit test requirements."}

Analyze the error traceback and code carefully.
Identify the single most likely root cause.
If the evidence is ambiguous, missing, or insufficient to identify a specific defect, categorize as "inconclusive".

You MUST return a JSON object with this exact schema:
{{
    "diagnosis": {{
        "category": "syntax_error" | "import_error" | "type_error" | "logic_error" | "boundary_error" | "validation_error" | "runtime_error" | "inconclusive",
        "failure_description": "Clear explanation of what failed",
        "root_cause": "Exact technical explanation of why the failure occurred",
        "relevant_file": "path/to/buggy_file.py",
        "code_location": "function_name or line number",
        "suggested_fix": "Description of the required code change",
        "confidence": "high" | "medium" | "low",
        "evidence": "Key lines from the traceback or test output supporting this diagnosis"
    }},
    "patch": {{
        "target_file": "path/to/buggy_file.py",
        "original_code": "exact string currently in the file to be replaced",
        "replacement_code": "exact new replacement code string",
        "explanation": "Why this specific replacement fixes the issue",
        "expected_effect": "Test will pass without breaking existing functionality"
    }}
}}

CRITICAL RULES FOR PATCH:
1. `original_code` must be an exact, literal substring from the existing target file (including exact whitespace/indentation).
2. `replacement_code` must be the complete drop-in replacement for that exact substring.
3. Keep the patch minimal: fix only the bug; do NOT rewrite the entire file.
4. If category is "inconclusive", set `patch` to null.
5. Do NOT use markdown code fences in the JSON response.
"""
        return prompt.strip()

    def _fallback_heuristic_diagnosis(
        self,
        test_result: TestResult,
        source_files: List[Dict[str, str]],
    ) -> Tuple[BugDiagnosis, Optional[PatchProposal]]:
        """Provides a safe, evidence-based diagnostic classification when LLM is unavailable."""
        raw = (test_result.raw_output + "\n" + (test_result.error_summary or "")).lower()

        category = BugCategoryEnum.INCONCLUSIVE
        failure_desc = "Test suite execution failed."
        root_cause = "Insufficient evidence to conclusively determine root cause automatically."
        confidence = "low"
        relevant_file = source_files[0].get("path") if source_files else None

        if "syntaxerror" in raw:
            category = BugCategoryEnum.SYNTAX_ERROR
            failure_desc = "Syntax error encountered during test loading or execution."
            root_cause = "Invalid Python syntax in source or test file."
            confidence = "high"
        elif "modulenotfounderror" in raw or "importerror" in raw:
            category = BugCategoryEnum.IMPORT_ERROR
            failure_desc = "Import or module resolution error."
            root_cause = "Missing dependency or unresolved import path."
            confidence = "high"
        elif "typeerror" in raw:
            category = BugCategoryEnum.TYPE_ERROR
            failure_desc = "Type mismatch or inappropriate argument type."
            root_cause = "Object passed with incompatible type or missing argument."
            confidence = "medium"
        elif "assertionerror" in raw:
            category = BugCategoryEnum.LOGIC_ERROR
            failure_desc = "Assertion condition failed during test execution."
            root_cause = "Output value did not match expected assertion criteria."
            confidence = "medium"
        elif "indexerror" in raw or "keyerror" in raw or "zerodivisionerror" in raw:
            category = BugCategoryEnum.BOUNDARY_ERROR
            failure_desc = "Boundary or lookup failure."
            root_cause = "Index out of range, missing dictionary key, or divide-by-zero."
            confidence = "medium"
        elif "validationerror" in raw:
            category = BugCategoryEnum.VALIDATION_ERROR
            failure_desc = "Data schema validation failed."
            root_cause = "Input data did not conform to expected schema."
            confidence = "medium"

        diagnosis = BugDiagnosis(
            category=category,
            failure_description=failure_desc,
            root_cause=root_cause,
            relevant_file=relevant_file,
            code_location=None,
            suggested_fix="Review error traceback and inspect relevant function implementation.",
            confidence=confidence,
            evidence=(test_result.error_summary or test_result.raw_output)[:300],
        )
        return diagnosis, None

    def diagnose_failure(
        self,
        test_result: TestResult,
        source_files: List[Dict[str, str]],
        requirement_spec: Optional[Dict[str, Any]] = None,
    ) -> Tuple[BugDiagnosis, Optional[PatchProposal]]:
        """
        Diagnoses test failures using Gemini and extracts structured patch proposals.

        Args:
            test_result: TestResult from failed Pytest run.
            source_files: List of generated source files.
            requirement_spec: Optional requirement specifications.

        Returns:
            Tuple[BugDiagnosis, Optional[PatchProposal]]: Structured diagnosis and patch (if available).
        """
        prompt = self.build_diagnosis_prompt(
            test_result=test_result,
            source_files=source_files,
            requirement_spec=requirement_spec,
        )

        try:
            model_instance = self.model
            response = model_instance.generate_content(prompt)

            if not response or not hasattr(response, "text") or not response.text:
                raise ValueError("Empty response received from Gemini model.")

            raw_text = response.text.strip()
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\n?", "", raw_text)
                raw_text = re.sub(r"\n?```$", "", raw_text)

            parsed = json.loads(raw_text)
            diag_dict = parsed.get("diagnosis", {})
            diagnosis = BugDiagnosis(**diag_dict)

            patch: Optional[PatchProposal] = None
            patch_dict = parsed.get("patch")
            if patch_dict and isinstance(patch_dict, dict) and diagnosis.category != BugCategoryEnum.INCONCLUSIVE:
                patch = PatchProposal(**patch_dict)

            return diagnosis, patch

        except Exception as exc:
            logger.warning(f"[Bug Agent] Model diagnosis failed: {str(exc)}. Using heuristic fallback.")
            return self._fallback_heuristic_diagnosis(test_result, source_files)

    def self_correct(self, state: AgentState, single_step: bool = False) -> AgentState:
        """
        Coordinates the bounded self-correction loop.

        Args:
            state: Central AgentState dictionary.
            single_step: If True, execute only one repair attempt per call (useful for LangGraph looping).

        Returns:
            AgentState: Updated AgentState after self-correction attempts.
        """
        if "logs" not in state or state["logs"] is None:
            state["logs"] = []

        state["current_agent"] = "Bug Detection Agent"
        state["logs"].append("[Bug Agent] Initiating failure diagnosis and self-correction coordinator.")

        # If tests are already passing, no action required
        if state.get("is_test_passed", False):
            state["logs"].append("[Bug Agent] Tests are already passing. No correction needed.")
            state["status"] = "reviewing"
            return state

        # If retry limit already exhausted, halt
        current_retries = state.get("retry_count", 0)
        if current_retries >= self.max_retries:
            state["logs"].append(
                f"[Bug Agent] Maximum retry limit ({self.max_retries}) already reached. Halting self-correction."
            )
            state["status"] = "failed"
            return state

        workspace_str = state.get("workspace_path") or config.WORKSPACE_DIR
        workspace_path = Path(workspace_str).resolve()

        if "patched_files" not in state or state["patched_files"] is None:
            state["patched_files"] = []

        # Read current source files from state or workspace
        source_files = state.get("generated_files") or []
        req_spec = state.get("requirement_spec")

        # Bounded repair loop
        while state.get("retry_count", 0) < self.max_retries:
            test_res_dict = state.get("test_results") or {}
            test_result = TestResult(**test_res_dict) if test_res_dict else TestResult(passed=False)

            # 1. Diagnose Root Cause
            state["logs"].append(
                f"[Bug Agent] Analyzing failure (Attempt {state.get('retry_count', 0) + 1}/{self.max_retries})..."
            )
            diagnosis, patch = self.diagnose_failure(
                test_result=test_result,
                source_files=source_files,
                requirement_spec=req_spec,
            )

            state["bug_diagnosis"] = f"[{diagnosis.category.value}] {diagnosis.root_cause}"
            state["logs"].append(
                f"[Bug Agent] Diagnosis: {diagnosis.failure_description} "
                f"(Root Cause: {diagnosis.root_cause}, Category: {diagnosis.category.value}, Confidence: {diagnosis.confidence})"
            )

            # Halt if diagnosis is inconclusive or no patch was provided
            if diagnosis.category == BugCategoryEnum.INCONCLUSIVE or patch is None:
                state["logs"].append("[Bug Agent] Inconclusive diagnosis or no safe patch proposal. Halting repair.")
                state["status"] = "failed"
                break

            # 2. Validate Patch
            is_valid, val_err = self.patch_manager.validate_patch(workspace_path, patch)
            if not is_valid:
                state["logs"].append(f"[Bug Agent] Proposed patch rejected: {val_err}")
                state["status"] = "failed"
                break

            # 3. Apply Patch Atomically
            success, audit_rec, app_err = self.patch_manager.apply_patch(workspace_path, patch)
            if not success or audit_rec is None:
                state["logs"].append(f"[Bug Agent] Failed to apply patch: {app_err}")
                state["status"] = "failed"
                break

            state["logs"].append(
                f"[Bug Agent] Applied verified patch to '{patch.target_file}': {patch.explanation}"
            )

            # Update patched_files in AgentState
            target_file_path = (workspace_path / patch.target_file).resolve()
            new_file_content = target_file_path.read_text(encoding="utf-8", errors="replace")
            self._update_patched_files_state(state, patch.target_file, new_file_content)

            # Increment retry count
            state["retry_count"] = state.get("retry_count", 0) + 1

            # 4. Rerun Tests via TestExecutor
            state["logs"].append(
                f"[Bug Agent] Rerunning test suite (Attempt {state['retry_count']}/{self.max_retries})..."
            )
            try:
                rerun_result: TestResult = self.executor.run_tests(workspace_path=workspace_path)
                state["test_results"] = rerun_result.model_dump()
                state["is_test_passed"] = rerun_result.passed

                if rerun_result.passed:
                    state["logs"].append(
                        f"[Bug Agent] Self-correction SUCCESSFUL! All {rerun_result.total_tests} tests passed on attempt {state['retry_count']}."
                    )
                    state["status"] = "reviewing"
                    break
                else:
                    state["logs"].append(
                        f"[Bug Agent] Tests still failing after patch attempt {state['retry_count']}: "
                        f"{rerun_result.failed_tests}/{rerun_result.total_tests} failed."
                    )
                    if state["retry_count"] >= self.max_retries:
                        state["logs"].append(
                            f"[Bug Agent] Maximum retry limit ({self.max_retries}) reached. Self-correction halted."
                        )
                        state["status"] = "failed"
                        break
                    else:
                        state["status"] = "debugging"

            except Exception as rerun_exc:
                state["logs"].append(f"[Bug Agent Error] Test rerun failed: {str(rerun_exc)}")
                state["status"] = "failed"
                break

            if single_step:
                break

        return state

    @staticmethod
    def _update_patched_files_state(
        state: AgentState,
        rel_path: str,
        content: str,
    ) -> None:
        """Records modified file in state.patched_files without duplicates."""
        existing = state.get("patched_files") or []
        updated = [f for f in existing if f.get("path") != rel_path]
        updated.append({"path": rel_path, "content": content})
        state["patched_files"] = updated


def run_bug_agent(state: AgentState) -> AgentState:
    """
    Standard entrypoint function for Member 3 Bug Detection Agent.
    Compatible with LangGraph StateGraph node registration.
    """
    agent = BugDetectionAgent()
    return agent.self_correct(state)
