"""
AI Testing Agent for Member 3: Testing & Debugging Subsystem.

Uses Google Gemini to analyze requirements and generated source code,
synthesizes comprehensive unit, positive, negative, boundary, and edge test cases,
statically validates code security, and executes tests in the Docker Sandbox via TestExecutor.
"""

import ast
import json
import logging
import re
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
    GeneratedTestSuite,
    PytestExecutionDetails,
    TestCaseSpec,
    TestingAgentResult,
)
from backend.testing_debugging.test_executor import TestExecutor

logger = logging.getLogger(__name__)


# ============================================================================
# Security Policy for Generated Test Code
# ============================================================================

DISALLOWED_MODULES = {
    "subprocess",
    "pty",
    "ctypes",
    "winreg",
    "msvcrt",
    "socket",
    "urllib.request",
    "http.client",
    "ftplib",
    "smtplib",
    "telnetlib",
    "xmlrpc",
    "pickle",
    "shelve",
}

DISALLOWED_CALLS = {
    "eval",
    "exec",
    "__import__",
    "compile",
}

DISALLOWED_ATTR_CALLS = {
    ("os", "system"),
    ("os", "popen"),
    ("os", "spawn"),
    ("os", "spawnl"),
    ("os", "spawnle"),
    ("os", "spawnlp"),
    ("os", "spawnlpe"),
    ("os", "spawnv"),
    ("os", "spawnve"),
    ("os", "spawnvp"),
    ("os", "spawnvpe"),
    ("os", "execl"),
    ("os", "execle"),
    ("os", "execlp"),
    ("os", "execlpe"),
    ("os", "execv"),
    ("os", "execve"),
    ("os", "execvp"),
    ("os", "execvpe"),
    ("os", "remove"),
    ("os", "unlink"),
    ("os", "rmdir"),
    ("shutil", "rmtree"),
}

FORBIDDEN_STRING_TOKENS = [
    "docker.sock",
    "docker_engine",
    "/etc/passwd",
    "/etc/shadow",
    "c:\\windows",
    "c:\\program files",
]


class TestCodeValidationError(Exception):
    """Raised when generated test code fails static syntax or security validation."""
    pass


class TestCodeSecurityValidator:
    """
    Static AST analyzer to validate generated test code before execution.
    Rejects syntax errors, empty code, dangerous imports, process spawns, and escape tokens.
    
    IMPORTANT DEFENSE-IN-DEPTH NOTICE:
    Static analysis is a preliminary hygiene filter, not a complete security boundary.
    Containerized execution inside DockerSandbox remains strictly mandatory.
    """

    @classmethod
    def validate(cls, code: str, is_test_file: bool = True) -> Tuple[bool, List[str]]:
        """
        Validates Python code against syntax rules and security constraints.

        Args:
            code: Python source code string.
            is_test_file: If True, requires at least one function definition.

        Returns:
            Tuple[bool, List[str]]: (is_valid, list of violation descriptions).
        """
        if not code or not code.strip():
            return False, ["Test code is empty."]

        # 1. Verify Syntax
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, [f"Invalid Python syntax: {e.msg} at line {e.lineno}"]

        violations: List[str] = []

        # 2. Verify Presence of Functions (only required for test files)
        if is_test_file:
            functions = [
                node for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if not functions:
                violations.append("No functions defined in generated test code.")

        # 3. Walk AST to Detect Security Violations
        for node in ast.walk(tree):
            # Check module imports: import X
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod_name = alias.name.split(".")[0].lower()
                    if mod_name in DISALLOWED_MODULES:
                        violations.append(f"Security violation: Disallowed import '{alias.name}'.")

            # Check from X import Y
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    mod_name = node.module.split(".")[0].lower()
                    if mod_name in DISALLOWED_MODULES:
                        violations.append(f"Security violation: Disallowed import from '{node.module}'.")

            # Check function calls: eval(), exec(), os.system()
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in DISALLOWED_CALLS:
                        violations.append(f"Security violation: Disallowed call to '{node.func.id}()'.")
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        caller = node.func.value.id
                        method = node.func.attr
                        if (caller, method) in DISALLOWED_ATTR_CALLS or (
                            caller == "os" and ("exec" in method or "spawn" in method)
                        ):
                            violations.append(f"Security violation: Disallowed system call '{caller}.{method}()'.")

            # Check string literals for sensitive tokens
            elif isinstance(node, ast.Constant):
                if isinstance(node.value, str):
                    val_lower = node.value.lower()
                    for forbidden in FORBIDDEN_STRING_TOKENS:
                        if forbidden in val_lower:
                            violations.append(
                                f"Security violation: Forbidden string token '{forbidden}' found in test code."
                            )

        if violations:
            return False, violations

        return True, []


# ============================================================================
# Gemini Configuration & Model Initializer
# ============================================================================

def get_gemini_model(model_name: Optional[str] = None) -> Any:
    """
    Initializes and returns a Gemini GenerativeModel configured for JSON output.

    Args:
        model_name: Optional override for LLM model name.

    Returns:
        GenerativeModel: Configured Gemini instance.

    Raises:
        ValueError: If GEMINI_API_KEY is not configured in backend.config.
    """
    api_key = config.GEMINI_API_KEY
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not configured. Set GEMINI_API_KEY in your environment or .env file."
        )

    if genai is None:
        raise ImportError(
            "google-generativeai is not installed in the current environment."
        )

    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 0.2,
        "response_mime_type": "application/json",
    }

    selected_model = model_name or config.LLM_MODEL
    return genai.GenerativeModel(
        model_name=selected_model,
        generation_config=generation_config,
    )


# ============================================================================
# AI Testing Agent Implementation
# ============================================================================

class TestingAgent:
    """
    AI-powered Testing Agent.
    Synthesizes unit, negative, boundary, and edge test cases from requirements,
    validates the code statically, and executes it via TestExecutor in DockerSandbox.
    """

    __test__ = False  # Prevent pytest collection

    def __init__(
        self,
        model: Optional[Any] = None,
        executor: Optional[TestExecutor] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the Testing Agent.

        Args:
            model: Optional injected Gemini model instance (for testing and mocking).
            executor: Optional injected TestExecutor instance.
            model_name: Optional override for Gemini model name.
        """
        self._injected_model = model
        self.executor = executor or TestExecutor()
        self.model_name = model_name or config.LLM_MODEL

    @property
    def model(self) -> Any:
        """Lazily initialize or return the Gemini model."""
        if self._injected_model is not None:
            return self._injected_model
        return get_gemini_model(self.model_name)

    def build_system_prompt(
        self,
        requirement_spec: Optional[Dict[str, Any]] = None,
        generated_files: Optional[List[Dict[str, str]]] = None,
        acceptance_criteria: Optional[List[str]] = None,
    ) -> str:
        """Constructs structured prompt instructing Gemini to produce valid Pytest suites."""
        req_title = "Software Project"
        func_reqs = []
        acc_criteria = acceptance_criteria or []

        if requirement_spec:
            req_title = requirement_spec.get("project_title", req_title)
            func_reqs = requirement_spec.get("functional_requirements", [])
            if not acc_criteria:
                acc_criteria = requirement_spec.get("acceptance_criteria", [])

        code_sections: List[str] = []
        if generated_files:
            for gf in generated_files:
                path = gf.get("path", "file.py")
                content = gf.get("content", "")
                code_sections.append(f"--- File: {path} ---\n{content}")
        code_str = "\n\n".join(code_sections) if code_sections else "# No generated source files available."

        func_str = "\n".join(f"- {r}" for r in func_reqs) if func_reqs else "- Implement standard CRUD and business logic."
        crit_str = "\n".join(f"- {c}" for c in acc_criteria) if acc_criteria else "- All unit and edge test cases pass cleanly."

        prompt = f"""
You are an expert Senior QA Automation Engineer and Security Testing Specialist.
Analyze the software requirements and generated source code below, and generate a comprehensive Pytest test suite.

PROJECT: {req_title}

FUNCTIONAL REQUIREMENTS:
{func_str}

ACCEPTANCE CRITERIA:
{crit_str}

SOURCE CODE FILES:
{code_str}

You MUST return a JSON object with this exact schema:
{{
    "test_file_path": "tests/test_api.py",
    "scenarios_summary": "Comprehensive unit test suite covering positive, negative, boundary, and edge test cases",
    "test_cases": [
        {{
            "name": "test_feature_positive",
            "category": "positive",
            "description": "Verify standard happy path execution with valid inputs",
            "expected_outcome": "Returns expected status and correct output"
        }},
        {{
            "name": "test_feature_negative",
            "category": "negative",
            "description": "Verify error handling and rejection of invalid data",
            "expected_outcome": "Raises expected error or returns 4xx status"
        }},
        {{
            "name": "test_feature_boundary",
            "category": "boundary",
            "description": "Verify min, max, zero, and boundary values",
            "expected_outcome": "Handles boundary values accurately without crashing"
        }},
        {{
            "name": "test_feature_edge_case",
            "category": "edge_case",
            "description": "Verify special characters, empty inputs, and nulls",
            "expected_outcome": "Safe failure or validation message"
        }}
    ],
    "test_code": "import pytest\\n\\ndef test_sample():\\n    assert True\\n"
}}

STRICT RULES FOR GENERATED TEST CODE:
1. Output MUST be valid Python code using standard `pytest`.
2. Every test function name MUST start with `test_`.
3. NEVER import or use `subprocess`, `os.system`, or shell commands.
4. NEVER use `socket` or raw external network requests.
5. Use mocks (`unittest.mock.MagicMock`, `patch`) for external services, files, or databases.
6. Target test_file_path should always be relative, e.g. 'tests/test_main.py'.
7. Do not include markdown code fences (```json or ```python) in the JSON fields.
"""
        return prompt.strip()

    def generate_fallback_test_suite(
        self,
        generated_files: Optional[List[Dict[str, str]]] = None,
        error_context: str = "Fallback test suite generated due to LLM unavailability",
    ) -> GeneratedTestSuite:
        """Generates a reliable, safe baseline test suite when Gemini is offline or unreachable."""
        test_cases = [
            TestCaseSpec(
                name="test_environment_sanity",
                category="positive",
                description="Verify testing environment and pytest execution sanity",
                expected_outcome="Basic assertions succeed cleanly",
            ),
            TestCaseSpec(
                name="test_generated_files_presence",
                category="boundary",
                description="Verify that expected project files exist in the workspace",
                expected_outcome="All generated project files exist on disk",
            ),
        ]

        file_checks = []
        if generated_files:
            for gf in generated_files:
                p = gf.get("path", "")
                if p:
                    file_checks.append(f'    assert (workspace / "{p}").exists(), "Missing file: {p}"')

        file_check_body = "\n".join(file_checks) if file_checks else "    assert True"

        test_code = f'''"""
Baseline Pytest Suite (Fallback).
Auto-generated test suite for workspace verification.
"""

from pathlib import Path
import pytest


def test_environment_sanity():
    """Verify standard test assertions run cleanly."""
    assert 1 + 1 == 2
    assert True is not False


def test_generated_files_presence():
    """Verify that expected source files are present in the workspace."""
    workspace = Path(".")
{file_check_body}
'''
        return GeneratedTestSuite(
            test_file_path="tests/test_sanity.py",
            scenarios_summary=f"Fallback baseline test suite: {error_context}",
            test_cases=test_cases,
            test_code=test_code,
        )

    def generate_tests(
        self,
        requirement_spec: Optional[Dict[str, Any]] = None,
        generated_files: Optional[List[Dict[str, str]]] = None,
        acceptance_criteria: Optional[List[str]] = None,
    ) -> GeneratedTestSuite:
        """
        Invokes Gemini to synthesize a structured test suite from requirements and source code.

        Args:
            requirement_spec: Requirement specification dictionary.
            generated_files: List of generated source files.
            acceptance_criteria: Optional list of acceptance criteria.

        Returns:
            GeneratedTestSuite: Structured test suite with validated Python code.
        """
        prompt = self.build_system_prompt(
            requirement_spec=requirement_spec,
            generated_files=generated_files,
            acceptance_criteria=acceptance_criteria,
        )

        try:
            model_instance = self.model
            response = model_instance.generate_content(prompt)

            if not response or not hasattr(response, "text") or not response.text:
                raise ValueError("Empty response received from Gemini model.")

            raw_text = response.text.strip()
            # Remove any markdown code fences if accidentally included
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\n?", "", raw_text)
                raw_text = re.sub(r"\n?```$", "", raw_text)

            parsed_dict = json.loads(raw_text)
            suite = GeneratedTestSuite(**parsed_dict)

            # Validate static security & syntax
            is_valid, violations = TestCodeSecurityValidator.validate(suite.test_code)
            if not is_valid:
                err_msg = "; ".join(violations)
                logger.warning(f"[Testing Agent] Generated code failed security validation: {err_msg}")
                raise TestCodeValidationError(f"Generated test code violated security rules: {err_msg}")

            return suite

        except (TestCodeValidationError, Exception) as exc:
            err_str = str(exc)
            logger.warning(f"[Testing Agent] Test generation fallback triggered: {err_str}")
            return self.generate_fallback_test_suite(
                generated_files=generated_files,
                error_context=f"LLM generation failed: {err_str}",
            )

    def save_test_file(
        self,
        workspace_path: Union[str, Path],
        test_suite: GeneratedTestSuite,
    ) -> Path:
        """
        Safely writes the validated test code into the designated workspace under tests/.

        Args:
            workspace_path: Target project directory.
            test_suite: Validated GeneratedTestSuite instance.

        Returns:
            Path: Full resolved path to the written test file.

        Raises:
            DockerSecurityViolationError: If test_file_path attempts path traversal outside workspace.
        """
        resolved_ws = Path(workspace_path).resolve()
        test_rel = Path(test_suite.test_file_path)

        # Enforce relative path within workspace
        dest_path = (resolved_ws / test_rel).resolve()
        try:
            dest_path.relative_to(resolved_ws)
        except ValueError as exc:
            raise DockerSecurityViolationError(
                f"Security violation: Test path '{test_suite.test_file_path}' escapes workspace boundary."
            ) from exc

        # Ensure parent directories exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(test_suite.test_code, encoding="utf-8")
        return dest_path

    def run(self, state: AgentState) -> AgentState:
        """
        Executes the Testing Agent node in the LangGraph Multi-Agent pipeline.

        Args:
            state: Central AgentState dictionary.

        Returns:
            AgentState: Updated AgentState with test results, logs, and next agent status.
        """
        # Initialize logs safely
        if "logs" not in state or state["logs"] is None:
            state["logs"] = []

        state["current_agent"] = "Testing Agent"
        state["logs"].append("[Testing Agent] Initiating AI-driven test synthesis and sandbox execution.")

        workspace_str = state.get("workspace_path") or config.WORKSPACE_DIR
        workspace_path = Path(workspace_str).resolve()
        workspace_path.mkdir(parents=True, exist_ok=True)

        req_spec = state.get("requirement_spec")
        generated_files = state.get("generated_files") or []

        # 1. Write any generated files to workspace if present and missing
        for gf in generated_files:
            rel_p = gf.get("path")
            content = gf.get("content")
            if rel_p and content is not None:
                target_file = (workspace_path / rel_p).resolve()
                try:
                    target_file.relative_to(workspace_path)
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    target_file.write_text(content, encoding="utf-8")
                except Exception:
                    pass

        # 2. Synthesize test suite with Gemini
        state["logs"].append(
            f"[Testing Agent] Generating Pytest test cases using model '{self.model_name}'..."
        )
        test_suite = self.generate_tests(
            requirement_spec=req_spec,
            generated_files=generated_files,
        )

        state["logs"].append(
            f"[Testing Agent] Generated {len(test_suite.test_cases)} test scenarios: "
            f"{test_suite.scenarios_summary[:100]}"
        )

        # 3. Save test file safely into workspace
        try:
            saved_path = self.save_test_file(workspace_path, test_suite)
            state["logs"].append(f"[Testing Agent] Saved verified test suite to '{test_suite.test_file_path}'.")
        except Exception as e:
            state["logs"].append(f"[Testing Agent Error] Failed to write test file: {str(e)}")
            saved_path = None

        # 4. Execute tests inside Docker Sandbox via TestExecutor
        state["logs"].append("[Testing Agent] Executing Pytest suite inside Docker Sandbox...")
        try:
            rel_test_path = test_suite.test_file_path if saved_path else None
            test_paths = [rel_test_path] if rel_test_path else None

            test_result: TestResult = self.executor.run_tests(
                workspace_path=workspace_path,
                test_file_paths=test_paths,
            )

            # Update state with outcome
            state = self.executor.update_agent_state(state, test_result)

        except Exception as e:
            error_msg = f"[Testing Agent Error] Sandbox execution failed: {str(e)}"
            state["logs"].append(error_msg)
            fallback_res = TestResult(
                passed=False,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                raw_output=str(e),
                error_summary=error_msg,
            )
            state = self.executor.update_agent_state(state, fallback_res)

        return state


def run_testing_agent(state: AgentState) -> AgentState:
    """
    Standard entrypoint function for Member 3 Testing Agent.
    Compatible with LangGraph StateGraph node registration.
    """
    agent = TestingAgent()
    return agent.run(state)
