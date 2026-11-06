"""
AgentState Definition for Multi-Agent AI DevTeam / AutoDev.
This file serves as the unified data contract between all 6 specialized agents:
1. Requirement Analysis Agent
2. System Design Agent
3. Code Generation Agent
4. Testing Agent
5. Bug Detection Agent
6. Code Review Agent
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class GeneratedFile(BaseModel):
    """Schema for an individual generated code file."""
    path: str = Field(description="Relative file path, e.g. 'app/main.py' or 'tests/test_api.py'")
    content: str = Field(description="Source code content of the file")
    language: str = Field(default="python", description="Programming language of the file")


class RequirementSpec(BaseModel):
    """Structured output from Requirement Analysis Agent."""
    project_title: str
    functional_requirements: List[str]
    non_functional_requirements: List[str]
    tech_stack: List[str]
    acceptance_criteria: List[str]


class ArchitectureBlueprint(BaseModel):
    """Structured output from System Design Agent."""
    directory_tree: List[str]
    api_routes: List[Dict[str, str]]
    database_tables: List[str]
    files_to_generate: List[str]


class TestResult(BaseModel):
    """Execution results from Testing Agent running Pytest in Docker."""
    passed: bool
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    raw_output: str = ""
    error_summary: Optional[str] = None


class SecurityReviewReport(BaseModel):
    """Security audit & quality report from Code Review Agent."""
    quality_score: int = Field(default=100, description="Score out of 100")
    owasp_compliance: bool = True
    vulnerabilities: List[str] = []
    suggestions: List[str] = []


class AgentState(TypedDict):
    """
    LangGraph Central State Object passed across all nodes in the state graph machine.
    """
    # User Input
    project_id: str
    user_prompt: str
    
    # Member 1 Outputs
    requirement_spec: Optional[Dict[str, Any]]
    architecture_blueprint: Optional[Dict[str, Any]]
    
    # Member 2 Outputs
    generated_files: List[Dict[str, str]]  # List of {"path": ..., "content": ...}
    workspace_path: str
    
    # Member 3 Outputs
    test_results: Optional[Dict[str, Any]]
    retry_count: int
    is_test_passed: bool
    
    # Member 3 (Bug Agent) Outputs
    bug_diagnosis: Optional[str]
    patched_files: List[Dict[str, str]]
    
    # Member 4 Outputs
    security_report: Optional[Dict[str, Any]]
    
    # Logging & Stepper Status
    current_agent: str
    logs: List[str]
    status: str  # "analyzing" | "designing" | "coding" | "testing" | "debugging" | "reviewing" | "completed" | "failed"