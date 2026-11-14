
import pytest
from pathlib import Path
from unittest.mock import Mock

from backend.agents import code_gen_agent as agent
from backend.agents.code_gen_agent import (
    GeneratedFilesResponse,
    validate_generated_files,
)
from backend.schemas.state import GeneratedFile
from backend.core import file_manager


# --------------------------------------------------
# Test data
# --------------------------------------------------

@pytest.fixture
def sample_state():
    """Provide a valid AgentState for testing."""

    return {
        "project_id": "test_calculator",
        "user_prompt": "Create a Python calculator",

        "requirement_spec": {
            "project_title": "Python Calculator",
            "functional_requirements": [
                "Addition",
                "Subtraction",
            ],
            "non_functional_requirements": [
                "Readable code",
            ],
            "tech_stack": ["Python"],
            "acceptance_criteria": [
                "Calculator performs arithmetic operations",
            ],
        },

        "architecture_blueprint": {
            "directory_tree": ["main.py"],
            "api_routes": [],
            "database_tables": [],
            "files_to_generate": ["main.py"],
        },

        "generated_files": [],
        "workspace_path": "",
        "test_results": None,
        "retry_count": 0,
        "is_test_passed": False,
        "bug_diagnosis": None,
        "patched_files": [],
        "security_report": None,
        "current_agent": "",
        "logs": [],
        "status": "pending",
    }


@pytest.fixture
def mock_gemini(monkeypatch):
    """Mock Gemini so tests never make real API calls."""

    generated_response = GeneratedFilesResponse(
        files=[
            GeneratedFile(
                path="main.py",
                content="print('Hello AI DevTeam')",
                language="python",
            )
        ]
    )

    structured_llm = Mock()
    structured_llm.invoke.return_value = generated_response

    fake_llm = Mock()
    fake_llm.with_structured_output.return_value = structured_llm

    monkeypatch.setattr(
        agent,
        "ChatGoogleGenerativeAI",
        lambda **kwargs: fake_llm,
    )

    monkeypatch.setattr(
        agent.config,
        "GEMINI_API_KEY",
        "test-api-key",
    )

    return structured_llm


# --------------------------------------------------
# 1. Successful code generation
# --------------------------------------------------

def test_successful_code_generation(
    sample_state,
    mock_gemini,
    monkeypatch,
    tmp_path,
):

    monkeypatch.setattr(
        agent,
        "create_project_directory",
        lambda project_id: tmp_path / project_id,
    )

    write_mock = Mock(
        side_effect=lambda project_id, file_path, content:
        tmp_path / project_id / file_path
    )

    monkeypatch.setattr(agent, "write_file", write_mock)

    result = agent.run_code_generation_agent(sample_state)

    assert result["status"] == "completed"
    assert result["current_agent"] == "Code Generation Agent"
    assert len(result["generated_files"]) == 1
    assert result["generated_files"][0]["path"] == "main.py"
    assert result["workspace_path"] == str(
        tmp_path / "test_calculator"
    )

    write_mock.assert_called_once()


# --------------------------------------------------
# 2. Missing project ID
# --------------------------------------------------

def test_missing_project_id(sample_state, monkeypatch):

    sample_state["project_id"] = ""

    monkeypatch.setattr(
        agent.config,
        "GEMINI_API_KEY",
        "test-api-key",
    )

    result = agent.run_code_generation_agent(sample_state)

    assert result["status"] == "failed"
    assert "Project ID is missing." in result["logs"][-1]


# --------------------------------------------------
# 3. Missing requirement specification
# --------------------------------------------------

def test_missing_requirement_spec(sample_state, monkeypatch):

    sample_state["requirement_spec"] = None

    monkeypatch.setattr(
        agent.config,
        "GEMINI_API_KEY",
        "test-api-key",
    )

    result = agent.run_code_generation_agent(sample_state)

    assert result["status"] == "failed"
    assert "Requirement specification is missing." in result["logs"][-1]


# --------------------------------------------------
# 4. Missing architecture blueprint
# --------------------------------------------------

def test_missing_architecture(sample_state, monkeypatch):

    sample_state["architecture_blueprint"] = None

    monkeypatch.setattr(
        agent.config,
        "GEMINI_API_KEY",
        "test-api-key",
    )

    result = agent.run_code_generation_agent(sample_state)

    assert result["status"] == "failed"
    assert "Architecture blueprint is missing." in result["logs"][-1]


# --------------------------------------------------
# 5. Missing Gemini API key
# --------------------------------------------------

def test_missing_api_key(sample_state, monkeypatch):

    monkeypatch.setattr(
        agent.config,
        "GEMINI_API_KEY",
        None,
    )

    result = agent.run_code_generation_agent(sample_state)

    assert result["status"] == "failed"
    assert "Gemini API key is not configured." in result["logs"][-1]


# --------------------------------------------------
# 6. Invalid Gemini response
# --------------------------------------------------

def test_invalid_gemini_response(sample_state, mock_gemini):

    mock_gemini.invoke.return_value = "Invalid response"

    result = agent.run_code_generation_agent(sample_state)

    assert result["status"] == "failed"
    assert (
        "Gemini returned an invalid structured response."
        in result["logs"][-1]
    )


# --------------------------------------------------
# 7. Gemini generates no files
# --------------------------------------------------

def test_empty_generated_files(sample_state, mock_gemini):

    mock_gemini.invoke.return_value = GeneratedFilesResponse(
        files=[]
    )

    result = agent.run_code_generation_agent(sample_state)

    assert result["status"] == "failed"
    assert "Gemini generated no project files." in result["logs"][-1]


# --------------------------------------------------
# 8. Missing required file
# --------------------------------------------------

def test_missing_required_file(sample_state, mock_gemini):

    mock_gemini.invoke.return_value = GeneratedFilesResponse(
        files=[
            GeneratedFile(
                path="app.py",
                content="print('Hello')",
            )
        ]
    )

    result = agent.run_code_generation_agent(sample_state)

    assert result["status"] == "failed"
    assert "Missing required files" in result["logs"][-1]


# --------------------------------------------------
# 9. Unexpected generated file
# --------------------------------------------------

def test_unexpected_generated_file(sample_state, mock_gemini):

    mock_gemini.invoke.return_value = GeneratedFilesResponse(
        files=[
            GeneratedFile(
                path="main.py",
                content="print('Hello')",
            ),
            GeneratedFile(
                path="extra.py",
                content="print('Extra')",
            ),
        ]
    )

    result = agent.run_code_generation_agent(sample_state)

    assert result["status"] == "failed"
    assert "Unexpected files generated" in result["logs"][-1]


# --------------------------------------------------
# 10. Duplicate generated file paths
# --------------------------------------------------

def test_duplicate_generated_paths(sample_state, mock_gemini):

    mock_gemini.invoke.return_value = GeneratedFilesResponse(
        files=[
            GeneratedFile(
                path="main.py",
                content="print('First')",
            ),
            GeneratedFile(
                path="main.py",
                content="print('Second')",
            ),
        ]
    )

    result = agent.run_code_generation_agent(sample_state)

    assert result["status"] == "failed"
    assert "Gemini generated duplicate file paths." in result["logs"][-1]


# --------------------------------------------------
# 11. File writing failure
# --------------------------------------------------

def test_file_writing_failure(
    sample_state,
    mock_gemini,
    monkeypatch,
    tmp_path,
):

    monkeypatch.setattr(
        agent,
        "create_project_directory",
        lambda project_id: tmp_path / project_id,
    )

    def failing_write(*args, **kwargs):
        raise OSError("Unable to write file")

    monkeypatch.setattr(agent, "write_file", failing_write)

    result = agent.run_code_generation_agent(sample_state)

    assert result["status"] == "failed"
    assert "Unable to write file" in result["logs"][-1]


# --------------------------------------------------
# 12. Empty generated file content
# --------------------------------------------------

def test_empty_file_content(
    sample_state,
    mock_gemini,
    monkeypatch,
    tmp_path,
):

    mock_gemini.invoke.return_value = GeneratedFilesResponse(
        files=[
            GeneratedFile(
                path="main.py",
                content="   ",
            )
        ]
    )

    monkeypatch.setattr(
        agent,
        "create_project_directory",
        lambda project_id: tmp_path / project_id,
    )

    result = agent.run_code_generation_agent(sample_state)

    assert result["status"] == "failed"
    assert "Generated file is empty" in result["logs"][-1]


# --------------------------------------------------
# 13. Valid generated Python file
# --------------------------------------------------

def test_validate_valid_python_file(tmp_path, monkeypatch):

    monkeypatch.setattr(
        file_manager,
        "BASE_WORKSPACE",
        tmp_path,
    )

    files = [
        GeneratedFile(
            path="main.py",
            content="def main():\n    print('Hello')\n\nmain()",
        )
    ]

    result = validate_generated_files(
        project_id="test_project",
        expected_files=["main.py"],
        generated_files=files,
    )

    assert len(result) == 1
    assert result[0].path == "main.py"


# --------------------------------------------------
# 14. Invalid Python syntax
# --------------------------------------------------

def test_reject_invalid_python_syntax(tmp_path, monkeypatch):

    monkeypatch.setattr(
        file_manager,
        "BASE_WORKSPACE",
        tmp_path,
    )

    files = [
        GeneratedFile(
            path="main.py",
            content="def main(\n    print('Hello')",
        )
    ]

    with pytest.raises(ValueError, match="Python syntax error"):

        validate_generated_files(
            project_id="test_project",
            expected_files=["main.py"],
            generated_files=files,
        )


# --------------------------------------------------
# 15. Empty generated file path
# --------------------------------------------------

def test_reject_empty_file_path():

    files = [
        GeneratedFile(
            path="   ",
            content="print('Hello')",
        )
    ]

    with pytest.raises(ValueError, match="path cannot be empty"):

        validate_generated_files(
            project_id="test_project",
            expected_files=["main.py"],
            generated_files=files,
        )


# --------------------------------------------------
# 16. Empty architecture file list
# --------------------------------------------------

def test_reject_empty_architecture_file_list():

    files = [
        GeneratedFile(
            path="main.py",
            content="print('Hello')",
        )
    ]

    with pytest.raises(ValueError, match="Architecture does not specify"):

        validate_generated_files(
            project_id="test_project",
            expected_files=[],
            generated_files=files,
        )


# --------------------------------------------------
# 17. Duplicate architecture paths
# --------------------------------------------------

def test_reject_duplicate_architecture_paths():

    files = [
        GeneratedFile(
            path="main.py",
            content="print('Hello')",
        )
    ]

    with pytest.raises(ValueError, match="Architecture contains duplicate"):

        validate_generated_files(
            project_id="test_project",
            expected_files=["main.py", "main.py"],
            generated_files=files,
        )


# --------------------------------------------------
# 18. Reject unsafe generated file path
# --------------------------------------------------

def test_reject_unsafe_generated_file_path(tmp_path, monkeypatch):

    monkeypatch.setattr(
        file_manager,
        "BASE_WORKSPACE",
        tmp_path,
    )

    files = [
        GeneratedFile(
            path="../secret.py",
            content="print('Hello')",
        )
    ]

    with pytest.raises(ValueError, match="Path traversal"):

        validate_generated_files(
            project_id="test_project",
            expected_files=["../secret.py"],
            generated_files=files,
        )


# --------------------------------------------------
# 19. Normalize Windows-style file paths
# --------------------------------------------------

def test_normalize_windows_file_path(tmp_path, monkeypatch):

    monkeypatch.setattr(
        file_manager,
        "BASE_WORKSPACE",
        tmp_path,
    )

    files = [
        GeneratedFile(
            path="models\\student.py",
            content="class Student:\n    pass",
        )
    ]

    result = validate_generated_files(
        project_id="test_project",
        expected_files=["models\\student.py"],
        generated_files=files,
    )

    assert result[0].path == "models/student.py"


# --------------------------------------------------
# 20. Invalid Python prevents all file writes
# --------------------------------------------------

def test_invalid_python_prevents_all_file_writes(
    sample_state,
    mock_gemini,
    monkeypatch,
    tmp_path,
):

    mock_gemini.invoke.return_value = GeneratedFilesResponse(
        files=[
            GeneratedFile(
                path="main.py",
                content="print('Valid file')",
            ),
            GeneratedFile(
                path="helper.py",
                content="def broken(\n",
            ),
        ]
    )

    sample_state["architecture_blueprint"]["files_to_generate"] = [
        "main.py",
        "helper.py",
    ]

    write_mock = Mock()

    create_directory_mock = Mock(
        return_value=tmp_path / "test_project"
    )

    monkeypatch.setattr(
        agent,
        "write_file",
        write_mock,
    )

    monkeypatch.setattr(
        agent,
        "create_project_directory",
        create_directory_mock,
    )

    monkeypatch.setattr(
        agent,
        "get_safe_file_path",
        lambda project_id, file_path: tmp_path / file_path,
    )

    result = agent.run_code_generation_agent(sample_state)

    assert result["status"] == "failed"
    assert "Python syntax error" in result["logs"][-1]

    # No file should be saved if any generated file is invalid.
    write_mock.assert_not_called()

    # Workspace creation should not start.
    create_directory_mock.assert_not_called()

    assert result["generated_files"] == []
