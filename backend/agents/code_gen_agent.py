
"""
Code Generation Agent for Multi-Agent AI DevTeam.

Generates source code using Google Gemini API,
validates generated files, and stores them securely
inside the project workspace.
"""

import ast
import json
from typing import List

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from backend.schemas.state import AgentState, GeneratedFile
from backend.config import config

from backend.core.file_manager import (
    write_file,
    create_project_directory,
    get_safe_file_path,
)


class GeneratedFilesResponse(BaseModel):
    """Schema for the complete code generation response."""

    files: List[GeneratedFile] = Field(
        description="List of all generated project files"
    )


def normalize_file_path(file_path: str) -> str:
    """Normalize file paths to use forward slashes."""

    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("Generated file path cannot be empty.")

    return file_path.strip().replace("\\", "/")


def validate_generated_files(
    project_id: str,
    expected_files: List[str],
    generated_files: List[GeneratedFile],
) -> List[GeneratedFile]:
    """
    Validate every generated file before saving.

    Checks:
    1. Architecture file list.
    2. Path normalization.
    3. Duplicate paths.
    4. Missing required files.
    5. Unexpected files.
    6. Empty file content.
    7. Unsafe file paths.
    8. Python syntax.
    """

    # 1. Validate architecture file list.
    if not isinstance(expected_files, list) or not expected_files:
        raise ValueError(
            "Architecture does not specify valid files to generate."
        )

    if not isinstance(generated_files, list) or not generated_files:
        raise ValueError("Gemini generated no project files.")

    # 2. Normalize expected architecture paths.
    expected_paths = [
        normalize_file_path(path)
        for path in expected_files
    ]

    # 3. Reject duplicate architecture paths.
    if len(expected_paths) != len(set(expected_paths)):
        raise ValueError(
            "Architecture contains duplicate file paths."
        )

    # 4. Normalize and validate generated files.
    normalized_files = []

    for generated_file in generated_files:

        normalized_path = normalize_file_path(
            generated_file.path
        )

        file_name = normalized_path.rsplit("/", 1)[-1]

        # Empty __init__.py files are valid package markers.
        if (
            not generated_file.content.strip()
            and file_name != "__init__.py"
        ):
            raise ValueError(
                f"Generated file is empty: {normalized_path}"
            )

        normalized_files.append(
            generated_file.model_copy(
                update={"path": normalized_path}
            )
        )

    # 5. Extract generated paths.
    generated_paths = [
        file.path for file in normalized_files
    ]

    # 6. Reject duplicate generated paths.
    if len(generated_paths) != len(set(generated_paths)):
        raise ValueError(
            "Gemini generated duplicate file paths."
        )

    expected_path_set = set(expected_paths)
    generated_path_set = set(generated_paths)

    # 7. Check for missing required files.
    missing_files = expected_path_set - generated_path_set

    if missing_files:
        raise ValueError(
            f"Missing required files: {sorted(missing_files)}"
        )

    # 8. Check unexpected files.
    unexpected_files = generated_path_set - expected_path_set

    # Allow only additional package markers and .env.example.
    allowed_support_files = {
        path
        for path in unexpected_files
        if (
            path.rsplit("/", 1)[-1] == "__init__.py"
            or path == ".env.example"
        )
    }

    rejected_files = unexpected_files - allowed_support_files

    if rejected_files:
        raise ValueError(
            f"Unexpected files generated: {sorted(rejected_files)}"
        )

    # 9. Validate path safety and Python syntax.
    # All files are validated before any file is saved.
    for generated_file in normalized_files:

        get_safe_file_path(
            project_id=project_id,
            file_path=generated_file.path,
        )

        # Parse Python files without executing them.
        if generated_file.path.endswith(".py"):

            try:
                ast.parse(
                    generated_file.content,
                    filename=generated_file.path,
                )

            except SyntaxError as error:
                raise ValueError(
                    f"Python syntax error in "
                    f"{generated_file.path}: {error}"
                ) from error

    return normalized_files


def run_code_generation_agent(state: AgentState) -> AgentState:
    """
    Generate project source code using Gemini.

    Steps:
    1. Read requirements and architecture.
    2. Send information to Gemini.
    3. Receive structured generated files.
    4. Validate generated files.
    5. Save validated files in workspace.
    6. Update AgentState.
    """

    # STEP 1: Initialize agent state.
    state["current_agent"] = "Code Generation Agent"
    state["status"] = "coding"

    if "logs" not in state or state["logs"] is None:
        state["logs"] = []

    state["generated_files"] = []
    state["workspace_path"] = ""

    state["logs"].append(
        "[Code Generation Agent] Starting code generation..."
    )

    try:

        # STEP 2: Extract information from AgentState.
        project_id = state.get("project_id")
        user_prompt = state.get("user_prompt", "")

        requirement_spec = state.get("requirement_spec")

        architecture_blueprint = state.get(
            "architecture_blueprint"
        )

        # STEP 3: Validate project ID.
        if project_id is None or project_id == "":
            raise ValueError("Project ID is missing.")

        if not isinstance(project_id, str) or not project_id.strip():
            raise ValueError("Project ID is invalid.")

        # STEP 4: Validate requirements and architecture.
        if not requirement_spec:
            raise ValueError(
                "Requirement specification is missing."
            )

        if not architecture_blueprint:
            raise ValueError(
                "Architecture blueprint is missing."
            )

        if not isinstance(architecture_blueprint, dict):
            raise ValueError(
                "Architecture blueprint must be a dictionary."
            )

        # STEP 5: Validate Gemini API key.
        if not config.GEMINI_API_KEY:
            raise ValueError(
                "Gemini API key is not configured."
            )

        # STEP 6: Initialize Gemini LLM.
        llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            google_api_key=config.GEMINI_API_KEY,
        )

        # STEP 7: Enable structured output.
        structured_llm = llm.with_structured_output(
            GeneratedFilesResponse
        )

        # STEP 8: Create code generation prompt.
        prompt_template = PromptTemplate(
            template="""You are an expert Senior Software Engineer.

Your task is to generate complete, functional source code
based on the requirements and architecture provided below.

USER REQUEST:
{user_prompt}

REQUIREMENT SPECIFICATION:
{requirement_spec}

ARCHITECTURE BLUEPRINT:
{architecture_blueprint}

INSTRUCTIONS:

1. Follow the provided architecture strictly.
2. Generate every file listed in files_to_generate.
3. Generate complete source code, not pseudocode.
4. Follow the specified technology stack.
5. Use proper imports, naming conventions and modularity.
6. Include appropriate error handling.
7. Ensure all generated files are compatible with each other.
8. Do not generate unnecessary files.
9. Put complete source code inside each file's content field.
10. Do not include Markdown code fences inside file content.
11. Ensure every Python file has valid Python syntax.
12. Empty __init__.py files are allowed when appropriate.
13. Additional __init__.py package marker files are allowed.
14. An optional .env.example configuration template is allowed.
15. Do not leave other required source files empty.
16. Do not generate any other files outside the architecture,
    except the allowed support files above.
17. Return output matching this JSON schema:

{format_instructions}
""",
            input_variables=[
                "user_prompt",
                "requirement_spec",
                "architecture_blueprint",
            ],
            partial_variables={
                "format_instructions": json.dumps(
                    GeneratedFilesResponse.model_json_schema(),
                    indent=2,
                )
            },
        )

        # STEP 9: Format prompt with project information.
        formatted_prompt = prompt_template.format(
            user_prompt=user_prompt,
            requirement_spec=json.dumps(
                requirement_spec,
                indent=2,
                default=str,
            ),
            architecture_blueprint=json.dumps(
                architecture_blueprint,
                indent=2,
                default=str,
            ),
        )

        state["logs"].append(
            "[Code Generation Agent] Sending architecture to Gemini..."
        )

        # STEP 10: Generate structured response.
        parsed_response = structured_llm.invoke(
            formatted_prompt
        )

        # Handle dictionary responses.
        if isinstance(parsed_response, dict):
            parsed_response = GeneratedFilesResponse.model_validate(
                parsed_response
            )

        if not isinstance(
            parsed_response,
            GeneratedFilesResponse,
        ):
            raise ValueError(
                "Gemini returned an invalid structured response."
            )

        # STEP 11: Validate generated files BEFORE saving.
        expected_files = architecture_blueprint.get(
            "files_to_generate",
            [],
        )

        validated_files = validate_generated_files(
            project_id=project_id,
            expected_files=expected_files,
            generated_files=parsed_response.files,
        )

        state["logs"].append(
            "[Code Generation Agent] All generated files passed validation."
        )

        state["logs"].append(
            f"[Code Generation Agent] Gemini generated "
            f"{len(validated_files)} valid files."
        )

        # STEP 12: Create project workspace.
        project_path = create_project_directory(project_id)

        # STEP 13: Save validated files.
        saved_files = []

        for generated_file in validated_files:

            saved_path = write_file(
                project_id=project_id,
                file_path=generated_file.path,
                content=generated_file.content,
            )

            saved_files.append(
                generated_file.model_dump()
            )

            state["logs"].append(
                f"[Code Generation Agent] Saved: {saved_path}"
            )

        # STEP 14: Update AgentState.
        state["generated_files"] = saved_files
        state["workspace_path"] = str(project_path)
        state["status"] = "completed"

        state["logs"].append(
            "[Code Generation Agent] Code generation completed successfully."
        )

    except Exception as error:

        state["status"] = "failed"

        state["logs"].append(
            f"[Code Generation Agent Error] {str(error)}"
        )

    return state
