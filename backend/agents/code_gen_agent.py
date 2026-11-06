
"""
Code Generation Agent for Multi-Agent AI DevTeam.

Generates source code using Google Gemini API,
validates generated files, and saves them securely
into the project workspace.
"""

import json
from typing import List

from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from backend.schemas.state import AgentState, GeneratedFile
from backend.config import config

from backend.core.file_manager import (
    write_file,
    create_project_directory
)


class GeneratedFilesResponse(BaseModel):
    """Schema for the complete code generation response."""

    files: List[GeneratedFile] = Field(
        description="List of all generated project files"
    )


def run_code_generation_agent(state: AgentState) -> AgentState:
    """
    Generates project source code using Gemini,
    validates the generated files, and saves them
    into the project workspace.
    """

    # Step 1: Initialize agent state
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

        # Step 2: Extract information from AgentState
        project_id = state.get("project_id")
        user_prompt = state.get("user_prompt", "")

        requirement_spec = state.get("requirement_spec")
        architecture_blueprint = state.get("architecture_blueprint")

        if not project_id:
            raise ValueError("Project ID is missing.")

        if not requirement_spec:
            raise ValueError("Requirement specification is missing.")

        if not architecture_blueprint:
            raise ValueError("Architecture blueprint is missing.")

        # Step 3: Validate Gemini API key
        if not config.GEMINI_API_KEY:
            raise ValueError("Gemini API key is not configured.")

        # Step 4: Initialize Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            google_api_key=config.GEMINI_API_KEY,
        )

        # Step 5: Enable structured output
        structured_llm = llm.with_structured_output(
            GeneratedFilesResponse
        )

        # Step 6: Create code generation prompt
        prompt_template = PromptTemplate(
            template="""You are an expert Senior Software Engineer.

Generate complete, functional source code based on
the requirements and architecture provided.

USER REQUEST:
{user_prompt}

REQUIREMENT SPECIFICATION:
{requirement_spec}

ARCHITECTURE BLUEPRINT:
{architecture_blueprint}

INSTRUCTIONS:

1. Follow the provided architecture strictly.
2. Generate every file listed in files_to_generate.
3. Generate complete and executable code, not pseudocode.
4. Follow the specified technology stack.
5. Ensure proper imports, naming conventions and modularity.
6. Include necessary error handling.
7. Ensure all generated files are compatible with each other.
8. Do not generate unnecessary files.
9. Put complete source code inside each file's content field.
10. Do not include Markdown code fences inside content.
11. Return output matching this JSON schema:

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
                    indent=2
                )
            },
        )

        # Step 7: Format the prompt
        formatted_prompt = prompt_template.format(
            user_prompt=user_prompt,
            requirement_spec=json.dumps(
                requirement_spec,
                indent=2
            ),
            architecture_blueprint=json.dumps(
                architecture_blueprint,
                indent=2
            ),
        )

        state["logs"].append(
            "[Code Generation Agent] Sending architecture to Gemini..."
        )

        # Step 8: Generate structured response
        parsed_response = structured_llm.invoke(
            formatted_prompt
        )

        if isinstance(parsed_response, dict):
            parsed_response = GeneratedFilesResponse.model_validate(
                parsed_response
            )

        if not isinstance(parsed_response, GeneratedFilesResponse):
            raise ValueError(
                "Gemini returned an invalid structured response."
            )

        if not parsed_response.files:
            raise ValueError(
                "Gemini generated no project files."
            )

        # Step 9: Validate generated files against architecture
        expected_files = architecture_blueprint.get(
            "files_to_generate", []
        )

        if not expected_files:
            raise ValueError(
                "Architecture does not specify files to generate."
            )

        expected_paths = {
            path.strip().replace("\\", "/")
            for path in expected_files
        }

        generated_paths = [
            file.path.strip().replace("\\", "/")
            for file in parsed_response.files
        ]

        # Check for duplicate paths
        if len(generated_paths) != len(set(generated_paths)):
            raise ValueError(
                "Gemini generated duplicate file paths."
            )

        generated_path_set = set(generated_paths)

        # Check for missing files
        missing_files = expected_paths - generated_path_set

        # Check for unexpected files
        unexpected_files = generated_path_set - expected_paths

        if missing_files:
            raise ValueError(
                f"Missing required files: {sorted(missing_files)}"
            )

        if unexpected_files:
            raise ValueError(
                f"Unexpected files generated: {sorted(unexpected_files)}"
            )

        state["logs"].append(
            "[Code Generation Agent] Architecture validation passed."
        )

        state["logs"].append(
            f"[Code Generation Agent] Gemini generated "
            f"{len(parsed_response.files)} files."
        )

        # Step 10: Create project workspace
        project_path = create_project_directory(project_id)

        generated_files = []

        # Step 11: Save generated files
        for generated_file in parsed_response.files:

            if not generated_file.path.strip():
                raise ValueError(
                    "Generated file path cannot be empty."
                )

            if not generated_file.content.strip():
                raise ValueError(
                    f"Generated file is empty: {generated_file.path}"
                )

            saved_path = write_file(
                project_id=project_id,
                file_path=generated_file.path,
                content=generated_file.content,
            )

            generated_files.append(
                generated_file.model_dump()
            )

            state["logs"].append(
                f"[Code Generation Agent] Saved: {saved_path}"
            )

        # Step 12: Update AgentState
        state["generated_files"] = generated_files
        state["workspace_path"] = str(project_path)
        state["status"] = "completed"

        state["logs"].append(
            "[Code Generation Agent] Code generation completed successfully."
        )

    except Exception as e:

        state["status"] = "failed"

        state["logs"].append(
            f"[Code Generation Agent Error] {str(e)}"
        )

    return state
