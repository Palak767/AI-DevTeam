
"""
Code Generation Agent for Multi-Agent AI DevTeam.

Generates source code using Google Gemini API
and saves generated files into the project workspace.
"""

import json
from typing import List

from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

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
    Generates project source code using Gemini and saves
    the generated files into the project workspace.
    """

    # Step 1: Update agent status and initialize logs
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
        # Step 2: Extract information from shared AgentState
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

        # Step 5: Initialize Pydantic output parser
        parser = PydanticOutputParser(
            pydantic_object=GeneratedFilesResponse
        )

        # Step 6: Create code generation prompt
        prompt_template = PromptTemplate(
            template="""You are an expert Senior Software Engineer.

Generate complete, functional source code based on the
requirements and architecture provided.

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
9. Return source code as plain text inside the content field.
10. Do not wrap source code in Markdown code fences.
11. Return valid structured output according to the given schema.

{format_instructions}
""",
            input_variables=[
                "user_prompt",
                "requirement_spec",
                "architecture_blueprint",
            ],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            },
        )

        # Step 7: Format the prompt
        formatted_prompt = prompt_template.format(
            user_prompt=user_prompt,
            requirement_spec=json.dumps(
                requirement_spec, indent=2
            ),
            architecture_blueprint=json.dumps(
                architecture_blueprint, indent=2
            ),
        )

        state["logs"].append(
            "[Code Generation Agent] Sending architecture to Gemini..."
        )

        # Step 8: Invoke Gemini
        response = llm.invoke(formatted_prompt)

        # Step 9: Extract text from Gemini response
        response_text = response.content

        if isinstance(response_text, list):
            text_blocks = []

            for block in response_text:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_blocks.append(block.get("text", ""))

                elif isinstance(block, str):
                    text_blocks.append(block)

            response_text = "\n".join(text_blocks)

        if not isinstance(response_text, str) or not response_text.strip():
            raise ValueError("Gemini returned an empty response.")

        # Step 10: Parse and validate generated files
        parsed_response = parser.parse(response_text)

        if not parsed_response.files:
            raise ValueError("Gemini generated no project files.")

        state["logs"].append(
            f"[Code Generation Agent] Gemini generated "
            f"{len(parsed_response.files)} files."
        )

        # Step 11: Create project workspace
        project_path = create_project_directory(project_id)

        generated_files = []

        # Step 12: Save generated files
        for generated_file in parsed_response.files:

            if not generated_file.path.strip():
                raise ValueError(
                    "Generated file path cannot be empty."
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

        # Step 13: Update shared AgentState
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
