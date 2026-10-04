"""
Requirement Analysis Agent for Multi-Agent AI DevTeam.
Parses natural language prompts into structured requirements using Google Gemini API.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from backend.schemas.state import AgentState, RequirementSpec
from backend.config import config


def run_requirement_agent(state: AgentState) -> AgentState:
    """
    Parses user prompt and updates AgentState with a structured RequirementSpec.
    """
    user_prompt = state.get("user_prompt", "")
    
    # Update status and log
    state["current_agent"] = "Requirement Analysis Agent"
    state["status"] = "analyzing"
    if "logs" not in state or state["logs"] is None:
        state["logs"] = []
    state["logs"].append(f"[Requirement Agent] Analyzing user prompt: '{user_prompt}'...")

    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        temperature=0.2,
    )

    # Setup Pydantic parser for structured output
    parser = PydanticOutputParser(pydantic_object=RequirementSpec)

    # Prompt Template
    prompt_template = PromptTemplate(
        template="""You are an expert Senior Software Systems Architect.
Analyze the following user request and generate detailed, professional software engineering requirements.

User Request: {user_prompt}

{format_instructions}

Be thorough, precise, and ensure acceptance criteria cover edge cases and testing requirements.
""",
        input_variables=["user_prompt"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Format prompt and invoke Gemini LLM
    formatted_prompt = prompt_template.format(user_prompt=user_prompt)
    response = llm.invoke(formatted_prompt)

    # Parse LLM response into RequirementSpec schema
    try:
        parsed_spec: RequirementSpec = parser.parse(response.content)
        state["requirement_spec"] = parsed_spec.model_dump()
        state["logs"].append(
            f"[Requirement Agent] Successfully generated specs for '{parsed_spec.project_title}'."
        )
    except Exception as e:
        state["logs"].append(f"[Requirement Agent Error] Failed to parse JSON: {str(e)}")
        # Fallback dictionary if parsing fails
        state["requirement_spec"] = {
            "project_title": "Software Project",
            "functional_requirements": [user_prompt],
            "non_functional_requirements": ["High performance", "Secure API design"],
            "tech_stack": ["Python 3.11", "FastAPI", "Pytest"],
            "acceptance_criteria": ["Project must pass automated pytest suite"],
        }

    return state