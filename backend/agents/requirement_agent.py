"""
Requirement Analysis Agent for Multi-Agent AI DevTeam / AutoDev.
Uses Google Gemini API to analyze user requests and produce structured RequirementSpec.
"""

import json
import google.generativeai as genai
from backend.config import config
from backend.schemas.state import AgentState, RequirementSpec


def get_gemini_model():
    """Configures and returns the Gemini GenerativeModel with JSON output mode."""
    genai.configure(api_key=config.GEMINI_API_KEY)
    
    # Enforces guaranteed valid JSON from Gemini
    generation_config = {
        "temperature": 0.2,
        "response_mime_type": "application/json",
    }
    
    return genai.GenerativeModel(
        model_name=config.LLM_MODEL,
        generation_config=generation_config
    )


def run_requirement_agent(state: AgentState) -> AgentState:
    """
    Analyzes natural language prompt from state and updates state with structured RequirementSpec.
    """
    user_prompt = state.get("user_prompt", "")
    
    # Initialize logs if not present
    if "logs" not in state or state["logs"] is None:
        state["logs"] = []

    # Update agent status
    state["current_agent"] = "Requirement Analysis Agent"
    state["status"] = "analyzing"
    state["logs"].append(f"[Requirement Agent] Analyzing request: '{user_prompt}'")

    system_prompt = f"""
You are an expert Senior Software Systems Architect.
Analyze the following user request and generate detailed, professional software engineering requirements.

User Request: {user_prompt}

You MUST return a JSON object matching this exact schema:
{{
    "project_title": "Concise, professional project title",
    "functional_requirements": [
        "Requirement 1...",
        "Requirement 2..."
    ],
    "non_functional_requirements": [
        "Performance / security / scalability requirement 1...",
        "Requirement 2..."
    ],
    "tech_stack": [
        "e.g. Python 3.11",
        "FastAPI",
        "PostgreSQL",
        "Pytest",
        "Docker"
    ],
    "acceptance_criteria": [
        "Criterion 1...",
        "Criterion 2..."
    ]
}}
Ensure the requirements are specific, modern, and production-grade.
"""

    try:
        model = get_gemini_model()
        response = model.generate_content(system_prompt)
        
        # Parse returned JSON
        spec_dict = json.loads(response.text)
        
        # Validate against Pydantic schema
        validated_spec = RequirementSpec(**spec_dict)
        state["requirement_spec"] = validated_spec.model_dump()
        
        state["logs"].append(
            f"[Requirement Agent] Successfully generated specifications for: {validated_spec.project_title}"
        )
        state["logs"].append(
            f"[Requirement Agent] Identified {len(validated_spec.functional_requirements)} functional requirements and {len(validated_spec.acceptance_criteria)} acceptance criteria."
        )

    except Exception as e:
        error_msg = f"[Requirement Agent Error] Generation failed: {str(e)}"
        state["logs"].append(error_msg)
        
        # Safe fallback so pipeline never crashes
        fallback_spec = RequirementSpec(
            project_title="Software Project",
            functional_requirements=[user_prompt, "CRUD operations", "RESTful API endpoints"],
            non_functional_requirements=["High reliability", "Clean modular code structure", "Fast response times"],
            tech_stack=["Python 3.11", "FastAPI", "Pytest", "Docker"],
            acceptance_criteria=["All endpoints respond with appropriate HTTP status codes", "Unit and integration tests pass cleanly"]
        )
        state["requirement_spec"] = fallback_spec.model_dump()

    return state


# Self-contained testing block to test the agent independently
if __name__ == "__main__":
    sample_state: AgentState = {
        "project_id": "test_001",
        "user_prompt": "Build a Student Attendance Management REST API with FastAPI, PostgreSQL, and Pytest",
        "requirement_spec": None,
        "architecture_blueprint": None,
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
        "status": "pending"
    }
    
    print("Running Requirement Agent test...")
    result_state = run_requirement_agent(sample_state)
    print("\n--- AGENT LOGS ---")
    for log in result_state["logs"]:
        print(log)
    print("\n--- GENERATED REQUIREMENT SPEC ---")
    print(json.dumps(result_state["requirement_spec"], indent=2))