"""
System Design Agent for Multi-Agent AI DevTeam / AutoDev.
Transforms RequirementSpec into an ArchitectureBlueprint (directory tree, REST routes, DB schema, files to generate).
"""

import json
import google.generativeai as genai
from backend.config import config
from backend.schemas.state import AgentState, ArchitectureBlueprint


def run_design_agent(state: AgentState) -> AgentState:
    """
    Reads requirement_spec from state and generates a production-grade ArchitectureBlueprint.
    """
    req_spec = state.get("requirement_spec") or {}
    project_title = req_spec.get("project_title", "Software Project")
    tech_stack = req_spec.get("tech_stack", ["Python 3.11", "FastAPI", "PostgreSQL", "Pytest", "Docker"])
    functional_reqs = req_spec.get("functional_requirements", [])

    if "logs" not in state or state["logs"] is None:
        state["logs"] = []

    # Update agent status & logs
    state["current_agent"] = "System Design Agent"
    state["status"] = "designing"
    state["logs"].append(f"[System Design Agent] Designing architecture blueprint for: '{project_title}'")

    system_prompt = f"""
You are a Principal Software Architect and DevOps Lead.
Given the following project specifications, generate a complete, modular, production-ready system architecture blueprint.

Project Title: {project_title}
Tech Stack: {', '.join(tech_stack)}
Functional Requirements: {json.dumps(functional_reqs)}

You MUST return a JSON object matching this exact schema:
{{
    "directory_tree": [
        "app/",
        "app/main.py",
        "app/models.py",
        "app/schemas.py",
        "app/database.py",
        "tests/",
        "tests/test_api.py",
        "Dockerfile",
        "requirements.txt"
    ],
    "api_routes": [
        {{"method": "GET", "path": "/api/v1/attendance", "description": "Retrieve all attendance records"}},
        {{"method": "POST", "path": "/api/v1/attendance", "description": "Mark new attendance entry"}},
        {{"method": "GET", "path": "/api/v1/attendance/{{student_id}}", "description": "Get student attendance by ID"}}
    ],
    "database_tables": [
        "students (id SERIAL PRIMARY KEY, roll_no VARCHAR UNIQUE, name VARCHAR, email VARCHAR)",
        "attendance_records (id SERIAL PRIMARY KEY, student_id INTEGER REFERENCES students(id), status VARCHAR, date DATE, timestamp TIMESTAMP)"
    ],
    "files_to_generate": [
        "app/main.py",
        "app/models.py",
        "app/schemas.py",
        "app/database.py",
        "tests/test_api.py",
        "Dockerfile",
        "requirements.txt"
    ]
}}

Ensure:
1. Every API route required by the functional requirements is represented.
2. The database schema defines all necessary relational tables, foreign keys, and constraints.
3. 'files_to_generate' includes all files needed to run, test, and containerize the application.
"""

    try:
        genai.configure(api_key=config.GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name=config.LLM_MODEL,
            generation_config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
            }
        )
        response = model.generate_content(system_prompt)
        blueprint_dict = json.loads(response.text)

        # Validate against ArchitectureBlueprint schema
        validated_blueprint = ArchitectureBlueprint(**blueprint_dict)
        state["architecture_blueprint"] = validated_blueprint.model_dump()

        state["logs"].append(
            f"[System Design Agent] Successfully generated blueprint with {len(validated_blueprint.files_to_generate)} files."
        )
        state["logs"].append(
            f"[System Design Agent] Designed {len(validated_blueprint.api_routes)} REST endpoints and {len(validated_blueprint.database_tables)} database tables."
        )

    except Exception as e:
        error_msg = f"[System Design Agent Error] Blueprint generation failed: {str(e)}"
        state["logs"].append(error_msg)

        # Resilient fallback blueprint
        fallback_blueprint = ArchitectureBlueprint(
            directory_tree=[
                "app/",
                "app/main.py",
                "app/models.py",
                "app/schemas.py",
                "app/database.py",
                "tests/",
                "tests/test_api.py",
                "Dockerfile",
                "requirements.txt"
            ],
            api_routes=[
                {"method": "GET", "path": "/api/v1/attendance", "description": "Retrieve all attendance records"},
                {"method": "POST", "path": "/api/v1/attendance", "description": "Mark attendance for a student"},
                {"method": "GET", "path": "/api/v1/attendance/report", "description": "Generate aggregated attendance report"}
            ],
            database_tables=[
                "students (id SERIAL PRIMARY KEY, roll_no VARCHAR UNIQUE, name VARCHAR, email VARCHAR)",
                "attendance_records (id SERIAL PRIMARY KEY, student_id INTEGER, status VARCHAR, timestamp TIMESTAMP)"
            ],
            files_to_generate=[
                "app/main.py",
                "app/models.py",
                "app/schemas.py",
                "app/database.py",
                "tests/test_api.py",
                "Dockerfile",
                "requirements.txt"
            ]
        )
        state["architecture_blueprint"] = fallback_blueprint.model_dump()

    return state


# Standalone test block
if __name__ == "__main__":
    sample_state: AgentState = {
        "project_id": "test_002",
        "user_prompt": "Build a Student Attendance Management REST API with FastAPI, PostgreSQL, and Pytest",
        "requirement_spec": {
            "project_title": "Student Attendance Management REST API",
            "functional_requirements": [
                "User Authentication & Authorization with JWT and RBAC",
                "Student Profile Management with enrollment tracking",
                "Batch Attendance Recording (Present, Absent, Late)",
                "Attendance Reporting & Analytics with 75% threshold alerts"
            ],
            "non_functional_requirements": [
                "Sub-100ms API response time",
                "Bcrypt password hashing and Pydantic v2 input validation",
                "Composite unique constraints to avoid duplicate attendance entries"
            ],
            "tech_stack": [
                "Python 3.11+",
                "FastAPI",
                "PostgreSQL 15+",
                "SQLAlchemy 2.0",
                "Pytest",
                "Docker"
            ],
            "acceptance_criteria": [
                "Swagger OpenAPI documentation generated",
                "Teacher can batch-submit 100 students in single request",
                "Pytest suite passes with >= 90% code coverage"
            ]
        },
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
        "status": "analyzing"
    }

    print("Running System Design Agent test...")
    result_state = run_design_agent(sample_state)
    print("\n--- AGENT LOGS ---")
    for log in result_state["logs"]:
        print(log)
    print("\n--- GENERATED ARCHITECTURE BLUEPRINT ---")
    print(json.dumps(result_state["architecture_blueprint"], indent=2))