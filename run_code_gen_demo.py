
from backend.agents.code_gen_agent import run_code_generation_agent

state = {
    "project_id": "calculator_test",
    "user_prompt": "Create a simple Python calculator CLI that supports addition, subtraction, multiplication and division.",

    "requirement_spec": {
        "project_title": "Python Calculator",
        "functional_requirements": [
            "Add two numbers",
            "Subtract two numbers",
            "Multiply two numbers",
            "Divide two numbers",
            "Handle division by zero"
        ],
        "non_functional_requirements": [
            "Readable and modular code",
            "Proper error handling"
        ],
        "tech_stack": ["Python"],
        "acceptance_criteria": [
            "All four arithmetic operations work correctly",
            "Division by zero is handled safely"
        ]
    },

    "architecture_blueprint": {
        "directory_tree": [
            "main.py"
        ],
        "api_routes": [],
        "database_tables": [],
        "files_to_generate": [
            "main.py"
        ]
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
    "status": "pending"
}

result = run_code_generation_agent(state)

print("\n========== AGENT RESULT ==========")
print("Status:", result["status"])
print("Workspace:", result["workspace_path"])

print("\n========== GENERATED FILES ==========")
for file in result["generated_files"]:
    print("File:", file["path"])
    print("Language:", file["language"])

print("\n========== EXECUTION LOGS ==========")
for log in result["logs"]:
    print(log)
