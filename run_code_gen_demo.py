
from backend.agents.code_gen_agent import run_code_generation_agent

state = {
    "project_id": "student_management_test",

    "user_prompt": """
    Create a console-based Student Management System in Python.

    The application should support:
    1. Adding a new student.
    2. Viewing all students.
    3. Searching for a student using their ID.
    4. Updating student details.
    5. Deleting a student.
    6. Exiting the application.

    Use clean, modular, object-oriented Python code.
    All files must work together without external dependencies.
    """,

    "requirement_spec": {
        "project_title": "Student Management System",

        "functional_requirements": [
            "Add a student with ID, name and marks",
            "View all registered students",
            "Search for a student using their unique ID",
            "Update student name and marks",
            "Delete a student using their ID",
            "Provide a menu-driven command-line interface",
            "Handle invalid inputs gracefully",
            "Prevent duplicate student IDs"
        ],

        "non_functional_requirements": [
            "Readable and modular code",
            "Use object-oriented programming",
            "Proper error handling",
            "No external dependencies"
        ],

        "tech_stack": ["Python"],

        "acceptance_criteria": [
            "All CRUD operations work correctly",
            "Student IDs must be unique",
            "Invalid student IDs are handled safely",
            "The application should not crash on invalid input"
        ]
    },

    "architecture_blueprint": {
        "directory_tree": [
            "main.py",
            "student.py",
            "student_manager.py",
            "README.md"
        ],

        "api_routes": [],

        "database_tables": [],

        "files_to_generate": [
            "main.py",
            "student.py",
            "student_manager.py",
            "README.md"
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


# Run the Code Generation Agent
result = run_code_generation_agent(state)


# Display Agent Result
print("\n========== AGENT RESULT ==========")
print("Status:", result["status"])
print("Workspace:", result["workspace_path"])


# Display Generated Files
print("\n========== GENERATED FILES ==========")

for file in result["generated_files"]:
    print("File:", file["path"])
    print("Language:", file["language"])


# Display Execution Logs
print("\n========== EXECUTION LOGS ==========")

for log in result["logs"]:
    print(log)


# Display Error if Generation Failed
if result["status"] != "completed":
    print("\nCode generation failed. Check the logs above.")
