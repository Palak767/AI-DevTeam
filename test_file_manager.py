
from backend.core.file_manager import (
    create_project_directory,
    write_file,
    read_file,
    apply_patch,
)

project_id = "file_manager_test"

# Test 1: Create project directory
project_path = create_project_directory(project_id)
print("Directory created:", project_path)

# Test 2: Write a file
write_file(project_id, "src/main.py", "print('Hello AI DevTeam')")

print("File written successfully.")

# Test 3: Read the file
content = read_file(project_id, "src/main.py")
print("File content:", content)

# Test 4: Update the file
apply_patch(project_id, "src/main.py", "print('File updated successfully')")

updated_content = read_file(project_id, "src/main.py")
print("Updated content:", updated_content)

# Test 5: Check path traversal protection
try:
    write_file(project_id, "../unsafe.py", "malicious content")
except ValueError as error:
    print("Security check passed:", error)

print("\nAll File Manager tests completed.")
