
"""
File Manager for Multi-Agent AI DevTeam.

Handles project directory creation, file writing,
file reading, and safe file updates.
"""

from pathlib import Path, PureWindowsPath

# Base directory where all generated projects are stored
BASE_WORKSPACE = Path("workspace_output")


def validate_project_id(project_id: str) -> None:
    """Validate project ID to prevent unsafe directory paths."""

    if not isinstance(project_id, str) or not project_id.strip():
        raise ValueError("Project ID cannot be empty.")

    if project_id in (".", ".."):
        raise ValueError("Invalid project ID.")

    if "/" in project_id or "\\" in project_id or ":" in project_id:
        raise ValueError("Project ID must be a single directory name.")


def create_project_directory(project_id: str) -> Path:
    """Create and return a project's workspace directory."""

    validate_project_id(project_id)

    workspace_root = BASE_WORKSPACE.resolve()
    project_path = (workspace_root / project_id).resolve()

    # Ensure project directory remains inside workspace
    try:
        project_path.relative_to(workspace_root)
    except ValueError:
        raise ValueError("Unsafe project directory path.")

    project_path.mkdir(parents=True, exist_ok=True)

    return project_path


def get_safe_file_path(project_id: str, file_path: str) -> Path:
    """Return a validated file path inside the project directory."""

    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("File path cannot be empty.")

    # Reject null bytes
    if "\0" in file_path:
        raise ValueError("Null bytes are not allowed in file paths.")

    # Reject absolute paths
    if Path(file_path).is_absolute() or PureWindowsPath(file_path).is_absolute():
        raise ValueError("Absolute file paths are not allowed.")

    # Reject Windows drive-based paths
    if PureWindowsPath(file_path).drive:
        raise ValueError("Drive-based paths are not allowed.")

    # Reject Windows alternate data stream syntax
    if ":" in file_path:
        raise ValueError("Colon characters are not allowed in file paths.")

    # Reject parent-directory traversal
    if ".." in Path(file_path).parts or ".." in PureWindowsPath(file_path).parts:
        raise ValueError("Path traversal is not allowed.")

    project_path = create_project_directory(project_id)

    target_path = (project_path / file_path).resolve()

    # Ensure the resolved path remains inside the project directory
    try:
        target_path.relative_to(project_path)
    except ValueError:
        raise ValueError(f"Unsafe file path: {file_path}")

    if target_path == project_path:
        raise ValueError("File path must specify a file.")

    return target_path


def write_file(project_id: str, file_path: str, content: str) -> Path:
    """Create or overwrite a file inside the project workspace."""

    if not isinstance(content, str):
        raise TypeError("File content must be a string.")

    target_path = get_safe_file_path(project_id, file_path)

    # Create parent directories if they do not exist
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Write file using UTF-8 encoding
    target_path.write_text(content, encoding="utf-8")

    return target_path


def read_file(project_id: str, file_path: str) -> str:
    """Read a file safely from the project workspace."""

    target_path = get_safe_file_path(project_id, file_path)

    if not target_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    return target_path.read_text(encoding="utf-8")


def apply_patch(project_id: str, file_path: str, new_content: str) -> Path:
    """Update an existing file with new content."""

    if not isinstance(new_content, str):
        raise TypeError("New content must be a string.")

    target_path = get_safe_file_path(project_id, file_path)

    if not target_path.is_file():
        raise FileNotFoundError(
            f"Cannot patch a file that does not exist: {file_path}"
        )

    return write_file(
        project_id=project_id,
        file_path=file_path,
        content=new_content
    )
