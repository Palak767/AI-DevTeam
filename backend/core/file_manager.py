from pathlib import Path


BASE_WORKSPACE = Path("workspace_output")


def create_project_directory(project_id: str) -> Path:
    """
    Create and return the workspace directory for a project.
    """

    project_path = BASE_WORKSPACE / project_id
    project_path.mkdir(parents=True, exist_ok=True)

    return project_path


def write_file(project_id: str, file_path: str, content: str) -> Path:
    """
    Create or overwrite a file safely inside the project's workspace.
    """

    project_path = create_project_directory(project_id)

    target_path = (project_path / file_path).resolve()
    workspace_root = project_path.resolve()

    # Security check: file must remain inside the project workspace
    if workspace_root not in target_path.parents:
        raise ValueError(f"Unsafe file path: {file_path}")

    # Create parent folders if they don't exist
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the generated code
    target_path.write_text(content, encoding="utf-8")

    return target_path


def read_file(project_id: str, file_path: str) -> str:
    """
    Read a file from the project's workspace.
    """

    project_path = create_project_directory(project_id)

    target_path = project_path / file_path

    return target_path.read_text(encoding="utf-8")


def apply_patch(project_id: str, file_path: str, new_content: str) -> Path:
    """
    Replace a broken file with patched code.
    """

    return write_file(
        project_id=project_id,
        file_path=file_path,
        content=new_content
    )