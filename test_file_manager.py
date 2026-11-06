
import pytest
from backend.core import file_manager


@pytest.fixture
def project_workspace(tmp_path, monkeypatch):
    """Use a temporary workspace for isolated tests."""
    monkeypatch.setattr(file_manager, "BASE_WORKSPACE", tmp_path)
    return "test_project"


def test_create_project_directory(project_workspace):
    project_path = file_manager.create_project_directory(project_workspace)

    assert project_path.exists()
    assert project_path.is_dir()


def test_write_and_read_file(project_workspace):
    file_manager.write_file(
        project_workspace,
        "src/main.py",
        "print('Hello AI DevTeam')"
    )

    content = file_manager.read_file(project_workspace, "src/main.py")

    assert content == "print('Hello AI DevTeam')"


def test_apply_patch(project_workspace):
    file_manager.write_file(
        project_workspace,
        "main.py",
        "print('Old content')"
    )

    file_manager.apply_patch(
        project_workspace,
        "main.py",
        "print('Updated content')"
    )

    content = file_manager.read_file(project_workspace, "main.py")

    assert content == "print('Updated content')"


def test_reject_path_traversal(project_workspace):
    with pytest.raises(ValueError):
        file_manager.write_file(
            project_workspace,
            "../unsafe.py",
            "malicious content"
        )


def test_reject_absolute_path(project_workspace, tmp_path):
    absolute_path = str(tmp_path / "outside.py")

    with pytest.raises(ValueError):
        file_manager.write_file(
            project_workspace,
            absolute_path,
            "unsafe content"
        )


def test_reject_invalid_project_id():
    invalid_ids = ["", ".", "..", "../outside", "folder/name"]

    for project_id in invalid_ids:
        with pytest.raises(ValueError):
            file_manager.create_project_directory(project_id)


def test_read_missing_file(project_workspace):
    with pytest.raises(FileNotFoundError):
        file_manager.read_file(project_workspace, "missing.py")


def test_patch_missing_file(project_workspace):
    with pytest.raises(FileNotFoundError):
        file_manager.apply_patch(
            project_workspace,
            "missing.py",
            "new content"
        )


def test_reject_non_string_content(project_workspace):
    with pytest.raises(TypeError):
        file_manager.write_file(
            project_workspace,
            "main.py",
            123
        )


# Additional Windows security tests

def test_reject_windows_path_traversal(project_workspace):
    with pytest.raises(ValueError):
        file_manager.write_file(
            project_workspace,
            r"..\..\unsafe.py",
            "unsafe content"
        )


def test_reject_windows_drive_path(project_workspace):
    with pytest.raises(ValueError):
        file_manager.write_file(
            project_workspace,
            r"C:\Windows\system32\unsafe.py",
            "unsafe content"
        )


def test_reject_windows_alternate_data_stream(project_workspace):
    with pytest.raises(ValueError):
        file_manager.write_file(
            project_workspace,
            "main.py:secret",
            "unsafe content"
        )
