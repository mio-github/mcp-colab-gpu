"""Tests for Drive code injection (fetch/save code generation for Colab runtime)."""

from unittest.mock import MagicMock, patch

import pytest

from mcp_colab_gpu.drive import (
    generate_drive_fetch_code,
    generate_drive_save_code,
    resolve_file_id,
)


@pytest.fixture
def mock_creds():
    creds = MagicMock()
    creds.token = "fake-token-12345"
    creds.valid = True
    return creds


class TestResolveFileId:
    def test_file_in_root(self, mock_creds):
        file_resp = MagicMock()
        file_resp.status_code = 200
        file_resp.json.return_value = {
            "files": [{"id": "file-abc", "name": "data.csv"}]
        }
        file_resp.raise_for_status = MagicMock()

        with patch("mcp_colab_gpu.drive.requests.get", return_value=file_resp):
            fid = resolve_file_id("data.csv", mock_creds)

        assert fid == "file-abc"

    def test_file_in_nested_folder(self, mock_creds):
        folder_resp = MagicMock()
        folder_resp.status_code = 200
        folder_resp.json.return_value = {
            "files": [{"id": "folder-results", "name": "results"}]
        }
        folder_resp.raise_for_status = MagicMock()

        file_resp = MagicMock()
        file_resp.status_code = 200
        file_resp.json.return_value = {
            "files": [{"id": "file-xyz", "name": "model.pt"}]
        }
        file_resp.raise_for_status = MagicMock()

        with patch("mcp_colab_gpu.drive.requests.get", side_effect=[folder_resp, file_resp]):
            fid = resolve_file_id("results/model.pt", mock_creds)

        assert fid == "file-xyz"

    def test_file_not_found(self, mock_creds):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"files": []}
        resp.raise_for_status = MagicMock()

        with patch("mcp_colab_gpu.drive.requests.get", return_value=resp):
            with pytest.raises(FileNotFoundError, match="not found on Google Drive"):
                resolve_file_id("missing.csv", mock_creds)


class TestGenerateDriveFetchCode:
    def test_single_file(self):
        code = generate_drive_fetch_code(
            [{"file_id": "abc123", "dest_path": "/content/train.csv"}],
            token="test-token",
        )

        assert "abc123" in code
        assert "/content/train.csv" in code
        assert "test-token" in code
        assert "requests" in code
        assert "def _mcp_drive_fetch" in code

    def test_multiple_files(self):
        code = generate_drive_fetch_code(
            [
                {"file_id": "f1", "dest_path": "/content/a.csv"},
                {"file_id": "f2", "dest_path": "/content/b.csv"},
            ],
            token="tok",
        )

        assert code.count("_mcp_drive_fetch(") >= 3  # def + 2 calls
        assert "f1" in code
        assert "f2" in code

    def test_token_cleanup(self):
        code = generate_drive_fetch_code(
            [{"file_id": "x", "dest_path": "/tmp/x"}],
            token="secret",
        )

        assert "del _mcp_drive_fetch" in code

    def test_generated_code_is_syntactically_valid(self):
        code = generate_drive_fetch_code(
            [{"file_id": "abc", "dest_path": "/content/file.csv"}],
            token="token123",
        )

        compile(code, "<test>", "exec")


class TestGenerateDriveSaveCode:
    def test_single_file(self):
        code = generate_drive_save_code(
            [{"local_path": "/content/model.pt", "drive_folder": "results", "filename": "model.pt"}],
            token="save-token",
        )

        assert "/content/model.pt" in code
        assert "results" in code
        assert "model.pt" in code
        assert "save-token" in code
        assert "def _mcp_drive_save" in code

    def test_multiple_files(self):
        code = generate_drive_save_code(
            [
                {"local_path": "/content/a.pt", "drive_folder": "out", "filename": "a.pt"},
                {"local_path": "/content/b.csv", "drive_folder": "out", "filename": "b.csv"},
            ],
            token="tok",
        )

        assert code.count("_mcp_drive_save(") >= 3  # def + 2 calls

    def test_token_cleanup(self):
        code = generate_drive_save_code(
            [{"local_path": "/tmp/x", "drive_folder": "d", "filename": "x"}],
            token="secret",
        )

        assert "del _mcp_drive_save" in code

    def test_generated_code_is_syntactically_valid(self):
        code = generate_drive_save_code(
            [{"local_path": "/content/out.csv", "drive_folder": "results", "filename": "out.csv"}],
            token="token123",
        )

        compile(code, "<test>", "exec")
