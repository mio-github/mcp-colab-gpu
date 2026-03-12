"""Tests for Google Drive integration (colab_drive_upload / colab_drive_download)."""

import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from mcp_colab_gpu import drive as drive_mod
from mcp_colab_gpu.drive import (
    DRIVE_API_BASE,
    DRIVE_UPLOAD_BASE,
    _drive_query_escape,
    _validate_local_path,
    download_from_drive,
    find_or_create_folder,
    get_drive_credentials,
    resolve_drive_path,
    upload_to_drive,
)


@pytest.fixture
def mock_creds():
    creds = MagicMock()
    creds.token = "fake-access-token"
    creds.valid = True
    return creds


@pytest.fixture
def tmp_file(tmp_path):
    p = tmp_path / "test_data.csv"
    p.write_text("col1,col2\n1,2\n3,4\n")
    return str(p)


class TestDriveQueryEscape:
    def test_plain_name(self):
        assert _drive_query_escape("colab_data") == "colab_data"

    def test_single_quote(self):
        assert _drive_query_escape("it's") == "it\\'s"

    def test_backslash(self):
        assert _drive_query_escape("a\\b") == "a\\\\b"

    def test_injection_attempt(self):
        result = _drive_query_escape("x' or '1'='1")
        assert "'" not in result or result.count("\\'") == result.count("'")


class TestValidateLocalPath:
    def test_valid_path(self, tmp_path):
        p = tmp_path / "file.txt"
        p.touch()
        result = _validate_local_path(str(p))
        assert result.is_file()

    def test_rejects_path_traversal(self):
        with pytest.raises(ValueError, match="Path traversal rejected"):
            _validate_local_path("/tmp/../../etc/passwd")


class TestFindOrCreateFolder:
    def test_finds_existing_folder(self, mock_creds):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "files": [{"id": "folder-123", "name": "colab_data"}]
        }

        with patch("mcp_colab_gpu.drive.requests.get", return_value=resp) as mock_get:
            folder_id = find_or_create_folder("colab_data", mock_creds)

        assert folder_id == "folder-123"
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert "colab_data" in call_args.kwargs["params"]["q"]

    def test_creates_folder_when_not_found(self, mock_creds):
        list_resp = MagicMock()
        list_resp.status_code = 200
        list_resp.json.return_value = {"files": []}

        create_resp = MagicMock()
        create_resp.status_code = 200
        create_resp.json.return_value = {"id": "new-folder-456"}
        create_resp.raise_for_status = MagicMock()

        with (
            patch("mcp_colab_gpu.drive.requests.get", return_value=list_resp),
            patch("mcp_colab_gpu.drive.requests.post", return_value=create_resp) as mock_post,
        ):
            folder_id = find_or_create_folder("my_results", mock_creds)

        assert folder_id == "new-folder-456"
        mock_post.assert_called_once()
        call_body = mock_post.call_args.kwargs["json"]
        assert call_body["name"] == "my_results"
        assert call_body["mimeType"] == "application/vnd.google-apps.folder"

    def test_creates_nested_folder(self, mock_creds):
        """find_or_create_folder with parent_id passes 'parents' in create request."""
        list_resp = MagicMock()
        list_resp.status_code = 200
        list_resp.json.return_value = {"files": []}

        create_resp = MagicMock()
        create_resp.status_code = 200
        create_resp.json.return_value = {"id": "nested-789"}
        create_resp.raise_for_status = MagicMock()

        with (
            patch("mcp_colab_gpu.drive.requests.get", return_value=list_resp),
            patch("mcp_colab_gpu.drive.requests.post", return_value=create_resp) as mock_post,
        ):
            folder_id = find_or_create_folder("sub", mock_creds, parent_id="parent-id")

        assert folder_id == "nested-789"
        call_body = mock_post.call_args.kwargs["json"]
        assert call_body["parents"] == ["parent-id"]

    def test_escapes_single_quote_in_name(self, mock_creds):
        """Folder names with single quotes are properly escaped in the query."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"files": [{"id": "f-1", "name": "it's"}]}

        with patch("mcp_colab_gpu.drive.requests.get", return_value=resp) as mock_get:
            folder_id = find_or_create_folder("it's", mock_creds)

        assert folder_id == "f-1"
        query = mock_get.call_args.kwargs["params"]["q"]
        assert "it\\'s" in query


class TestResolveDrivePath:
    def test_single_folder(self, mock_creds):
        with patch("mcp_colab_gpu.drive.find_or_create_folder", return_value="f1") as mock_find:
            folder_id = resolve_drive_path("data", mock_creds)

        assert folder_id == "f1"
        mock_find.assert_called_once_with("data", mock_creds, parent_id=None)

    def test_nested_path(self, mock_creds):
        with patch("mcp_colab_gpu.drive.find_or_create_folder", side_effect=["f1", "f2", "f3"]) as mock_find:
            folder_id = resolve_drive_path("data/train/images", mock_creds)

        assert folder_id == "f3"
        assert mock_find.call_count == 3
        mock_find.assert_any_call("data", mock_creds, parent_id=None)
        mock_find.assert_any_call("train", mock_creds, parent_id="f1")
        mock_find.assert_any_call("images", mock_creds, parent_id="f2")

    def test_empty_path_returns_none(self, mock_creds):
        result = resolve_drive_path("", mock_creds)
        assert result is None


class TestUploadToDrive:
    def test_upload_success(self, mock_creds, tmp_file):
        with patch("mcp_colab_gpu.drive.resolve_drive_path", return_value="folder-id"):
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"id": "file-abc", "name": "test_data.csv"}
            resp.raise_for_status = MagicMock()

            with patch("mcp_colab_gpu.drive.requests.post", return_value=resp):
                result = upload_to_drive(tmp_file, "colab_data", mock_creds)

        assert result["id"] == "file-abc"
        assert result["name"] == "test_data.csv"

    def test_upload_file_not_found(self, mock_creds):
        with pytest.raises(FileNotFoundError):
            upload_to_drive("/nonexistent/file.csv", "colab_data", mock_creds)

    def test_upload_to_root_when_no_folder(self, mock_creds, tmp_file):
        """When drive_folder is empty, upload to MyDrive root (no parents in metadata)."""
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"id": "file-root", "name": "test_data.csv"}
        resp.raise_for_status = MagicMock()

        with patch("mcp_colab_gpu.drive.requests.post", return_value=resp) as mock_post:
            result = upload_to_drive(tmp_file, "", mock_creds)

        assert result["id"] == "file-root"
        body = mock_post.call_args.kwargs["data"]
        assert b'"parents"' not in body


class TestDownloadFromDrive:
    def test_download_success(self, mock_creds, tmp_path):
        local_path = str(tmp_path / "downloaded.csv")

        folder_resp = MagicMock()
        folder_resp.status_code = 200
        folder_resp.json.return_value = {
            "files": [{"id": "folder-results", "name": "results"}]
        }
        folder_resp.raise_for_status = MagicMock()

        file_resp = MagicMock()
        file_resp.status_code = 200
        file_resp.json.return_value = {
            "files": [{"id": "file-xyz", "name": "model.pt", "size": "1024"}]
        }
        file_resp.raise_for_status = MagicMock()

        dl_resp = MagicMock()
        dl_resp.status_code = 200
        dl_resp.content = b"col1,col2\n1,2\n"
        dl_resp.raise_for_status = MagicMock()

        with (
            patch("mcp_colab_gpu.drive.requests.get", side_effect=[folder_resp, file_resp, dl_resp]),
        ):
            result = download_from_drive("results/model.pt", local_path, mock_creds)

        assert os.path.exists(local_path)
        with open(local_path, "rb") as f:
            assert f.read() == b"col1,col2\n1,2\n"

    def test_download_file_not_found_on_drive(self, mock_creds, tmp_path):
        local_path = str(tmp_path / "missing.csv")

        list_resp = MagicMock()
        list_resp.status_code = 200
        list_resp.json.return_value = {"files": []}

        with patch("mcp_colab_gpu.drive.requests.get", return_value=list_resp):
            with pytest.raises(FileNotFoundError, match="not found on Google Drive"):
                download_from_drive("data/missing.csv", local_path, mock_creds)

    def test_download_rejects_path_traversal(self, mock_creds, tmp_path):
        """local_path with path traversal should be rejected."""
        with pytest.raises(ValueError, match="Path traversal rejected"):
            download_from_drive("data/file.csv", "/tmp/../../etc/passwd", mock_creds)

    def test_download_simple_file_no_folder(self, mock_creds, tmp_path):
        """Download a file from MyDrive root (no folder path)."""
        local_path = str(tmp_path / "root_file.csv")

        file_resp = MagicMock()
        file_resp.status_code = 200
        file_resp.json.return_value = {
            "files": [{"id": "file-root", "name": "data.csv", "size": "256"}]
        }
        file_resp.raise_for_status = MagicMock()

        dl_resp = MagicMock()
        dl_resp.status_code = 200
        dl_resp.content = b"hello"
        dl_resp.raise_for_status = MagicMock()

        with patch("mcp_colab_gpu.drive.requests.get", side_effect=[file_resp, dl_resp]):
            result = download_from_drive("data.csv", local_path, mock_creds)

        assert os.path.exists(local_path)
        with open(local_path, "rb") as f:
            assert f.read() == b"hello"


class TestTokenMaxAge:
    """Tests for MCP_DRIVE_TOKEN_MAX_AGE forced refresh behaviour."""

    def _make_creds(self):
        creds = MagicMock()
        creds.token = "fresh-token"
        creds.valid = True
        creds.expired = False
        creds.refresh_token = "refresh-tok"
        return creds

    def test_forced_refresh_when_max_age_exceeded(self):
        """Token is refreshed when _last_drive_token_time exceeds max_age."""
        creds = self._make_creds()

        # Simulate: token was obtained 120 seconds ago, max_age = 60
        drive_mod._last_drive_token_time = time.time() - 120

        with (
            patch.dict(os.environ, {"MCP_DRIVE_TOKEN_MAX_AGE": "60"}),
            patch("mcp_colab_gpu.drive.os.path.exists", return_value=True),
            patch(
                "mcp_colab_gpu.drive.Credentials.from_authorized_user_file",
                return_value=creds,
            ),
            patch("mcp_colab_gpu.drive._save_drive_credentials"),
        ):
            result = get_drive_credentials()

        creds.refresh.assert_called_once()
        assert result is creds

    def test_no_forced_refresh_within_max_age(self):
        """Token is not refreshed when still within max_age."""
        creds = self._make_creds()

        # Token obtained 10 seconds ago, max_age = 60
        drive_mod._last_drive_token_time = time.time() - 10

        with (
            patch.dict(os.environ, {"MCP_DRIVE_TOKEN_MAX_AGE": "60"}),
            patch("mcp_colab_gpu.drive.os.path.exists", return_value=True),
            patch(
                "mcp_colab_gpu.drive.Credentials.from_authorized_user_file",
                return_value=creds,
            ),
        ):
            result = get_drive_credentials()

        creds.refresh.assert_not_called()
        assert result is creds

    def test_no_forced_refresh_without_env_var(self):
        """Without MCP_DRIVE_TOKEN_MAX_AGE, no forced refresh."""
        creds = self._make_creds()

        drive_mod._last_drive_token_time = time.time() - 9999

        with (
            patch.dict(os.environ, {}, clear=False),
            patch("mcp_colab_gpu.drive.os.path.exists", return_value=True),
            patch(
                "mcp_colab_gpu.drive.Credentials.from_authorized_user_file",
                return_value=creds,
            ),
        ):
            # Ensure env var is not set
            os.environ.pop("MCP_DRIVE_TOKEN_MAX_AGE", None)
            result = get_drive_credentials()

        creds.refresh.assert_not_called()
        assert result is creds
