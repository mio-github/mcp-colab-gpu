"""Google Drive integration for mcp-colab-gpu.

Provides upload/download operations via the Drive API v3.
Uses a separate token cache with drive.file scope so that
existing Colab-only authentication is not disrupted.
"""

import json
import mimetypes
import os
import pathlib
import sys
import time

import requests
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from .colab_runtime import TOKEN_CACHE_DIR

os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

DRIVE_SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
]

DRIVE_TOKEN_CACHE_PATH = os.path.join(TOKEN_CACHE_DIR, "drive_token.json")

# OAuth2 client credentials for Drive access (Desktop app type).
# These are intentionally public — same pattern as Colab's "ClientNotSoSecret".
# The client_id/secret alone cannot access user data; each user must still
# authenticate with their own Google account and grant consent.
DRIVE_CLIENT_CONFIG = {
    "installed": {
        "client_id": "269125631449-cbel0pvim52pqtqiadq630fntobc8u7g.apps.googleusercontent.com",
        "project_id": "mcp-colab-gpu",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": "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",
        "redirect_uris": ["http://localhost"],
    }
}

# Optional override: users can provide their own OAuth client JSON via
# environment variable or by placing a file at ~/.config/colab-exec/drive_client.json.
DRIVE_CLIENT_JSON_PATH = os.environ.get(
    "MCP_DRIVE_CLIENT_JSON",
    os.path.join(TOKEN_CACHE_DIR, "drive_client.json"),
)

DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"
DRIVE_UPLOAD_BASE = "https://www.googleapis.com/upload/drive/v3"

FOLDER_MIME = "application/vnd.google-apps.folder"

# Track when the Drive token was last obtained/refreshed (epoch seconds).
_last_drive_token_time: float = 0.0


def _drive_query_escape(value: str) -> str:
    """Escape a value for use in a Google Drive API query string."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def get_drive_credentials() -> Credentials:
    """Load cached Drive credentials or run the browser OAuth2 flow.

    Uses a separate token cache from the Colab-only credentials so that
    adding the drive.file scope does not force re-authentication for
    users who only use colab_execute.

    Environment variable ``MCP_DRIVE_TOKEN_MAX_AGE`` (seconds) controls
    the maximum age of the cached access token.  When set, the token is
    forcibly refreshed if it was obtained more than *max_age* seconds ago,
    even when the underlying ``Credentials`` object still reports itself
    as valid.  This is used for E2E testing with a short TTL (e.g. 60 s).
    """
    global _last_drive_token_time  # noqa: PLW0603

    max_age_env = os.environ.get("MCP_DRIVE_TOKEN_MAX_AGE")
    max_token_age: int | None = int(max_age_env) if max_age_env else None

    creds = None

    if os.path.exists(DRIVE_TOKEN_CACHE_PATH):
        creds = Credentials.from_authorized_user_file(DRIVE_TOKEN_CACHE_PATH, DRIVE_SCOPES)

    # Force refresh if the token is older than max_token_age.
    if (
        creds
        and creds.valid
        and max_token_age is not None
        and _last_drive_token_time > 0
        and (time.time() - _last_drive_token_time) > max_token_age
    ):
        print(
            f"[colab-gpu] Drive token age {time.time() - _last_drive_token_time:.0f}s "
            f"> max_age {max_token_age}s — forcing refresh",
            file=sys.stderr,
        )
        if creds.refresh_token:
            try:
                creds.refresh(GoogleRequest())
                _save_drive_credentials(creds)
                _last_drive_token_time = time.time()
                return creds
            except Exception:
                creds = None
        else:
            creds = None

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(GoogleRequest())
            _save_drive_credentials(creds)
            _last_drive_token_time = time.time()
        except Exception as e:
            print(
                f"[colab-gpu] Warning: Drive token refresh failed ({e}), re-authenticating...",
                file=sys.stderr,
            )
            creds = None

    if not creds or not creds.valid:
        # Use external JSON if provided, otherwise fall back to embedded config.
        if os.path.exists(DRIVE_CLIENT_JSON_PATH):
            flow = InstalledAppFlow.from_client_secrets_file(
                DRIVE_CLIENT_JSON_PATH, DRIVE_SCOPES,
            )
        else:
            flow = InstalledAppFlow.from_client_config(
                DRIVE_CLIENT_CONFIG, DRIVE_SCOPES,
            )
        creds = flow.run_local_server(
            port=0,
            access_type="offline",
            prompt="consent",
            success_message="Drive authentication successful! You can close this tab.",
        )
        _save_drive_credentials(creds)
        _last_drive_token_time = time.time()

    # First call in this process — record the time without forcing refresh.
    if _last_drive_token_time == 0.0:
        _last_drive_token_time = time.time()

    return creds


def _save_drive_credentials(creds: Credentials) -> None:
    os.makedirs(TOKEN_CACHE_DIR, exist_ok=True)
    fd = os.open(DRIVE_TOKEN_CACHE_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(creds.to_json())


def _drive_headers(creds: Credentials) -> dict:
    return {"Authorization": f"Bearer {creds.token}"}


def find_or_create_folder(
    name: str,
    creds: Credentials,
    parent_id: str | None = None,
) -> str:
    """Find a folder by name in Drive, or create it if it doesn't exist.

    Returns the folder ID.
    """
    escaped = _drive_query_escape(name)
    query = f"name='{escaped}' and mimeType='{FOLDER_MIME}' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"

    resp = requests.get(
        f"{DRIVE_API_BASE}/files",
        headers=_drive_headers(creds),
        params={"q": query, "fields": "files(id,name)", "spaces": "drive"},
        timeout=30,
    )
    resp.raise_for_status()
    files = resp.json().get("files", [])

    if files:
        return files[0]["id"]

    body = {"name": name, "mimeType": FOLDER_MIME}
    if parent_id:
        body["parents"] = [parent_id]

    resp = requests.post(
        f"{DRIVE_API_BASE}/files",
        headers={**_drive_headers(creds), "Content-Type": "application/json"},
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def resolve_drive_path(drive_folder: str, creds: Credentials) -> str | None:
    """Resolve a nested folder path (e.g. 'data/train/images') into a folder ID.

    Creates intermediate folders as needed.

    Returns:
        The final folder ID, or None if drive_folder is empty.
    """
    parts = [p for p in drive_folder.strip("/").split("/") if p]
    parent_id = None
    for part in parts:
        parent_id = find_or_create_folder(part, creds, parent_id=parent_id)
    return parent_id


def _validate_local_path(raw: str) -> pathlib.Path:
    """Resolve and validate a local file path to prevent path traversal."""
    if ".." in pathlib.PurePosixPath(raw).parts:
        raise ValueError(f"Path traversal rejected: {raw}")
    return pathlib.Path(os.path.expanduser(raw)).resolve()


def upload_to_drive(
    local_path: str,
    drive_folder: str,
    creds: Credentials,
) -> dict:
    """Upload a local file to Google Drive.

    Args:
        local_path: Path to the local file.
        drive_folder: Drive folder path (e.g. 'colab_data' or 'data/train').
                      Empty string uploads to MyDrive root.
        creds: Google OAuth2 credentials with drive.file scope.

    Returns:
        Dict with 'id' and 'name' of the uploaded file.

    Raises:
        FileNotFoundError: If local_path does not exist.
    """
    resolved = _validate_local_path(local_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"File not found: {resolved}")

    filename = resolved.name
    content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"

    metadata = {"name": filename}
    if drive_folder:
        folder_id = resolve_drive_path(drive_folder, creds)
        if folder_id:
            metadata["parents"] = [folder_id]

    boundary = "colab_gpu_boundary"
    meta_json = json.dumps(metadata)

    file_data = resolved.read_bytes()

    body = (
        f"--{boundary}\r\n"
        f"Content-Type: application/json; charset=UTF-8\r\n\r\n"
        f"{meta_json}\r\n"
        f"--{boundary}\r\n"
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8") + file_data + f"\r\n--{boundary}--".encode("utf-8")

    resp = requests.post(
        f"{DRIVE_UPLOAD_BASE}/files",
        headers={
            **_drive_headers(creds),
            "Content-Type": f"multipart/related; boundary={boundary}",
        },
        params={"uploadType": "multipart", "fields": "id,name,size"},
        data=body,
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def download_from_drive(
    drive_path: str,
    local_path: str,
    creds: Credentials,
) -> dict:
    """Download a file from Google Drive to a local path.

    Args:
        drive_path: Path on Drive relative to MyDrive (e.g. 'results/model.pt').
        local_path: Local destination path.
        creds: Google OAuth2 credentials with drive.file scope.

    Returns:
        Dict with 'local_path', 'drive_file_id', and 'size'.

    Raises:
        FileNotFoundError: If the file is not found on Drive.
        ValueError: If local_path contains path traversal.
    """
    dest = _validate_local_path(local_path)

    parts = drive_path.strip("/").split("/")
    filename = parts[-1]
    folder_parts = parts[:-1]

    escaped_filename = _drive_query_escape(filename)
    query = f"name='{escaped_filename}' and trashed=false"
    if folder_parts:
        parent_id = None
        for part in folder_parts:
            escaped_part = _drive_query_escape(part)
            folder_query = f"name='{escaped_part}' and mimeType='{FOLDER_MIME}' and trashed=false"
            if parent_id:
                folder_query += f" and '{parent_id}' in parents"
            resp = requests.get(
                f"{DRIVE_API_BASE}/files",
                headers=_drive_headers(creds),
                params={"q": folder_query, "fields": "files(id,name)", "spaces": "drive"},
                timeout=30,
            )
            resp.raise_for_status()
            folders = resp.json().get("files", [])
            if not folders:
                raise FileNotFoundError(
                    f"'{drive_path}' not found on Google Drive (folder '{part}' does not exist)"
                )
            parent_id = folders[0]["id"]
        query += f" and '{parent_id}' in parents"

    resp = requests.get(
        f"{DRIVE_API_BASE}/files",
        headers=_drive_headers(creds),
        params={"q": query, "fields": "files(id,name,size)", "spaces": "drive"},
        timeout=30,
    )
    resp.raise_for_status()
    files = resp.json().get("files", [])

    if not files:
        raise FileNotFoundError(f"'{drive_path}' not found on Google Drive")

    file_id = files[0]["id"]
    file_size = files[0].get("size", "0")

    resp = requests.get(
        f"{DRIVE_API_BASE}/files/{file_id}",
        headers=_drive_headers(creds),
        params={"alt": "media"},
        timeout=300,
        stream=True,
    )
    resp.raise_for_status()

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    return {"local_path": str(dest), "drive_file_id": file_id, "size": int(file_size)}


# ---------------------------------------------------------------------------
# Colab runtime injection: code generation for Drive fetch/save
# ---------------------------------------------------------------------------


def resolve_file_id(drive_path: str, creds: Credentials) -> str:
    """Resolve a Drive path (e.g. 'results/model.pt') to a file ID.

    Raises:
        FileNotFoundError: If the file or any parent folder is not found.
    """
    parts = drive_path.strip("/").split("/")
    filename = parts[-1]
    folder_parts = parts[:-1]

    parent_id = None
    for part in folder_parts:
        escaped = _drive_query_escape(part)
        query = f"name='{escaped}' and mimeType='{FOLDER_MIME}' and trashed=false"
        if parent_id:
            query += f" and '{parent_id}' in parents"
        resp = requests.get(
            f"{DRIVE_API_BASE}/files",
            headers=_drive_headers(creds),
            params={"q": query, "fields": "files(id,name)", "spaces": "drive"},
            timeout=30,
        )
        resp.raise_for_status()
        folders = resp.json().get("files", [])
        if not folders:
            raise FileNotFoundError(
                f"'{drive_path}' not found on Google Drive (folder '{part}' does not exist)"
            )
        parent_id = folders[0]["id"]

    escaped_filename = _drive_query_escape(filename)
    query = f"name='{escaped_filename}' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"

    resp = requests.get(
        f"{DRIVE_API_BASE}/files",
        headers=_drive_headers(creds),
        params={"q": query, "fields": "files(id,name)", "spaces": "drive"},
        timeout=30,
    )
    resp.raise_for_status()
    files = resp.json().get("files", [])

    if not files:
        raise FileNotFoundError(f"'{drive_path}' not found on Google Drive")

    return files[0]["id"]


def generate_drive_fetch_code(
    file_mappings: list[dict],
    token: str,
) -> str:
    """Generate Python code for Colab to download files from Drive.

    Args:
        file_mappings: List of {"file_id": str, "dest_path": str}.
        token: Fresh OAuth access token (short-lived).

    Returns:
        Python code string to be executed on Colab before the user's code.
    """
    escaped_token = token.replace("\\", "\\\\").replace("'", "\\'")

    lines = [
        "import requests as _mcp_req, os as _mcp_os",
        "",
        "def _mcp_drive_fetch(_fid, _dest):",
        "    _mcp_os.makedirs(_mcp_os.path.dirname(_dest) or '.', exist_ok=True)",
        "    _r = _mcp_req.get(f'https://www.googleapis.com/drive/v3/files/{_fid}?alt=media',",
        f"                      headers={{'Authorization': 'Bearer {escaped_token}'}}, timeout=300)",
        "    _r.raise_for_status()",
        "    with open(_dest, 'wb') as _f:",
        "        _f.write(_r.content)",
        "    print(f'[mcp-drive] Fetched {_dest} ({len(_r.content)} bytes)')",
        "",
    ]

    for m in file_mappings:
        fid = m["file_id"].replace("'", "\\'")
        dest = m["dest_path"].replace("'", "\\'")
        lines.append(f"_mcp_drive_fetch('{fid}', '{dest}')")

    lines.append("del _mcp_drive_fetch, _mcp_req, _mcp_os")
    return "\n".join(lines)


def generate_drive_save_code(
    save_mappings: list[dict],
    token: str,
) -> str:
    """Generate Python code for Colab to upload files to Drive.

    Args:
        save_mappings: List of {"local_path": str, "drive_folder": str, "filename": str}.
        token: Fresh OAuth access token (short-lived).

    Returns:
        Python code string to be executed on Colab after the user's code.
    """
    escaped_token = token.replace("\\", "\\\\").replace("'", "\\'")

    lines = [
        "import requests as _mcp_req, json as _mcp_json, os as _mcp_os",
        "",
        "def _mcp_drive_save(_lpath, _folder, _fname):",
        f"    _hdr = {{'Authorization': 'Bearer {escaped_token}'}}",
        "    _pid = None",
        "    for _part in [p for p in _folder.strip('/').split('/') if p]:",
        "        _esc = _part.replace('\\\\', '\\\\\\\\').replace(\"'\", \"\\\\'\")",
        "        _q = f\"name='{_esc}' and mimeType='application/vnd.google-apps.folder' and trashed=false\"",
        "        if _pid:",
        "            _q += f\" and '{_pid}' in parents\"",
        "        _r = _mcp_req.get('https://www.googleapis.com/drive/v3/files',",
        "                          headers=_hdr, params={'q': _q, 'fields': 'files(id,name)', 'spaces': 'drive'}, timeout=30)",
        "        _r.raise_for_status()",
        "        _fs = _r.json().get('files', [])",
        "        if _fs:",
        "            _pid = _fs[0]['id']",
        "        else:",
        "            _body = {'name': _part, 'mimeType': 'application/vnd.google-apps.folder'}",
        "            if _pid:",
        "                _body['parents'] = [_pid]",
        "            _r = _mcp_req.post('https://www.googleapis.com/drive/v3/files',",
        "                               headers={**_hdr, 'Content-Type': 'application/json'}, json=_body, timeout=30)",
        "            _r.raise_for_status()",
        "            _pid = _r.json()['id']",
        "    _meta = {'name': _fname}",
        "    if _pid:",
        "        _meta['parents'] = [_pid]",
        "    _bnd = 'mcp_save_boundary'",
        "    _mj = _mcp_json.dumps(_meta)",
        "    with open(_lpath, 'rb') as _f:",
        "        _fd = _f.read()",
        "    _bd = (f'--{_bnd}\\r\\nContent-Type: application/json; charset=UTF-8\\r\\n\\r\\n{_mj}\\r\\n'",
        "           f'--{_bnd}\\r\\nContent-Type: application/octet-stream\\r\\n\\r\\n').encode() + _fd + f'\\r\\n--{_bnd}--'.encode()",
        "    _r = _mcp_req.post('https://www.googleapis.com/upload/drive/v3/files',",
        "                       headers={**_hdr, 'Content-Type': f'multipart/related; boundary={_bnd}'},",
        "                       params={'uploadType': 'multipart', 'fields': 'id,name,size'}, data=_bd, timeout=600)",
        "    _r.raise_for_status()",
        "    _res = _r.json()",
        "    print(f'[mcp-drive] Saved {_lpath} -> {_folder}/{_fname} (id={_res[\"id\"]}, {_mcp_os.path.getsize(_lpath)} bytes)')",
        "",
    ]

    for m in save_mappings:
        lp = m["local_path"].replace("'", "\\'")
        df = m["drive_folder"].replace("'", "\\'")
        fn = m["filename"].replace("'", "\\'")
        lines.append(f"_mcp_drive_save('{lp}', '{df}', '{fn}')")

    lines.append("del _mcp_drive_save, _mcp_req, _mcp_json, _mcp_os")
    return "\n".join(lines)
