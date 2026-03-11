"""Colab GPU runtime engine --- auth, allocate, execute, cleanup.

Provides a clean importable API for allocating Google Colab GPU runtimes
and executing Python code on them via the Jupyter kernel WebSocket protocol.

Original: mcp-server-colab-exec v0.1.0 by Paritosh Dwivedi (MIT License)
Extended: mcp-colab-gpu by Masaya Hirano --- all Colab Pro GPUs, high-memory, security fixes.
"""

import io
import json
import os
import re
import sys
import threading
import time
import uuid

import requests
import websocket
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

COLAB_API = "https://colab.research.google.com"
SCOPES = [
    "https://www.googleapis.com/auth/colaboratory",
    "profile",
    "email",
]

# OAuth2 client credentials from the Colab VS Code extension (google.colab@0.3.0)
# These are intentionally public (the extension names them "ClientNotSoSecret")
CLIENT_CONFIG = {
    "installed": {
        "client_id": "1014160490159-cvot3bea7tgkp72a4m29h20d9ddo6bne.apps.googleusercontent.com",
        "client_secret": "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",
        "auth_uri": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost"],
    }
}

TOKEN_CACHE_DIR = os.path.expanduser("~/.config/colab-exec")
TOKEN_CACHE_PATH = os.path.join(TOKEN_CACHE_DIR, "token.json")

HIGHMEM_REQUIRED_ACCELERATORS = {"V5E1", "V6E1"}
VALID_ACCELERATORS = {"T4", "L4", "A100", "H100", "G4", "V5E1", "V6E1"}
MAX_TIMEOUT = 3600
MIN_TIMEOUT = 10
EPHEMERAL_AUTH_TYPES = {"dfs_ephemeral", "auth_user_ephemeral"}


def validate_params(accelerator: str, timeout: int) -> None:
    """Validate accelerator and timeout parameters."""
    if accelerator not in VALID_ACCELERATORS:
        raise ValueError(
            f"Invalid accelerator '{accelerator}'. "
            f"Must be one of: {sorted(VALID_ACCELERATORS)}"
        )
    if not (MIN_TIMEOUT <= timeout <= MAX_TIMEOUT):
        raise ValueError(
            f"timeout must be between {MIN_TIMEOUT} and {MAX_TIMEOUT}, got {timeout}"
        )


def get_credentials() -> Credentials:
    """Load cached credentials or run the browser OAuth2 flow."""
    creds = None

    if os.path.exists(TOKEN_CACHE_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_CACHE_PATH, SCOPES)

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(GoogleRequest())
            _save_credentials(creds)
        except Exception as e:
            print(f"[colab-gpu] Warning: token refresh failed ({e}), re-authenticating...", file=sys.stderr)
            creds = None

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_config(CLIENT_CONFIG, SCOPES)
        creds = flow.run_local_server(
            port=0,
            access_type="offline",
            prompt="consent",
            success_message="Authentication successful! You can close this tab.",
        )
        _save_credentials(creds)

    return creds


def _save_credentials(creds: Credentials):
    os.makedirs(TOKEN_CACHE_DIR, exist_ok=True)
    fd = os.open(TOKEN_CACHE_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(creds.to_json())


def _strip_xssi(text: str) -> dict:
    """Strip the )]}' XSSI prefix and parse JSON."""
    if text.startswith(")]}'"):
        text = text[text.index("\n") + 1:]
    return json.loads(text)


def _colab_headers(token: str, extra: dict = None) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "X-Colab-Client-Agent": "vscode",
    }
    if extra:
        headers.update(extra)
    return headers


def generate_notebook_hash() -> str:
    """Generate a Colab-valid notebook hash."""
    raw_uuid = str(uuid.uuid4())
    return raw_uuid.replace("-", "_") + "." * (44 - len(raw_uuid))


def _build_assign_params(
    nbh: str,
    accelerator: str,
    high_memory: bool = False,
) -> dict:
    params = {
        "nbh": nbh,
        "authuser": "0",
    }
    if accelerator:
        params["variant"] = "GPU"
        params["accelerator"] = accelerator
        if high_memory or accelerator in HIGHMEM_REQUIRED_ACCELERATORS:
            params["shape"] = "hm"
    return params


def _parse_assignment(assignment: dict) -> dict:
    endpoint = assignment.get("endpoint")
    proxy_info = assignment.get("runtimeProxyInfo", {})
    proxy_url = (proxy_info.get("url") or "").rstrip("/")
    proxy_token = proxy_info.get("token")
    return {
        "endpoint": endpoint,
        "proxy_url": proxy_url,
        "proxy_token": proxy_token,
    }


def allocate_runtime(
    token: str,
    accelerator: str = "T4",
    high_memory: bool = False,
) -> dict:
    """Allocate a Colab GPU runtime. Returns dict with endpoint + proxy info."""
    nbh = generate_notebook_hash()
    params = _build_assign_params(nbh, accelerator, high_memory)
    headers = _colab_headers(token)

    hm_label = " (high-memory)" if high_memory else ""
    print(f"[colab-gpu] Requesting {accelerator}{hm_label} runtime...", file=sys.stderr)
    r = requests.get(f"{COLAB_API}/tun/m/assign", params=params, headers=headers, timeout=30)
    r.raise_for_status()
    data = _strip_xssi(r.text)

    parsed = _parse_assignment(data)
    if parsed["endpoint"] and parsed["proxy_url"] and parsed["proxy_token"]:
        print(f"[colab-gpu] Reusing existing runtime: endpoint={parsed['endpoint']}", file=sys.stderr)
        return {
            **parsed,
            "xsrf_token": None,
            "nbh": nbh,
            "reused": True,
        }

    xsrf_token = data.get("token") or data.get("xsrfToken")
    if not xsrf_token:
        raise RuntimeError("No XSRF token in Colab assign response")

    post_headers = _colab_headers(token, {"X-Goog-Colab-Token": xsrf_token})
    r = requests.post(
        f"{COLAB_API}/tun/m/assign",
        params=params,
        headers=post_headers,
        timeout=30,
    )
    r.raise_for_status()
    assignment = _strip_xssi(r.text)

    parsed = _parse_assignment(assignment)
    if not parsed["endpoint"] or not parsed["proxy_url"] or not parsed["proxy_token"]:
        raise RuntimeError("Incomplete assignment response from Colab")

    print(f"[colab-gpu] Runtime allocated: endpoint={parsed['endpoint']}", file=sys.stderr)
    return {
        **parsed,
        "xsrf_token": xsrf_token,
        "nbh": nbh,
        "reused": False,
    }


def unassign_runtime(token: str, endpoint: str) -> bool:
    """Release the runtime."""
    headers = _colab_headers(token)
    url = f"{COLAB_API}/tun/m/unassign/{endpoint}"
    params = {"authuser": "0"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        data = _strip_xssi(r.text)
        xsrf = data.get("token", "")
        post_headers = _colab_headers(token, {"X-Goog-Colab-Token": xsrf})
        r = requests.post(url, headers=post_headers, params=params, timeout=20)
        r.raise_for_status()
        print(f"[colab-gpu] Runtime {endpoint} released.", file=sys.stderr)
        return True
    except Exception as e:
        print(f"[colab-gpu] Warning: failed to unassign runtime: {e}", file=sys.stderr)
        return False


def propagate_credentials(token: str, endpoint: str, auth_type: str, dry_run: bool) -> dict:
    """Call Colab credentials propagation API for ephemeral auth challenges."""
    url = f"{COLAB_API}/tun/m/credentials-propagation/{endpoint}"
    params = {
        "authuser": "0",
        "authtype": auth_type,
        "version": "2",
        "dryrun": str(bool(dry_run)).lower(),
        "propagate": "true",
        "record": "false",
    }
    headers = _colab_headers(token)
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    data = _strip_xssi(r.text)
    xsrf = data.get("token") or data.get("xsrfToken")
    if not xsrf:
        raise RuntimeError("No XSRF token from credentials propagation")
    post_headers = _colab_headers(token, {"X-Goog-Colab-Token": xsrf})
    r = requests.post(url, headers=post_headers, params=params, timeout=30)
    r.raise_for_status()
    return _strip_xssi(r.text)


def start_keepalive(token: str, endpoint: str) -> threading.Event:
    """Start a background thread that pings keep-alive every 60s."""
    stop_event = threading.Event()
    def loop():
        headers = _colab_headers(token, {"X-Colab-Tunnel": "Google"})
        url = f"{COLAB_API}/tun/m/{endpoint}/keep-alive/"
        params = {"authuser": "0"}
        while not stop_event.is_set():
            try:
                requests.get(url, headers=headers, params=params, timeout=10)
            except Exception:
                pass
            stop_event.wait(60)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return stop_event


def create_session(proxy_url: str, proxy_token: str, startup_timeout: int = 180) -> str:
    """Create a Jupyter session and return the kernel ID."""
    headers = {
        "X-Colab-Runtime-Proxy-Token": proxy_token,
        "X-Colab-Client-Agent": "vscode",
        "Content-Type": "application/json",
    }
    body = {
        "kernel": {"name": "python3"},
        "name": "colab-exec",
        "path": "colab-exec",
        "type": "notebook",
    }
    last_error = None
    deadline = time.time() + startup_timeout
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            r = requests.post(f"{proxy_url}/api/sessions", headers=headers, json=body, timeout=30)
            r.raise_for_status()
            data = r.json()
            kernel_id = data["kernel"]["id"]
            print(f"[colab-gpu] Kernel ready: {kernel_id}", file=sys.stderr)
            return kernel_id
        except Exception as e:
            last_error = e
            remaining = int(deadline - time.time())
            if remaining <= 0:
                break
            print(f"[colab-gpu] Waiting for runtime readiness (attempt {attempt}, {remaining}s left)...", file=sys.stderr)
            time.sleep(3)
    raise RuntimeError(f"Timed out creating kernel session: {last_error}")


def _make_colab_input_reply(client_session_id: str, colab_msg_id, err: str = None) -> dict:
    value = {"type": "colab_reply", "colab_msg_id": colab_msg_id}
    if err:
        value["error"] = err
    return {
        "header": {
            "msg_id": uuid.uuid4().hex,
            "msg_type": "input_reply",
            "session": client_session_id,
            "date": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "username": "username",
            "version": "5.0",
        },
        "content": {"value": value},
        "channel": "stdin",
        "metadata": {},
        "parent_header": {},
    }


def execute_code(
    proxy_url: str,
    proxy_token: str,
    kernel_id: str,
    code: str,
    timeout: int = 300,
    access_token: str = None,
    endpoint: str = None,
) -> tuple[str, str, int]:
    """Execute code on a Colab kernel via WebSocket.
    Returns (stdout, stderr, exit_code).
    """
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    execute_session_id = uuid.uuid4().hex
    ws_url = proxy_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_url}/api/kernels/{kernel_id}/channels?session_id={execute_session_id}"

    ws = websocket.create_connection(
        ws_url,
        header=[f"X-Colab-Runtime-Proxy-Token: {proxy_token}", "X-Colab-Client-Agent: vscode"],
        timeout=timeout,
    )

    msg_id = uuid.uuid4().hex
    execute_msg = {
        "header": {
            "msg_id": msg_id, "msg_type": "execute_request",
            "username": "colab-exec", "session": execute_session_id, "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code, "silent": False, "store_history": True,
            "user_expressions": {}, "allow_stdin": False, "stop_on_error": True,
        },
        "channel": "shell",
    }
    ws.send(json.dumps(execute_msg))

    had_error = False
    saw_idle = False
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            raw = ws.recv()
        except websocket.WebSocketTimeoutException:
            continue
        if not raw:
            continue

        msg = json.loads(raw)
        msg_type = msg.get("msg_type") or msg.get("header", {}).get("msg_type", "")
        content = msg.get("content", {})

        if msg_type == "colab_request":
            metadata = msg.get("metadata", {})
            request_type = metadata.get("colab_request_type")
            colab_msg_id = metadata.get("colab_msg_id")
            auth_type = (content.get("request", {}).get("authType", "") if isinstance(content, dict) else "")
            auth_type = str(auth_type).lower()
            if request_type == "request_auth" and colab_msg_id is not None:
                error_text = None
                if not access_token or not endpoint:
                    error_text = "missing auth context for credentials propagation"
                elif auth_type not in EPHEMERAL_AUTH_TYPES:
                    error_text = f"unsupported auth type: {auth_type}"
                else:
                    try:
                        dry = propagate_credentials(access_token, endpoint, auth_type, dry_run=True)
                        if dry.get("success"):
                            propagate_credentials(access_token, endpoint, auth_type, dry_run=False)
                        elif dry.get("unauthorizedRedirectUri"):
                            error_text = f"{auth_type} requires interactive browser consent: {dry['unauthorizedRedirectUri']}"
                        else:
                            error_text = f"{auth_type} dry-run failed: {dry}"
                    except Exception as e:
                        error_text = f"{auth_type} propagation failed: {e}"
                reply = _make_colab_input_reply(execute_session_id, colab_msg_id, error_text)
                ws.send(json.dumps(reply))
                if error_text:
                    stderr_buf.write(f"[colab-gpu] Warning: {error_text}\n")
            continue

        parent_msg_id = msg.get("parent_header", {}).get("msg_id")
        if parent_msg_id != msg_id:
            continue

        if msg_type == "stream":
            text = content.get("text", "")
            if content.get("name", "stdout") == "stdout":
                stdout_buf.write(text)
            else:
                stderr_buf.write(text)
        elif msg_type == "execute_result":
            text = content.get("data", {}).get("text/plain", "")
            if text:
                stdout_buf.write(text + "\n")
        elif msg_type == "display_data":
            text = content.get("data", {}).get("text/plain", "")
            if text:
                stdout_buf.write(text + "\n")
        elif msg_type == "error":
            had_error = True
            ename = content.get("ename", "Error")
            evalue = content.get("evalue", "")
            stderr_buf.write(f"\n{ename}: {evalue}\n")
            for line in content.get("traceback", []):
                stderr_buf.write(re.sub(r"\x1b\[[0-9;]*m", "", line) + "\n")
        elif msg_type == "status":
            if content.get("execution_state") == "idle":
                saw_idle = True
                break

    ws.close()
    if not saw_idle:
        stderr_buf.write("[colab-gpu] ERROR: Timed out waiting for kernel execution to finish.\n")
        had_error = True

    return stdout_buf.getvalue(), stderr_buf.getvalue(), 1 if had_error else 0
