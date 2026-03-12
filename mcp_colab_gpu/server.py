"""MCP server for executing Python code on Google Colab GPU/TPU runtimes.

Exposes three tools via the FastMCP API:
- colab_execute: Run inline Python code on a Colab GPU/TPU
- colab_execute_file: Run a local .py file on a Colab GPU/TPU
- colab_execute_notebook: Run code and collect generated artifacts

Original: mcp-server-colab-exec v0.1.0 by Paritosh Dwivedi (MIT License)
Extended: mcp-colab-gpu by Masaya Hirano --- all Colab Pro GPUs, high-memory, security fixes.
"""

import base64
import json
import os
import pathlib
import re
import zipfile

from mcp.server.fastmcp import FastMCP

from .colab_runtime import (
    allocate_runtime,
    create_session,
    execute_code,
    get_credentials,
    start_keepalive,
    unassign_runtime,
    validate_params,
)

mcp = FastMCP("colab-gpu")

CELL_START = "===CELL_START_{n}==="
CELL_END = "===CELL_END_{n}==="
ARTIFACT_B64_START = "ARTIFACT_BASE64_START"
ARTIFACT_B64_END = "ARTIFACT_BASE64_END"


def _wrap_cells(code: str) -> tuple[str, int]:
    """Wrap code in cell-boundary markers so output can be parsed per-cell."""
    raw_cells = re.split(r"\n{2,}", code.strip())
    cells = [c.strip() for c in raw_cells if c.strip()]
    if not cells:
        cells = [code]
    wrapped_parts = []
    for i, cell in enumerate(cells):
        marker_start = CELL_START.format(n=i)
        marker_end = CELL_END.format(n=i)
        wrapped_parts.append(
            f'print("{marker_start}", flush=True)\n'
            f"{cell}\n"
            f'print("{marker_end}", flush=True)'
        )
    return "\n\n".join(wrapped_parts), len(cells)


def _parse_cell_output(stdout: str, num_cells: int) -> list[dict]:
    cells = []
    for i in range(num_cells):
        start_marker = CELL_START.format(n=i)
        end_marker = CELL_END.format(n=i)
        pattern = re.escape(start_marker) + r"\n?(.*?)\n?" + re.escape(end_marker)
        match = re.search(pattern, stdout, re.DOTALL)
        cell_stdout = match.group(1).strip() if match else ""
        cells.append({"cell_num": i, "stdout": cell_stdout, "status": "ok" if match else "no_output"})
    return cells


def _extract_artifact_b64(stdout: str) -> str | None:
    """Extract base64-encoded artifact data from stdout."""
    pattern = re.escape(ARTIFACT_B64_START) + r"\n(.*?)\n" + re.escape(ARTIFACT_B64_END)
    match = re.search(pattern, stdout, re.DOTALL)
    return match.group(1).strip() if match else None


def _strip_artifact_b64(stdout: str) -> str:
    """Remove the base64 artifact block from stdout to prevent context pollution."""
    pattern = re.escape(ARTIFACT_B64_START) + r"\n.*?\n" + re.escape(ARTIFACT_B64_END)
    return re.sub(pattern, "", stdout, flags=re.DOTALL)


def _run_on_colab(code: str, accelerator: str, high_memory: bool, timeout: int) -> tuple[str, str, int]:
    validate_params(accelerator, timeout)
    creds = get_credentials()
    access_token = creds.token
    assignment = allocate_runtime(access_token, accelerator, high_memory)
    stop_event = start_keepalive(access_token, assignment["endpoint"])
    try:
        kernel_id = create_session(assignment["proxy_url"], assignment["proxy_token"])
        stdout, stderr, exit_code = execute_code(
            assignment["proxy_url"], assignment["proxy_token"], kernel_id, code,
            timeout=timeout, access_token=access_token, endpoint=assignment["endpoint"],
        )
    finally:
        stop_event.set()
        unassign_runtime(access_token, assignment["endpoint"])
    return stdout, stderr, exit_code


def _validate_file_path(raw: str) -> pathlib.Path:
    """Validate file path: must exist, must be .py."""
    resolved = pathlib.Path(os.path.expanduser(raw)).resolve()
    if resolved.suffix != ".py":
        raise ValueError(f"Only .py files are allowed, got: '{resolved.suffix}'")
    if not resolved.is_file():
        raise FileNotFoundError(f"File not found: {resolved}")
    return resolved


def _safe_extract_zip(zip_path: str, output_dir: str) -> list[str]:
    """Extract zip with zip-slip protection. Returns list of extracted filenames."""
    resolved_output = pathlib.Path(output_dir).resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = (resolved_output / member.filename).resolve()
            if not str(member_path).startswith(str(resolved_output)):
                raise ValueError(f"Zip slip detected: {member.filename}")
        artifact_files = zf.namelist()
        zf.extractall(output_dir)
    return artifact_files


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def colab_execute(
    code: str,
    accelerator: str = "T4",
    high_memory: bool = False,
    timeout: int = 300,
) -> str:
    """Execute Python code on a Google Colab GPU/TPU runtime.

    Allocates a GPU or TPU, runs the code, and returns structured JSON
    with per-cell output, errors, and stderr.

    Args:
        code: Python code to execute on the Colab runtime.
        accelerator: Hardware accelerator type. Default: "T4".
            GPU types:
              "T4"   - NVIDIA Tesla T4 (16 GB, free-tier)
              "L4"   - NVIDIA L4 (24 GB, Colab Pro)
              "A100"  - NVIDIA A100 (40 GB, Colab Pro/Pro+)
              "H100"  - NVIDIA H100 (80 GB, Colab Pro+)
              "G4"   - NVIDIA G4 (Colab Pro+)
            TPU types:
              "V5E1" - TPU v5e-1 (Colab Pro+)
              "V6E1" - TPU v6e-1 (Colab Pro+)
        high_memory: Enable high-memory runtime (more RAM). Default: False.
        timeout: Max execution time in seconds. Default: 300.
    """
    wrapped, num_cells = _wrap_cells(code)
    stdout, stderr, rc = _run_on_colab(wrapped, accelerator, high_memory, timeout)
    cells = _parse_cell_output(stdout, num_cells)
    errors = [c for c in cells if c["status"] != "ok"] if rc != 0 else []
    return json.dumps({"cells": cells, "errors": errors, "stderr": stderr, "exit_code": rc}, indent=2)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def colab_execute_file(
    file_path: str,
    accelerator: str = "T4",
    high_memory: bool = False,
    timeout: int = 300,
) -> str:
    """Execute a local Python file on a Google Colab GPU/TPU runtime.

    Reads the file contents and sends them for execution on a Colab runtime.
    Only .py files are allowed for security.

    Args:
        file_path: Path to a local .py file to execute on Colab.
        accelerator: Hardware accelerator type. Default: "T4".
            GPU types:
              "T4"   - NVIDIA Tesla T4 (16 GB, free-tier)
              "L4"   - NVIDIA L4 (24 GB, Colab Pro)
              "A100"  - NVIDIA A100 (40 GB, Colab Pro/Pro+)
              "H100"  - NVIDIA H100 (80 GB, Colab Pro+)
              "G4"   - NVIDIA G4 (Colab Pro+)
            TPU types:
              "V5E1" - TPU v5e-1 (Colab Pro+)
              "V6E1" - TPU v6e-1 (Colab Pro+)
        high_memory: Enable high-memory runtime (more RAM). Default: False.
        timeout: Max execution time in seconds. Default: 300.
    """
    try:
        resolved = _validate_file_path(file_path)
    except (ValueError, FileNotFoundError) as e:
        return json.dumps({"error": str(e)})

    code = resolved.read_text()
    wrapped, num_cells = _wrap_cells(code)
    stdout, stderr, rc = _run_on_colab(wrapped, accelerator, high_memory, timeout)
    cells = _parse_cell_output(stdout, num_cells)
    errors = [c for c in cells if c["status"] != "ok"] if rc != 0 else []
    return json.dumps({"cells": cells, "errors": errors, "stderr": stderr, "exit_code": rc}, indent=2)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def colab_execute_notebook(
    code: str,
    output_dir: str,
    accelerator: str = "T4",
    high_memory: bool = False,
    timeout: int = 300,
) -> str:
    """Execute Python code on Colab GPU/TPU and collect generated artifacts.

    Runs the code, then scans the runtime for output files (images, CSVs,
    models, etc.), zips them, and downloads to a local directory.

    Args:
        code: Python code to execute on the Colab runtime.
        output_dir: Local directory to save the artifacts zip and extracted files.
        accelerator: Hardware accelerator type. Default: "T4".
            GPU types:
              "T4"   - NVIDIA Tesla T4 (16 GB, free-tier)
              "L4"   - NVIDIA L4 (24 GB, Colab Pro)
              "A100"  - NVIDIA A100 (40 GB, Colab Pro/Pro+)
              "H100"  - NVIDIA H100 (80 GB, Colab Pro+)
              "G4"   - NVIDIA G4 (Colab Pro+)
            TPU types:
              "V5E1" - TPU v5e-1 (Colab Pro+)
              "V6E1" - TPU v6e-1 (Colab Pro+)
        high_memory: Enable high-memory runtime (more RAM). Default: False.
        timeout: Max execution time in seconds. Default: 300.
    """
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    artifact_code = '''

# --- colab-exec artifact collection ---
import os, zipfile, base64, glob

_artifact_dir = "/tmp/colab_artifacts"
os.makedirs(_artifact_dir, exist_ok=True)

_scan_dirs = ["/tmp", os.getcwd(), "/content"]
_skip_prefixes = ["/tmp/colab_artifacts", "/tmp/."]
_collected = []
for _sd in _scan_dirs:
    if not os.path.isdir(_sd):
        continue
    for _root, _dirs, _files in os.walk(_sd):
        _dirs[:] = [d for d in _dirs if not d.startswith('.')]
        if any(_root.startswith(p) for p in _skip_prefixes):
            continue
        for _f in _files:
            _fp = os.path.join(_root, _f)
            if _f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg', '.csv', '.json',
                           '.txt', '.pt', '.pth', '.h5', '.pkl', '.npy', '.npz',
                           '.onnx', '.mp4', '.wav', '.mp3', '.pdf')):
                try:
                    if os.path.getsize(_fp) < 50 * 1024 * 1024:
                        _collected.append(_fp)
                except OSError:
                    pass

if _collected:
    _zip_path = "/tmp/colab_artifacts.zip"
    with zipfile.ZipFile(_zip_path, 'w', zipfile.ZIP_DEFLATED) as _zf:
        for _fp in _collected:
            _zf.write(_fp, os.path.basename(_fp))
    with open(_zip_path, 'rb') as _zfh:
        _b64 = base64.b64encode(_zfh.read()).decode('ascii')
    print("ARTIFACT_BASE64_START")
    print(_b64)
    print("ARTIFACT_BASE64_END")
    print(f"[colab-gpu] Collected {len(_collected)} artifact(s)", flush=True)
else:
    print("[colab-gpu] No artifacts found to collect", flush=True)
'''
    full_code = code + "\n\n" + artifact_code
    wrapped, num_cells = _wrap_cells(full_code)
    stdout, stderr, rc = _run_on_colab(wrapped, accelerator, high_memory, timeout)

    # Extract artifact base64 before stripping it from stdout.
    b64_data = _extract_artifact_b64(stdout)
    # Strip the base64 block so it does not leak into cell output JSON,
    # which would pollute the AI client's context window.
    clean_stdout = _strip_artifact_b64(stdout)

    cells = _parse_cell_output(clean_stdout, num_cells)
    errors = [c for c in cells if c["status"] != "ok"] if rc != 0 else []

    artifact_files = []
    artifacts_zip_path = None
    if b64_data:
        try:
            zip_bytes = base64.b64decode(b64_data)
            artifacts_zip_path = os.path.join(output_dir, "colab_artifacts.zip")
            with open(artifacts_zip_path, "wb") as f:
                f.write(zip_bytes)
            artifact_files = _safe_extract_zip(artifacts_zip_path, output_dir)
        except Exception as e:
            errors.append({"artifact_error": str(e)})

    return json.dumps({
        "cells": cells, "errors": errors, "artifacts_zip": artifacts_zip_path,
        "artifact_files": artifact_files, "stderr": stderr, "exit_code": rc,
    }, indent=2)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
