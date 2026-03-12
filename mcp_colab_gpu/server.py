"""MCP server for executing Python code on Google Colab GPU/TPU runtimes.

Exposes tools via the FastMCP API:
- colab_execute: Run inline Python code on a Colab GPU/TPU (sync or background)
- colab_execute_file: Run a local .py file on a Colab GPU/TPU
- colab_execute_notebook: Run code and collect generated artifacts
- colab_poll: Poll a background job for status and results
- colab_jobs: List all tracked background jobs
- colab_drive_upload: Upload a local file to Google Drive
- colab_drive_download: Download a file from Google Drive
- colab_version: Return the server version

Original: mcp-server-colab-exec v0.1.0 by Paritosh Dwivedi (MIT License)
Extended: mcp-colab-gpu by Masaya Hirano --- all Colab Pro GPUs, high-memory, security fixes.
"""

import asyncio
import base64
import json
import os
import pathlib
import re
import zipfile

from mcp.server.fastmcp import FastMCP

from .background import JobStatus, JobStore, run_background_job
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

_job_store = JobStore()


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
    """Remove the base64 artifact block from stdout to prevent context pollution.

    Handles both complete blocks (START...END) and truncated blocks where
    the end marker is missing due to timeout or mid-execution errors.
    """
    pattern = (
        re.escape(ARTIFACT_B64_START)
        + r"\n.*?"
        + r"(?:\n" + re.escape(ARTIFACT_B64_END) + r"|$)"
    )
    return re.sub(pattern, "", stdout, flags=re.DOTALL)


def _run_on_colab(code: str, accelerator: str, high_memory: bool, timeout: int) -> tuple[str, str, int]:
    """Blocking function that allocates a runtime, executes code, and releases.

    This runs in a thread via asyncio.to_thread() for both sync and background paths.
    The runtime is ALWAYS released in the finally block.
    """
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


def _format_sync_result(stdout: str, stderr: str, rc: int, num_cells: int) -> str:
    """Format execution results into the standard JSON response."""
    cells = _parse_cell_output(stdout, num_cells)
    errors = [c for c in cells if c["status"] != "ok"] if rc != 0 else []
    return json.dumps({"cells": cells, "errors": errors, "stderr": stderr, "exit_code": rc}, indent=2)


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


def _parse_drive_json(raw: str) -> dict[str, str]:
    """Parse a drive_fetch/drive_save JSON string into a dict."""
    if not raw:
        return {}
    return json.loads(raw)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def colab_execute(
    code: str,
    accelerator: str = "T4",
    high_memory: bool = False,
    timeout: int = 300,
    background: bool = False,
    drive_fetch: str = "",
    drive_save: str = "",
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
        background: Run in background (non-blocking). Default: False.
            When True, returns immediately with a job_id that can be polled
            via colab_poll. Incompatible with drive_fetch/drive_save.
        drive_fetch: JSON mapping Drive paths to Colab paths. Files are
            downloaded from Google Drive BEFORE your code runs.
            Example: '{"colab_data/train.csv": "/content/train.csv"}'
            Requires prior colab_drive_upload to place files on Drive.
        drive_save: JSON mapping Colab paths to Drive paths. Files are
            uploaded to Google Drive AFTER your code finishes, using
            a freshly obtained token (safe for long-running tasks).
            Example: '{"/content/model.pt": "results/model.pt"}'
    """
    fetch_map = _parse_drive_json(drive_fetch)
    save_map = _parse_drive_json(drive_save)
    if background:
        if fetch_map or save_map:
            return json.dumps({
                "error": "background=True is incompatible with drive_fetch/drive_save"
            })
        job_id = await _job_store.create_if_no_active(accelerator)
        if job_id is None:
            active = await _job_store.active_job_id()
            return json.dumps({
                "error": "A background job is already running",
                "active_job_id": active,
            })
        wrapped, num_cells = _wrap_cells(code)
        def _format_bg_result(stdout: str, stderr: str, rc: int) -> str:
            return _format_sync_result(stdout, stderr, rc, num_cells)
        asyncio.create_task(run_background_job(
            _job_store,
            job_id,
            _run_on_colab,
            code=wrapped,
            accelerator=accelerator,
            high_memory=high_memory,
            timeout=timeout,
            format_result_fn=_format_bg_result,
        ))
        return json.dumps({"job_id": job_id, "status": "starting"})
    if fetch_map or save_map:
        return await asyncio.to_thread(
            _execute_with_drive, code, accelerator, high_memory, timeout, fetch_map, save_map,
        )
    wrapped, num_cells = _wrap_cells(code)
    stdout, stderr, rc = await asyncio.to_thread(
        _run_on_colab, wrapped, accelerator, high_memory, timeout,
    )
    return _format_sync_result(stdout, stderr, rc, num_cells)


def _execute_with_drive(
    code: str,
    accelerator: str,
    high_memory: bool,
    timeout: int,
    fetch_map: dict[str, str],
    save_map: dict[str, str],
) -> str:
    """Execute code on Colab with Drive fetch (pre) and save (post) steps.

    All three steps run on the same runtime allocation:
    1. [Pre]  Fresh token -> download files from Drive to Colab filesystem
    2. [Main] Execute user code (may run for hours)
    3. [Post] Fresh token -> upload results from Colab filesystem to Drive
    """
    from .drive import (
        generate_drive_fetch_code,
        generate_drive_save_code,
        get_drive_credentials,
        resolve_file_id,
    )

    drive_errors = []
    validate_params(accelerator, timeout)
    creds = get_credentials()
    access_token = creds.token
    assignment = allocate_runtime(access_token, accelerator, high_memory)
    stop_event = start_keepalive(access_token, assignment["endpoint"])
    try:
        kernel_id = create_session(assignment["proxy_url"], assignment["proxy_token"])
        # --- Step 1: Drive fetch (pre-execution) ---
        if fetch_map:
            try:
                drive_creds = get_drive_credentials()
                file_mappings = []
                for drive_path, colab_path in fetch_map.items():
                    fid = resolve_file_id(drive_path, drive_creds)
                    file_mappings.append({"file_id": fid, "dest_path": colab_path})
                fetch_code = generate_drive_fetch_code(file_mappings, drive_creds.token)
                _, fetch_stderr, fetch_rc = execute_code(
                    assignment["proxy_url"], assignment["proxy_token"],
                    kernel_id, fetch_code, timeout=120,
                    access_token=access_token, endpoint=assignment["endpoint"],
                )
                if fetch_rc != 0:
                    drive_errors.append({"drive_fetch_error": fetch_stderr.strip()})
            except Exception as e:
                drive_errors.append({"drive_fetch_error": str(e)})
        # --- Step 2: User code ---
        wrapped, num_cells = _wrap_cells(code)
        stdout, stderr, rc = execute_code(
            assignment["proxy_url"], assignment["proxy_token"],
            kernel_id, wrapped, timeout=timeout,
            access_token=access_token, endpoint=assignment["endpoint"],
        )
        cells = _parse_cell_output(stdout, num_cells)
        errors = [c for c in cells if c["status"] != "ok"] if rc != 0 else []
        # --- Step 3: Drive save (post-execution) ---
        drive_saved = []
        if save_map and rc == 0:
            try:
                drive_creds = get_drive_credentials()
                save_mappings = []
                for colab_path, drive_path in save_map.items():
                    parts = drive_path.strip("/").split("/")
                    filename = parts[-1]
                    folder = "/".join(parts[:-1]) if len(parts) > 1 else ""
                    save_mappings.append({
                        "local_path": colab_path,
                        "drive_folder": folder,
                        "filename": filename,
                    })
                save_code = generate_drive_save_code(save_mappings, drive_creds.token)
                _save_stdout, save_stderr, save_rc = execute_code(
                    assignment["proxy_url"], assignment["proxy_token"],
                    kernel_id, save_code, timeout=300,
                    access_token=access_token, endpoint=assignment["endpoint"],
                )
                if save_rc != 0:
                    drive_errors.append({"drive_save_error": save_stderr.strip()})
                else:
                    drive_saved = [
                        {"colab_path": m["local_path"], "drive_path": dp}
                        for m, dp in zip(save_mappings, save_map.values())
                    ]
            except Exception as e:
                drive_errors.append({"drive_save_error": str(e)})
    finally:
        stop_event.set()
        unassign_runtime(access_token, assignment["endpoint"])
    result = {
        "cells": cells, "errors": errors + drive_errors,
        "stderr": stderr, "exit_code": rc,
    }
    if drive_saved:
        result["drive_saved"] = drive_saved
    return json.dumps(result, indent=2)


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False})
async def colab_poll(job_id: str) -> str:
    """Poll a background job for its current status and results.

    Use this after launching a background execution with
    colab_execute(..., background=True) to check progress
    and retrieve results when complete.

    Args:
        job_id: The job identifier returned by colab_execute.
    """
    record = await _job_store.get(job_id)
    if record is None:
        return json.dumps({"error": f"Unknown job_id: {job_id}"})
    info: dict = {
        "job_id": record.job_id,
        "status": record.status.value,
        "accelerator": record.accelerator,
        "created_at": record.created_at,
    }
    if record.completed_at is not None:
        info["completed_at"] = record.completed_at
    if record.status == JobStatus.COMPLETED and record.result is not None:
        info["result"] = json.loads(record.result)
    if record.status == JobStatus.FAILED and record.error is not None:
        info["error"] = record.error
    return json.dumps(info, indent=2)


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False})
async def colab_jobs() -> str:
    """List all tracked background jobs.

    Returns a JSON array of job summaries including job_id, status,
    accelerator type, and timestamps.
    """
    records = await _job_store.list_all()
    jobs = []
    for record in records:
        entry: dict = {
            "job_id": record.job_id,
            "status": record.status.value,
            "accelerator": record.accelerator,
            "created_at": record.created_at,
        }
        if record.completed_at is not None:
            entry["completed_at"] = record.completed_at
        jobs.append(entry)
    return json.dumps({"jobs": jobs, "count": len(jobs)}, indent=2)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def colab_execute_file(
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
    stdout, stderr, rc = await asyncio.to_thread(
        _run_on_colab, wrapped, accelerator, high_memory, timeout,
    )
    return _format_sync_result(stdout, stderr, rc, num_cells)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def colab_execute_notebook(
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
    stdout, stderr, rc = await asyncio.to_thread(
        _run_on_colab, wrapped, accelerator, high_memory, timeout,
    )
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


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def colab_drive_upload(
    local_path: str,
    drive_folder: str = "colab_data",
) -> str:
    """Upload a local file to Google Drive.

    The file will be accessible from Colab via Drive mount:
      from google.colab import drive
      drive.mount('/content/drive')
      # Access at /content/drive/MyDrive/<drive_folder>/<filename>

    Args:
        local_path: Path to the local file to upload.
        drive_folder: Target folder path on Google Drive (relative to MyDrive).
            Nested paths like 'data/train' are supported.
            Folders are created automatically if they don't exist.
            Default: "colab_data".
    """
    from .drive import get_drive_credentials, upload_to_drive

    try:
        creds = get_drive_credentials()
        result = await asyncio.to_thread(upload_to_drive, local_path, drive_folder, creds)
        return json.dumps({
            "status": "uploaded",
            "drive_file_id": result["id"],
            "filename": result["name"],
            "drive_folder": drive_folder,
            "colab_path": f"/content/drive/MyDrive/{drive_folder}/{result['name']}" if drive_folder else f"/content/drive/MyDrive/{result['name']}",
        }, indent=2)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Drive upload failed: {e}"})


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def colab_drive_download(
    drive_path: str,
    local_path: str,
) -> str:
    """Download a file from Google Drive to a local path.

    Use this to retrieve results saved to Drive by Colab execution:
      colab_execute(code=\"\"\"
        import torch
        torch.save(model, '/content/drive/MyDrive/results/model.pt')
      \"\"\")
      colab_drive_download(drive_path='results/model.pt', local_path='./model.pt')

    Args:
        drive_path: File path on Google Drive relative to MyDrive
            (e.g. 'results/model.pt' or 'colab_data/output.csv').
        local_path: Local destination path where the file will be saved.
    """
    from .drive import get_drive_credentials, download_from_drive

    try:
        creds = get_drive_credentials()
        result = await asyncio.to_thread(download_from_drive, drive_path, local_path, creds)
        return json.dumps({
            "status": "downloaded",
            "local_path": result["local_path"],
            "drive_file_id": result["drive_file_id"],
            "size_bytes": result["size"],
        }, indent=2)
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        return json.dumps({"error": f"Drive download failed: {e}"})


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False})
async def colab_version() -> str:
    """Return the mcp-colab-gpu server version."""
    from . import __version__
    return json.dumps({"version": __version__})


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
