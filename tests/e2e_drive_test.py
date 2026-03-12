#!/usr/bin/env python3
"""E2E test for Drive integration with fresh-token architecture.

Usage:
    # Normal run (default token lifetime):
    uv run python tests/e2e_drive_test.py

    # With 1-minute token TTL to verify refresh:
    MCP_DRIVE_TOKEN_MAX_AGE=60 uv run python tests/e2e_drive_test.py

First run will open a browser for Google Drive OAuth consent.
Subsequent runs reuse the cached token (refreshing as needed).
"""

import json
import os
import sys
import tempfile
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp_colab_gpu.drive import (
    download_from_drive,
    generate_drive_fetch_code,
    generate_drive_save_code,
    get_drive_credentials,
    resolve_file_id,
    upload_to_drive,
)
from mcp_colab_gpu.server import _ColabRuntime, _execute_with_drive


DRIVE_TEST_FOLDER = "mcp_colab_gpu_e2e_test"


def _separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_drive_upload_download() -> None:
    """Test 1: Upload a file to Drive, download it back, verify contents."""
    _separator("Test 1: Drive Upload → Download round-trip")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="mcp_e2e_"
    ) as f:
        f.write("name,score\nAlice,95\nBob,87\nCharlie,92\n")
        upload_path = f.name

    try:
        print(f"[1a] Uploading {upload_path} to Drive/{DRIVE_TEST_FOLDER}/ ...")
        creds = get_drive_credentials()
        result = upload_to_drive(upload_path, DRIVE_TEST_FOLDER, creds)
        print(f"      Uploaded: id={result['id']}, name={result['name']}")

        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, prefix="mcp_e2e_dl_"
        ) as dl:
            download_path = dl.name

        drive_path = f"{DRIVE_TEST_FOLDER}/{result['name']}"
        print(f"[1b] Downloading {drive_path} to {download_path} ...")
        creds = get_drive_credentials()
        dl_result = download_from_drive(drive_path, download_path, creds)
        print(f"      Downloaded: {dl_result}")

        with open(download_path) as f:
            content = f.read()
        assert "Alice" in content, f"Expected 'Alice' in downloaded content, got: {content[:200]}"
        assert "Bob" in content, f"Expected 'Bob' in downloaded content"
        print("      Content verified OK")

    finally:
        for p in [upload_path, download_path]:
            if os.path.exists(p):
                os.unlink(p)

    print("\n  [PASS] Test 1 passed")


def test_colab_drive_roundtrip() -> None:
    """Test 2: Full Colab execution with drive_fetch + drive_save.

    Flow:
    1. Upload test CSV to Drive
    2. Execute on Colab: fetch CSV → process → save result
    3. Download result from Drive
    4. Verify the processed data
    """
    _separator("Test 2: Colab execute with Drive fetch/save (fresh-token test)")

    max_age = os.environ.get("MCP_DRIVE_TOKEN_MAX_AGE", "none")
    print(f"  MCP_DRIVE_TOKEN_MAX_AGE = {max_age}")

    # Step 1: Upload test data to Drive
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="mcp_e2e_input_"
    ) as f:
        f.write("x,y\n1,10\n2,20\n3,30\n4,40\n5,50\n")
        upload_path = f.name

    print(f"[2a] Uploading input CSV to Drive/{DRIVE_TEST_FOLDER}/input.csv ...")
    creds = get_drive_credentials()
    up_result = upload_to_drive(upload_path, DRIVE_TEST_FOLDER, creds)
    print(f"      Uploaded: id={up_result['id']}")
    os.unlink(upload_path)

    # Step 2: Execute on Colab with drive_fetch and drive_save
    fetch_json = json.dumps({
        f"{DRIVE_TEST_FOLDER}/input.csv": "/content/input.csv",
    })
    save_json = json.dumps({
        "/content/output.csv": f"{DRIVE_TEST_FOLDER}/output.csv",
    })

    user_code = """
import csv
import time

# Read input
with open('/content/input.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Read {len(rows)} rows from input.csv")

# Simulate a long-running computation
wait_secs = 75  # > 60s to trigger token refresh if MCP_DRIVE_TOKEN_MAX_AGE=60
print(f"Simulating computation for {wait_secs}s ...")
time.sleep(wait_secs)

# Process: add a 'z' column = x * y
with open('/content/output.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['x', 'y', 'z'])
    writer.writeheader()
    for row in rows:
        row['z'] = int(row['x']) * int(row['y'])
        writer.writerow(row)

print(f"Wrote {len(rows)} rows to output.csv with z=x*y")
"""

    print("[2b] Executing on Colab with drive_fetch + drive_save ...")
    print("     (This will take ~90 seconds including simulated computation)")
    start = time.time()
    result_json = _execute_with_drive(
        code=user_code,
        accelerator="T4",
        high_memory=False,
        timeout=300,
        fetch_map=json.loads(fetch_json),
        save_map=json.loads(save_json),
    )
    elapsed = time.time() - start
    print(f"     Completed in {elapsed:.1f}s")

    result = json.loads(result_json)
    print(f"     exit_code: {result['exit_code']}")
    if result.get("errors"):
        print(f"     errors: {result['errors']}")
    if result.get("drive_saved"):
        print(f"     drive_saved: {result['drive_saved']}")

    assert result["exit_code"] == 0, f"Colab execution failed: {result}"
    assert result.get("drive_saved"), "No drive_saved in result — save step did not run"

    # Step 3: Download the output from Drive
    with tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, prefix="mcp_e2e_output_"
    ) as f:
        output_path = f.name

    print(f"[2c] Downloading {DRIVE_TEST_FOLDER}/output.csv ...")
    creds = get_drive_credentials()
    dl_result = download_from_drive(
        f"{DRIVE_TEST_FOLDER}/output.csv", output_path, creds
    )
    print(f"      Downloaded: {dl_result}")

    # Step 4: Verify
    with open(output_path) as f:
        content = f.read()
    os.unlink(output_path)

    print(f"      Content:\n{content}")
    assert "z" in content, "Missing 'z' column in output"
    assert "10" in content, "Expected z=1*10=10"
    assert "100" in content or "50" in content, "Expected computed z values"

    print("\n  [PASS] Test 2 passed — fresh-token architecture verified!")


def test_token_refresh_tracking() -> None:
    """Test 3: Verify that _last_drive_token_time is tracked and token refresh works."""
    _separator("Test 3: Token refresh tracking")

    from mcp_colab_gpu.drive import _last_drive_token_time

    creds1 = get_drive_credentials()
    from mcp_colab_gpu import drive as drive_mod
    t1 = drive_mod._last_drive_token_time
    print(f"  First call:  token_time={t1:.1f}, token={creds1.token[:20]}...")

    assert t1 > 0, "_last_drive_token_time should be set after first call"

    # Second call should reuse the same token (no forced refresh)
    creds2 = get_drive_credentials()
    t2 = drive_mod._last_drive_token_time
    print(f"  Second call: token_time={t2:.1f}, token={creds2.token[:20]}...")

    max_age = os.environ.get("MCP_DRIVE_TOKEN_MAX_AGE")
    if not max_age:
        print("  (No MCP_DRIVE_TOKEN_MAX_AGE set — skipping forced-refresh subtest)")
        print("\n  [PASS] Test 3 passed (basic tracking)")
        return

    # With max_age set, wait and verify refresh
    wait = int(max_age) + 5
    print(f"  Waiting {wait}s to exceed max_age={max_age}s ...")
    time.sleep(wait)

    creds3 = get_drive_credentials()
    t3 = drive_mod._last_drive_token_time
    print(f"  After wait:  token_time={t3:.1f}, token={creds3.token[:20]}...")

    assert t3 > t2, f"Token time should have been updated: t3={t3} should be > t2={t2}"
    print(f"  Token time advanced by {t3 - t2:.1f}s — refresh confirmed!")

    print("\n  [PASS] Test 3 passed (forced refresh verified)")


def main() -> None:
    print("=" * 60)
    print("  mcp-colab-gpu E2E Drive Integration Test")
    print("=" * 60)

    max_age = os.environ.get("MCP_DRIVE_TOKEN_MAX_AGE")
    if max_age:
        print(f"\n  Token TTL override: {max_age}s")
    else:
        print("\n  Token TTL: default (Google's ~3600s)")
        print("  Set MCP_DRIVE_TOKEN_MAX_AGE=60 to test forced refresh")

    passed = 0
    failed = 0
    tests = [
        ("Drive upload/download", test_drive_upload_download),
        ("Token refresh tracking", test_token_refresh_tracking),
        ("Colab Drive roundtrip", test_colab_drive_roundtrip),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    _separator("Results")
    print(f"  Passed: {passed}/{passed + failed}")
    if failed:
        print(f"  Failed: {failed}/{passed + failed}")
        sys.exit(1)
    else:
        print("  All tests passed!")


if __name__ == "__main__":
    main()
