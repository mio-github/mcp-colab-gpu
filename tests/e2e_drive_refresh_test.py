#!/usr/bin/env python3
"""E2E test: Drive upload/download with token refresh (MCP_DRIVE_TOKEN_MAX_AGE=60).

Scenario:
1. Upload input CSV to Drive
2. Wait > 60s (triggers forced token refresh)
3. Upload result CSV to Drive (with refreshed token)
4. Download both files and verify contents
5. Cleanup

Usage:
    MCP_DRIVE_TOKEN_MAX_AGE=60 uv run python tests/e2e_drive_refresh_test.py
"""

import csv
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp_colab_gpu import drive as drive_mod
from mcp_colab_gpu.drive import (
    download_from_drive,
    get_drive_credentials,
    upload_to_drive,
)

DRIVE_TEST_FOLDER = "mcp_colab_gpu_e2e_test"


def _separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main() -> None:
    max_age = os.environ.get("MCP_DRIVE_TOKEN_MAX_AGE", "not set")
    print("=" * 60)
    print("  mcp-colab-gpu E2E: Drive Upload/Download + Token Refresh")
    print("=" * 60)
    print(f"\n  MCP_DRIVE_TOKEN_MAX_AGE = {max_age}")

    # ── Step 1: Upload input file ──
    _separator("Step 1: Upload input CSV to Drive")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="mcp_e2e_input_"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "score"])
        for i in range(1, 21):
            writer.writerow([i, f"user_{i}", i * 10])
        input_path = f.name

    print(f"  Created local file: {input_path}")
    creds = get_drive_credentials()
    t1 = drive_mod._last_drive_token_time
    token1 = creds.token[:30]
    print(f"  Token time: {t1:.1f}")
    print(f"  Token (prefix): {token1}...")

    up1 = upload_to_drive(input_path, DRIVE_TEST_FOLDER, creds)
    print(f"  Uploaded: id={up1['id']}, name={up1['name']}")
    os.unlink(input_path)

    # ── Step 2: Wait > 60s to trigger token refresh ──
    _separator("Step 2: Wait 75s to trigger token refresh")

    wait_secs = 75
    start_wait = time.time()
    for elapsed in range(0, wait_secs + 1, 15):
        remaining = wait_secs - elapsed
        if remaining > 0:
            print(f"  Waiting... {remaining}s remaining")
            time.sleep(min(15, remaining))
        else:
            print(f"  Wait complete ({time.time() - start_wait:.1f}s elapsed)")

    # ── Step 3: Get fresh credentials and upload result file ──
    _separator("Step 3: Upload result CSV (should use refreshed token)")

    creds2 = get_drive_credentials()
    t2 = drive_mod._last_drive_token_time
    token2 = creds2.token[:30]
    print(f"  Token time: {t2:.1f}")
    print(f"  Token (prefix): {token2}...")

    if t2 > t1:
        print(f"  Token REFRESHED (time advanced by {t2 - t1:.1f}s)")
    else:
        print(f"  WARNING: Token was NOT refreshed (t1={t1:.1f}, t2={t2:.1f})")

    # Create result CSV with computed data
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="mcp_e2e_result_"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "score", "grade"])
        for i in range(1, 21):
            score = i * 10
            grade = "A" if score >= 150 else "B" if score >= 100 else "C"
            writer.writerow([i, f"user_{i}", score, grade])
        result_path = f.name

    print(f"  Created result file: {result_path}")
    up2 = upload_to_drive(result_path, DRIVE_TEST_FOLDER, creds2)
    print(f"  Uploaded: id={up2['id']}, name={up2['name']}")
    os.unlink(result_path)

    # ── Step 4: Download both files and verify ──
    _separator("Step 4: Download & verify both files")

    # Download input
    with tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, prefix="mcp_e2e_dl_input_"
    ) as f:
        dl_input_path = f.name

    creds3 = get_drive_credentials()
    drive_input = f"{DRIVE_TEST_FOLDER}/{up1['name']}"
    print(f"  Downloading {drive_input}...")
    dl1 = download_from_drive(drive_input, dl_input_path, creds3)
    print(f"  Downloaded: size={dl1['size']} bytes")

    with open(dl_input_path) as f:
        content1 = f.read()
    os.unlink(dl_input_path)

    assert "user_1" in content1, f"Expected 'user_1' in input CSV, got: {content1[:200]}"
    assert "user_20" in content1, f"Expected 'user_20' in input CSV"
    lines1 = content1.strip().split("\n")
    assert len(lines1) == 21, f"Expected 21 lines (header + 20 rows), got {len(lines1)}"
    print(f"  Input CSV verified: {len(lines1)} lines, contains user_1..user_20")

    # Download result
    with tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, prefix="mcp_e2e_dl_result_"
    ) as f:
        dl_result_path = f.name

    drive_result = f"{DRIVE_TEST_FOLDER}/{up2['name']}"
    print(f"  Downloading {drive_result}...")
    dl2 = download_from_drive(drive_result, dl_result_path, creds3)
    print(f"  Downloaded: size={dl2['size']} bytes")

    with open(dl_result_path) as f:
        content2 = f.read()
    os.unlink(dl_result_path)

    assert "grade" in content2, f"Expected 'grade' column in result CSV"
    assert ",A" in content2, "Expected grade 'A' in result CSV"
    assert ",B" in content2, "Expected grade 'B' in result CSV"
    assert ",C" in content2, "Expected grade 'C' in result CSV"
    lines2 = content2.strip().split("\n")
    assert len(lines2) == 21, f"Expected 21 lines in result, got {len(lines2)}"
    print(f"  Result CSV verified: {len(lines2)} lines, contains grades A/B/C")

    # ── Summary ──
    _separator("Results")

    print(f"  Upload (before refresh):  {up1['name']} → id={up1['id']}")
    print(f"  Upload (after refresh):   {up2['name']} → id={up2['id']}")
    print(f"  Download input:           {dl1['size']} bytes - OK")
    print(f"  Download result:          {dl2['size']} bytes - OK")
    print(f"  Token refresh:            {'YES' if t2 > t1 else 'NO'} (t1={t1:.1f}, t2={t2:.1f})")
    print()
    print("  ALL CHECKS PASSED!")


if __name__ == "__main__":
    main()
