# mcp-colab-gpu

Extended MCP server for Google Colab GPU/TPU runtimes

> Based on [mcp-server-colab-exec](https://github.com/pdwi2020/mcp-server-colab-exec) by [Paritosh Dwivedi](https://github.com/pdwi2020) (MIT License). Thank you for the original idea and implementation! / 素晴らしいアイデアと実装に感謝します！

MCP server that allocates Google Colab GPU/TPU runtimes and executes Python code on them. Lets any MCP-compatible AI assistant -- Claude Code, Claude Desktop, Gemini CLI, Cline, and others -- run GPU/TPU-accelerated code (CUDA, PyTorch, TensorFlow, JAX) without local GPU hardware.

## What's different from the original

| Feature | mcp-server-colab-exec | mcp-colab-gpu |
|---|---|---|
| GPU support | T4, L4 | **T4, L4, A100, H100, G4** |
| TPU support | -- | **V5E1, V6E1** |
| High-memory runtime | -- | **Supported** |
| Google Drive integration | -- | **Upload / Download / Fetch / Save** |
| Background execution | -- | **Non-blocking with poll** |
| Runtime release | Manual | **Automatic on completion** |
| Input validation | -- | **Accelerator + timeout validation** |
| Path traversal protection | -- | **.py-only + resolved symlinks** |
| Zip slip protection | -- | **Member path validation** |
| Token file permissions | Default | **0600 (owner-only)** |
| Token refresh error logging | Silent | **Logged with re-auth fallback** |

## Supported accelerators

| Accelerator | VRAM / Memory | Tier |
|---|---|---|
| `T4` | 16 GB | Free |
| `L4` | 22 GB | Colab Pro |
| `A100` | 40 GB | Colab Pro / Pro+ |
| `H100` | 80 GB | Colab Pro+ |
| `G4` | 95 GB | Colab Pro+ |
| `V5E1` | TPU v5e-1 | Colab Pro+ |
| `V6E1` | TPU v6e-1 | Colab Pro+ |

## Prerequisites

- Python 3.10+
- A Google account with access to [Google Colab](https://colab.research.google.com)
- On first run, a browser window opens for OAuth2 consent. The token is cached at `~/.config/colab-exec/token.json` for subsequent runs.

## Installation

### With uvx (recommended)

```bash
uvx mcp-colab-gpu
```

### With pip

```bash
pip install mcp-colab-gpu
```

### Claude Code configuration

Add to your project's `.mcp.json` or `~/.claude/.mcp.json`:

```json
{
  "mcpServers": {
    "colab-gpu": {
      "command": "uvx",
      "args": ["mcp-colab-gpu"]
    }
  }
}
```

Or via the CLI:

```bash
# Project-local (this project only)
claude mcp add colab-gpu -- uvx mcp-colab-gpu

# User-global (available in all projects)
claude mcp add --scope user colab-gpu -- uvx mcp-colab-gpu
```

### Claude Desktop configuration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "colab-gpu": {
      "command": "uvx",
      "args": ["mcp-colab-gpu"]
    }
  }
}
```

## Tools

### `colab_execute`

Execute inline Python code on a Colab GPU/TPU runtime.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `code` | string | -- | Python code to execute (required) |
| `accelerator` | string | `"T4"` | GPU/TPU type: `T4`, `L4`, `A100`, `H100`, `G4`, `V5E1`, `V6E1` |
| `high_memory` | bool | `false` | Enable high-memory runtime (more RAM) |
| `timeout` | int | `300` | Max execution time in seconds (10--3600) |
| `background` | bool | `false` | Run in background (non-blocking). Returns a `job_id` to poll via `colab_poll`. Incompatible with `drive_fetch`/`drive_save`. |
| `drive_fetch` | string | `""` | JSON mapping Drive paths to Colab paths. Files are downloaded from Google Drive **before** your code runs. Example: `'{"colab_data/train.csv": "/content/train.csv"}'` |
| `drive_save` | string | `""` | JSON mapping Colab paths to Drive paths. Files are uploaded to Google Drive **after** your code finishes (with a freshly obtained token). Example: `'{"/content/model.pt": "results/model.pt"}'` |

Returns JSON with per-cell output, errors, and stderr.

### `colab_execute_file`

Execute a local `.py` file on a Colab GPU/TPU runtime.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file_path` | string | -- | Path to a local `.py` file (required) |
| `accelerator` | string | `"T4"` | GPU/TPU type: `T4`, `L4`, `A100`, `H100`, `G4`, `V5E1`, `V6E1` |
| `high_memory` | bool | `false` | Enable high-memory runtime (more RAM) |
| `timeout` | int | `300` | Max execution time in seconds (10--3600) |

Returns JSON with per-cell output, errors, and stderr.

### `colab_execute_notebook`

Execute code and collect all generated artifacts (images, CSVs, models, etc.).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `code` | string | -- | Python code to execute (required) |
| `output_dir` | string | -- | Local directory for downloaded artifacts (required) |
| `accelerator` | string | `"T4"` | GPU/TPU type: `T4`, `L4`, `A100`, `H100`, `G4`, `V5E1`, `V6E1` |
| `high_memory` | bool | `false` | Enable high-memory runtime (more RAM) |
| `timeout` | int | `300` | Max execution time in seconds (10--3600) |

Artifacts are downloaded as a zip and extracted into `output_dir`.

### `colab_poll`

Poll a background job for its current status and results.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `job_id` | string | -- | The job identifier returned by `colab_execute(..., background=true)` (required) |

Returns JSON with `job_id`, `status` (`starting`, `running`, `completed`, `failed`), `accelerator`, timestamps, and `result` (when completed) or `error` (when failed).

### `colab_jobs`

List all tracked background jobs.

No parameters. Returns a JSON array of job summaries including `job_id`, `status`, `accelerator`, and timestamps.

### `colab_drive_upload`

Upload a local file to Google Drive.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `local_path` | string | -- | Path to the local file to upload (required) |
| `drive_folder` | string | `"colab_data"` | Target folder path on Google Drive (relative to MyDrive). Nested paths like `data/train` are supported. Folders are created automatically. |

Returns JSON with `drive_file_id`, `filename`, `drive_folder`, and `colab_path`.

### `colab_drive_download`

Download a file from Google Drive to a local path.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `drive_path` | string | -- | File path on Google Drive relative to MyDrive (e.g. `results/model.pt`) (required) |
| `local_path` | string | -- | Local destination path (required) |

Returns JSON with `local_path`, `drive_file_id`, and `size_bytes`.

### `colab_version`

Return the mcp-colab-gpu server version. No parameters.

## Examples

### Just ask in natural language / プロンプトで話しかけるだけ

Once this MCP server is configured, you can use GPU/TPU from any MCP-compatible AI assistant just by asking in natural language. No API calls, no boilerplate -- just describe what you want.

このMCPサーバーを設定すれば、Claude CodeなどのMCP対応AIアシスタントに日本語で話しかけるだけでGPU/TPUが使えます。APIコールもボイラープレートも不要です。

**Claude Code / Claude Desktop / Gemini CLI / Cline:**

> "A100でResNet-50をCIFAR-10で10エポック学習させて、精度を教えて"

> "Train a ResNet-50 on CIFAR-10 for 10 epochs using A100 and report the accuracy"

> "H100でphi-2を動かして、量子コンピューティングについて説明させて"

> "Run phi-2 on H100 and ask it to explain quantum computing"

> "手元のtrain.csvをDriveにアップして、GPUで前処理してから結果をダウンロードして"

> "Upload my local train.csv to Drive, preprocess it on GPU, and download the results"

> "バックグラウンドでStable Diffusionの画像を100枚生成して、終わったら教えて"

> "Generate 100 images with Stable Diffusion in the background and let me know when it's done"

The AI assistant automatically selects the right tool, writes the code, executes it on Colab, and returns the results -- all from a single prompt.

AIアシスタントが適切なツールを自動選択し、コードを書き、Colabで実行し、結果を返します。すべてひとつのプロンプトから。

---

### Check GPU availability

```
colab_execute(code="import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))")
```

### Matrix multiplication benchmark on A100

```
colab_execute(
    code="""
import torch
import time

device = torch.device('cuda')
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# Benchmark: large matrix multiplication
a = torch.randn(8192, 8192, device=device)
b = torch.randn(8192, 8192, device=device)

torch.cuda.synchronize()
start = time.time()
c = torch.mm(a, b)
torch.cuda.synchronize()
elapsed = time.time() - start

tflops = 2 * 8192**3 / elapsed / 1e12
print(f"8192x8192 matmul: {elapsed:.3f}s ({tflops:.1f} TFLOPS)")
""",
    accelerator="A100",
    high_memory=True,
)
```

### Background execution with polling

```
# 1. Start a long-running job in the background
colab_execute(
    code="""
import torch
model = torch.hub.load('pytorch/vision', 'resnet152', pretrained=True)
model = model.cuda().eval()
dummy = torch.randn(64, 3, 224, 224, device='cuda')
with torch.no_grad():
    for i in range(100):
        output = model(dummy)
print(f"Completed 100 inference batches")
""",
    accelerator="A100",
    background=True,
)
# Returns: {"job_id": "abc123def456", "status": "starting"}

# 2. Poll for results
colab_poll(job_id="abc123def456")
# Returns: {"job_id": "...", "status": "running", ...}

# 3. When complete
colab_poll(job_id="abc123def456")
# Returns: {"job_id": "...", "status": "completed", "result": {...}}

# 4. List all jobs
colab_jobs()
# Returns: {"jobs": [...], "count": 1}
```

### Google Drive: upload data, process on GPU, download results

```
# 1. Upload training data to Drive
colab_drive_upload(local_path="./data/train.csv", drive_folder="colab_data")

# 2. Execute on Colab with Drive fetch + save
colab_execute(
    code="""
import pandas as pd
import torch

df = pd.read_csv('/content/train.csv')
print(f"Loaded {len(df)} rows")

# ... GPU training ...

torch.save(model.state_dict(), '/content/model.pt')
print("Model saved")
""",
    accelerator="A100",
    drive_fetch='{"colab_data/train.csv": "/content/train.csv"}',
    drive_save='{"/content/model.pt": "results/model.pt"}',
    timeout=600,
)

# 3. Download results from Drive
colab_drive_download(drive_path="results/model.pt", local_path="./model.pt")
```

### LLM inference on H100

```
colab_execute(
    code="""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)

prompt = "Explain quantum computing in one paragraph:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
""",
    accelerator="H100",
    timeout=600,
)
```

### Train and download model weights

```
colab_execute_notebook(
    code="""
import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
model = model.cuda()
# ... training code ...
torch.save(model.state_dict(), '/tmp/model.pt')
print("Model saved!")
""",
    output_dir="./outputs",
    accelerator="T4",
)
```

## Authentication

### Colab runtime

On first use, the server opens a browser window for Google OAuth2 consent. The access token and refresh token are cached at `~/.config/colab-exec/token.json`. Subsequent runs use the cached token and refresh it automatically.

The OAuth2 client credentials are the same ones used by the official Google Colab VS Code extension (`google.colab@0.3.0`). They are intentionally public.

### Google Drive (v0.3.0+)

Drive tools (`colab_drive_upload`, `colab_drive_download`, `drive_fetch`, `drive_save`) use a **separate OAuth2 token** with the `drive.file` scope. On first Drive operation, a second browser window opens for consent. The token is cached at `~/.config/colab-exec/drive_token.json`.

This separate token means:
- Users who only use `colab_execute` are never asked for Drive permissions.
- Drive access is limited to the `drive.file` scope (only files created by this app).

To use your own OAuth client instead of the built-in one, set the environment variable `MCP_DRIVE_CLIENT_JSON` or place a `drive_client.json` file at `~/.config/colab-exec/drive_client.json`.

## Design: zero CU waste

Colab compute units (CU) are consumed while a runtime is allocated. This server releases runtimes **immediately** after code execution finishes -- not when the client disconnects or the session times out.

- **Sync execution**: allocate -> execute -> release (all in one call)
- **Background execution**: allocate -> execute -> release on completion (not on poll)
- **Drive fetch/save**: all three steps (fetch, execute, save) run on the same allocation, released once all steps finish

This means you only pay for actual computation time, never for idle runtimes.

## Security improvements

This fork includes the following security hardening over the original:

- **Path validation in `colab_execute_file`**: Only `.py` files are accepted. Paths are resolved through `pathlib.Path.resolve()` to prevent symlink-based traversal attacks.
- **Zip slip protection in `colab_execute_notebook`**: Every member in a downloaded artifact zip is validated to ensure its resolved path stays within the target `output_dir`, preventing directory traversal via crafted zip entries.
- **Path traversal protection in Drive operations**: Local paths in `colab_drive_upload` and `colab_drive_download` reject `..` segments to prevent directory traversal.
- **Drive query injection prevention**: Folder and file names are escaped before being used in Drive API query strings.
- **Token file permissions**: The OAuth token cache files (`token.json`, `drive_token.json`) are created with `0o600` permissions (owner read/write only) using `os.open` instead of plain `open`.
- **Input validation**: The `accelerator` parameter is validated against the known set of supported accelerators, and `timeout` is bounded to 10--3600 seconds, before any network calls are made.
- **Token refresh error logging**: When automatic token refresh fails, the error is logged to stderr with a warning message before falling back to re-authentication, instead of silently discarding the error.

## Troubleshooting

**"GPU quota exceeded"** -- Colab has usage limits. Wait and retry, or use a different Google account.

**"Timed out creating kernel session"** -- The runtime took too long to start. Retry -- Colab sometimes has delays during peak usage.

**"Authentication failed"** -- Delete `~/.config/colab-exec/token.json` and re-authenticate.

**"Drive authentication failed"** -- Delete `~/.config/colab-exec/drive_token.json` and re-authenticate.

**"A background job is already running"** -- Only one background job can run at a time (Colab single-GPU constraint). Wait for the current job to finish or poll its status with `colab_poll`.

**OAuth browser window doesn't open** -- Ensure you're running in an environment with a browser. For headless servers, authenticate on a machine with a browser first and copy the token file.

## Acknowledgments / 謝辞

### English

Special thanks to [Paritosh Dwivedi (@pdwi2020)](https://github.com/pdwi2020) for creating [mcp-server-colab-exec](https://github.com/pdwi2020/mcp-server-colab-exec). Your original idea of bridging MCP-compatible AI assistants with Google Colab GPU runtimes was brilliant and made this extended fork possible. We hope this project helps more developers leverage cloud GPUs from their local AI workflows.

### 日本語

[Paritosh Dwivedi (@pdwi2020)](https://github.com/pdwi2020) 氏の [mcp-server-colab-exec](https://github.com/pdwi2020/mcp-server-colab-exec) に心より感謝いたします。MCP対応のAIアシスタントとGoogle ColabのGPUランタイムを橋渡しするという素晴らしいアイデアと実装のおかげで、この拡張フォークを実現することができました。このプロジェクトが、より多くの開発者がローカルのAIワークフローからクラウドGPUを活用する助けになることを願っています。

## About the maintainer

Masaya Hirano -- CEO of [Mio System Co., Ltd.](https://miosys.co.jp/), CRO of [TrustedAI Corporation](https://www.trusted-ai.co/en)

## License

[MIT](LICENSE)

Original work: Copyright (c) 2026 Paritosh Dwivedi
Extended fork: Copyright (c) 2026 Masaya Hirano
