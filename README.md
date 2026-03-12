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
claude mcp add colab-gpu -- uvx mcp-colab-gpu
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

## Examples

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

On first use, the server opens a browser window for Google OAuth2 consent. The access token and refresh token are cached at `~/.config/colab-exec/token.json`. Subsequent runs use the cached token and refresh it automatically.

The OAuth2 client credentials are the same ones used by the official Google Colab VS Code extension (`google.colab@0.3.0`). They are intentionally public.

## Security improvements

This fork includes the following security hardening over the original:

- **Path validation in `colab_execute_file`**: Only `.py` files are accepted. Paths are resolved through `pathlib.Path.resolve()` to prevent symlink-based traversal attacks.
- **Zip slip protection in `colab_execute_notebook`**: Every member in a downloaded artifact zip is validated to ensure its resolved path stays within the target `output_dir`, preventing directory traversal via crafted zip entries.
- **Token file permissions**: The OAuth token cache file (`~/.config/colab-exec/token.json`) is created with `0o600` permissions (owner read/write only) using `os.open` instead of plain `open`.
- **Input validation**: The `accelerator` parameter is validated against the known set of supported accelerators, and `timeout` is bounded to 10--3600 seconds, before any network calls are made.
- **Token refresh error logging**: When automatic token refresh fails, the error is logged to stderr with a warning message before falling back to re-authentication, instead of silently discarding the error.

## Troubleshooting

**"GPU quota exceeded"** -- Colab has usage limits. Wait and retry, or use a different Google account.

**"Timed out creating kernel session"** -- The runtime took too long to start. Retry -- Colab sometimes has delays during peak usage.

**"Authentication failed"** -- Delete `~/.config/colab-exec/token.json` and re-authenticate.

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
