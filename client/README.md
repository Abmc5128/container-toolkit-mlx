# mlx-container Python Client

Python client for the **MLX Container Toolkit** — runs inside a Linux container and
offloads inference to the Apple Silicon GPU on the macOS host over vsock.

```
Linux container (guest VM)           macOS host
┌─────────────────────────┐         ┌────────────────────────────┐
│  your Python code        │  vsock  │  mlx-container-daemon      │
│  mlx_container.generate ◄─────────► Swift gRPC + MLX inference  │
│                          │  :2048  │  Metal GPU                  │
└─────────────────────────┘         └────────────────────────────┘
```

## Install

```bash
pip install mlx-container
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add mlx-container
```

Requires Python 3.10+ and `grpcio>=1.60`.  No `protobuf` package needed — the
client uses JSON-over-gRPC to match the Swift server's `JSONEncoder` serialisation.

## Quick start

```python
from mlx_container import load_model, generate, list_models

# Load a model onto the host GPU (downloads from HuggingFace if needed)
load_model("mlx-community/Llama-3.2-1B-4bit")

# Generate text — blocks until complete
result = generate(
    "Explain quantum computing in one paragraph",
    model="mlx-community/Llama-3.2-1B-4bit",
    max_tokens=256,
    temperature=0.7,
)
print(result.text)
print(f"Generated {result.completion_tokens} tokens at {result.tokens_per_second:.1f} tok/s")

# Stream tokens as they are generated
for token in generate("Write a haiku about Apple Silicon", model="mlx-community/Llama-3.2-1B-4bit", stream=True):
    print(token, end="", flush=True)
print()

# List loaded models
for m in list_models():
    print(m.model_id, "—", m.memory_used_bytes // 1024 // 1024, "MB")
```

## Chat messages

```python
from mlx_container import generate
from mlx_container.types import ChatMessage

result = generate(
    model="mlx-community/Llama-3.2-1B-4bit",
    messages=[
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="What is MLX?"),
    ],
    max_tokens=512,
)
print(result.text)
```

## OpenAI-compatible interface

```python
from mlx_container.compat.openai import ChatCompletion

response = ChatCompletion.create(
    model="mlx-community/Llama-3.2-1B-4bit",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    max_tokens=256,
)
print(response.choices[0].message.content)
print(f"Usage: {response.usage.total_tokens} tokens")
```

Streaming:

```python
for chunk in ChatCompletion.create(..., stream=True):
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

## mlx-lm drop-in replacement

```python
# Instead of: from mlx_lm import load, generate
from mlx_container.compat.mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-1B-4bit")
text = generate(model, tokenizer, prompt="Hello world", max_tokens=64)
print(text)
```

## Direct client

```python
from mlx_container import MLXContainerClient

with MLXContainerClient() as client:
    # Health check
    pong = client.ping()
    print(pong)  # {"status": "ok", "version": "0.1.0", "uptime_seconds": 42.0}

    # GPU info
    gpu = client.get_gpu_status()
    print(f"{gpu.device_name}: {gpu.available_memory_bytes // 1024**3} GB free")
```

### Custom target (development / TCP)

```bash
# On the host, start the daemon bound to a TCP port for testing
export MLX_DAEMON_HOST=localhost
export MLX_DAEMON_PORT=50051
```

```python
client = MLXContainerClient(target="localhost:50051")
```

### vsock environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLX_VSOCK_CID` | `2` | vsock context ID (2 = host) |
| `MLX_VSOCK_PORT` | `2048` | vsock port |
| `MLX_DAEMON_HOST` | `localhost` | TCP host (fallback outside container) |
| `MLX_DAEMON_PORT` | `50051` | TCP port (fallback outside container) |

## API reference

### `generate(prompt, model, messages, max_tokens, temperature, top_p, stream)`

Returns `GenerateResult` or `Iterator[str]` when `stream=True`.

**`GenerateResult` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Full generated text |
| `prompt_tokens` | `int` | Input token count |
| `completion_tokens` | `int` | Output token count |
| `prompt_time_seconds` | `float` | Time to process prompt |
| `generation_time_seconds` | `float` | Time to generate tokens |
| `tokens_per_second` | `float` | Generation throughput |

### `load_model(model_id, alias, memory_budget_bytes)`

Loads a model on the host GPU.  Returns `True` on success, raises `RuntimeError`
on failure.

### `unload_model(model_id)`

Unloads a model from the host GPU.

### `list_models()`

Returns `list[ModelInfo]`.

**`ModelInfo` fields:** `model_id`, `alias`, `memory_used_bytes`, `is_loaded`, `model_type`

## Architecture

The client communicates over **vsock** (AF_VSOCK, CID 2, port 2048) — the same
low-latency host-guest channel used by Apple's container runtime.

Because `grpcio` does not natively support AF_VSOCK, the client transparently
bridges the vsock connection through a local TCP loopback proxy started in a
daemon thread.  No configuration is required.

The wire protocol is **JSON-over-gRPC** matching the Swift server's
`JSONMessageSerializer` / `JSONMessageDeserializer`.  Field names follow Swift's
camelCase convention (`modelID`, `maxTokens`, `topP`, etc.).

## Development

```bash
# Install with dev extras
uv pip install -e ".[dev]"

# Lint
ruff check client/

# Type-check
mypy client/mlx_container/

# Tests
pytest client/tests/
```
