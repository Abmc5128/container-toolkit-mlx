#!/usr/bin/env python3
"""
Hello MLX — Simple inference example for MLX Container Toolkit.

This script runs INSIDE a Linux container but uses the host's
Apple GPU for inference via the MLX Container Daemon.  Outside
a container it falls back to TCP (set MLX_DAEMON_HOST / MLX_DAEMON_PORT
or pass --host / --port).

Usage (inside container):
    container run --gpu --gpu-model mlx-community/Llama-3.2-1B-4bit \\
        ghcr.io/robotflow-labs/mlx-container:latest \\
        python3 inference.py

Usage (local TCP dev mode):
    python3 inference.py --host localhost --port 50051
    python3 inference.py --model mlx-community/Llama-3.2-3B-4bit
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import NoReturn


DEFAULT_MODEL = "mlx-community/Llama-3.2-1B-4bit"


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

class DaemonNotRunningError(RuntimeError):
    """Raised when the MLX Container Daemon cannot be reached."""


class ModelNotFoundError(RuntimeError):
    """Raised when the requested model is not available."""


def _die(message: str, exit_code: int = 1) -> NoReturn:
    print(f"\n[ERROR] {message}", file=sys.stderr)
    sys.exit(exit_code)


# ---------------------------------------------------------------------------
# Transport setup
# ---------------------------------------------------------------------------

def _configure_transport(host: str | None, port: int | None) -> None:
    """
    Override the default client target via env vars when explicit TCP
    coordinates are provided on the CLI.  The client library reads
    MLX_DAEMON_HOST / MLX_DAEMON_PORT on first use, so setting them here
    is sufficient.
    """
    if host is not None:
        os.environ["MLX_DAEMON_HOST"] = host
    if port is not None:
        os.environ["MLX_DAEMON_PORT"] = str(port)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hello MLX — GPU inference from a Linux container",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HuggingFace model ID to load and run",
    )
    parser.add_argument(
        "--host",
        default=None,
        help=(
            "Daemon TCP host for local dev mode. "
            "Inside a container vsock is used automatically."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Daemon TCP port for local dev mode.",
    )
    args = parser.parse_args()

    _configure_transport(args.host, args.port)

    # Import after env vars are set so the client picks up the right target.
    try:
        from mlx_container import generate, load_model, list_models
        from mlx_container._grpc_client import get_client
    except ImportError as exc:
        _die(
            f"mlx_container package not found: {exc}\n"
            "Install it with: pip install -e client/"
        )

    model = args.model

    print("=" * 60)
    print("MLX Container Toolkit — Hello World")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 0: Verify daemon is reachable
    # ------------------------------------------------------------------
    print("\n[0] Pinging daemon...")
    try:
        ping = get_client().ping()
        print(f"    Daemon status : {ping.get('status', 'unknown')}")
        print(f"    Version       : {ping.get('version', 'unknown')}")
        print(f"    Uptime        : {ping.get('uptime_seconds', 0):.0f}s")
    except Exception as exc:
        _die(
            f"Cannot reach MLX Container Daemon: {exc}\n\n"
            "Checklist:\n"
            "  - Inside a container: was --gpu passed to `container run`?\n"
            "  - Local dev mode    : is mlx-container-daemon running?\n"
            "  - TCP override      : set --host / --port or MLX_DAEMON_HOST / MLX_DAEMON_PORT"
        )

    # ------------------------------------------------------------------
    # Step 1: Load a model on the host GPU
    # ------------------------------------------------------------------
    print(f"\n[1] Loading model: {model}")
    try:
        load_model(model)
        print("    Model loaded successfully.")
    except RuntimeError as exc:
        _die(
            f"Failed to load model '{model}': {exc}\n\n"
            "Possible causes:\n"
            "  - The model ID is incorrect or not on HuggingFace.\n"
            "  - Insufficient GPU memory (try a smaller / more quantised model).\n"
            "  - The daemon has no internet access to download the model."
        )

    # ------------------------------------------------------------------
    # Step 2: List loaded models
    # ------------------------------------------------------------------
    print("\n[2] Loaded models:")
    try:
        loaded = list_models()
        if not loaded:
            print("  (none)")
        for m in loaded:
            mem_mb = m.memory_used_bytes / 1024 / 1024
            status = "loaded" if m.is_loaded else "unloaded"
            print(f"  - {m.model_id}  [{status}, {mem_mb:.0f} MB]")
    except Exception as exc:
        print(f"  Warning: could not list models: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 3: Generate text (non-streaming)
    # ------------------------------------------------------------------
    print("\n[3] Generating text (non-streaming)...")
    prompt = "Explain what makes Apple Silicon unique in 3 sentences."
    try:
        result = generate(
            prompt=prompt,
            model=model,
            max_tokens=150,
            temperature=0.7,
        )
    except RuntimeError as exc:
        _die(f"Generation failed: {exc}")

    print(f"\n  Prompt   : {prompt}")
    print(f"  Response :\n{result.text}")
    print(f"\n  Stats:")
    print(f"    Tokens/sec      : {result.tokens_per_second:.1f}")
    print(f"    Prompt time     : {result.prompt_time_seconds:.3f}s")
    print(f"    Generation time : {result.generation_time_seconds:.3f}s")
    print(f"    Prompt tokens   : {result.prompt_tokens}")
    print(f"    Completion tokens: {result.completion_tokens}")

    # ------------------------------------------------------------------
    # Step 4: Streaming generation
    # ------------------------------------------------------------------
    print("\n[4] Streaming generation:")
    print("Q: Write a haiku about containers")
    print("A: ", end="", flush=True)
    try:
        for token in generate(
            prompt="Write a haiku about containers",
            model=model,
            max_tokens=50,
            stream=True,
        ):
            print(token, end="", flush=True)
        print("\n")
    except Exception as exc:
        print(f"\n  Warning: streaming interrupted: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Done! GPU inference from inside a Linux container.")
    print("=" * 60)


if __name__ == "__main__":
    main()
