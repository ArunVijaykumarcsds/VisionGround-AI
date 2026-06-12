#!/usr/bin/env python3
"""
scripts/download_model.py
==========================
Standalone script to pre-download nvidia/LocateAnything-3B weights
from HuggingFace Hub to the local cache before starting the server.

This is useful for:
  - Docker builds that need weights baked in (not recommended — too large)
  - Pre-warming the HF cache on a fresh server before the first request
  - CI/CD pipelines that cache HF_HOME between runs

Usage
-----
    # Download to default HF cache (~/.cache/huggingface/hub/)
    python scripts/download_model.py

    # Download to a specific directory
    python scripts/download_model.py --cache-dir /data/models

    # Verify an existing download without re-downloading
    python scripts/download_model.py --verify-only

    # Use a specific HF token
    python scripts/download_model.py --token hf_xxx

Environment variables
---------------------
    HF_TOKEN        HuggingFace API token (alternative to --token flag)
    HF_HOME         Cache directory (alternative to --cache-dir flag)
    MODEL_PATH      Model ID to download (default: nvidia/LocateAnything-3B)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-download nvidia/LocateAnything-3B weights.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", "nvidia/LocateAnything-3B"),
        help="HuggingFace model ID or local path. (default: nvidia/LocateAnything-3B)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace API token. Can also be set via HF_TOKEN env var.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_HOME"),
        help="Local directory for HF Hub cache. Default: ~/.cache/huggingface/hub",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify that all files are present; do not download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist.",
    )
    return parser.parse_args()


def check_requirements() -> None:
    """Ensure required packages are importable."""
    missing = []
    for pkg in ("transformers", "huggingface_hub"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print("        Run: pip install -r requirements.txt")
        sys.exit(1)


def download_model(
    model_path: str,
    token: str | None,
    cache_dir: str | None,
    force: bool,
) -> None:
    """Download all model files using snapshot_download."""
    from huggingface_hub import snapshot_download, model_info
    from huggingface_hub.utils import HfHubHTTPError

    print(f"[INFO]  Model:     {model_path}")
    print(f"[INFO]  Cache dir: {cache_dir or '~/.cache/huggingface/hub'}")
    print(f"[INFO]  Token:     {'set' if token else 'NOT SET (may fail for gated models)'}")
    print()

    # Verify model exists and is accessible.
    print("[INFO]  Checking model accessibility on HuggingFace Hub...")
    try:
        info = model_info(model_path, token=token)
        print(f"[INFO]  Model found: {info.id}  ({info.private=})")
    except HfHubHTTPError as e:
        if "401" in str(e) or "403" in str(e):
            print("[ERROR] Authentication failed.")
            print("        This model is gated. Set HF_TOKEN to a valid token.")
            print("        Generate one at: https://huggingface.co/settings/tokens")
            sys.exit(1)
        elif "404" in str(e):
            print(f"[ERROR] Model not found: {model_path}")
            sys.exit(1)
        else:
            print(f"[ERROR] Hub error: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Could not reach HuggingFace Hub: {e}")
        sys.exit(1)

    # Download all files.
    print()
    print("[INFO]  Starting download (this may take 10–30 minutes on first run)...")
    print("[INFO]  Files: model-00001-of-00002.safetensors (~3.4 GB)")
    print("[INFO]         model-00002-of-00002.safetensors (~3.4 GB)")
    print("[INFO]         model.safetensors.index.json")
    print("[INFO]         tokenizer, processor, config files")
    print()

    t0 = time.perf_counter()

    try:
        local_dir = snapshot_download(
            repo_id=model_path,
            token=token,
            cache_dir=cache_dir,
            force_download=force,
            ignore_patterns=["*.md", "*.txt", "*.png"],   # skip docs/images
        )
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        sys.exit(1)

    elapsed = (time.perf_counter() - t0) / 60
    print()
    print(f"[OK]    Download complete in {elapsed:.1f} minutes.")
    print(f"[OK]    Local path: {local_dir}")
    return local_dir


def verify_model(local_dir: str) -> bool:
    """Check that all critical files exist in the local cache."""
    required_files = [
        "config.json",
        "tokenizer_config.json",
        "processor_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "modeling_locateanything.py",
        "processing_locateanything.py",
        "configuration_locateanything.py",
    ]

    print(f"[INFO]  Verifying files in: {local_dir}")
    all_ok = True
    for fname in required_files:
        path = Path(local_dir) / fname
        exists = path.exists()
        size   = path.stat().st_size if exists else 0
        marker = "OK  " if exists and size > 0 else "MISS"
        if not (exists and size > 0):
            all_ok = False
        size_str = f"{size / 1e9:.2f} GB" if size > 1e6 else f"{size:,} B"
        print(f"  [{marker}] {fname}  ({size_str})")

    print()
    if all_ok:
        print("[OK]    All required files present.")
    else:
        print("[WARN]  Some files are missing. Re-run without --verify-only to download.")
    return all_ok


def main() -> None:
    check_requirements()
    args = parse_args()

    # Set token in environment for transformers to pick up.
    if args.token:
        os.environ["HF_TOKEN"] = args.token
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir

    if args.verify_only:
        # Find the local cache directory.
        from huggingface_hub import try_to_load_from_cache
        import re
        # Try to locate the cached snapshot
        cache_base = args.cache_dir or Path.home() / ".cache" / "huggingface" / "hub"
        model_slug = re.sub(r"[/\\]", "--", args.model_path)
        model_dir  = Path(cache_base) / f"models--{model_slug}"

        if not model_dir.exists():
            print(f"[WARN]  No local cache found at {model_dir}")
            print("        Run without --verify-only to download.")
            sys.exit(1)

        # Find the latest snapshot
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            print(f"[WARN]  No snapshots directory found in {model_dir}")
            sys.exit(1)

        snapshots = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime)
        if not snapshots:
            print("[WARN]  No snapshots found.")
            sys.exit(1)

        latest = snapshots[-1]
        ok = verify_model(str(latest))
        sys.exit(0 if ok else 1)

    else:
        local_dir = download_model(
            model_path=args.model_path,
            token=args.token,
            cache_dir=args.cache_dir,
            force=args.force,
        )
        verify_model(local_dir)
        print()
        print("[DONE]  You can now start the server:")
        print(f"        MODEL_PATH={local_dir} uvicorn app.main:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
