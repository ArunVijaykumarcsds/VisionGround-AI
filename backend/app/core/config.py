"""
app/core/config.py
==================
Centralised application settings loaded from environment variables / .env file.
All settings are typed and validated by Pydantic Settings v2.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration derived from environment variables.

    Variable precedence (highest → lowest):
        1. Process environment variables
        2. .env file in the working directory
        3. Default values declared here
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_path: str = Field(
        default="nvidia/LocateAnything-3B",
        description="HF model ID or local path to LocateAnything-3B.",
    )
    hf_token: str | None = Field(
        default=None,
        alias="HF_TOKEN",
        description="Hugging Face API token for gated models.",
    )
    device: str = Field(
        default="auto",
        description="Inference device: 'cuda', 'cpu', or 'auto'.",
    )
    torch_dtype: Literal["bfloat16", "float16", "float32"] = Field(
        default="bfloat16",
        description="Torch dtype for model weights.",
    )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    generation_mode: Literal["hybrid", "fast", "slow"] = Field(
        default="hybrid",
        description="LocateAnything generation mode.",
    )
    max_new_tokens: int = Field(
        default=2048,
        ge=64,
        le=8192,
        description="Max tokens generated per request.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p nucleus sampling.",
    )
    repetition_penalty: float = Field(
        default=1.1,
        ge=1.0,
        le=2.0,
        description="Repetition penalty.",
    )

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)
    log_level: Literal["debug", "info", "warning", "error"] = Field(default="info")

    # ------------------------------------------------------------------
    # Upload limits
    # ------------------------------------------------------------------
    max_image_size_mb: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum image upload size in megabytes.",
    )
    max_image_dimension: int = Field(
        default=2560,
        ge=64,
        le=8192,
        description="Maximum image width or height in pixels.",
    )
    allowed_image_types: str = Field(
        default="image/jpeg,image/png,image/webp,image/bmp",
        description="Comma-separated allowed MIME types.",
    )

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    cors_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        description="Comma-separated allowed CORS origins.",
    )

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------
    enable_memory_logging: bool = Field(default=True)
    slow_inference_threshold_ms: int = Field(default=5000)

    # ------------------------------------------------------------------
    # Computed helpers
    # ------------------------------------------------------------------

    @property
    def cors_origins_list(self) -> List[str]:
        """Return CORS origins as a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_image_types_list(self) -> List[str]:
        """Return allowed MIME types as a list."""
        return [t.strip() for t in self.allowed_image_types.split(",") if t.strip()]

    @property
    def max_image_size_bytes(self) -> int:
        """Return maximum upload size in bytes."""
        return self.max_image_size_mb * 1024 * 1024

    @field_validator("device")
    @classmethod
    def resolve_auto_device(cls, v: str) -> str:
        """Replace 'auto' with the actual available device at import time."""
        if v == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the global Settings singleton.

    Using @lru_cache ensures the .env file is parsed exactly once,
    even across multiple module imports.
    """
    return Settings()
