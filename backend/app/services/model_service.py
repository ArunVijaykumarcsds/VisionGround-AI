"""
app/services/model_service.py
==============================
Singleton service that owns the LocateAnything-3B model lifecycle.

Implementation follows the official LocateAnythingWorker pattern from:
https://huggingface.co/nvidia/LocateAnything-3B

Responsibilities
----------------
* Load AutoTokenizer, AutoProcessor, and AutoModel once at startup.
* Keep all components resident in GPU (or CPU) memory.
* Expose a unified predict() method and task-specific helpers.
* Parse model outputs into structured bounding-box / point lists.
* Log inference timing and memory usage on every call.
* Raise clear, typed exceptions on failure.
"""

from __future__ import annotations

import os
import re
import time
import threading
from typing import Any, Dict, List, Optional

import psutil
import torch
from PIL import Image

from app.core.config import get_settings
from app.core.logging import logger
from app.schemas.detection import BoundingBox, DetectionPoint

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ModelNotLoadedError(RuntimeError):
    """Raised when inference is attempted before the model is ready."""


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails size or format checks."""


# ---------------------------------------------------------------------------
# Singleton guard
# ---------------------------------------------------------------------------
_instance_lock = threading.Lock()
_service_instance: Optional["ModelService"] = None


def get_model_service() -> "ModelService":
    """
    Return the global ModelService singleton.

    Thread-safe: the first call creates the instance; subsequent calls
    return the cached object immediately.
    """
    global _service_instance
    if _service_instance is None:
        with _instance_lock:
            if _service_instance is None:
                _service_instance = ModelService()
    return _service_instance


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------

class ModelService:
    """
    Wraps LocateAnything-3B for production inference.

    Mirrors the official LocateAnythingWorker from the HF model card.
    Do not instantiate directly — use get_model_service().
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: Any = None
        self._tokenizer: Any = None
        self._processor: Any = None
        self._device: str = self.settings.device
        self._dtype: torch.dtype = self._resolve_dtype()
        self._loaded: bool = False
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public: lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load the model, tokenizer, and processor into memory.

        Idempotent — safe to call multiple times; only loads once.
        Raises RuntimeError if loading fails.
        """
        with self._load_lock:
            if self._loaded:
                logger.info("Model already loaded — skipping duplicate load.")
                return

            logger.info(
                "Loading LocateAnything-3B | path={} | device={} | dtype={}",
                self.settings.model_path,
                self._device,
                self.settings.torch_dtype,
            )

            t0 = time.perf_counter()

            # Set HF token in environment so transformers picks it up.
            if self.settings.hf_token:
                os.environ["HF_TOKEN"] = self.settings.hf_token
                logger.debug("HF_TOKEN set from settings.")

            try:
                from transformers import AutoModel, AutoProcessor, AutoTokenizer

                # --- Tokenizer ---
                logger.info("Loading tokenizer ...")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.model_path,
                    trust_remote_code=True,
                )

                # --- Processor ---
                logger.info("Loading processor ...")
                self._processor = AutoProcessor.from_pretrained(
                    self.settings.model_path,
                    trust_remote_code=True,
                )

                # --- Model ---
                logger.info("Loading model weights (this may take a while) ...")
                self._model = AutoModel.from_pretrained(
                    self.settings.model_path,
                    torch_dtype=self._dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,  # reduces peak RAM during shard loading
                )
                self._model = self._model.to(self._device).eval()

            except Exception as exc:
                logger.exception("Failed to load LocateAnything-3B: {}", exc)
                raise RuntimeError(
                    f"Model loading failed: {exc}. "
                    "Check MODEL_PATH, HF_TOKEN, and available VRAM."
                ) from exc

            elapsed = (time.perf_counter() - t0) * 1000
            self._loaded = True
            logger.info("Model ready | load_time={:.0f}ms", elapsed)
            self._log_memory("post-load")

    def unload(self) -> None:
        """Release model from memory (e.g., on server shutdown)."""
        with self._load_lock:
            if self._model is not None:
                del self._model
                del self._tokenizer
                del self._processor
                self._model = None
                self._tokenizer = None
                self._processor = None
                self._loaded = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Model unloaded and GPU cache cleared.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Public: core inference  (mirrors LocateAnythingWorker.predict)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run a single inference pass.

        Follows the exact pattern from the official LocateAnythingWorker:
          1. Build messages dict with image + text
          2. Apply py_apply_chat_template
          3. Call process_vision_info to get image tensors
          4. Call processor to build model inputs
          5. Call model.generate() with the documented kwargs
          6. Return raw answer string + timing metadata

        Parameters
        ----------
        image:            PIL Image (RGB).
        question:         Natural-language prompt.
        generation_mode:  Override for this call; falls back to settings.
        max_new_tokens:   Override for this call; falls back to settings.

        Returns dict with:
            answer (str), inference_time_ms (float), generation_mode_used (str)
        """
        if not self._loaded:
            raise ModelNotLoadedError(
                "Model is not loaded. Call ModelService.load() first."
            )

        # Validate and pre-process image.
        image = self._validate_and_resize_image(image)

        mode = generation_mode or self.settings.generation_mode
        tokens = max_new_tokens or self.settings.max_new_tokens

        logger.debug(
            "Inference | mode={} | max_new_tokens={} | img={}x{} | query={!r}",
            mode,
            tokens,
            image.width,
            image.height,
            question[:80],
        )

        t0 = time.perf_counter()

        # Build multimodal conversation messages.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        # Step 1: Apply chat template (custom method on LocateAnything processor).
        text = self._processor.py_apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Step 2: Extract image tensors from messages dict.
        images, videos = self._processor.process_vision_info(messages)

        # Step 3: Tokenise + build pixel_values.
        inputs = self._processor(
            text=[text],
            images=images,
            videos=videos,
            return_tensors="pt",
        ).to(self._device)

        # Step 4: Cast pixel values to the correct dtype.
        pixel_values = inputs["pixel_values"].to(self._dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        # Step 5: Generate — using the exact signature from the model card.
        response = self._model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            image_grid_hws=image_grid_hws,
            tokenizer=self._tokenizer,
            max_new_tokens=tokens,
            use_cache=True,
            generation_mode=mode,
            temperature=self.settings.temperature,
            do_sample=True,
            top_p=self.settings.top_p,
            repetition_penalty=self.settings.repetition_penalty,
            verbose=False,
        )

        # Step 6: Unpack return value.
        # Model card: response[0] if tuple, else response directly.
        if isinstance(response, tuple):
            raw_answer = response[0]
        else:
            raw_answer = response

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("Inference complete | time={:.0f}ms | mode={}", elapsed_ms, mode)

        if elapsed_ms > self.settings.slow_inference_threshold_ms:
            logger.warning(
                "Slow inference: {:.0f}ms > threshold {}ms",
                elapsed_ms,
                self.settings.slow_inference_threshold_ms,
            )

        if self.settings.enable_memory_logging:
            self._log_memory("post-inference")

        return {
            "answer": raw_answer,
            "inference_time_ms": elapsed_ms,
            "generation_mode_used": mode,
        }

    # ------------------------------------------------------------------
    # Public: task-specific helpers (mirrors LocateAnythingWorker)
    # ------------------------------------------------------------------

    def detect(self, image: Image.Image, categories: List[str], **kwargs: Any) -> Dict[str, Any]:
        """Object detection for a list of categories."""
        cats = "</c>".join(categories)
        prompt = f"Locate all the instances that matches the following description: {cats}."
        return self.predict(image, prompt, **kwargs)

    def ground_single(self, image: Image.Image, phrase: str, **kwargs: Any) -> Dict[str, Any]:
        """Phrase grounding — single instance."""
        prompt = f"Locate a single instance that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_multi(self, image: Image.Image, phrase: str, **kwargs: Any) -> Dict[str, Any]:
        """Phrase grounding — multiple instances."""
        prompt = f"Locate all the instances that match the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_text(self, image: Image.Image, phrase: str, **kwargs: Any) -> Dict[str, Any]:
        """Text grounding."""
        prompt = f"Please locate the text referred as {phrase}."
        return self.predict(image, prompt, **kwargs)

    def detect_text(self, image: Image.Image, **kwargs: Any) -> Dict[str, Any]:
        """Scene text / OCR detection."""
        return self.predict(image, "Detect all the text in box format.", **kwargs)

    def ground_gui(self, image: Image.Image, phrase: str, output_type: str = "box", **kwargs: Any) -> Dict[str, Any]:
        """GUI element grounding (box or point output)."""
        if output_type == "point":
            prompt = f"Point to: {phrase}."
        else:
            prompt = f"Locate the region that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def point(self, image: Image.Image, phrase: str, **kwargs: Any) -> Dict[str, Any]:
        """Pointing task."""
        prompt = f"Point to: {phrase}."
        return self.predict(image, prompt, **kwargs)

    # ------------------------------------------------------------------
    # Public: output parsing  (exact patterns from official model card)
    # ------------------------------------------------------------------

    @staticmethod
    def parse_boxes(
        answer: str,
        image_width: int,
        image_height: int,
        label: str = "object",
    ) -> List[BoundingBox]:
        """
        Parse model output into pixel-coordinate BoundingBox objects.

        LocateAnything encodes boxes as:
            <box><x1><y1><x2><y2></box>
        where coordinates are integers normalised to [0, 1000].

        NOTE: The model does NOT embed per-box labels in its output;
        labels are derived from the query / calling context.
        """
        boxes: List[BoundingBox] = []

        # Official regex from LocateAnythingWorker.parse_boxes
        pattern = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")

        for m in pattern.finditer(answer):
            x1_n, y1_n, x2_n, y2_n = (int(g) for g in m.groups())

            # Convert from [0,1000] normalised space to pixel coordinates.
            x1 = x1_n / 1000.0 * image_width
            y1 = y1_n / 1000.0 * image_height
            x2 = x2_n / 1000.0 * image_width
            y2 = y2_n / 1000.0 * image_height

            # Skip degenerate boxes.
            if x2 <= x1 or y2 <= y1:
                logger.debug("Skipping degenerate box: [{},{},{},{}]", x1_n, y1_n, x2_n, y2_n)
                continue

            boxes.append(
                BoundingBox(
                    label=label,
                    confidence=1.0,  # Model does not produce per-box scores.
                    bbox=[x1, y1, x2, y2],
                    bbox_normalised=[
                        x1_n / 1000.0,
                        y1_n / 1000.0,
                        x2_n / 1000.0,
                        y2_n / 1000.0,
                    ],
                )
            )

        logger.debug("Parsed {} bounding boxes from model output.", len(boxes))
        return boxes

    @staticmethod
    def parse_points(
        answer: str,
        image_width: int,
        image_height: int,
        label: str = "point",
    ) -> List[DetectionPoint]:
        """
        Parse 2-coordinate point tokens from model output.

        Point format (official):  <box><x><y></box>
        """
        points: List[DetectionPoint] = []

        # Official regex from LocateAnythingWorker.parse_points
        pattern = re.compile(r"<box><(\d+)><(\d+)></box>")

        for m in pattern.finditer(answer):
            x_n, y_n = int(m.group(1)), int(m.group(2))
            points.append(
                DetectionPoint(
                    label=label,
                    x=x_n / 1000.0 * image_width,
                    y=y_n / 1000.0 * image_height,
                    x_normalised=x_n / 1000.0,
                    y_normalised=y_n / 1000.0,
                )
            )

        logger.debug("Parsed {} points from model output.", len(points))
        return points

    # ------------------------------------------------------------------
    # Public: diagnostics
    # ------------------------------------------------------------------

    def memory_info(self) -> Dict[str, Any]:
        """Return current CPU and GPU memory diagnostics."""
        info: Dict[str, Any] = {}

        vm = psutil.virtual_memory()
        info["system_ram_total_gb"] = round(vm.total / 1e9, 2)
        info["system_ram_used_gb"] = round(vm.used / 1e9, 2)
        info["system_ram_available_gb"] = round(vm.available / 1e9, 2)

        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            info["gpu_name"] = torch.cuda.get_device_name(device_idx)
            info["gpu_vram_allocated_gb"] = round(
                torch.cuda.memory_allocated(device_idx) / 1e9, 3
            )
            info["gpu_vram_reserved_gb"] = round(
                torch.cuda.memory_reserved(device_idx) / 1e9, 3
            )
            total = torch.cuda.get_device_properties(device_idx).total_memory
            info["gpu_vram_total_gb"] = round(total / 1e9, 2)
        else:
            info["gpu_name"] = "N/A"

        return info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_dtype(self) -> torch.dtype:
        dtype_map: Dict[str, torch.dtype] = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.settings.torch_dtype, torch.bfloat16)

    def _validate_and_resize_image(self, image: Image.Image) -> Image.Image:
        """Ensure image is RGB and within the configured maximum dimension."""
        if image.mode != "RGB":
            logger.debug("Converting image from {} to RGB.", image.mode)
            image = image.convert("RGB")

        max_dim = self.settings.max_image_dimension
        w, h = image.size

        if w > max_dim or h > max_dim:
            scale = max_dim / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            logger.info(
                "Resizing image {}x{} -> {}x{} (max_dim={}).",
                w, h, new_w, new_h, max_dim,
            )
            image = image.resize((new_w, new_h), Image.LANCZOS)

        return image

    def _log_memory(self, tag: str = "") -> None:
        if not self.settings.enable_memory_logging:
            return
        info = self.memory_info()
        if torch.cuda.is_available():
            logger.debug(
                "[{}] GPU VRAM: {}/{} GB | System RAM: {}/{} GB",
                tag,
                info.get("gpu_vram_allocated_gb"),
                info.get("gpu_vram_total_gb"),
                info.get("system_ram_used_gb"),
                info.get("system_ram_total_gb"),
            )
        else:
            logger.debug(
                "[{}] System RAM: {}/{} GB used",
                tag,
                info.get("system_ram_used_gb"),
                info.get("system_ram_total_gb"),
            )
