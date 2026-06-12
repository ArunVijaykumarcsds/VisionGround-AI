# Locate Anything Assistant

> Production-grade full-stack application for **nvidia/LocateAnything-3B** — a 3-billion-parameter vision-language model for fast, accurate visual grounding, object detection, scene-text detection, GUI grounding, and pointing tasks.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Repository Structure](#3-repository-structure)
4. [System Requirements](#4-system-requirements)
5. [Quick Start — Local (No Docker)](#5-quick-start--local-no-docker)
6. [Quick Start — Docker Compose](#6-quick-start--docker-compose)
7. [Environment Variables Reference](#7-environment-variables-reference)
8. [API Reference](#8-api-reference)
9. [Model Details](#9-model-details)
10. [Frontend Guide](#10-frontend-guide)
11. [Testing Guide](#11-testing-guide)
12. [Docker Testing Guide](#12-docker-testing-guide)
13. [GitHub Push Guide](#13-github-push-guide)
14. [Deployment Guide](#14-deployment-guide)
15. [CURL Examples](#15-curl-examples)
16. [Python Client](#16-python-client)
17. [JavaScript Client](#17-javascript-client)
18. [Swagger / OpenAPI Testing](#18-swagger--openapi-testing)
19. [Troubleshooting](#19-troubleshooting)
20. [Resource Requirements](#20-resource-requirements)
21. [Roadmap](#21-roadmap)
22. [License](#22-license)

---

## 1. Project Overview

Locate Anything Assistant wraps **nvidia/LocateAnything-3B** in a production-ready web application. The model accepts an image and a natural-language query and returns structured bounding boxes or point coordinates identifying the requested objects.

### Supported Tasks

| Task | Example Query | Output |
|------|--------------|--------|
| Multi-category detection | `find car, person, bicycle` | Bounding boxes per category |
| Free-form phrase grounding | `locate the person wearing a red hat` | Bounding boxes |
| Single-instance grounding | `find the tallest building` | Single bounding box |
| Scene text detection | `detect all text` | Bounding boxes around text |
| GUI element grounding | `locate the submit button` | Bounding box or point |
| Pointing | `point to the dog` | X/Y coordinate |

### Key Features

- **React frontend** with drag-and-drop image upload, live bounding-box canvas overlay, chat interface, and health status bar
- **FastAPI backend** with multipart upload, structured JSON responses, singleton model loading, and full OpenAPI docs
- **Production inference pipeline** mirroring the official `LocateAnythingWorker` from the HuggingFace model card
- **Docker Compose** stack with GPU support, health checks, and volume-mounted model cache
- **Zero mock outputs** — every code path runs real inference

---

## 2. Architecture

```
Browser (React + Vite)
        │
        │  HTTP  multipart/form-data
        ▼
┌─────────────────────────────────────────────┐
│              Nginx (port 80)                │  ← serves built React SPA
│  /detect  /chat  /health  → proxy :8000     │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│           FastAPI  (Uvicorn :8000)          │
│                                             │
│  POST /detect  ──►  detect.py              │
│  POST /chat    ──►  chat.py                │
│  GET  /health  ──►  health.py              │
│                                             │
│  All routes ──► ModelService (singleton)   │
│                                             │
│  ModelService                               │
│    ├── AutoTokenizer.from_pretrained()      │
│    ├── AutoProcessor.from_pretrained()      │
│    ├── AutoModel.from_pretrained()          │  trust_remote_code=True
│    ├── model.generate()                     │  custom generate() from repo
│    └── parse_boxes() / parse_points()       │  regex on <box> tokens
└─────────────────────────────────────────────┘
                    │
                    ▼
       nvidia/LocateAnything-3B weights
       (HuggingFace Hub or local mount)
       model-00001-of-00002.safetensors
       model-00002-of-00002.safetensors
       model.safetensors.index.json
```

### Data Flow — Single Request

```
1. User drops image + types query in React UI
2. React POSTs multipart/form-data to /detect
3. FastAPI validates image (MIME type, size, decodability)
4. prompt_utils.build_detect_prompt() converts query to model template
5. ModelService.predict():
      a. _validate_and_resize_image() — ensure RGB, cap at max_dimension
      b. Build messages dict  {role:user, content:[{image},{text}]}
      c. processor.py_apply_chat_template()  →  text string
      d. processor.process_vision_info()     →  pixel tensors
      e. processor()                         →  input_ids, attention_mask
      f. model.generate()                    →  raw answer string
6. parse_boxes()  →  List[BoundingBox]  (regex on <box><x1><y1><x2><y2></box>)
7. parse_points() →  List[DetectionPoint]
8. Return DetectResponse JSON
9. React renders boxes on canvas overlay
```

---

## 3. Repository Structure

```
locate-anything-assistant/
│
├── .gitignore
├── docker-compose.yml              ← full-stack compose (CPU + GPU profiles)
├── README.md
│
├── backend/
│   ├── .dockerignore
│   ├── .env.example                ← copy to .env and fill in values
│   ├── Dockerfile                  ← multi-stage: builder → runtime
│   ├── requirements.txt
│   ├── logs/                       ← log files written here at runtime
│   └── app/
│       ├── __init__.py
│       ├── main.py                 ← FastAPI app factory + lifespan
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── detect.py           ← POST /detect
│       │   ├── chat.py             ← POST /chat
│       │   └── health.py           ← GET  /health
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py           ← Pydantic Settings (reads .env)
│       │   └── logging.py          ← Loguru setup + stdlib intercept
│       │
│       ├── schemas/
│       │   ├── __init__.py
│       │   └── detection.py        ← BoundingBox, DetectResponse, ChatResponse…
│       │
│       ├── services/
│       │   ├── __init__.py
│       │   └── model_service.py    ← singleton model loader + inference engine
│       │
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── image_utils.py      ← upload validation, resize helpers
│       │   └── prompt_utils.py     ← query → LocateAnything prompt templates
│       │
│       └── models/
│           └── __init__.py         ← reserved for future ORM / DB models
│
└── frontend/
    ├── .env.example
    ├── Dockerfile                  ← Vite build → Nginx serve
    ├── index.html
    ├── nginx.conf                  ← SPA fallback + API proxy
    ├── package.json
    ├── postcss.config.js
    ├── tailwind.config.js
    ├── vite.config.js
    └── src/
        ├── main.jsx                ← ReactDOM.createRoot entry
        ├── App.jsx                 ← root layout, tab switcher
        │
        ├── components/
        │   ├── ChatPanel.jsx       ← multi-turn chat UI
        │   ├── DetectionPanel.jsx  ← structured results list
        │   ├── ErrorBanner.jsx     ← dismissable error banner
        │   ├── Header.jsx          ← branding + tab switcher
        │   ├── ImageCanvas.jsx     ← image + bounding-box canvas overlay
        │   ├── ImageUpload.jsx     ← drag-and-drop uploader
        │   ├── QueryInput.jsx      ← detection query form + mode selector
        │   └── StatusBar.jsx       ← live backend / model health bar
        │
        ├── hooks/
        │   ├── useDetection.js     ← manages image, results, API calls
        │   └── useHealth.js        ← polls /health every 10 s
        │
        ├── utils/
        │   └── api.js              ← Axios client for /detect /chat /health
        │
        └── styles/
            └── index.css           ← Tailwind directives + custom component classes
```

---

## 4. System Requirements

### Minimum (CPU inference — slow but functional)

| Component | Minimum |
|-----------|---------|
| CPU | 8-core x86-64 |
| RAM | 24 GB |
| Storage | 20 GB free (model ~7 GB + OS + deps) |
| OS | Ubuntu 22.04 / macOS 13+ / Windows 11 WSL2 |
| Python | 3.11 |
| Node.js | 20 LTS |
| Docker | 24+ (optional) |

### Recommended (GPU inference)

| Component | Recommended |
|-----------|-------------|
| GPU | NVIDIA RTX 3090 / A10 / A100 (24 GB VRAM) |
| VRAM | 8 GB minimum (BF16) · 16 GB comfortable · 24 GB ideal |
| RAM | 32 GB |
| CUDA | 12.1+ |
| cuDNN | 8.9+ |
| Driver | 525+ |

### Per-platform VRAM budget

| Dtype | Approximate VRAM |
|-------|-----------------|
| BF16 (default) | ~7 GB |
| FP16 | ~7 GB |
| FP32 | ~14 GB |

> **Note:** With `generation_mode=hybrid` and BF16, the model fits on an 8 GB GPU with careful batch sizing. Use `TORCH_DTYPE=float16` if BF16 is not supported (pre-Ampere cards).

---

## 5. Quick Start — Local (No Docker)

### 5.1 Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/locate-anything-assistant.git
cd locate-anything-assistant
```

### 5.2 Backend setup

```bash
cd backend

# Create and activate a virtual environment
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install PyTorch first (choose the correct variant for your system)

# ── GPU (CUDA 12.1) ──
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ── GPU (CUDA 11.8) ──
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# ── CPU only ──
pip install torch torchvision

# Install all other dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
```

Edit `backend/.env`:

```dotenv
MODEL_PATH=nvidia/LocateAnything-3B
HF_TOKEN=your_huggingface_token_here   # required — model is gated
DEVICE=auto                             # auto-detects CUDA
TORCH_DTYPE=bfloat16                   # bfloat16 for Ampere+, float16 for older GPUs
```

### 5.3 Start the backend

```bash
# From backend/ with venv active
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

On first run, `transformers` will download ~7 GB of model weights from HuggingFace Hub into `~/.cache/huggingface/hub/`. This is a one-time operation.

You will see:

```
INFO  Loading LocateAnything-3B | path=nvidia/LocateAnything-3B | device=cuda | dtype=bfloat16
INFO  Loading tokenizer ...
INFO  Loading processor ...
INFO  Loading model weights (this may take a while) ...
INFO  Model ready | load_time=18432ms
INFO  Application startup complete.
```

Visit **http://localhost:8000/docs** for the Swagger UI.

### 5.4 Frontend setup

Open a **new terminal**:

```bash
cd locate-anything-assistant/frontend

# Install Node dependencies
npm install

# Copy env
cp .env.example .env
# VITE_API_URL is empty by default — Vite proxies /detect /chat /health to :8000

# Start development server
npm run dev
```

Visit **http://localhost:5173**

---

## 6. Quick Start — Docker Compose

### 6.1 Prerequisites

- Docker Engine 24+
- Docker Compose v2 (`docker compose` not `docker-compose`)
- For GPU: [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed and configured

### 6.2 CPU deployment

```bash
cd locate-anything-assistant

# Copy and edit backend env
cp backend/.env.example backend/.env
# Set HF_TOKEN in backend/.env

# Build and start everything
docker compose up --build
```

Services started:
- **Backend** → http://localhost:8000
- **Frontend** → http://localhost:3000
- **Swagger**  → http://localhost:8000/docs

### 6.3 GPU deployment

```bash
docker compose --profile gpu up --build
```

This starts the `backend-gpu` service which reserves one NVIDIA GPU.

### 6.4 With a locally cached model (no download)

If you have already downloaded the model weights:

```bash
# Weights are in /data/LocateAnything-3B/
MODEL_PATH=/models/LocateAnything-3B docker compose up --build
```

Add this to `docker-compose.yml` under `backend.volumes`:
```yaml
- /data/LocateAnything-3B:/models/LocateAnything-3B:ro
```

Then set `MODEL_PATH=/models/LocateAnything-3B` in `backend/.env`.

### 6.5 Useful Compose commands

```bash
# View live logs for both services
docker compose logs -f

# Backend logs only
docker compose logs -f backend

# Stop everything and remove containers
docker compose down

# Stop and remove volumes (clears HF model cache)
docker compose down -v

# Rebuild without cache
docker compose build --no-cache
```

---

## 7. Environment Variables Reference

All variables live in `backend/.env`. Copy from `backend/.env.example`.

### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `nvidia/LocateAnything-3B` | HuggingFace model ID **or** absolute path to local weights directory |
| `HF_TOKEN` | *(required)* | HuggingFace API token. Generate at https://huggingface.co/settings/tokens. Required for gated model access. |
| `DEVICE` | `auto` | Inference device: `cuda`, `cpu`, or `auto` (auto-detects GPU at startup) |
| `TORCH_DTYPE` | `bfloat16` | Weight precision: `bfloat16` (Ampere+ GPUs), `float16` (older GPUs), `float32` (CPU or max precision) |

### Inference Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GENERATION_MODE` | `hybrid` | LocateAnything generation mode: `hybrid` (balanced), `fast` (lower latency), `slow` (higher accuracy) |
| `MAX_NEW_TOKENS` | `2048` | Maximum tokens the model generates per request. Higher values allow more detections per image. |
| `TEMPERATURE` | `0.7` | Sampling temperature. `0` = fully deterministic greedy decoding. |
| `TOP_P` | `0.9` | Nucleus sampling probability mass. |
| `REPETITION_PENALTY` | `1.1` | Penalises repeated tokens. Values >1.0 reduce repetition in model output. |

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Uvicorn bind address. Use `127.0.0.1` to restrict to localhost. |
| `PORT` | `8000` | Uvicorn listen port. |
| `WORKERS` | `1` | Number of Uvicorn worker processes. **Must be 1** when the model is loaded in-process to avoid loading the model N times. |
| `LOG_LEVEL` | `info` | Loguru / Uvicorn log level: `debug`, `info`, `warning`, `error`. |

### Upload Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_IMAGE_SIZE_MB` | `20` | Maximum allowed upload size in megabytes. Requests exceeding this receive HTTP 413. |
| `MAX_IMAGE_DIMENSION` | `2560` | Maximum image width or height in pixels. Images exceeding this are proportionally downscaled before inference. |
| `ALLOWED_IMAGE_TYPES` | `image/jpeg,image/png,image/webp,image/bmp` | Comma-separated list of accepted MIME types. |

### CORS

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Comma-separated list of allowed frontend origins. Use `*` only in development. |

### Monitoring

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_MEMORY_LOGGING` | `true` | Log GPU VRAM and system RAM usage after each inference call. |
| `SLOW_INFERENCE_THRESHOLD_MS` | `5000` | Requests slower than this value (in ms) are flagged with a WARNING log entry. |

### Cache

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HOME` | `~/.cache/huggingface` | Directory where HuggingFace Hub caches downloaded model files. Set to a mounted volume in Docker. |

---

## 8. API Reference

### GET /health

Returns backend liveness and model readiness.

**Response 200:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "model_path": "nvidia/LocateAnything-3B",
  "memory_info": {
    "system_ram_total_gb": 31.99,
    "system_ram_used_gb": 18.4,
    "system_ram_available_gb": 13.59,
    "gpu_name": "NVIDIA GeForce RTX 3090",
    "gpu_vram_allocated_gb": 7.124,
    "gpu_vram_reserved_gb": 7.5,
    "gpu_vram_total_gb": 24.0
  }
}
```

---

### POST /detect

Upload an image and a natural-language query. Returns structured bounding boxes and/or points.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | File | ✅ | Image file (JPEG, PNG, WebP, BMP, max 20 MB) |
| `query` | string | ✅ | Natural-language detection query |
| `generation_mode` | string | ❌ | Override: `hybrid` / `fast` / `slow` |
| `max_new_tokens` | integer | ❌ | Override token limit (64–8192) |

**Response 200:**
```json
{
  "query": "find all cars",
  "detections": [
    {
      "label": "find all cars",
      "confidence": 1.0,
      "bbox": [142.4, 310.8, 587.2, 634.1],
      "bbox_normalised": [0.142, 0.310, 0.587, 0.634]
    },
    {
      "label": "find all cars",
      "confidence": 1.0,
      "bbox": [820.1, 290.3, 1100.5, 580.7],
      "bbox_normalised": [0.820, 0.290, 1.100, 0.580]
    }
  ],
  "points": [],
  "raw_answer": "<box><142><310><587><634></box><box><820><290><1100><580></box>",
  "image_width": 1000,
  "image_height": 1000,
  "inference_time_ms": 2341.7,
  "generation_mode_used": "hybrid"
}
```

**Error responses:**

| Code | Condition |
|------|-----------|
| 400 | Malformed request |
| 413 | Image exceeds size limit |
| 422 | Unsupported image type or validation failure |
| 503 | Model not yet loaded (still initialising) |
| 500 | Inference error |

---

### POST /chat

Chat-style visual grounding. Returns a natural-language assistant reply plus structured detections.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | File | ✅ | Image file |
| `message` | string | ✅ | User chat message |
| `generation_mode` | string | ❌ | Override generation mode |

**Response 200:**
```json
{
  "assistant": "I found 3 objects in the image.",
  "detections": [
    {
      "label": "locate all buildings",
      "confidence": 1.0,
      "bbox": [50.0, 120.0, 400.0, 890.0],
      "bbox_normalised": [0.05, 0.12, 0.40, 0.89]
    }
  ],
  "points": [],
  "raw_answer": "<box><50><120><400><890></box>",
  "image_width": 1000,
  "image_height": 1000,
  "inference_time_ms": 2187.3
}
```

---

## 9. Model Details

### nvidia/LocateAnything-3B

| Property | Value |
|----------|-------|
| Architecture | Vision-Language Model (VLM) |
| Parameters | ~3 billion |
| Base | Qwen2 language model + ViT visual encoder |
| Input | Image + text prompt |
| Output | Text with embedded `<box>` coordinate tokens |
| Coordinate format | `<box><x1><y1><x2><y2></box>` normalised to [0, 1000] |
| Point format | `<box><x><y></box>` normalised to [0, 1000] |
| Generation modes | `hybrid` · `fast` · `slow` |
| Weight files | `model-00001-of-00002.safetensors` + `model-00002-of-00002.safetensors` |
| Index file | `model.safetensors.index.json` (maps layers to shards) |
| HuggingFace ID | `nvidia/LocateAnything-3B` |
| License | NVIDIA Research |

### Prompt Templates

The model requires exact prompt phrasing. The `prompt_utils.py` module handles this automatically:

```python
# Multi-category detection
"Locate all the instances that matches the following description: car</c>person</c>bicycle."

# Free-form phrase grounding (multiple)
"Locate all the instances that match the following description: person wearing a red hat."

# Single instance grounding
"Locate a single instance that matches the following description: the tallest building."

# Scene text detection
"Detect all the text in box format."

# GUI grounding (box)
"Locate the region that matches the following description: submit button."

# GUI grounding (point)
"Point to: submit button."

# Text grounding
"Please locate the text referred as Hello World."
```

### Loading Mechanism

```python
from transformers import AutoModel, AutoProcessor, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "nvidia/LocateAnything-3B",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(
    "nvidia/LocateAnything-3B",
    trust_remote_code=True,
)
model = AutoModel.from_pretrained(
    "nvidia/LocateAnything-3B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
).to("cuda").eval()
```

`trust_remote_code=True` is **required** because the model ships custom Python modules (`modeling_locateanything.py`, `processing_locateanything.py`, `configuration_locateanything.py`) that must execute locally. The `model.safetensors.index.json` file is consumed automatically by `from_pretrained()` — it maps weight tensors to their respective shard files. No manual merging is needed.

---

## 10. Frontend Guide

### Technology Stack

| Library | Version | Purpose |
|---------|---------|---------|
| React | 18.3 | UI framework |
| Vite | 5.3 | Build tool + dev server |
| Tailwind CSS | 3.4 | Utility-first styling |
| Axios | 1.7 | HTTP client |
| react-dropzone | 14.2 | Drag-and-drop file upload |

### Component Map

```
App.jsx
├── StatusBar          polls /health every 10 s; shows backend/model status
├── Header             branding + Detect / Chat tab switcher
├── (left column)
│   ├── ImageUpload    drag-and-drop zone (react-dropzone)
│   └── ImageCanvas    HTML5 Canvas overlay with bbox drawing + hover tooltips
├── (right column, Detect tab)
│   ├── QueryInput     text input + mode buttons + example chips
│   └── DetectionPanel structured results list + raw output toggle
├── (right column, Chat tab)
│   └── ChatPanel      message feed + input + typing indicator
└── ErrorBanner        dismissable error display
```

### Canvas Overlay

`ImageCanvas.jsx` uses a `<canvas>` absolutely positioned over the `<img>` tag. On every render it:

1. Scales bounding-box pixel coordinates by `(displayWidth / modelWidth)` and `(displayHeight / modelHeight)`.
2. Draws a semi-transparent fill and coloured border for each box.
3. Renders a label pill above each box.
4. Performs hit-testing on `mousemove` to highlight the hovered box.
5. Draws circular point markers for point-output tasks.

### Development

```bash
cd frontend
npm install
npm run dev        # → http://localhost:5173  (hot module reload)
npm run build      # production bundle → dist/
npm run preview    # preview the production build locally
```

---

## 11. Testing Guide

### 11.1 Backend unit tests

```bash
cd backend
source venv/bin/activate
pip install pytest pytest-asyncio httpx

# Run all tests
pytest tests/ -v

# Run with coverage
pip install pytest-cov
pytest tests/ -v --cov=app --cov-report=term-missing
```

### 11.2 Manual endpoint testing

**Health check:**
```bash
curl http://localhost:8000/health | python3 -m json.tool
```

**Detection (with a real image):**
```bash
curl -X POST http://localhost:8000/detect \
  -F "image=@/path/to/your/image.jpg" \
  -F "query=find all cars" \
  | python3 -m json.tool
```

**Chat:**
```bash
curl -X POST http://localhost:8000/chat \
  -F "image=@/path/to/your/image.jpg" \
  -F "message=Locate all buildings in this image" \
  | python3 -m json.tool
```

### 11.3 Test without GPU (mock model)

To test the API layer without loading the model, set `MODEL_PATH` to a tiny valid HF model and override the inference method. The application will start, all endpoints will be reachable, and the `/health` endpoint will reflect `model_loaded: true` once loading completes.

### 11.4 Swagger UI

Navigate to **http://localhost:8000/docs** for the auto-generated Swagger interface. All three endpoints (`/health`, `/detect`, `/chat`) are fully documented with request/response schemas and can be called directly from the browser using the "Try it out" button.

---

## 12. Docker Testing Guide

### 12.1 Build images

```bash
cd locate-anything-assistant

# Build both images
docker compose build

# Build backend only
docker compose build backend

# Build frontend only
docker compose build frontend

# Build without Docker layer cache (clean build)
docker compose build --no-cache
```

### 12.2 Start and verify

```bash
# Start in foreground (CTRL+C to stop)
docker compose up

# Start in background
docker compose up -d

# Check container status
docker compose ps

# View backend logs
docker compose logs -f backend

# View frontend logs
docker compose logs -f frontend
```

### 12.3 Test inside containers

```bash
# Shell into backend container
docker compose exec backend bash

# Shell into frontend container
docker compose exec frontend sh

# Test health from inside backend container
docker compose exec backend curl http://localhost:8000/health

# Test health from host
curl http://localhost:8000/health
```

### 12.4 GPU verification

```bash
# Verify GPU is accessible inside the backend container
docker compose exec backend-gpu python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
    print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
"
```

### 12.5 Cleanup

```bash
# Stop containers, keep volumes (model cache preserved)
docker compose down

# Stop containers, remove volumes (model cache deleted — re-download required)
docker compose down -v

# Remove all images
docker compose down --rmi all

# Full nuclear reset
docker system prune -a --volumes
```

---

## 13. GitHub Push Guide

### 13.1 Initialise and configure

```bash
cd locate-anything-assistant

git init
git config user.name  "Your Name"
git config user.email "you@example.com"
```

### 13.2 Set up secrets properly

Before committing, verify `.gitignore` excludes secrets:

```bash
# These must NOT appear in git status output:
cat backend/.env              # should not exist in git
ls backend/*.safetensors      # model weights must be gitignored

git status                    # verify .env and *.safetensors are NOT listed
```

The `.gitignore` already excludes `.env`, `*.safetensors`, `*.bin`, and `**/huggingface/hub/`.

### 13.3 First commit

```bash
git add .
git commit -m "feat: initial Locate Anything Assistant

- FastAPI backend with LocateAnything-3B inference
- React/Vite frontend with canvas detection overlay
- Docker Compose with CPU and GPU profiles
- Full OpenAPI documentation"
```

### 13.4 Push to GitHub

```bash
# Create repository on GitHub first (https://github.com/new)
# Then:

git remote add origin https://github.com/YOUR_USERNAME/locate-anything-assistant.git
git branch -M main
git push -u origin main
```

### 13.5 Add GitHub Actions CI (optional)

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on: [push, pull_request]

jobs:
  syntax-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: |
          cd backend
          python -m py_compile app/main.py app/services/model_service.py
          echo "Syntax OK"

  frontend-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: "20" }
      - run: |
          cd frontend
          npm install
          npm run build
```

---

## 14. Deployment Guide

### 14.1 Local GPU Server (Recommended)

**Requirements:** Ubuntu 22.04, NVIDIA GPU (8 GB+ VRAM), CUDA 12.1+

```bash
# 1. Install CUDA drivers and toolkit
# https://developer.nvidia.com/cuda-downloads

# 2. Clone and configure
git clone https://github.com/YOUR_USERNAME/locate-anything-assistant.git
cd locate-anything-assistant
cp backend/.env.example backend/.env
# Edit backend/.env: set HF_TOKEN, DEVICE=cuda, TORCH_DTYPE=bfloat16

# 3. Install nvidia-container-toolkit (for Docker GPU)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
# (follow official guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

# 4. Start with GPU
docker compose --profile gpu up -d
```

**Cost:** Hardware cost only (~$0/month running cost beyond electricity)
**Latency:** 1–4 seconds per inference on RTX 3090

---

### 14.2 Google Cloud VM (A100 / T4)

```bash
# 1. Create a GPU VM
gcloud compute instances create locate-anything-vm \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE

# 2. SSH in
gcloud compute ssh locate-anything-vm

# 3. Install Docker + CUDA drivers
# (follow GCP's GPU driver install guide)

# 4. Clone repo and start
git clone https://github.com/YOUR_USERNAME/locate-anything-assistant.git
cd locate-anything-assistant
cp backend/.env.example backend/.env   # add HF_TOKEN
docker compose --profile gpu up -d

# 5. Open firewall ports
gcloud compute firewall-rules create allow-locate-anything \
  --allow tcp:8000,tcp:3000 \
  --target-tags locate-anything
```

| VM Type | VRAM | vCPU | RAM | Cost (USD/hr) |
|---------|------|------|-----|---------------|
| a2-highgpu-1g (A100) | 40 GB | 12 | 85 GB | ~$3.67 |
| n1-standard-8 + T4 | 16 GB | 8 | 30 GB | ~$0.77 |
| n1-standard-4 + T4 (min) | 16 GB | 4 | 15 GB | ~$0.54 |

---

### 14.3 Render.com

Render does not currently offer GPU instances for public deployment. Use Render for CPU-only testing:

```yaml
# render.yaml
services:
  - type: web
    name: locate-anything-backend
    env: docker
    dockerfilePath: ./backend/Dockerfile
    plan: standard           # 2 GB RAM — insufficient for model; use pro (7 GB+)
    envVars:
      - key: MODEL_PATH
        value: nvidia/LocateAnything-3B
      - key: HF_TOKEN
        sync: false          # set in Render dashboard
      - key: DEVICE
        value: cpu
      - key: TORCH_DTYPE
        value: float32
```

**Limitations:** CPU inference only; ~60–120 seconds per inference; 7 GB RAM plan required; cold-start downloads ~7 GB model.

---

### 14.4 HuggingFace Spaces (ZeroGPU)

```python
# app.py (Gradio interface for HF Spaces)
import gradio as gr
# ... wrap the model in a Gradio interface
# See: https://huggingface.co/docs/hub/spaces-sdks-gradio
```

HuggingFace Spaces with ZeroGPU provides free A100 access with a queue. The FastAPI backend cannot run directly on Spaces — use a Gradio wrapper instead, or deploy the Docker container to Spaces using the Docker SDK.

---

### 14.5 RunPod / Vast.ai (Cheapest GPU Cloud)

```bash
# 1. Select a pod: RTX 3090 (24 GB) ~$0.22/hr on RunPod
# 2. Choose PyTorch 2.1 template (includes CUDA)
# 3. SSH in and run:

git clone https://github.com/YOUR_USERNAME/locate-anything-assistant.git
cd locate-anything-assistant
cp backend/.env.example backend/.env   # add HF_TOKEN
pip install -r backend/requirements.txt
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 &
cd ../frontend && npm install && npm run build
```

---

## 15. CURL Examples

### Health check

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

### Object detection

```bash
curl -s -X POST http://localhost:8000/detect \
  -F "image=@street.jpg" \
  -F "query=find all cars" \
  | python3 -m json.tool
```

### Multi-category detection

```bash
curl -s -X POST http://localhost:8000/detect \
  -F "image=@scene.jpg" \
  -F "query=find car, person, bicycle, traffic light" \
  | python3 -m json.tool
```

### Phrase grounding

```bash
curl -s -X POST http://localhost:8000/detect \
  -F "image=@photo.jpg" \
  -F "query=locate the person wearing a red jacket" \
  | python3 -m json.tool
```

### Scene text detection

```bash
curl -s -X POST http://localhost:8000/detect \
  -F "image=@sign.jpg" \
  -F "query=detect all text" \
  | python3 -m json.tool
```

### Chat interface

```bash
curl -s -X POST http://localhost:8000/chat \
  -F "image=@aerial.jpg" \
  -F "message=How many buildings can you see? Locate them all." \
  | python3 -m json.tool
```

### Override generation mode

```bash
curl -s -X POST http://localhost:8000/detect \
  -F "image=@photo.jpg" \
  -F "query=find all people" \
  -F "generation_mode=fast" \
  -F "max_new_tokens=512" \
  | python3 -m json.tool
```

---

## 16. Python Client

Save as `client.py` and run from anywhere:

```python
"""
Python client for Locate Anything Assistant.
Usage:
    python client.py --image photo.jpg --query "find all cars"
    python client.py --image photo.jpg --chat "locate all buildings"
"""

import argparse
import json
import sys
import requests

BASE_URL = "http://localhost:8000"


def health_check() -> dict:
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    r.raise_for_status()
    return r.json()


def detect(image_path: str, query: str, generation_mode: str = "hybrid") -> dict:
    with open(image_path, "rb") as f:
        files = {"image": (image_path, f, "image/jpeg")}
        data  = {"query": query, "generation_mode": generation_mode}
        r = requests.post(
            f"{BASE_URL}/detect",
            files=files,
            data=data,
            timeout=120,
        )
    r.raise_for_status()
    return r.json()


def chat(image_path: str, message: str) -> dict:
    with open(image_path, "rb") as f:
        files = {"image": (image_path, f, "image/jpeg")}
        data  = {"message": message}
        r = requests.post(
            f"{BASE_URL}/chat",
            files=files,
            data=data,
            timeout=120,
        )
    r.raise_for_status()
    return r.json()


def print_detections(result: dict) -> None:
    print(f"\nQuery:  {result['query']}")
    print(f"Image:  {result['image_width']} x {result['image_height']} px")
    print(f"Time:   {result['inference_time_ms']:.0f} ms")
    print(f"Mode:   {result['generation_mode_used']}")
    print()

    boxes = result.get("detections", [])
    pts   = result.get("points", [])

    if boxes:
        print(f"Bounding Boxes ({len(boxes)}):")
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = [round(v) for v in b["bbox"]]
            print(f"  [{i+1}] {b['label']}  "
                  f"bbox=[{x1},{y1},{x2},{y2}]  "
                  f"conf={b['confidence']:.0%}")
    else:
        print("No bounding boxes detected.")

    if pts:
        print(f"\nPoints ({len(pts)}):")
        for i, p in enumerate(pts):
            print(f"  [{i+1}] {p['label']}  x={p['x']:.1f} y={p['y']:.1f}")

    print(f"\nRaw answer:\n  {result['raw_answer']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Locate Anything client")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--query", help="Detection query")
    parser.add_argument("--chat",  help="Chat message")
    parser.add_argument("--mode",  default="hybrid", choices=["hybrid","fast","slow"])
    parser.add_argument("--health", action="store_true")
    args = parser.parse_args()

    if args.health:
        h = health_check()
        print(json.dumps(h, indent=2))
        sys.exit(0)

    if not args.query and not args.chat:
        print("Error: provide --query or --chat")
        sys.exit(1)

    if args.query:
        result = detect(args.image, args.query, args.mode)
        print_detections(result)

    if args.chat:
        result = chat(args.image, args.chat)
        print(f"\nAssistant: {result['assistant']}")
        if result.get("detections"):
            print_detections({**result, "query": args.chat,
                               "generation_mode_used": "hybrid"})
```

**Usage:**
```bash
pip install requests

python client.py --health
python client.py --image street.jpg --query "find all cars"
python client.py --image scene.jpg  --query "find car, person" --mode fast
python client.py --image photo.jpg  --chat  "locate all buildings"
```

---

## 17. JavaScript Client

```javascript
// locate-anything-client.js
// Works in Node.js 18+ or a browser with FormData support.

const BASE_URL = 'http://localhost:8000';

/**
 * GET /health
 */
export async function getHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

/**
 * POST /detect
 * @param {File|Blob} imageFile
 * @param {string} query
 * @param {{ generation_mode?: string, max_new_tokens?: number }} [opts]
 */
export async function detect(imageFile, query, opts = {}) {
  const form = new FormData();
  form.append('image', imageFile, imageFile.name ?? 'image.jpg');
  form.append('query', query);
  if (opts.generation_mode) form.append('generation_mode', opts.generation_mode);
  if (opts.max_new_tokens)  form.append('max_new_tokens', String(opts.max_new_tokens));

  const res = await fetch(`${BASE_URL}/detect`, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.detail?.message ?? `HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * POST /chat
 * @param {File|Blob} imageFile
 * @param {string} message
 * @param {{ generation_mode?: string }} [opts]
 */
export async function chat(imageFile, message, opts = {}) {
  const form = new FormData();
  form.append('image', imageFile, imageFile.name ?? 'image.jpg');
  form.append('message', message);
  if (opts.generation_mode) form.append('generation_mode', opts.generation_mode);

  const res = await fetch(`${BASE_URL}/chat`, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.detail?.message ?? `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Example usage ────────────────────────────────────────────────────────────

// Node.js example (requires node-fetch or Node 18+ native fetch):
// import { readFileSync } from 'fs';
// const imageBlob = new Blob([readFileSync('photo.jpg')], { type: 'image/jpeg' });
// const result = await detect(imageBlob, 'find all cars');
// console.log(`Found ${result.detections.length} detections in ${result.inference_time_ms}ms`);

// Browser example:
// const input = document.querySelector('input[type=file]');
// const result = await detect(input.files[0], 'locate all people');
// result.detections.forEach(d => console.log(d.label, d.bbox));
```

---

## 18. Swagger / OpenAPI Testing

Swagger UI is served at **http://localhost:8000/docs** when the backend is running.

### Testing /detect via Swagger

1. Navigate to http://localhost:8000/docs
2. Expand the **POST /detect** endpoint
3. Click **Try it out**
4. Click **Choose File** next to the `image` field and upload any JPEG/PNG
5. Enter a query in the `query` field (e.g., `find all cars`)
6. Leave `generation_mode` and `max_new_tokens` empty to use defaults
7. Click **Execute**
8. The response body shows the full `DetectResponse` JSON

### Testing /chat via Swagger

1. Expand **POST /chat**
2. Click **Try it out**
3. Upload an image and enter a message (e.g., `Locate all buildings`)
4. Click **Execute**

### OpenAPI JSON

The raw OpenAPI schema is available at:
```
http://localhost:8000/openapi.json
```

Import this URL into Postman, Insomnia, or any OpenAPI-compatible client.

---

## 19. Troubleshooting

### Model fails to load: `OSError: nvidia/LocateAnything-3B is not a valid model ID`

**Cause:** Missing or invalid `HF_TOKEN`.
**Fix:** Generate a token at https://huggingface.co/settings/tokens (Read access is sufficient) and set `HF_TOKEN=...` in `backend/.env`.

---

### `CUDA out of memory` during model load

**Cause:** Insufficient VRAM. LocateAnything-3B requires ~7 GB in BF16.
**Fix options:**
1. Set `TORCH_DTYPE=float16` (same memory footprint, slightly different numerics)
2. If only 6 GB VRAM: set `DEVICE=cpu` and `TORCH_DTYPE=float32` (slow but works)
3. Upgrade to a GPU with more VRAM

---

### `trust_remote_code=True required` warning in logs

This is expected and safe. The warning appears because the model ships custom Python code alongside its weights. The application sets `trust_remote_code=True` explicitly on all three `from_pretrained()` calls.

---

### Frontend shows "Model is still loading…" indefinitely

**Cause:** Model download is in progress or failed silently.
**Fix:**
```bash
# Check backend logs
docker compose logs -f backend
# or
tail -f backend/logs/locate_anything_*.log
```

If you see download progress, wait. If you see an error, check `HF_TOKEN` and network connectivity.

---

### CORS error in browser

**Cause:** Frontend origin not in `CORS_ORIGINS`.
**Fix:** Add your frontend URL to `CORS_ORIGINS` in `backend/.env`:
```dotenv
CORS_ORIGINS=http://localhost:5173,http://localhost:3000,http://your-domain.com
```

---

### `/detect` returns 0 detections for an obvious query

**Possible causes and fixes:**

1. **Wrong prompt format** — use exact lowercase queries like `find all cars`, not `Find Cars`.
2. **Image too small** — the model needs reasonable resolution. Use images ≥ 512×512.
3. **Try `generation_mode=slow`** — more accurate at the cost of speed.
4. **Increase `MAX_NEW_TOKENS`** — if the model output is being cut off, raise to 4096.
5. **Check `raw_answer`** in the response — if it contains `<box>` tokens, the parse may be failing; open a GitHub issue.

---

### `decord` installation fails on Python 3.11

```bash
# Install system dependencies first (Ubuntu/Debian)
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev

# Then reinstall
pip install decord==0.6.0 --break-system-packages
```

On macOS:
```bash
brew install ffmpeg
pip install decord==0.6.0
```

If decord still fails, it can be safely omitted — it is only required for video input, which this application does not use.

---

### Docker build fails: `no space left on device`

The model weights are ~7 GB. Combined with base images and build layers, total disk usage approaches 25–30 GB.
```bash
# Free space
docker system prune -a
# Then rebuild
docker compose build
```

---

## 20. Resource Requirements

### Disk Storage

| Component | Size |
|-----------|------|
| Model weights (2 safetensors shards) | ~6.8 GB |
| Python dependencies | ~4 GB |
| Docker images (backend) | ~8 GB |
| Docker image (frontend) | ~50 MB |
| Frontend node_modules (dev) | ~400 MB |
| Logs (daily rotation, 7-day retention) | <100 MB |
| **Total (Docker)** | **~20 GB** |
| **Total (Local venv)** | **~12 GB** |

### RAM Usage

| Scenario | System RAM |
|----------|-----------|
| BF16 on GPU | ~4 GB (model stays on GPU) |
| FP32 on CPU | ~16 GB (model fully in RAM) |
| During model load (peak) | ~12 GB spike (de-sharding) |

### VRAM Usage

| Mode | VRAM |
|------|------|
| BF16 inference | ~7.2 GB |
| FP16 inference | ~7.2 GB |
| FP32 inference | ~14.4 GB |
| Peak during load | +1–2 GB above inference figure |

### Inference Latency (approximate)

| Hardware | Mode | Latency |
|----------|------|---------|
| RTX 3090 (BF16) | hybrid | 1–3 s |
| RTX 3090 (BF16) | fast | 0.5–1.5 s |
| A100 40 GB (BF16) | hybrid | 0.5–1.5 s |
| CPU (FP32, 16-core) | hybrid | 60–120 s |

### Supported Operating Systems

| OS | Local Dev | Docker |
|----|-----------|--------|
| Ubuntu 22.04 LTS | ✅ | ✅ |
| Ubuntu 20.04 LTS | ✅ | ✅ |
| macOS 13+ (Apple Silicon) | ✅ (CPU only, MPS experimental) | ✅ (CPU) |
| macOS 12 Intel | ✅ (CPU only) | ✅ (CPU) |
| Windows 11 WSL2 | ✅ | ✅ |
| Windows 11 native | ⚠️ (path issues) | ✅ |

---

## 21. Roadmap

- [ ] **OCR pipeline** — integrate Tesseract or PaddleOCR for text extraction from detected text regions
- [ ] **Segmentation** — add SAM (Segment Anything Model) as a second-stage step to convert boxes to pixel masks
- [ ] **SAR analysis** — extend prompt templates for synthetic-aperture radar imagery tasks
- [ ] **Report generation** — PDF export of annotated images with detection tables
- [ ] **Batch inference** — `/detect/batch` endpoint accepting multiple images
- [ ] **Streaming** — Server-Sent Events for progressive token-by-token output
- [ ] **Authentication** — API key middleware for production deployments
- [ ] **Rate limiting** — per-IP and per-key limits via slowapi
- [ ] **PostgreSQL logging** — persist all inference requests and results
- [ ] **Video support** — frame-by-frame detection using decord for video uploads

---

## 22. License

This application code is released under the **MIT License**.

The model weights (`nvidia/LocateAnything-3B`) are subject to **NVIDIA's model license**. Review the license at https://huggingface.co/nvidia/LocateAnything-3B before commercial use.

Dependencies are subject to their respective licenses (Apache 2.0, MIT, BSD — see each package's homepage).

---

*Built with nvidia/LocateAnything-3B · FastAPI · React · Vite · Tailwind CSS · Docker*
