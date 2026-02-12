# PaddleOCR-VL

This repo contains the PaddleOCR-VL deployment in Modal.

## Project Overview

PaddleOCR-VL deployment on Modal.com - a serverless GPU-backed OCR service using Baidu's PaddleOCR Vision-Language model. The service provides document parsing, layout detection, and OCR via a FastAPI endpoint.

## Development Commands

### Package Management (uv)
```bash
uv sync                    # Install dependencies
uv add <package>           # Add new dependency
```

### Modal Deployment
```bash
modal serve api.py        # Local dev server with hot-reload
modal deploy api.py       # Deploy to Modal cloud
```

### Testing
```bash
# Load test against deployed API
python test_api.py --url https://diwakarbasnet--paddleocr-vl-fastapi-app.modal.run/predict --requests 10 --concurrency 5 --image-dir test_images
```

```bash
# Single image inference
python single_image_client.py test_images/test-9.jpg
```

```bash
# Test PDF with tables
python pdf_test_client.py test_pdf/test_pdf.pdf -o output/test_pdf_result.md -p table
```

### MCP Server
```bash
python mcp_deploy.py       # Run MCP server (SSE transport)
```

## Architecture

### Core Components

**`main.py`** - Modal app definition and OCR service
- `PaddleOCRService`: Modal class with GPU snapshots (`enable_memory_snapshot=True`, `enable_gpu_snapshot=True`)
- Uses `@modal.enter(snap=True)` to initialize model before snapshot
- Model: PaddleOCRVL with UVDoc (unwarping), PP-DocLayoutV2 (layout detection)
- Mounted volume at `/models` for persistent model caching

**`api.py`** - FastAPI web layer
- POST `/predict`: Single image OCR (base64 input → markdown output)
- GET `/health`: Health check
- Calls `PaddleOCRService.predict.remote()` for inference

**`utils.py`** - Shared utilities
- `decode_image()`: Handles base64/bytes → OpenCV BGR conversion
- `generate_markdown()`: Converts OCR results to markdown
- `clean_text()`: Strips PaddleOCR internal tags (`<fcel>`, `<nl>`, `<frow>`)
- `setup_model_cache()`: Creates volume directories for model persistence

**`config.py`** - Constants
- `APP_NAME`, `GPU_TYPE`, `ENV_VARS` for Modal configuration
- Model cache paths: `PADDLE_HOME`, `PADDLEOCR_BASE_DIR`, `HF_HOME`

**`mcp_deploy.py`** - MCP server exposing OCR tools
- `ocr_remote(image_url)`: Fetch URL and run OCR
- `ocr_base64(image_base64)`: Direct base64 OCR

### Data Flow
```
Client -> FastAPI (/predict) -> PaddleOCRService.predict.remote() -> PaddleOCRVL -> Markdown
```

### Key Configuration

GPU: T4, Memory: 16GB, Max containers: 4, Scale-down: 600s (GPU) / 300s (API)

Model cache volume: `paddle-ocr-models` mounted at `/models`

PaddleX cache symlinked: `/root/.paddlex` -> `/models/paddlex`
