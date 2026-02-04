import os
import time
import modal
import asyncio
import pathlib
import multiprocessing
from typing import Dict, Any
from config import APP_NAME, GPU_TYPE, ENV_VARS
from utils import decode_image, clean_text, generate_markdown, setup_model_cache

os.environ["PADDLEX_PADDLE_INFERENCE_PARALLEL"] = "True"

# ============================================================================
# GLOBAL WORKER STATE & UTILITIES
# ============================================================================


def _init_ocr():
    """Shared initializer for the OCR model."""
    import os
    import sys
    import paddle

    # Setup model cache directories
    setup_model_cache()

    # Setup symlink for PaddleX cache
    try:
        paddlex_root = "/root/.paddlex"
        paddlex_volume = "/models/paddlex"

        # Remove existing symlink/directory
        if os.path.islink(paddlex_root):
            os.unlink(paddlex_root)
        elif os.path.exists(paddlex_root):
            import shutil
            shutil.rmtree(paddlex_root)

        # Create symlink to volume
        os.symlink(paddlex_volume, paddlex_root)
        print(f"INIT: Symlinked {paddlex_root} -> {paddlex_volume}")
    except Exception as e:
        print(f"INIT WARN: Failed to symlink paddlex storage: {e}")

    try:
        import paddle.base.libpaddle
        import paddle.tensor.manipulation

        actual_bool_type = paddle.to_tensor([False], dtype='bool').dtype

        # Patch every potential location of the bool constant
        paddle.bool = actual_bool_type
        paddle.tensor.bool = actual_bool_type
        paddle.tensor.manipulation.bool = actual_bool_type

        # Also try to reach into the base library if possible
        try:
            paddle.base.libpaddle.DataType.BOOL = actual_bool_type
        except:
            pass
            
    except Exception as e:
        print(f"INIT ERROR: Failed to deep sync paddle.bool: {e}")

    # Initialize model with correct parameters
    try:
        from paddleocr import PaddleOCRVL
        print(f"INIT: Loading PaddleOCR-VL model...")
        model = PaddleOCRVL(
            doc_unwarping_model_name="UVDoc",
            layout_detection_model_name="PP-DocLayoutV2",
            use_layout_detection=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=True
        )
        print(f"INIT: Model loaded successfully")
        return model
    except Exception as e:
        print(f"INIT ERROR: Failed to load model: {e}")
        import traceback
        print(f"INIT TRACEBACK: {traceback.format_exc()}")
        return None


def _predict_with_model(model, kwargs):
    """Core prediction logic using the provided model."""
    import time

    if model is None:
        return {"success": False, "error": "Model not initialized"}

    try:
        start_time = time.time()
        results = model.predict(**kwargs)
        results_list = list(results)
        if not results_list:
            return {"success": False, "error": "No results returned from model"}

        raw_result = results_list[0]
        inf_time_ms = (time.time() - start_time) * 1000

        markdown_content = generate_markdown(raw_result)

        return {
            "success": True,
            "inference_time_ms": round(inf_time_ms, 2),
            "markdown": markdown_content
        }
    except Exception as e:
        import traceback
        return {"success": False, "error": f"{type(e).__name__}: {str(e)}", "traceback": traceback.format_exc()}
    except Exception as e:
        import traceback
        return {"success": False, "error": f"{type(e).__name__}: {str(e)}", "traceback": traceback.format_exc()}


# ============================================================================
# MODAL APP SETUP
# ============================================================================

app = modal.App(APP_NAME)

# User-requested image definition
paddle_image = (
    modal.Image.from_registry("paddlepaddle/paddle:3.3.0-gpu-cuda11.8-cudnn8.9")
    .apt_install("libgl1", "libglib2.0-0", "git-lfs")
    .pip_install("paddleocr[doc-parser]", "fastapi[standard]", "pydantic-settings", "httpx", "numpy", "orjson")
    .env({
        **ENV_VARS,
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "1",
        # Add PaddleX specific cache dir
        "PADDLEX_HOME": "/models/paddlex",
    })
    .add_local_python_source("main", "api", "utils", "config")
)

# Create a persistent volume for model caching
volume = modal.Volume.from_name("paddle-ocr-models", create_if_missing=True)

# ============================================================================
# OCR SERVICE CLASS
# ============================================================================


@app.cls(
    image=paddle_image,
    gpu=GPU_TYPE,
    cpu=4.0,
    memory=16384,
    max_containers=4,
    scaledown_window=600,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    volumes={"/models": volume},  # Mount volume at /models
    timeout=300,
)
class PaddleOCRService:
    """
    PaddleOCR service with persistent model caching and GPU snapshots
    """

    @modal.enter(snap=True)
    def load_model(self):
        """
        Runs once when container starts.
        Sets up the multiprocessing pool for parallel inference.
        This method runs before the snapshot is taken.
        """
        print("=" * 60)
        print("Starting PaddleOCR Service with persistent caching and GPU snapshots")
        print("=" * 60)

        # Set start method to spawn for CUDA compatibility
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except:
            pass

        # Setup cache directories first
        setup_model_cache()

        # Setup symlink for PaddleX cache
        try:
            paddlex_root = "/root/.paddlex"
            paddlex_volume = "/models/paddlex"

            # Remove existing symlink/directory
            if os.path.islink(paddlex_root):
                os.unlink(paddlex_root)
            elif os.path.exists(paddlex_root):
                import shutil
                shutil.rmtree(paddlex_root)

            # Create symlink to volume
            os.symlink(paddlex_volume, paddlex_root)
            print(f"Symlinked {paddlex_root} -> {paddlex_volume}")

        except Exception as e:
            print(f"WARN: Failed to setup PaddleX symlink: {e}")

        # Maintain a worker for isolated GPU access
        # Note: The actual worker initialization happens in the snapshot phase
        # This means the worker process and model will be part of the snapshot
        # Initialize the model directly in the main process
        # This allows Modal to capture the model state in memory for the GPU snapshot.
        self.ocr = _init_ocr()
        print(f"Model initialization complete. Success: {self.ocr is not None}")

        # Force disable PIR before importing paddle
        os.environ["FLAGS_enable_pir_api"] = "0"
        os.environ["FLAGS_enable_pir_in_executor"] = "0"
        os.environ["FLAGS_ir_optim_cache_disable"] = "1"

        start_time = time.time()

        import paddle
        import numpy as np

        try:
            import paddle.tensor.manipulation
            actual_bool_type = paddle.to_tensor([False], dtype='bool').dtype

            paddle.bool = actual_bool_type
            paddle.tensor.bool = actual_bool_type
            paddle.tensor.manipulation.bool = actual_bool_type

            print(f"Synchronized paddle.bool across modules: {actual_bool_type}")
        except Exception as e:
            print(f"WARN: Failed to sync paddle.bool: {e}")

        # Monkey-patch masked_scatter for extra safety
        try:
            _orig_masked_scatter = paddle.Tensor.masked_scatter
            def patched_masked_scatter(self, mask, value):
                if hasattr(mask, "dtype") and mask.dtype != paddle.bool:
                    try:
                        mask = mask.cast(paddle.bool)
                    except:
                        pass
                return _orig_masked_scatter(self, mask, value)
            paddle.Tensor.masked_scatter = patched_masked_scatter
            print("Applied safety patch for paddle.Tensor.masked_scatter")
        except Exception as e:
            print(f"WARN: Failed to patch masked_scatter: {e}")

        # Disable PIR API
        try:
            paddle.set_flags({
                "FLAGS_enable_pir_api": 0,
                "FLAGS_enable_pir_in_executor": 0,
                "FLAGS_ir_optim_cache_disable": 1
            })
        except Exception:
            pass

        try:
            print(f"Paddle (Main Process) device: {paddle.device.get_device()}")
        except Exception as e:
            print(f"Warning checking device: {e}")

        # No-op: model is already loaded in self.ocr

        elapsed = time.time() - start_time
        print("=" * 60)
        print(f"Container initialized in {elapsed:.2f}s")
        print("=" * 60)
        print("Ready to take GPU snapshot...")

    @modal.method()
    async def predict(
        self,
        image_data: Any,
        prompt: str = "ocr",
        use_layout_detection: bool = True,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = True,
        layout_merge_bboxes_mode: str = "small",
    ) -> Dict[str, Any]:
        """
        Predict method for a single image request.
        """
        try:
            # Decode image in the main thread (fast)
            img = await asyncio.to_thread(decode_image, image_data)
            if img is None:
                return {"success": False, "error": "Invalid image data"}

            # Prepare arguments for the model
            worker_kwargs = {
                "input": img,
                "use_queues": False,
                "prompt_label": prompt,
                "use_layout_detection": use_layout_detection,
                "use_doc_orientation_classify": use_doc_orientation_classify,
                "use_doc_unwarping": use_doc_unwarping,
                "layout_merge_bboxes_mode": layout_merge_bboxes_mode,
            }

            # Run prediction directly
            return await asyncio.to_thread(_predict_with_model, self.ocr, worker_kwargs)

        except Exception as e:
            import traceback
            err_msg = f"{type(e).__name__}: {str(e)}"
            print(f"Prediction error: {err_msg}")
            return {
                "success": False,
                "error": err_msg,
                "traceback": traceback.format_exc()
            }
