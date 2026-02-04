APP_NAME = "paddleocr-vl"
GPU_TYPE = "T4"

ENV_VARS = {
    # "PADDLE_INF_NUM_THREADS": "4",
    # "PADDLE_INF_THREADS": "4",
    # "PADDLE_PDX_VLM_PARALLEL": "4",
    # "OMP_NUM_THREADS": "4",
    "FLAGS_enable_pir_api": "0",
    "FLAGS_enable_pir_in_executor": "0",
    "FLAGS_ir_optim_cache_disable": "1",
    "DISABLE_MODEL_SOURCE_CHECK": "True",
    # Persistent model storage
    "PADDLE_HOME": "/models/paddle",
    "PADDLEOCR_BASE_DIR": "/models/paddleocr",
    "XDG_CACHE_HOME": "/models/cache",
    "HUGGINGFACE_HUB_CACHE": "/models/huggingface",
    "TRANSFORMERS_CACHE": "/models/huggingface/transformers",
    "HF_HOME": "/models/huggingface",
}
