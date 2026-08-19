import os
import sys
import logging

logger = logging.getLogger("vercel_entry")

# Vercel serverless: add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Only writable dir on Vercel is /tmp
os.environ.setdefault("UPLOAD_DIR", "/tmp/uploads")
os.makedirs("/tmp/uploads", exist_ok=True)

# Redirect HuggingFace model cache to /tmp (read-only filesystem elsewhere)
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/tmp/hf_cache")
os.makedirs("/tmp/hf_cache", exist_ok=True)

try:
    from main import app
    logger.info("[VERCEL] App loaded successfully")
except Exception as e:
    logger.error("[VERCEL] Failed to load app: %s", e)
    raise
