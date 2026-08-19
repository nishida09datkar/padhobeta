import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen/qwen3.6-27b")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "5"))
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

    LOWER_MODEL: str = os.getenv("LOWER_MODEL", "allam-2-7b")
    AVERAGE_MODEL: str = os.getenv("AVERAGE_MODEL", "qwen/qwen3.6-27b")
    HIGHER_MODEL: str = os.getenv("HIGHER_MODEL", "openai/gpt-oss-120b")
    ORCHESTRATOR_MODEL: str = os.getenv("ORCHESTRATOR_MODEL", "allam-2-7b")
    ORCHESTRATOR_DEBUG: bool = os.getenv("ORCHESTRATOR_DEBUG", "false").lower() == "true"
    AI_WARMUP_ENABLED: bool = os.getenv("AI_WARMUP_ENABLED", "false").lower() == "true"
    ROUTING_CACHE_ENABLED: bool = os.getenv("ROUTING_CACHE_ENABLED", "true").lower() == "true"
    MAX_MODEL_FALLBACKS: int = int(os.getenv("MAX_MODEL_FALLBACKS", "1"))

    ALLOWED_EXTENSIONS: dict[str, str] = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".pptx": "pptx",
        ".jpg": "image",
        ".jpeg": "image",
        ".png": "image",
        ".bmp": "image",
        ".tiff": "image",
        ".tif": "image",
    }

    MAX_FILE_SIZE_MB: int = 50


settings = Settings()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
