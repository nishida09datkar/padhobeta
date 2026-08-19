import logging
from config import settings

logger = logging.getLogger("model_router")


MODEL_MAP = {
    "lower": settings.LOWER_MODEL,
    "average": settings.AVERAGE_MODEL,
    "higher": settings.HIGHER_MODEL,
}

TIER_ORDER = ["lower", "average", "higher"]


def resolve_model_name(tier: str) -> str:
    model = MODEL_MAP.get(tier)
    if not model:
        logger.warning("[MODEL_ROUTER] Unknown tier '%s', falling back to average", tier)
        model = MODEL_MAP["average"]
    return model


def get_next_tier(current_tier: str) -> str | None:
    try:
        idx = TIER_ORDER.index(current_tier)
        if idx < len(TIER_ORDER) - 1:
            return TIER_ORDER[idx + 1]
    except ValueError:
        pass
    return None
