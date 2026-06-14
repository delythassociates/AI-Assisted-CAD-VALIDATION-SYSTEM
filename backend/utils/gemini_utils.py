import asyncio
import logging

logger = logging.getLogger(__name__)

async def call_gemini_with_timeout(func, *args, timeout_seconds=8.0, fallback=None, **kwargs):
    """Run a synchronous Gemini function in a thread with a hard timeout. Returns fallback on any failure."""
    try:
        coro = asyncio.to_thread(func, *args, **kwargs)
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except (asyncio.TimeoutError, Exception) as exc:
        logger.error(f"Gemini call to {func.__name__} timed out or failed: {exc}")
        return fallback
