"""
Video summarization using the Gemini Files API.
Uploads a local video, waits for processing, generates a natural-language summary, then cleans up.
"""
import asyncio
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

#MODEL = 'gemini-3.5-flash'
MODEL = 'gemini-3-flash-preview'
PROMPT = (
    "This video should provide an example of how a hypothetical tool for visual assistance should work"
    "Summarize what is happening in this video, including any relevant audio context. "
    "Break down the examples in presented in the video to articulate what kinds of input should generate specific outputs"
    "In the event the user is describing a hypothetical scenario/pretending with things in their environment, your description should match what they are pretending things are, not the actual item in the mock up"
    "If the user specifies a particular relationship between an action they take/a thing that comes onscreen and an output produced, include that relationship in your description"
    "Conclude with a sentence summarizing the key functionality and behavior of the desired tool"
)
POLL_INTERVAL = 2.0   # seconds between file-state checks
POLL_TIMEOUT  = 120.0  # max seconds to wait for file to become ACTIVE

# Mime types recognised as video by the Files API
_VIDEO_MIME = {
    '.mp4':  'video/mp4',
    '.m4v':  'video/mp4',
    '.mov':  'video/quicktime',
    '.mpeg': 'video/mpeg',
    '.mpg':  'video/mpeg',
    '.avi':  'video/avi',
    '.webm': 'video/webm',
}


def _make_client():
    """Return a google-genai Client, or None if unavailable."""
    try:
        from google import genai  # noqa: PLC0415
        api_key = os.environ.get('GEMINI_API_KEY', '')
        if not api_key:
            logger.warning("GEMINI_API_KEY not set — video summarization disabled")
            return None
        return genai.Client(api_key=api_key)
    except ImportError:
        logger.warning("google-genai not installed — video summarization disabled")
        return None


def _is_active(file_obj) -> bool:
    """Return True when the uploaded file's processing state is ACTIVE."""
    state = file_obj.state
    # google-genai exposes state as an enum; compare both .name and str() for safety
    if hasattr(state, 'name'):
        return state.name == 'ACTIVE'
    return str(state) in ('ACTIVE', 'FileState.ACTIVE')


async def summarize_video(video_path: str) -> str:
    """
    Upload *video_path* to Gemini Files API, summarize it, delete the remote file.

    Args:
        video_path: Absolute path to a local video file.

    Returns:
        A natural-language summary string, or "" on any failure (so callers
        can always proceed without blocking on the video step).
    """
    path = Path(video_path)
    if not path.exists():
        logger.error("Video file not found: %s", video_path)
        return ""

    client = _make_client()
    if client is None:
        return ""

    mime_type = _VIDEO_MIME.get(path.suffix.lower(), 'video/mp4')
    uploaded_file = None
    loop = asyncio.get_event_loop()

    try:
        from google.genai import types  # noqa: PLC0415

        logger.info("Uploading %s to Gemini Files API (%s) …", path.name, mime_type)

        # Upload is blocking I/O — offload to executor
        uploaded_file = await loop.run_in_executor(
            None,
            lambda: client.files.upload(
                file=str(path),
                config=types.UploadFileConfig(
                    mime_type=mime_type,
                    display_name=path.name,
                ),
            ),
        )

        logger.info("Upload complete — name=%s  state=%s", uploaded_file.name, uploaded_file.state)

        # Poll until the file is processed
        elapsed = 0.0
        file_name = uploaded_file.name  # capture so closures don't rebind
        while not _is_active(uploaded_file):
            if elapsed >= POLL_TIMEOUT:
                logger.error("Timed out waiting for Gemini file %s to become ACTIVE", file_name)
                return ""
            await asyncio.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
            uploaded_file = await loop.run_in_executor(
                None,
                lambda n=file_name: client.files.get(name=n),
            )
            logger.debug("File state: %s (%.0fs elapsed)", uploaded_file.state, elapsed)

        logger.info("File ACTIVE — generating summary …")

        # generate_content is also blocking
        file_ref = uploaded_file  # stable reference for lambda
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=MODEL,
                contents=[file_ref, PROMPT],
            ),
        )

        summary = (response.text or "").strip()
        logger.info("Video summary generated (%d chars)", len(summary))
        return summary

    except Exception:
        logger.error("Video summarization failed", exc_info=True)
        return ""

    finally:
        # Always attempt to delete the remote file to avoid quota waste
        if uploaded_file is not None:
            try:
                _name = uploaded_file.name
                await loop.run_in_executor(None, lambda n=_name: client.files.delete(name=n))
                logger.info("Deleted remote Gemini file: %s", _name)
            except Exception:
                logger.warning("Could not delete remote Gemini file", exc_info=True)
