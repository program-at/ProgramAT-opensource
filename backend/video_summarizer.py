"""
Video summarization using the Gemini Files API, with GCS fallback for Vertex AI keys.

Primary path: upload to Gemini Files API (Google AI Studio key).
Fallback path: upload to GCS and pass gs:// URI via Part.from_uri (Vertex AI key).
"""
import asyncio
import logging
import os
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

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

# Mime types recognised as video
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
    """Return a google-genai Client (AI Studio key), or None if unavailable."""
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


def _make_vertex_client():
    """Return a google-genai Client configured for Vertex AI, or None if unavailable."""
    try:
        from google import genai  # noqa: PLC0415
        project = os.environ.get('PROJECT_ID', '')
        if not project:
            return None
        return genai.Client(vertexai=True, project=project, location='us-central1')
    except ImportError:
        return None


def _is_active(file_obj) -> bool:
    """Return True when the uploaded file's processing state is ACTIVE."""
    state = file_obj.state
    if hasattr(state, 'name'):
        return state.name == 'ACTIVE'
    return str(state) in ('ACTIVE', 'FileState.ACTIVE')


async def _summarize_via_gcs(path: Path, mime_type: str, loop) -> str:
    """
    Upload video to GCS and summarize via Vertex AI using a gs:// URI.
    Reads BUCKET_NAME and PROJECT_ID from env. Deletes the GCS object after use.
    Returns summary string or "" on failure.
    """
    bucket_name = os.environ.get('BUCKET_NAME', '')
    if not bucket_name:
        logger.warning("BUCKET_NAME not set — GCS fallback disabled")
        return ""

    client = _make_vertex_client()
    if client is None:
        logger.warning("Vertex AI client unavailable — GCS fallback disabled")
        return ""

    blob_name = f"video_summarizer/{uuid.uuid4().hex}{path.suffix}"
    gs_uri = f"gs://{bucket_name}/{blob_name}"

    try:
        from google.cloud import storage as gcs  # noqa: PLC0415

        logger.info("Uploading %s to GCS: %s", path.name, gs_uri)

        def _upload():
            gcs_client = gcs.Client()
            bucket = gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(path), content_type=mime_type)

        await loop.run_in_executor(None, _upload)
        logger.info("GCS upload complete — %s", gs_uri)

    except Exception as e:
        logger.error("GCS upload failed (%s): %s", type(e).__name__, e, exc_info=True)
        return ""

    try:
        from google.genai import types  # noqa: PLC0415

        def _generate():
            part = types.Part.from_uri(file_uri=gs_uri, mime_type=mime_type)
            return client.models.generate_content(
                model=MODEL,
                contents=[part, PROMPT],
            )

        response = await loop.run_in_executor(None, _generate)
        summary = (response.text or "").strip()
        logger.info("GCS video summary generated (%d chars)", len(summary))
        return summary

    except Exception as e:
        logger.error("Vertex AI summarization via GCS failed (%s): %s", type(e).__name__, e, exc_info=True)
        return ""

    finally:
        try:
            from google.cloud import storage as gcs  # noqa: PLC0415

            def _delete():
                gcs_client = gcs.Client()
                bucket = gcs_client.bucket(bucket_name)
                bucket.blob(blob_name).delete()

            await loop.run_in_executor(None, _delete)
            logger.info("Deleted GCS object: %s", gs_uri)
        except Exception:
            logger.warning("Could not delete GCS object: %s", gs_uri, exc_info=True)


async def summarize_video(video_path: str) -> str:
    """
    Summarize a local video file using Gemini.

    Tries the Gemini Files API (AI Studio) first. If that fails, falls back to
    uploading to GCS and using Vertex AI (requires BUCKET_NAME and PROJECT_ID).

    Returns a natural-language summary string, or "" on total failure.
    """
    path = Path(video_path)
    if not path.exists():
        logger.error("Video file not found: %s", video_path)
        return ""

    mime_type = _VIDEO_MIME.get(path.suffix.lower(), 'video/mp4')
    loop = asyncio.get_event_loop()

    # --- Primary: Gemini Files API ---
    client = _make_client()
    if client is not None:
        uploaded_file = None
        try:
            from google.genai import types  # noqa: PLC0415

            logger.info("Uploading %s to Gemini Files API (%s) …", path.name, mime_type)

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

            elapsed = 0.0
            file_name = uploaded_file.name
            while not _is_active(uploaded_file):
                if elapsed >= POLL_TIMEOUT:
                    logger.error("Timed out waiting for Gemini file %s to become ACTIVE", file_name)
                    raise TimeoutError("File API polling timed out")
                await asyncio.sleep(POLL_INTERVAL)
                elapsed += POLL_INTERVAL
                uploaded_file = await loop.run_in_executor(
                    None,
                    lambda n=file_name: client.files.get(name=n),
                )
                logger.debug("File state: %s (%.0fs elapsed)", uploaded_file.state, elapsed)

            logger.info("File ACTIVE — generating summary …")

            file_ref = uploaded_file
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=MODEL,
                    contents=[file_ref, PROMPT],
                ),
            )

            summary = (response.text or "").strip()
            logger.info("Files API summary generated (%d chars)", len(summary))
            return summary

        except Exception:
            logger.warning("Gemini Files API summarization failed — trying GCS fallback", exc_info=True)

        finally:
            if uploaded_file is not None:
                try:
                    _name = uploaded_file.name
                    await loop.run_in_executor(None, lambda n=_name: client.files.delete(name=n))
                    logger.info("Deleted remote Gemini file: %s", _name)
                except Exception:
                    logger.warning("Could not delete remote Gemini file", exc_info=True)

    # --- Fallback: GCS + Vertex AI ---
    logger.info("Attempting GCS fallback for video summarization")
    return await _summarize_via_gcs(path, mime_type, loop)
