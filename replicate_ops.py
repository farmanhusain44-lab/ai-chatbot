"""Replicate-powered AI ops for BotifyAI's media editor.

These ops call hosted models on replicate.com and download the result.
Requires REPLICATE_API_TOKEN env var. If it's not set, ops raise a clear
"AI feature unavailable" error so the endpoint can degrade gracefully.

Each op takes an input file path and returns the path to a new file.
"""

import logging
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from typing import Any

import requests

logger = logging.getLogger(__name__)


class ReplicateUnavailable(Exception):
    """REPLICATE_API_TOKEN missing or client library missing."""


_TOKEN_ENV_NAMES = (
    "REPLICATE_API_TOKEN",
    "REPLICATE_API_KEY",
    "REPLICATE_TOKEN",
    "REPLICATE_KEY",
)

_TOKEN_VALUE_PREFIX = "r8_"


def _get_token() -> str | None:
    # 1. Preferred: standard env var names.
    for name in _TOKEN_ENV_NAMES:
        val = os.environ.get(name)
        if val and val.strip():
            return val.strip()
    # 2. Fallback: any env var whose value looks like a Replicate token.
    # Replicate tokens all start with "r8_". This lets us find the token even
    # if the user saved it under a custom variable name.
    for k, v in os.environ.items():
        if not v:
            continue
        stripped = v.strip()
        if stripped.startswith(_TOKEN_VALUE_PREFIX) and len(stripped) >= 20:
            logger.info("Using Replicate token from env var %s (matched by r8_ prefix)", k)
            return stripped
    return None


def _client():
    token = _get_token()
    if not token:
        raise ReplicateUnavailable(
            "Replicate token not set. Add REPLICATE_API_TOKEN in Railway → Variables to enable AI features."
        )
    try:
        import replicate
    except ImportError as e:
        raise ReplicateUnavailable(f"replicate library not installed: {e}")
    return replicate.Client(api_token=token)


# Model versions pinned so behavior is stable. Update deliberately.
# Only ops that have been end-to-end verified live are included.
# Deferred until working versions are pinned:
#   ai_beautify      — codeformer version removed from Replicate (was 422)
#   ai_cartoonify    — animegan variants removed (both 422s)
#   ai_object_remove — requires a mask image input; needs UX design
MODELS: dict[str, str] = {
    # Background removal from an image
    "ai_bg_remove": "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
    # 2x/4x super-resolution
    "ai_upscale": "nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa",
    # Text-to-image (Flux Schnell — fast and cheap)
    "ai_text_to_image": "black-forest-labs/flux-schnell",
    # Speech-to-text with timestamps for SRT captions.
    # Uses Whisper large-v3; produces srt-formatted output.
    "ai_transcribe": "openai/whisper",
    # Natural-language image transformation ("make him a bodybuilder",
    # "put him in a spacesuit", "cartoon style", etc). InstructPix2Pix is
    # purpose-built for prompt-driven photo edits, roughly $0.005 per call.
    "ai_transform_image": "timothybrooks/instruct-pix2pix:30c1d0b916a6f8efce20493f5d61ee27491ab2a60437c13c588468b9810ec23f",
    # Face restoration / enhancement — sharpens blurry / low-res faces
    # while preserving identity. CodeFormer is currently alive on Replicate.
    "ai_face_enhance": "sczhou/codeformer:cc4956dd26fa5a7185d5660cc2100fde1e761db2e5b62c831a44c00ef0f22e0b",
    # Voice / audio cleanup — Resemble Enhance denoises AND super-resolves
    # speech up to 44.1kHz. Great for cleaning mobile-recorded voiceovers.
    # No pinned version — Replicate resolves to the latest published one.
    "ai_voice_enhance": "resemble-ai/resemble-enhance",
}

# Which param key each model expects for its image input
_INPUT_IMAGE_KEY: dict[str, str] = {
    "ai_bg_remove": "image",
    "ai_upscale": "image",
    "ai_transform_image": "image",
    "ai_face_enhance": "image",
}

# Extra defaults per model (merged with user params)
_DEFAULTS: dict[str, dict[str, Any]] = {
    "ai_upscale": {"scale": 2},
    "ai_text_to_image": {"aspect_ratio": "1:1", "output_format": "png", "num_inference_steps": 4},
    # instruct-pix2pix defaults — moderate strength so identity is preserved
    # while the transform is visible. num_inference_steps=25 is a good
    # quality/speed tradeoff.
    "ai_transform_image": {
        "num_inference_steps": 25,
        "image_guidance_scale": 1.5,
        "guidance_scale": 7.5,
    },
    "ai_face_enhance": {
        "codeformer_fidelity": 0.7,
        "background_enhance": True,
        "face_upsample": True,
        "upscale": 2,
    },
}


def _guess_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def _download(url: str, out_dir: str, prefer_ext: str | None = None) -> str:
    resp = requests.get(url, stream=True, timeout=180)
    resp.raise_for_status()
    ext = prefer_ext
    if not ext:
        ctype = resp.headers.get("content-type", "").lower()
        if "png" in ctype:
            ext = ".png"
        elif "jpeg" in ctype or "jpg" in ctype:
            ext = ".jpg"
        elif "mp4" in ctype:
            ext = ".mp4"
        else:
            ext = os.path.splitext(url.split("?")[0])[1] or ".bin"
    out_path = os.path.join(out_dir, f"ai_{uuid.uuid4().hex}{ext}")
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 15):
            f.write(chunk)
    return out_path


def _first_url(output: Any) -> str:
    """Replicate returns either a URL string, a list of URLs, or a FileOutput-like
    object with a .url attribute. Normalize to a single URL string."""
    if isinstance(output, str):
        return output
    if isinstance(output, list) and output:
        return _first_url(output[0])
    if hasattr(output, "url"):
        return output.url
    if hasattr(output, "read"):
        # FileOutput — read to a temp file, but we still need a URL for _download.
        # Save directly instead.
        raise RuntimeError("Got FileOutput from Replicate — call save_output() instead")
    raise RuntimeError(f"Unexpected Replicate output type: {type(output)!r}")


def _save_output(output: Any, out_dir: str, prefer_ext: str | None = None) -> str:
    """Handles both URL outputs and FileOutput objects (newer replicate lib)."""
    ext = prefer_ext or ".png"

    # FileOutput case — newer replicate lib returns objects with .read()
    # that returns bytes when called with no args (unlike a file-like that
    # accepts a length). Also expose .url as a fallback.
    if not isinstance(output, (str, list)) and hasattr(output, "read"):
        out_path = os.path.join(out_dir, f"ai_{uuid.uuid4().hex}{ext}")
        try:
            data = output.read()
            with open(out_path, "wb") as f:
                f.write(data if isinstance(data, (bytes, bytearray)) else data.encode())
            return out_path
        except TypeError:
            # Fall through to URL download
            if hasattr(output, "url"):
                return _download(output.url, out_dir, prefer_ext)
            raise

    # list of FileOutput / URLs
    if isinstance(output, list) and output:
        return _save_output(output[0], out_dir, prefer_ext)

    # Plain URL string
    return _download(_first_url(output), out_dir, prefer_ext)


_RATE_LIMIT_MAX_RETRIES = 4


def _parse_reset_seconds(err_msg: str) -> float:
    """Extract 'resets in ~7s' style hints from Replicate's 429 message."""
    m = re.search(r"resets in\s*~?\s*(\d+(?:\.\d+)?)\s*s", err_msg, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return 0.0


def _run_model(op: str, image_path: str | None, params: dict[str, Any], out_dir: str) -> str:
    client = _client()
    if op not in MODELS:
        raise ValueError(f"Unknown Replicate op: {op}")
    model_id = MODELS[op]
    inputs: dict[str, Any] = {**_DEFAULTS.get(op, {}), **params}

    if image_path is not None:
        key = _INPUT_IMAGE_KEY.get(op, "image")
        inputs[key] = open(image_path, "rb")

    logger.info("Replicate %s: inputs keys=%s", op, list(inputs.keys()))

    # Retry with exponential backoff on 429 (rate-limit) errors. Replicate
    # throttles accounts under $5 lifetime spend to 1 request per burst — a
    # short wait usually clears it, so multi-op requests still succeed.
    last_err: Exception | None = None
    for attempt in range(_RATE_LIMIT_MAX_RETRIES):
        try:
            output = client.run(model_id, input=inputs)
            return _save_output(output, out_dir, ".png")
        except Exception as e:
            msg = str(e)
            is_rate_limit = "429" in msg or "throttled" in msg.lower() or "rate limit" in msg.lower()
            if not is_rate_limit or attempt == _RATE_LIMIT_MAX_RETRIES - 1:
                raise
            hinted = _parse_reset_seconds(msg)
            wait = max(hinted + 1.0, 2.0 * (2 ** attempt))  # honor server hint or backoff
            logger.warning(
                "Replicate %s rate-limited (attempt %d/%d), sleeping %.1fs then retrying",
                op, attempt + 1, _RATE_LIMIT_MAX_RETRIES, wait,
            )
            time.sleep(wait)
            # Reopen file handle if input was a file — some clients consume it
            if image_path is not None:
                key = _INPUT_IMAGE_KEY.get(op, "image")
                inputs[key] = open(image_path, "rb")
            last_err = e
    # Should not reach here — either returned or raised in the loop.
    raise last_err if last_err else RuntimeError("Replicate run failed without a specific error")


# Public entry points --------------------------------------------------------

def ai_bg_remove(image_path: str, params: dict[str, Any], out_dir: str) -> str:
    return _run_model("ai_bg_remove", image_path, params, out_dir)


def ai_upscale(image_path: str, params: dict[str, Any], out_dir: str) -> str:
    return _run_model("ai_upscale", image_path, params, out_dir)


def ai_text_to_image(prompt: str, params: dict[str, Any], out_dir: str) -> str:
    return _run_model("ai_text_to_image", None, {**params, "prompt": prompt}, out_dir)


def ai_transform_image(image_path: str, params: dict[str, Any], out_dir: str) -> str:
    """Apply a natural-language transformation to an image via
    InstructPix2Pix. The [params] dict must contain a "prompt" key with
    the transformation instruction (e.g. "make him a bodybuilder",
    "put him floating in the sky", "cartoon style"). Preserves the
    subject's identity while applying the transformation.
    """
    prompt = (params or {}).get("prompt", "").strip()
    if not prompt:
        raise RuntimeError("ai_transform_image requires a 'prompt' param")
    return _run_model("ai_transform_image", image_path, params, out_dir)


def ai_face_enhance(image_path: str, params: dict[str, Any], out_dir: str) -> str:
    """CodeFormer face restoration — sharpens faces while keeping identity."""
    return _run_model("ai_face_enhance", image_path, params, out_dir)


def ai_voice_enhance(video_or_audio_path: str, params: dict[str, Any], out_dir: str) -> str:
    """Clean up a voiceover / video's audio track via Resemble Enhance.

    Video input: extract audio → enhance → mux back with the original
    video track. Output filename mirrors the input.
    Audio input (mp3/wav/m4a/aac): enhance in place.
    """
    ext = os.path.splitext(video_or_audio_path)[1].lower().lstrip('.')
    is_video = ext in {"mp4", "mov", "mkv", "webm", "m4v", "avi", "3gp"}

    client = _client()
    os.makedirs(out_dir, exist_ok=True)

    # 1. Extract audio to WAV (Resemble likes clean PCM input).
    if is_video:
        audio_in = os.path.join(out_dir, f"vo_in_{uuid.uuid4().hex}.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_or_audio_path, "-vn",
             "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", audio_in],
            check=True, capture_output=True,
        )
    else:
        audio_in = video_or_audio_path

    # 2. Enhance via Replicate.
    logger.info("Replicate resemble-enhance on %s", audio_in)
    output = client.run(
        MODELS["ai_voice_enhance"],
        input={"input_audio": open(audio_in, "rb")},
    )
    enhanced_url = _first_url(output)
    enhanced_path = _download(enhanced_url, out_dir, prefer_ext="wav")

    # 3. If video, mux enhanced audio back with original video track.
    if is_video:
        out_video = os.path.join(out_dir, f"vo_out_{uuid.uuid4().hex}.mp4")
        subprocess.run(
            ["ffmpeg", "-y",
             "-i", video_or_audio_path,     # video (with old audio)
             "-i", enhanced_path,            # new audio
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-map", "0:v:0", "-map", "1:a:0",
             "-shortest", out_video],
            check=True, capture_output=True,
        )
        return out_video

    return enhanced_path


def ai_transcribe_to_srt(audio_path: str, out_dir: str) -> str:
    """Run Whisper on [audio_path] and return the path to a saved .srt file.
    Uses openai/whisper on Replicate."""
    client = _client()
    inputs: dict[str, Any] = {
        "audio": open(audio_path, "rb"),
        "model": "large-v3",
        "translate": False,
        "temperature": 0,
        "transcription": "srt",
        "condition_on_previous_text": True,
        "no_speech_threshold": 0.6,
    }
    logger.info("Replicate whisper: inputs=%s", list(inputs.keys()))
    output = client.run(MODELS["ai_transcribe"], input=inputs)
    # Whisper output is a dict with 'transcription' key (SRT string) OR
    # sometimes just the string directly. Handle both.
    srt_text: str | None = None
    if isinstance(output, dict):
        srt_text = output.get("transcription") or output.get("srt")
    elif isinstance(output, str):
        srt_text = output
    if not srt_text:
        raise RuntimeError(f"Whisper returned no transcription (got {type(output)})")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"captions_{uuid.uuid4().hex}.srt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(srt_text)
    return out_path


AI_IMAGE_OPS = {
    "ai_bg_remove": ai_bg_remove,
    "ai_upscale": ai_upscale,
    "ai_transform_image": ai_transform_image,
    "ai_face_enhance": ai_face_enhance,
}

# Video-scope AI ops. Same signature as image ops (input path + params →
# output path). Kept separate so the routing layer knows to accept a
# video input for these instead of blocking with the image-only check.
AI_VIDEO_OPS = {
    "ai_voice_enhance": ai_voice_enhance,
}
