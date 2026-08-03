"""Replicate-powered AI ops for BotifyAI's media editor.

These ops call hosted models on replicate.com and download the result.
Requires REPLICATE_API_TOKEN env var. If it's not set, ops raise a clear
"AI feature unavailable" error so the endpoint can degrade gracefully.

Each op takes an input file path and returns the path to a new file.
"""

import logging
import mimetypes
import os
import shutil
import tempfile
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
MODELS: dict[str, str] = {
    # Background removal from an image
    "ai_bg_remove": "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
    # 2x/4x super-resolution
    "ai_upscale": "nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa",
    # Face restoration & smoothing
    "ai_beautify": "sczhou/codeformer:cc4956dd26fa5a7185d5660cc2100fab1cfa4c076c81ff7ceb9d5bf9e1b6c31c",
    # Object / person removal (fills with plausible content)
    "ai_object_remove": "zylim0702/remove-object:0e3a841c913f597c1e4c321560aa69e2bc1f15c65f8c366caafc379240efd8ba",
    # Anime / cartoon stylization
    "ai_cartoonify": "cjwbw/animegan:1e8e7b4a97ea3f39d3a0c8b3f95f27c988bff2e2617e1bea0a7be36c9843c98d",
    # Text-to-image (Flux Schnell — fast and cheap)
    "ai_text_to_image": "black-forest-labs/flux-schnell",
}

# Which param key each model expects for its image input
_INPUT_IMAGE_KEY: dict[str, str] = {
    "ai_bg_remove": "image",
    "ai_upscale": "image",
    "ai_beautify": "image",
    "ai_object_remove": "image",
    "ai_cartoonify": "image",
}

# Extra defaults per model (merged with user params)
_DEFAULTS: dict[str, dict[str, Any]] = {
    "ai_upscale": {"scale": 2},
    "ai_beautify": {"codeformer_fidelity": 0.6, "background_enhance": True, "face_upsample": True},
    "ai_text_to_image": {"aspect_ratio": "1:1", "output_format": "png", "num_inference_steps": 4},
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
    output = client.run(model_id, input=inputs)

    prefer_ext = ".png"
    return _save_output(output, out_dir, prefer_ext)


# Public entry points --------------------------------------------------------

def ai_bg_remove(image_path: str, params: dict[str, Any], out_dir: str) -> str:
    return _run_model("ai_bg_remove", image_path, params, out_dir)


def ai_upscale(image_path: str, params: dict[str, Any], out_dir: str) -> str:
    return _run_model("ai_upscale", image_path, params, out_dir)


def ai_beautify(image_path: str, params: dict[str, Any], out_dir: str) -> str:
    return _run_model("ai_beautify", image_path, params, out_dir)


def ai_object_remove(image_path: str, params: dict[str, Any], out_dir: str) -> str:
    return _run_model("ai_object_remove", image_path, params, out_dir)


def ai_cartoonify(image_path: str, params: dict[str, Any], out_dir: str) -> str:
    return _run_model("ai_cartoonify", image_path, params, out_dir)


def ai_text_to_image(prompt: str, params: dict[str, Any], out_dir: str) -> str:
    return _run_model("ai_text_to_image", None, {**params, "prompt": prompt}, out_dir)


AI_IMAGE_OPS = {
    "ai_bg_remove": ai_bg_remove,
    "ai_upscale": ai_upscale,
    "ai_beautify": ai_beautify,
    "ai_object_remove": ai_object_remove,
    "ai_cartoonify": ai_cartoonify,
}
