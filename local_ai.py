"""Self-hosted AI ops that run on the server without paid API calls.

Currently just background removal via `rembg` (uses small ONNX models).
Kept separate from replicate_ops.py so we can toggle sources per-op.
"""

import logging
import os
import uuid

logger = logging.getLogger(__name__)

# Small model — ~4 MB on disk, ~200 MB RAM at runtime. Trades some accuracy
# for fitting in a 512 MB Railway container.
_MODEL_NAME = os.environ.get("REMBG_MODEL", "u2netp")

_session = None


def _get_session():
    """Lazy-load the rembg session so the model isn't loaded at import time."""
    global _session
    if _session is not None:
        return _session
    try:
        from rembg import new_session
    except ImportError as e:
        raise RuntimeError(
            f"rembg not installed: {e}. Add `rembg` to requirements.txt."
        )
    logger.info("Loading rembg model '%s' (first request may take a while)", _MODEL_NAME)
    _session = new_session(_MODEL_NAME)
    return _session


def local_bg_remove(image_path: str, params: dict, out_dir: str) -> str:
    """Remove background from an image, return path to a PNG with alpha."""
    from rembg import remove

    with open(image_path, "rb") as f:
        input_bytes = f.read()

    session = _get_session()
    output_bytes = remove(input_bytes, session=session)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"bg_{uuid.uuid4().hex}.png")
    with open(out_path, "wb") as f:
        f.write(output_bytes)
    return out_path
