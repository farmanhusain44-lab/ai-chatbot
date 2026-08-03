"""Server-side media editor for BotifyAI.

Executes a structured operation list against an input image or video and
returns the path to the edited output file.

Image ops use Pillow (PIL). Video ops shell out to ffmpeg — the binary must
be installed on the host (Railway: add via nixpacks.toml).
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from replicate_ops import AI_IMAGE_OPS as _REPLICATE_AI_OPS, ai_text_to_image, ReplicateUnavailable
from local_ai import local_bg_remove

# Route AI image ops: local (free) where possible, Replicate for the rest.
AI_IMAGE_OPS: dict[str, Any] = {
    **_REPLICATE_AI_OPS,
    "ai_bg_remove": local_bg_remove,  # free self-hosted, overrides Replicate
}

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".heic"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}

MAX_IMAGE_MB = 25
MAX_VIDEO_MB = 200


class MediaEditorError(Exception):
    pass


def kind_from_path(path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    raise MediaEditorError(f"Unsupported file extension: {ext}")


def apply_image_ops(input_path: str, output_path: str, ops: list[dict[str, Any]]) -> None:
    img = Image.open(input_path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.mode else "RGB")

    for op in ops:
        name = op.get("op")
        params = op.get("params", {})

        if name == "rotate":
            img = img.rotate(-float(params.get("degrees", 90)), expand=True)

        elif name == "flip":
            direction = params.get("direction", "horizontal")
            img = ImageOps.mirror(img) if direction == "horizontal" else ImageOps.flip(img)

        elif name == "resize":
            w = int(params.get("width", img.width))
            h = int(params.get("height", img.height))
            img = img.resize((w, h), Image.LANCZOS)

        elif name == "crop":
            l = int(params.get("left", 0))
            t = int(params.get("top", 0))
            r = int(params.get("right", img.width))
            b = int(params.get("bottom", img.height))
            img = img.crop((l, t, r, b))

        elif name == "brightness":
            img = ImageEnhance.Brightness(img).enhance(float(params.get("factor", 1.2)))

        elif name == "contrast":
            img = ImageEnhance.Contrast(img).enhance(float(params.get("factor", 1.2)))

        elif name == "saturation":
            img = ImageEnhance.Color(img).enhance(float(params.get("factor", 1.2)))

        elif name == "sharpness":
            img = ImageEnhance.Sharpness(img).enhance(float(params.get("factor", 1.5)))

        elif name == "grayscale":
            img = ImageOps.grayscale(img).convert("RGB")

        elif name == "sepia":
            gray = ImageOps.grayscale(img)
            sepia = ImageOps.colorize(gray, black="#3b1f0a", white="#f5e6c8")
            img = sepia.convert("RGB")

        elif name == "invert":
            if img.mode == "RGBA":
                r, g, b, a = img.split()
                rgb = Image.merge("RGB", (r, g, b))
                rgb = ImageOps.invert(rgb)
                img = Image.merge("RGBA", (*rgb.split(), a))
            else:
                img = ImageOps.invert(img)

        elif name == "blur":
            img = img.filter(ImageFilter.GaussianBlur(radius=float(params.get("radius", 4))))

        elif name == "sharpen":
            img = img.filter(ImageFilter.SHARPEN)

        elif name == "edge":
            img = img.filter(ImageFilter.FIND_EDGES)

        elif name == "auto_contrast":
            if img.mode == "RGBA":
                r, g, b, a = img.split()
                rgb = ImageOps.autocontrast(Image.merge("RGB", (r, g, b)))
                img = Image.merge("RGBA", (*rgb.split(), a))
            else:
                img = ImageOps.autocontrast(img)

        elif name == "watermark":
            from PIL import ImageDraw, ImageFont
            text = str(params.get("text", "LumaEdit"))
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", size=max(20, img.width // 30))
            except Exception:
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = img.width - tw - 20
            y = img.height - th - 20
            draw.text((x + 2, y + 2), text, fill=(0, 0, 0, 160), font=font)
            draw.text((x, y), text, fill=(255, 255, 255, 220), font=font)

        else:
            logger.warning("Unknown image op: %s", name)

    out_ext = os.path.splitext(output_path.lower())[1]
    save_kwargs: dict[str, Any] = {}
    if out_ext in (".jpg", ".jpeg"):
        if img.mode == "RGBA":
            img = img.convert("RGB")
        save_kwargs["quality"] = 92
    img.save(output_path, **save_kwargs)


def _run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
    logger.info("ffmpeg: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise MediaEditorError(f"ffmpeg failed: {proc.stderr.strip()[:500]}")


def apply_video_ops(input_path: str, output_path: str, ops: list[dict[str, Any]]) -> None:
    if not shutil.which("ffmpeg"):
        raise MediaEditorError(
            "ffmpeg is not installed on the server. On Railway add nixpacks.toml with `nixPkgs = ['ffmpeg']`."
        )

    current = input_path
    tmpdir = tempfile.mkdtemp(prefix="mediaedit_")
    try:
        for i, op in enumerate(ops):
            name = op.get("op")
            params = op.get("params", {})
            next_path = os.path.join(tmpdir, f"step_{i}.mp4")

            if name == "trim":
                start = float(params.get("start", 0))
                end = params.get("end")
                dur_args = ["-t", str(float(end) - start)] if end is not None else []
                _run_ffmpeg(["-ss", str(start), "-i", current, *dur_args, "-c", "copy", next_path])

            elif name == "rotate":
                deg = int(params.get("degrees", 90)) % 360
                transpose = {90: "transpose=1", 180: "transpose=2,transpose=2", 270: "transpose=2"}.get(deg)
                if not transpose:
                    shutil.copy(current, next_path)
                else:
                    _run_ffmpeg(["-i", current, "-vf", transpose, "-c:a", "copy", next_path])

            elif name == "flip":
                direction = params.get("direction", "horizontal")
                vf = "hflip" if direction == "horizontal" else "vflip"
                _run_ffmpeg(["-i", current, "-vf", vf, "-c:a", "copy", next_path])

            elif name == "speed":
                factor = float(params.get("factor", 2.0))
                if factor <= 0:
                    raise MediaEditorError("speed factor must be > 0")
                _run_ffmpeg([
                    "-i", current,
                    "-filter_complex", f"[0:v]setpts={1/factor}*PTS[v];[0:a]atempo={min(max(factor,0.5),2.0)}[a]",
                    "-map", "[v]", "-map", "[a]", next_path,
                ])

            elif name == "mute":
                _run_ffmpeg(["-i", current, "-an", "-c:v", "copy", next_path])

            elif name == "resize":
                w = int(params.get("width", -2))
                h = int(params.get("height", -2))
                _run_ffmpeg(["-i", current, "-vf", f"scale={w}:{h}", "-c:a", "copy", next_path])

            elif name == "compress":
                crf = int(params.get("crf", 28))
                _run_ffmpeg(["-i", current, "-vcodec", "libx264", "-crf", str(crf), "-preset", "fast", next_path])

            elif name == "extract_audio":
                _run_ffmpeg(["-i", current, "-vn", "-acodec", "copy", os.path.splitext(next_path)[0] + ".m4a"])
                next_path = os.path.splitext(next_path)[0] + ".m4a"

            else:
                logger.warning("Unknown video op: %s", name)
                shutil.copy(current, next_path)

            current = next_path

        shutil.copy(current, output_path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _apply_ai_image_op(current_path: str, op_name: str, params: dict[str, Any], out_dir: str) -> str:
    """Run a Replicate-backed image op and return the new file path."""
    fn = AI_IMAGE_OPS[op_name]
    try:
        return fn(current_path, params, out_dir)
    except ReplicateUnavailable as e:
        raise MediaEditorError(str(e))
    except Exception as e:
        raise MediaEditorError(f"AI op {op_name} failed: {e}")


def edit_media(input_path: str, ops: list[dict[str, Any]], out_dir: str) -> str:
    kind = kind_from_path(input_path)
    size_mb = os.path.getsize(input_path) / (1024 * 1024)
    if kind == "image" and size_mb > MAX_IMAGE_MB:
        raise MediaEditorError(f"Image too large ({size_mb:.1f} MB, max {MAX_IMAGE_MB} MB)")
    if kind == "video" and size_mb > MAX_VIDEO_MB:
        raise MediaEditorError(f"Video too large ({size_mb:.1f} MB, max {MAX_VIDEO_MB} MB)")

    os.makedirs(out_dir, exist_ok=True)
    ext = ".png" if kind == "image" else ".mp4"
    output_path = os.path.join(out_dir, f"edited_{uuid.uuid4().hex}{ext}")

    if kind == "image":
        # Split into AI ops (run one-at-a-time via Replicate) and local PIL ops
        # (accumulated). AI ops produce a new file each time, so we chain them.
        current = input_path
        local_ops: list[dict[str, Any]] = []
        for op in ops:
            name = op.get("op", "")
            if name in AI_IMAGE_OPS:
                # Flush any queued local ops first
                if local_ops:
                    tmp = os.path.join(out_dir, f"step_{uuid.uuid4().hex}.png")
                    apply_image_ops(current, tmp, local_ops)
                    if current != input_path:
                        try: os.remove(current)
                        except OSError: pass
                    current = tmp
                    local_ops = []
                new_path = _apply_ai_image_op(current, name, op.get("params", {}), out_dir)
                if current != input_path:
                    try: os.remove(current)
                    except OSError: pass
                current = new_path
            else:
                local_ops.append(op)

        if local_ops or current == input_path:
            apply_image_ops(current, output_path, local_ops)
            if current != input_path:
                try: os.remove(current)
                except OSError: pass
        else:
            os.rename(current, output_path)
    else:
        apply_video_ops(input_path, output_path, ops)
    return output_path


SUPPORTED_OPS_HINT = json.dumps({
    "image": {
        "rotate": {"degrees": "int (e.g. 90, 180, 270)"},
        "flip": {"direction": "'horizontal' | 'vertical'"},
        "resize": {"width": "int", "height": "int"},
        "crop": {"left": "int", "top": "int", "right": "int", "bottom": "int"},
        "brightness": {"factor": "float, 1.0=no change, 1.5=brighter, 0.5=darker"},
        "contrast": {"factor": "float"},
        "saturation": {"factor": "float"},
        "sharpness": {"factor": "float"},
        "grayscale": {},
        "sepia": {},
        "invert": {},
        "blur": {"radius": "float, default 4"},
        "sharpen": {},
        "edge": {},
        "auto_contrast": {},
        "watermark": {"text": "str"},
    },
    "video": {
        "trim": {"start": "float seconds", "end": "float seconds (optional)"},
        "rotate": {"degrees": "90, 180, or 270"},
        "flip": {"direction": "'horizontal' | 'vertical'"},
        "speed": {"factor": "float, e.g. 2.0=faster, 0.5=slow-mo"},
        "mute": {},
        "resize": {"width": "int (or -2 to auto)", "height": "int"},
        "compress": {"crf": "int 18-32, higher=smaller file"},
        "extract_audio": {},
    },
    "ai_image": {
        "ai_bg_remove": {"description": "Auto-remove background from a photo"},
        "ai_upscale": {"scale": "int 2 or 4"},
        "ai_beautify": {"codeformer_fidelity": "float 0-1 (default 0.6)"},
        "ai_object_remove": {"mask": "b64 image mask (optional, else auto)"},
        "ai_cartoonify": {"description": "Convert photo to anime/cartoon"},
    },
}, indent=2)
