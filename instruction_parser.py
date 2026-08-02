"""Parse natural-language edit instructions (Hindi/English) into structured ops.

Uses Claude to translate free-form user text like "background thoda bright karo,
end me thoda slow motion" into the JSON op list media_editor.py expects.
"""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the instruction parser for a photo/video editor.
Convert the user's natural-language instruction (English, Hindi, Hinglish, or Urdu)
into a JSON list of ops the editor can execute.

You will be told whether the input is an "image" or "video".

Return ONLY a JSON object, no prose, no markdown fences. Shape:
{
  "ops": [ {"op": "<name>", "params": { ... }}, ... ],
  "summary": "<one short line, in the user's language, describing what will happen>"
}

Supported ops per kind:
IMAGE (local, instant, free):
  rotate, flip, resize, crop, brightness, contrast, saturation, sharpness,
  grayscale, sepia, invert, blur, sharpen, edge, auto_contrast, watermark
IMAGE (AI, ~5-30 seconds, uses Replicate):
  ai_bg_remove         → auto-remove background
  ai_upscale           → 2x/4x sharpen and enlarge (params: {"scale": 2 or 4})
  ai_beautify          → face smooth / skin enhance / face restore
  ai_object_remove     → remove an object or person from the photo
  ai_cartoonify        → convert photo to anime/cartoon style
VIDEO: trim, rotate, flip, speed, mute, resize, compress, extract_audio

Guidelines:
- Prefer local ops (grayscale, brightness, etc.) — they are free and instant.
- Use AI ops ONLY when the user asks for something local ops cannot do:
  * "background hatao" / "background remove" / "no background"  → ai_bg_remove
  * "HD karo" / "clear karo" / "upscale" / "enhance quality"    → ai_upscale (scale 2)
  * "face smooth" / "beautify" / "skin fair" / "face clear"     → ai_beautify
  * "ye person hata do" / "object remove"                       → ai_object_remove
  * "cartoon" / "anime" / "ghibli"                              → ai_cartoonify
- Multiple ops can be combined ("BG hatao aur upscale karo" → both, in order).
- "Brighter"/"chamak badhao" → brightness with factor 1.3-1.6.
- "Darker"/"halka karo" → brightness with factor 0.6-0.8.
- "Slow motion"/"slow-mo" → speed with factor 0.5.
- "Fast"/"tez" → speed with factor 2.0.
- "Grayscale"/"black and white"/"kala safed" → grayscale.
- "Rotate 90"/"90 degree ghumao" → rotate degrees:90.
- "Trim first 5 seconds off"/"pehle 5 sec kaat do" → trim start:5.
- "Cut to 30 seconds"/"30 sec ka bana do" → trim start:0, end:30.
- If unclear, pick your best guess and mention it in summary.
- Never invent ops that aren't in the supported list.
- If the request is genuinely unsupported (e.g. text-to-video, 3D generation),
  return an empty ops list with summary explaining it's not available yet.
"""


def parse_instruction(client: Any, kind: str, instruction: str) -> dict[str, Any]:
    """Call Claude and return {'ops': [...], 'summary': '...'}. Never raises — returns
    a fallback dict on any error."""
    if not client:
        return {"ops": [], "summary": "Server error: AI client not configured"}

    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Kind: {kind}\nInstruction: {instruction}",
            }],
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        ).strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        parsed = json.loads(text)
    except Exception as e:
        logger.exception("instruction parse failed: %s", e)
        return {"ops": [], "summary": f"Could not understand instruction: {e}"}

    ops = parsed.get("ops")
    if not isinstance(ops, list):
        return {"ops": [], "summary": parsed.get("summary", "No ops returned")}

    cleaned: list[dict[str, Any]] = []
    for entry in ops:
        if isinstance(entry, dict) and "op" in entry:
            cleaned.append({"op": entry["op"], "params": entry.get("params", {})})

    return {"ops": cleaned, "summary": parsed.get("summary", "")}
