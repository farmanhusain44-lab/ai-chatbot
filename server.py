from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import anthropic
import os
import re
from datetime import datetime
from twilio.rest import Client as TwilioClient
from twilio.twiml.messaging_response import MessagingResponse
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import requests
import logging

# Configure logging so errors appear in Gunicorn/Railway logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app)
port = int(os.environ.get("PORT", 8080))

@app.route('/public/<path:filename>')
def serve_public(filename):
    return send_from_directory('public', filename)

# Warn at startup if the API key is missing so it shows up in deploy logs
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    logger.warning("ANTHROPIC_API_KEY environment variable is not set — API calls will fail")

client = anthropic.Anthropic(api_key=api_key)

twilio_sid = os.environ.get("TWILIO_ACCOUNT_SID")
twilio_token = os.environ.get("TWILIO_AUTH_TOKEN")
twilio_from = os.environ.get("TWILIO_WHATSAPP_NUMBER")
twilio = None
if twilio_sid and twilio_token:
    twilio = TwilioClient(twilio_sid, twilio_token)

elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY")

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "ur": "Urdu",
    "ar": "Arabic",
    "bn": "Bangla",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
    "mr": "Marathi",
    "gu": "Gujarati",
    "or": "Odia",
    "as": "Assamese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
    "tr": "Turkish"
}

# Male voice IDs for ElevenLabs (Adam is a clear male voice that works across languages)
# All languages use the same multilingual voice; ElevenLabs model detects language automatically.
ELEVENLABS_VOICES = {k: "pNInz6obpgDQGcFmaJgB" for k in LANGUAGE_NAMES}

def detect_language(text):
    # Script-based detection is more reliable for non-Latin scripts
    def has_chars(start, end):
        return any(start <= ord(c) <= end for c in text)

    # Urdu-specific characters (Arabic script with Persian/Urdu letters)
    if has_chars(0x067E, 0x067E) or has_chars(0x0686, 0x0686) or has_chars(0x0688, 0x0688) \
            or has_chars(0x0691, 0x0691) or has_chars(0x06A9, 0x06A9) or has_chars(0x06AF, 0x06AF) \
            or has_chars(0x06BA, 0x06BA) or has_chars(0x06CC, 0x06CC) or has_chars(0x06D2, 0x06D2) \
            or has_chars(0x06C1, 0x06C1) or has_chars(0x06BE, 0x06BE):
        return "ur"
    # Arabic/Persian script (without Urdu-specific characters)
    if has_chars(0x0600, 0x06FF) or has_chars(0x0750, 0x077F) or has_chars(0x08A0, 0x08FF):
        return "ar"
    if has_chars(0x0900, 0x097F):
        return "hi"  # Devanagari (Hindi, Marathi, Nepali)
    if has_chars(0x0980, 0x09FF):
        return "bn"  # Bengali
    if has_chars(0x0A00, 0x0A7F):
        return "pa"  # Gurmukhi (Punjabi)
    if has_chars(0x0A80, 0x0AFF):
        return "gu"  # Gujarati
    if has_chars(0x0B00, 0x0B7F):
        return "or"  # Oriya/Odia
    if has_chars(0x0B80, 0x0BFF):
        return "ta"  # Tamil
    if has_chars(0x0C00, 0x0C7F):
        return "te"  # Telugu
    if has_chars(0x0C80, 0x0CFF):
        return "kn"  # Kannada
    if has_chars(0x0D00, 0x0D7F):
        return "ml"  # Malayalam
    if has_chars(0x4E00, 0x9FFF):
        return "zh"  # Chinese
    if has_chars(0x3040, 0x309F) or has_chars(0x30A0, 0x30FF):
        return "ja"  # Japanese
    if has_chars(0xAC00, 0xD7AF) or has_chars(0x1100, 0x11FF):
        return "ko"  # Korean
    if has_chars(0x0400, 0x04FF):
        return "ru"  # Cyrillic

    # Roman Hindi detection (speech transcribed in English letters)
    roman_hindi_words = {
        "kya", "kaise", "kaun", "kahan", "kab", "kyun", "kitna", "kaunsa",
        "main", "tum", "aap", "woh", "yeh", "hum", "sab", "log",
        "hoon", "ho", "hai", "hain", "tha", "thi", "the", "raha", "rahi", "kar", "kiya", "gaya", "diya",
        "accha", "theek", "nahi", "bilkul", "bahut", "thoda", "zyada", "kam", "achha",
        "bhi", "lekin", "kyunki", "agar", "toh", "ya", "aur", "par", "se", "ko", "mein", "pe", "tak",
        "shukriya", "dhanyawad", "namaste", "bhai", "yaar", "chal", "karo", "dekho"
    }
    words = set(re.findall(r"\b[a-z]+\b", text.lower()))
    if len(words.intersection(roman_hindi_words)) >= 2:
        return "hi"

    try:
        lang = detect(text)
        if lang in LANGUAGE_NAMES:
            return lang
        # Fallback mappings for close languages
        if lang == "fa":
            return "ur"
        return "en"
    except LangDetectException:
        return "en"

def is_group_message(sender):
    # Twilio individual: whatsapp:+1234567890
    # Twilio group: whatsapp:1203630... (no +)
    if not sender:
        return True
    number = sender.replace("whatsapp:", "")
    return not number.startswith("+")

def send_whatsapp_reply(to, reply):
    if not twilio or not twilio_from:
        return False
    try:
        twilio.messages.create(
            from_=f"whatsapp:{twilio_from}",
            body=reply,
            to=to
        )
        return True
    except Exception as e:
        print(f"WhatsApp send error: {e}")
        return False

def get_ai_reply(message, language, timezone=None, history=None):
    # If history is provided, it already contains the current user message as the last item.
    messages = history if history else [{"role": "user", "content": message}]
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        system=get_system_prompt(language, timezone),
        messages=messages
    )
    return response.content[0].text

def get_system_prompt(language="en", timezone=None):
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(timezone) if timezone else None
    except Exception:
        tz = None
    now = datetime.now(tz) if tz else datetime.now()
    now_str = now.strftime("%A, %B %d, %Y at %I:%M %p %Z")
    return (
        "You are a smart, warm, and professional multilingual AI assistant. Talk like an intelligent, well-mannered human friend. "
        "You can speak many languages fluently. Whatever language the user writes or speaks in, reply directly in that same language. "
        "Switch languages instantly and naturally. Never say you can only speak one language. "
        "Never add English translations, never explain phrases, never explain emojis, and never quote the user's words back with definitions. "
        "Use common sense: a greeting like 'kya haal hai' means 'how are you' — simply reply naturally. "
        "When the user uses Hindi (or Roman Hindi), write your reply in Devanagari script (हिंदी). "
        "When the user uses Urdu, write your reply in Arabic/Persian script (اردو). "
        "When the user uses Arabic, write your reply in pure Arabic script (العربية الفصحى) only. "
        "When the user uses Bengali, write your reply in Bengali script. "
        "Keep answers short and friendly, 1-3 sentences when possible. "
        "If you are unsure about something, say so honestly. "
        f"The current date and time is: {now_str}. When asked about today's date, current time, or anything time-related, "
        "you must answer using this exact date and time. Do not say you lack real-time information. "
    )

@app.route("/")
def home():
    return app.send_static_file("index.html")

@app.route("/widget.js")
def widget_js():
    return app.send_static_file("widget.js")

@app.route("/widget.html")
def widget_html():
    return app.send_static_file("widget.html")

@app.route("/demo.html")
def demo_page():
    return app.send_static_file("demo.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data:
        logger.warning("Received request with no JSON body")
        return jsonify({"error": "Request body must be JSON"}), 400

    user_message = data.get("message", "").strip()
    if not user_message:
        logger.warning("Received request with missing or empty 'message' field")
        return jsonify({"error": "Field 'message' is required and cannot be empty"}), 400

    language = data.get("language") or detect_language(user_message)
    timezone = data.get("timezone")
    history = data.get("history", [])
    logger.info("Sending message to Claude (length=%d chars, lang=%s, tz=%s, history=%d)", len(user_message), language, timezone, len(history))

    try:
        reply = get_ai_reply(user_message, language, timezone, history)
    except anthropic.AuthenticationError as e:
        logger.error("Anthropic authentication failed — check ANTHROPIC_API_KEY: %s", e)
        return jsonify({"error": "API authentication failed. The server API key may be invalid or missing."}), 500
    except anthropic.RateLimitError as e:
        logger.error("Anthropic rate limit exceeded: %s", e)
        return jsonify({"error": "Rate limit exceeded. Please wait a moment and try again."}), 429
    except anthropic.APIStatusError as e:
        logger.error("Anthropic API returned status %s: %s", e.status_code, e.message)
        return jsonify({"error": f"Anthropic API error (status {e.status_code}): {e.message}"}), 502
    except anthropic.APIConnectionError as e:
        logger.error("Could not connect to Anthropic API: %s", e)
        return jsonify({"error": "Could not reach the Anthropic API. Check network connectivity."}), 502
    except Exception as e:
        logger.exception("Unexpected error while calling Anthropic API: %s", e)
        return jsonify({"error": f"Unexpected server error: {str(e)}"}), 500

    logger.info("Successfully received reply (length=%d chars)", len(reply))
    return jsonify({"reply": reply, "language": language})

@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json(silent=True)
    text = data.get("text", "") if data else ""
    language = data.get("language", "en") if data else "en"
    if not text or not elevenlabs_key:
        return jsonify({"error": "Missing text or API key"}), 400

    voice_id = ELEVENLABS_VOICES.get(language, ELEVENLABS_VOICES["en"])
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": elevenlabs_key
    }
    payload = {
        "text": text[:4000],
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code != 200:
            logger.error("ElevenLabs error: status=%s, body=%s", resp.status_code, resp.text[:500])
            return jsonify({"error": "ElevenLabs error", "details": resp.text[:200]}), 500
        return Response(resp.content, mimetype="audio/mpeg")
    except Exception as e:
        logger.error("ElevenLabs error: %s", e)
        return jsonify({"error": "Failed to generate audio"}), 500

@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    sender = request.form.get("From", "")
    message = request.form.get("Body", "").strip()

    if not message or is_group_message(sender):
        return str(MessagingResponse())

    language = detect_language(message)
    reply = get_ai_reply(message, language)

    if len(reply) > 1500:
        reply = reply[:1497] + "..."

    send_whatsapp_reply(sender, reply)
    return str(MessagingResponse())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=True)

