from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import anthropic
import os
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

def get_ai_reply(message, language, timezone=None):
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        system=get_system_prompt(language, timezone),
        messages=[{"role": "user", "content": message}]
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
    lang_name = LANGUAGE_NAMES.get(language, "English")
    return (
        "You are a highly intelligent, helpful, and knowledgeable AI assistant. "
        "You can answer questions on a wide range of topics including science, technology, history, "
        "general knowledge, coding, math, business, health, lifestyle, and more. "
        "Provide accurate, detailed, and useful answers. If you are unsure about something, say so honestly. "
        f"The current date and time is: {now_str}. When asked about today's date, current time, or anything time-related, "
        "you must answer using this exact date and time. Do not say you lack real-time information. "
        f"You must respond in {lang_name} language."
    )

@app.route("/")
def home():
    return app.send_static_file("index.html")

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
    logger.info("Sending message to Claude (length=%d chars, lang=%s, tz=%s)", len(user_message), language, timezone)

    try:
        reply = get_ai_reply(user_message, language, timezone)
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
            return jsonify({"error": "ElevenLabs error"}), 500
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

