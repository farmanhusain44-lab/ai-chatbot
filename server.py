from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import anthropic
import os
from datetime import datetime
from twilio.rest import Client as TwilioClient
from twilio.twiml.messaging_response import MessagingResponse
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

app = Flask(__name__, static_folder='static')
CORS(app)
port = int(os.environ.get("PORT", 8080))

@app.route('/public/<path:filename>')
def serve_public(filename):
    return send_from_directory('public', filename)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

twilio_sid = os.environ.get("TWILIO_ACCOUNT_SID")
twilio_token = os.environ.get("TWILIO_AUTH_TOKEN")
twilio_from = os.environ.get("TWILIO_WHATSAPP_NUMBER")
twilio = None
if twilio_sid and twilio_token:
    twilio = TwilioClient(twilio_sid, twilio_token)

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "ur": "Urdu",
    "ar": "Arabic"
}

def detect_language(text):
    try:
        lang = detect(text)
        if lang in LANGUAGE_NAMES:
            return lang
        if lang == "pa" or lang == "mr":
            return "hi"
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

def get_ai_reply(message, language):
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        system=get_system_prompt(language),
        messages=[{"role": "user", "content": message}]
    )
    return response.content[0].text

def get_system_prompt(language="en"):
    now = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    lang_name = LANGUAGE_NAMES.get(language, "English")
    return (
        "You are a highly intelligent, helpful, and knowledgeable AI assistant. "
        "You can answer questions on a wide range of topics including science, technology, history, "
        "general knowledge, coding, math, business, health, lifestyle, and more. "
        "Provide accurate, detailed, and useful answers. If you are unsure about something, say so honestly. "
        f"The current date and time is: {now}. When asked about today's date, current time, or anything time-related, "
        "you must answer using this exact date and time. Do not say you lack real-time information. "
        f"You must respond in {lang_name} language."
    )

@app.route("/")
def home():
    return app.send_static_file("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data["message"]
    language = data.get("language", "en")
    reply = get_ai_reply(user_message, language)
    return jsonify({"reply": reply})

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
    app.run(host="0.0.0.0", port=port, debug=True)  # 8080 ki jagah port

