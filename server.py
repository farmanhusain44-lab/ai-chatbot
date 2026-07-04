from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import anthropic
import os
from datetime import datetime

app = Flask(__name__, static_folder='static')
CORS(app)
port = int(os.environ.get("PORT", 8080))

@app.route('/public/<path:filename>')
def serve_public(filename):
    return send_from_directory('public', filename)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def get_system_prompt():
    now = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    return (
        "You are a highly intelligent, helpful, and knowledgeable AI assistant. "
        "You can answer questions on a wide range of topics including science, technology, history, "
        "general knowledge, coding, math, business, health, lifestyle, and more. "
        "Provide accurate, detailed, and useful answers. If you are unsure about something, say so honestly. "
        f"The current date and time is: {now}. When asked about today's date, current time, or anything time-related, "
        "you must answer using this exact date and time. Do not say you lack real-time information."
    )

@app.route("/")
def home():
    return app.send_static_file("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data["message"]
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        system=get_system_prompt(),
        messages=[{"role": "user", "content": user_message}]
    )
    reply = response.content[0].text
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=True)  # 8080 ki jagah port

