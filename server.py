from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import anthropic
import os

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

@app.route("/")
def home():
    with open('static/index.html', 'r') as f:
        return f.read()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data["message"]
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": user_message}]
    )
    reply = response.content[0].text
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

