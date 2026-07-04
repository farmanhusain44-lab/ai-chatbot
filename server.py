from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic
import os
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

# Warn at startup if the API key is missing so it shows up in deploy logs
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    logger.warning("ANTHROPIC_API_KEY environment variable is not set — API calls will fail")

client = anthropic.Anthropic(api_key=api_key)

@app.route("/")
def home():
    return app.send_static_file("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    # --- Request validation ---
    data = request.get_json(silent=True)
    if not data:
        logger.warning("Received request with no JSON body")
        return jsonify({"error": "Request body must be JSON"}), 400

    user_message = data.get("message", "").strip()
    if not user_message:
        logger.warning("Received request with missing or empty 'message' field")
        return jsonify({"error": "Field 'message' is required and cannot be empty"}), 400

    logger.info("Sending message to Claude (length=%d chars)", len(user_message))

    # --- API call with full error handling ---
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": user_message}]
        )
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

    # --- Response validation ---
    try:
        reply = response.content[0].text
    except (IndexError, AttributeError) as e:
        logger.error(
            "Unexpected response structure from Anthropic API. "
            "stop_reason=%s content=%r error=%s",
            getattr(response, "stop_reason", "unknown"),
            getattr(response, "content", None),
            e,
        )
        return jsonify({"error": "Received an unexpected response format from the AI model."}), 500

    logger.info("Successfully received reply (length=%d chars)", len(reply))
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=True)

