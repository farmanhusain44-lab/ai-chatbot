from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import anthropic
import os
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from twilio.rest import Client as TwilioClient
from twilio.twiml.messaging_response import MessagingResponse
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import requests
import logging
from io import BytesIO
import numpy as np

try:
    import openai
except ImportError:
    openai = None

# Document extraction libraries
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
try:
    from docx import Document
except ImportError:
    Document = None

# Configure logging so errors appear in Gunicorn/Railway logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app)
port = int(os.environ.get("PORT", 8080))

# In-memory document knowledge base
DOCUMENT_CHUNKS = []
MAX_DOCUMENT_CHUNKS = 2000
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# In-memory lead storage (in production, use database)
LEADS = []

# In-memory clients DB (payment records)
clients_db = []


def normalize_arabic(text):
    """Normalize Arabic text for better matching across dialects and diacritics."""
    if not text:
        return text
    # Remove tashkeel (diacritics)
    text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)
    # Normalize alef variants
    text = re.sub(r'[\u0622\u0623\u0625]', '\u0627', text)
    # Normalize yaa and alif maqsura
    text = text.replace('\u0649', '\u064A')
    return text


def extract_text_from_file(file_obj, filename):
    """Extract text from PDF, DOCX, or TXT files."""
    ext = os.path.splitext(filename.lower())[1]
    try:
        if ext == '.pdf' and PyPDF2:
            reader = PyPDF2.PdfReader(BytesIO(file_obj.read()))
            parts = []
            for page in reader.pages:
                try:
                    parts.append(page.extract_text() or '')
                except Exception:
                    pass
            return '\n'.join(parts)
        elif ext in ('.docx', '.doc') and Document:
            doc = Document(BytesIO(file_obj.read()))
            return '\n'.join(p.text for p in doc.paragraphs if p.text)
        elif ext in ('.txt', '.md', '.csv', '.json'):
            return file_obj.read().decode('utf-8', errors='ignore')
        else:
            return file_obj.read().decode('utf-8', errors='ignore')
    except Exception as e:
        logger.error("Document extraction failed for %s: %s", filename, e)
        return ""


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break at a sentence or space
        if end < len(text):
            for sep in ['. ', '? ', '! ', '\n', ' ']:
                pos = text.rfind(sep, start, end)
                if pos > start:
                    end = pos + len(sep)
                    break
        chunks.append(text[start:end].strip())
        start = end - overlap if end < len(text) else end
    return chunks


def embed_texts(texts):
    """Get OpenAI embeddings for a list of texts. Falls back to None if unavailable."""
    if not openai_client or not texts:
        return None
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logger.error("OpenAI embedding failed: %s", e)
        return None


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def add_document_to_knowledge_base(filename, text):
    """Chunk, embed, and store document text."""
    global DOCUMENT_CHUNKS
    chunks = chunk_text(text)
    valid_chunks = [chunk for chunk in chunks if chunk]
    if not valid_chunks:
        return 0

    embeddings = embed_texts(valid_chunks)
    for i, chunk in enumerate(valid_chunks):
        item = {"source": filename, "text": chunk, "norm": normalize_arabic(chunk.lower())}
        if embeddings:
            item["embedding"] = embeddings[i]
        DOCUMENT_CHUNKS.append(item)

    # Keep within memory limit
    if len(DOCUMENT_CHUNKS) > MAX_DOCUMENT_CHUNKS:
        DOCUMENT_CHUNKS = DOCUMENT_CHUNKS[-MAX_DOCUMENT_CHUNKS:]
    logger.info("Added document %s: %d chunks, total %d chunks, embeddings=%s", filename, len(valid_chunks), len(DOCUMENT_CHUNKS), bool(embeddings))
    return len(valid_chunks)


def keyword_search_context(query, top_k=3):
    """Keyword search as fallback."""
    if not DOCUMENT_CHUNKS:
        return ""
    query_norm = normalize_arabic(query.lower())
    query_words = set(re.findall(r"[\w'\u0600-\u06FF]+", query_norm))
    # Stop words for English and Arabic
    stop_words = {
        'the', 'is', 'a', 'an', 'and', 'or', 'to', 'of', 'in', 'for', 'on', 'with', 'what', 'how', 'who', 'where', 'when', 'why', 'me', 'my', 'your', 'this', 'that', 'are', 'do', 'does', 'can', 'you', 'was', 'were', 'did', 'will',
        'هذا', 'هذه', 'التي', 'الذي', 'من', 'في', 'على', 'إلى', 'عن', 'مع', 'كان', 'أن', 'أو', 'لم', 'قد', 'ما', 'كل', 'بعد', 'قبل', 'أي', 'هل', 'كيف', 'أين', 'متى', 'لماذا', 'لمن', 'هو', 'هي', 'هم'
    }
    query_words = query_words - stop_words
    if not query_words:
        return ""
    scored = []
    for chunk in DOCUMENT_CHUNKS:
        chunk_words = set(re.findall(r"[\w'\u0600-\u06FF]+", chunk.get("norm", chunk["text"].lower())))
        score = len(query_words & chunk_words)
        if score > 0:
            scored.append((score, chunk["text"]))
    scored.sort(reverse=True, key=lambda x: x[0])
    selected = []
    total_len = 0
    for score, text in scored[:top_k]:
        if total_len + len(text) > 2500:
            break
        selected.append(text)
        total_len += len(text)
    return "\n\n".join(selected)


def get_relevant_context(query, top_k=3):
    """Find relevant document chunks using embeddings if available, else keyword search."""
    if not DOCUMENT_CHUNKS:
        return ""

    # If we have embeddings, do vector search
    if openai_client and any("embedding" in c for c in DOCUMENT_CHUNKS):
        query_embedding = embed_texts([query])
        if query_embedding and query_embedding[0]:
            q_vec = np.array(query_embedding[0])
            scored = []
            for chunk in DOCUMENT_CHUNKS:
                if "embedding" not in chunk:
                    continue
                c_vec = np.array(chunk["embedding"])
                score = cosine_similarity(q_vec, c_vec)
                scored.append((score, chunk["text"]))
            scored.sort(reverse=True, key=lambda x: x[0])
            selected = []
            total_len = 0
            for score, text in scored[:top_k]:
                if total_len + len(text) > 2500:
                    break
                selected.append(text)
                total_len += len(text)
            if selected:
                return "\n\n".join(selected)

    # Fallback to keyword search
    return keyword_search_context(query, top_k)


@app.route('/public/<path:filename>')
def serve_public(filename):
    return send_from_directory('public', filename)

# Warn at startup if the API key is missing so it shows up in deploy logs
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    logger.warning("ANTHROPIC_API_KEY environment variable is not set — API calls will fail")

client = anthropic.Anthropic(api_key=api_key)

# Optional OpenAI client for document embeddings (vector search)
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_client = None
if openai and openai_api_key:
    try:
        openai_client = openai.OpenAI(api_key=openai_api_key)
    except Exception as e:
        logger.warning("OpenAI client failed to initialize: %s", e)

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

def get_ai_reply(message, language, timezone=None, history=None, context=None):
    # If history is provided, it already contains the current user message as the last item.
    messages = history if history else [{"role": "user", "content": message}]
    if context and messages and messages[-1]["role"] == "user":
        lang_name = LANGUAGE_NAMES.get(language, language)
        context_prompt = (
            f"You must reply in {lang_name}. "
            "Use the following document context to answer the question. "
            "If the answer is not in the context, say you don't know. "
            "Do not make up information.\n\n"
            f"Context:\n{context}\n\n"
            "Question:\n"
        )
        messages[-1]["content"] = context_prompt + messages[-1]["content"]
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

@app.route("/india")
def india_landing():
    return app.send_static_file("india-landing.html")

@app.route("/boss-demo")
def boss_demo():
    return app.send_static_file("boss-demo.html")


@app.route("/free-chatbot")
def free_chatbot():
    return app.send_static_file("free-chatbot.html")

@app.route("/uae")
def uae_landing():
    return app.send_static_file("uae-landing.html")

@app.route("/payment")
def payment():
    return app.send_static_file("payment.html")

@app.route("/dashboard")
def dashboard():
    return app.send_static_file("dashboard.html")

@app.route("/terms")
def terms():
    return app.send_static_file("terms.html")

@app.route("/privacy")
def privacy():
    return app.send_static_file("privacy.html")

@app.route("/refund")
def refund():
    return app.send_static_file("refund.html")

@app.route("/admin")
def admin():
    return app.send_static_file("admin.html")

@app.route("/deploy-guide")
def deploy_guide():
    return app.send_static_file("deploy-guide.html")

@app.route("/my-business")
def my_business():
    return app.send_static_file("my-business.html")



@app.route("/admin/clients", methods=["GET"])
def admin_clients():
    password = request.args.get("pw", "")
    if password != os.environ.get("ADMIN_PASSWORD", "farman2024"):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"clients": clients_db})

@app.route("/admin/deploy", methods=["POST"])
def admin_deploy():
    data = request.get_json()
    password = data.get("pw", "")
    if password != os.environ.get("ADMIN_PASSWORD", "farman2024"):
        return jsonify({"error": "Unauthorized"}), 401
    access_code = data.get("access_code", "")
    website_url = data.get("website_url", "")
    notes = data.get("notes", "")
    for client in clients_db:
        if client.get("access_code") == access_code:
            client["deployed"] = True
            client["website_url"] = website_url
            client["deploy_notes"] = notes
            client["deployed_at"] = datetime.now().isoformat()
            break
    return jsonify({"success": True})

@app.route("/save-lead", methods=["POST"])
def save_lead():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'email', 'phone']
        for field in required_fields:
            if not data.get(field, '').strip():
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Create lead record
        lead = {
            "id": len(LEADS) + 1,
            "name": data['name'].strip(),
            "email": data['email'].strip().lower(),
            "phone": data['phone'].strip(),
            "company": data.get('company', '').strip(),
            "timestamp": data.get('timestamp', datetime.now().isoformat()),
            "source": data.get('source', 'unknown'),
            "status": "new"
        }
        
        # Check for duplicate leads
        for existing_lead in LEADS:
            if (existing_lead['email'] == lead['email'] or 
                existing_lead['phone'] == lead['phone']):
                # Update existing lead instead of creating duplicate
                existing_lead.update(lead)
                logger.info(f"Updated existing lead: {lead['email']}")
                return jsonify({"success": True, "lead_id": existing_lead['id'], "updated": True})
        
        # Add new lead
        LEADS.append(lead)
        logger.info(f"New lead captured: {lead['name']} - {lead['email']}")
        
        return jsonify({
            "success": True, 
            "lead_id": lead['id'],
            "message": "Lead saved successfully"
        })
        
    except Exception as e:
        logger.error(f"Error saving lead: {e}")
        return jsonify({"error": "Failed to save lead"}), 500

@app.route("/leads", methods=["GET"])
def get_leads():
    # Simple endpoint to view leads (in production, add authentication)
    return jsonify({
        "leads": LEADS,
        "total": len(LEADS)
    })

@app.route("/create-payment", methods=["POST"])
def create_payment():
    """Create Razorpay payment order for Indian customers"""
    try:
        data = request.get_json()
        plan = data.get('plan', 'starter')
        
        # Plan pricing in INR
        pricing = {
            'starter': {'amount': 300000, 'name': 'Starter Plan'},  # ₹3,000 in paise
            'professional': {'amount': 1000000, 'name': 'Professional Plan'},  # ₹10,000 in paise
            'enterprise': {'amount': 2500000, 'name': 'Enterprise Plan'}  # ₹25,000 in paise
        }
        
        if plan not in pricing:
            return jsonify({"error": "Invalid plan selected"}), 400
        
        plan_details = pricing[plan]
        
        # In production, integrate with actual Razorpay API
        # For now, return mock payment order
        payment_order = {
            "id": f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "amount": plan_details['amount'],
            "currency": "INR",
            "name": "AI Chatbot India",
            "description": plan_details['name'],
            "customer_email": data.get('email', ''),
            "customer_phone": data.get('phone', ''),
            "callback_url": f"{request.url_root}payment-success",
            "notes": {
                "plan": plan,
                "customer_name": data.get('name', '')
            }
        }
        
        logger.info(f"Payment order created: {payment_order['id']} for plan {plan}")
        return jsonify({
            "success": True,
            "order": payment_order,
            "key": "rzp_test_1234567890"  # Test key in production
        })
        
    except Exception as e:
        logger.error(f"Error creating payment: {e}")
        return jsonify({"error": "Failed to create payment"}), 500

@app.route("/payment-success", methods=["POST"])
def payment_success():
    """Handle successful payment callback and grant automatic access"""
    try:
        data = request.get_json()
        payment_id = data.get('payment_id')
        order_id = data.get('order_id')
        plan = data.get('plan', 'starter')
        email = data.get('email', '')
        name = data.get('name', '')
        website = data.get('website', '')
        access_code_in = data.get('access_code', '')

        logger.info(f"Payment successful: {payment_id} for order {order_id} - Plan: {plan}")

        # Grant automatic access
        access_code = access_code_in if access_code_in else generate_access_code(email, plan)

        # Save to clients_db for admin panel
        clients_db.append({
            "name": name,
            "email": email,
            "website": website,
            "plan": plan,
            "access_code": access_code,
            "payment_id": payment_id,
            "order_id": order_id,
            "paid_at": datetime.now().isoformat(),
            "deployed": False,
            "deploy_notes": ""
        })
        
        # Store access record
        access_record = {
            "email": email,
            "name": name,
            "plan": plan,
            "access_code": access_code,
            "payment_id": payment_id,
            "granted_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=30)).isoformat()
        }
        
        # In production, save to database
        logger.info(f"Access granted: {access_code} for {email}")
        
        # Send access details via email (in production)
        send_access_email(email, name, access_code, plan)
        
        return jsonify({
            "success": True,
            "message": "Payment processed successfully",
            "access_code": access_code,
            "redirect_url": f"/chat?access={access_code}",
            "plan": plan,
            "expires_at": access_record["expires_at"]
        })
        
    except Exception as e:
        logger.error(f"Error processing payment success: {e}")
        return jsonify({"error": "Payment processing failed"}), 500

def generate_access_code(email, plan):
    """Generate unique access code for user"""
    import uuid
    import hashlib
    
    # Create unique code based on email and timestamp
    timestamp = str(datetime.now().timestamp())
    raw_string = f"{email}{plan}{timestamp}"
    access_code = hashlib.md5(raw_string.encode()).hexdigest()[:12].upper()
    
    return f"AI-{access_code}"

def send_access_email(email, name, access_code, plan):
    """Send access details via email (in production)"""
    logger.info(f"Email sent to {email}: Access Code: {access_code}, Plan: {plan}")
    send_welcome_email(email, name, access_code, plan)


def send_welcome_email(to_email, name, access_code, plan):
    """Send beautiful welcome email with embed code to new client"""
    gmail_user = os.environ.get('GMAIL_USER', '')
    gmail_pass = os.environ.get('GMAIL_APP_PASSWORD', '')
    if not gmail_user or not gmail_pass:
        logger.warning('GMAIL_USER or GMAIL_APP_PASSWORD not set — skipping welcome email')
        return

    plan_prices = {'starter': 'AED 499', 'professional': 'AED 1,499', 'enterprise': 'AED 3,499'}
    plan_label  = plan.title()
    price       = plan_prices.get(plan, '')
    embed_code  = f'<script src="https://ai-chatbot-production-2f3a.up.railway.app/widget.js" data-agent="AI Assistant" data-welcome="Hello! How can I help you?" data-access="{access_code}"></script>'
    dashboard_url = f'https://ai-chatbot-production-2f3a.up.railway.app/dashboard?access={access_code}&plan={plan}'
    guide_url = f'https://ai-chatbot-production-2f3a.up.railway.app/deploy-guide?access={access_code}'

    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
    <body style="margin:0;padding:0;background:#0a0a1a;font-family:Inter,Arial,sans-serif;">
      <div style="max-width:600px;margin:0 auto;padding:40px 20px;">

        <div style="text-align:center;margin-bottom:32px;">
          <div style="font-size:48px;">🤖</div>
          <h1 style="color:#a78bfa;font-size:1.8rem;margin:12px 0 4px;">Your AI Bot is Ready!</h1>
          <p style="color:#64748b;margin:0;">Payment confirmed — here's everything you need</p>
        </div>

        <div style="background:rgba(255,255,255,.05);border:1px solid rgba(167,139,250,.3);border-radius:16px;padding:24px;margin-bottom:20px;">
          <p style="color:#94a3b8;margin:0 0 4px;font-size:.85rem;text-transform:uppercase;letter-spacing:.5px;">Welcome</p>
          <p style="color:#e2e8f0;font-size:1.1rem;font-weight:600;margin:0;">Hi {name}!</p>
        </div>

        <div style="background:rgba(255,255,255,.05);border:1px solid rgba(167,139,250,.3);border-radius:16px;padding:24px;margin-bottom:20px;">
          <p style="color:#94a3b8;margin:0 0 8px;font-size:.85rem;text-transform:uppercase;letter-spacing:.5px;">Your Access Code</p>
          <div style="background:#0d0d1f;border-radius:8px;padding:14px;font-family:monospace;font-size:1.3rem;color:#a78bfa;letter-spacing:2px;text-align:center;">{access_code}</div>
          <p style="color:#475569;font-size:.8rem;margin:8px 0 0;text-align:center;">Keep this safe — it's your key to the dashboard</p>
        </div>

        <div style="background:rgba(255,255,255,.05);border:1px solid rgba(167,139,250,.3);border-radius:16px;padding:24px;margin-bottom:20px;">
          <p style="color:#94a3b8;margin:0 0 8px;font-size:.85rem;text-transform:uppercase;letter-spacing:.5px;">Plan</p>
          <p style="color:#34d399;font-size:1.1rem;font-weight:700;margin:0;">{plan_label} — {price}/month</p>
        </div>

        <div style="background:rgba(255,255,255,.05);border:1px solid rgba(167,139,250,.3);border-radius:16px;padding:24px;margin-bottom:20px;">
          <p style="color:#94a3b8;margin:0 0 12px;font-size:.85rem;text-transform:uppercase;letter-spacing:.5px;">🚀 Add Bot to Your Website</p>
          <p style="color:#94a3b8;font-size:.9rem;margin:0 0 12px;">Copy this one line and paste it in your website HTML before <code style="color:#a78bfa;">&lt;/body&gt;</code>:</p>
          <div style="background:#0d0d1f;border-radius:8px;padding:14px;font-family:monospace;font-size:.78rem;color:#60a5fa;word-break:break-all;line-height:1.6;">{embed_code}</div>
        </div>

        <div style="background:rgba(255,255,255,.05);border:1px solid rgba(167,139,250,.3);border-radius:16px;padding:24px;margin-bottom:28px;text-align:center;">
          <p style="color:#94a3b8;margin:0 0 6px;font-size:.85rem;">Step-by-step guide for WordPress, Shopify, Wix &amp; more:</p>
          <a href="{guide_url}" style="display:inline-block;background:rgba(167,139,250,.15);border:1px solid rgba(167,139,250,.3);color:#a78bfa;text-decoration:none;padding:12px 28px;border-radius:9px;font-weight:700;font-size:.95rem;">📖 View Full Deploy Guide →</a>
          <p style="color:#475569;font-size:.8rem;margin:12px 0 0;">Can't do it? Reply to this email — we'll add it for you free! ✅</p>
        </div>

        <div style="text-align:center;margin-bottom:32px;display:flex;gap:12px;justify-content:center;flex-wrap:wrap;">
          <a href="{dashboard_url}" style="display:inline-block;background:linear-gradient(135deg,#a78bfa,#60a5fa);color:#fff;text-decoration:none;padding:15px 36px;border-radius:10px;font-weight:700;font-size:1rem;">Go to My Dashboard →</a>
        </div>

        <p style="color:#334155;font-size:.82rem;text-align:center;">Questions? Reply to this email or contact <a href="mailto:support@aichatbot.ae" style="color:#60a5fa;">support@aichatbot.ae</a></p>
      </div>
    </body>
    </html>
    """

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f'🤖 Your AI Bot is Ready — {plan_label} Plan Activated!'
        msg['From']    = f'AI Chatbot <{gmail_user}>'
        msg['To']      = to_email
        msg.attach(MIMEText(html, 'html'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(gmail_user, gmail_pass)
            server.sendmail(gmail_user, to_email, msg.as_string())
        logger.info(f'Welcome email sent to {to_email}')
    except Exception as e:
        logger.error(f'Failed to send welcome email to {to_email}: {e}')

def notify_owner(name, email, website, plan, access_code):
    """Notify owner via WhatsApp + email when a new client pays"""
    plan_prices = {'starter':'AED 499','professional':'AED 1,499','enterprise':'AED 3,499'}
    price = plan_prices.get(plan, '')
    owner_phone = os.environ.get('OWNER_WHATSAPP', '')
    twilio_sid  = os.environ.get('TWILIO_ACCOUNT_SID', '')
    twilio_tok  = os.environ.get('TWILIO_AUTH_TOKEN', '')
    twilio_wa   = os.environ.get('TWILIO_WHATSAPP_NUMBER', 'whatsapp:+14155238886')

    # WhatsApp notification
    if owner_phone and twilio_sid and twilio_tok:
        try:
            from twilio.rest import Client as TwilioClient
            tc = TwilioClient(twilio_sid, twilio_tok)
            tc.messages.create(
                body=(
                    f"🎉 NEW CLIENT PAID!\n\n"
                    f"👤 Name: {name}\n"
                    f"📧 Email: {email}\n"
                    f"🌐 Website: {website or 'Not provided'}\n"
                    f"📦 Plan: {plan.title()} — {price}/month\n"
                    f"🔑 Code: {access_code}\n\n"
                    f"👉 Admin: https://ai-chatbot-production-2f3a.up.railway.app/admin"
                ),
                from_=twilio_wa,
                to=f'whatsapp:{owner_phone}'
            )
            logger.info(f'Owner WhatsApp notification sent for {email}')
        except Exception as e:
            logger.error(f'Owner WhatsApp notify failed: {e}')

    # Email notification to owner
    gmail_user = os.environ.get('GMAIL_USER', '')
    gmail_pass = os.environ.get('GMAIL_APP_PASSWORD', '')
    owner_email = os.environ.get('OWNER_EMAIL', gmail_user)
    if gmail_user and gmail_pass and owner_email:
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f'🎉 New Client Paid — {plan.title()} Plan — {name}'
            msg['From']    = f'AI Chatbot <{gmail_user}>'
            msg['To']      = owner_email
            body = f"""
            <div style="font-family:Arial;padding:20px;background:#0a0a1a;color:#e2e8f0;">
            <h2 style="color:#34d399;">🎉 New Client Paid!</h2>
            <table style="border-collapse:collapse;width:100%;">
              <tr><td style="padding:8px;color:#94a3b8;">Name</td><td style="padding:8px;color:#fff;font-weight:600;">{name}</td></tr>
              <tr><td style="padding:8px;color:#94a3b8;">Email</td><td style="padding:8px;color:#60a5fa;">{email}</td></tr>
              <tr><td style="padding:8px;color:#94a3b8;">Website</td><td style="padding:8px;color:#60a5fa;">{website or '—'}</td></tr>
              <tr><td style="padding:8px;color:#94a3b8;">Plan</td><td style="padding:8px;color:#a78bfa;font-weight:700;">{plan.title()} — {price}/month</td></tr>
              <tr><td style="padding:8px;color:#94a3b8;">Access Code</td><td style="padding:8px;font-family:monospace;color:#a78bfa;">{access_code}</td></tr>
            </table>
            <br>
            <a href="https://ai-chatbot-production-2f3a.up.railway.app/admin" style="background:linear-gradient(135deg,#a78bfa,#60a5fa);color:#fff;padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:700;">Open Admin Panel →</a>
            </div>
            """
            msg.attach(MIMEText(body, 'html'))
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(gmail_user, gmail_pass)
                server.sendmail(gmail_user, owner_email, msg.as_string())
            logger.info(f'Owner email notification sent for {email}')
        except Exception as e:
            logger.error(f'Owner email notify failed: {e}')


@app.route("/verify-access", methods=["POST"])
def verify_access():
    """Verify access code and grant chatbot access"""
    try:
        data = request.get_json()
        access_code = data.get('access_code')
        
        if not access_code:
            return jsonify({"error": "Access code required"}), 400
        
        # In production, verify against database
        # For now, accept any valid format
        if access_code.startswith("AI-") and len(access_code) == 15:
            return jsonify({
                "success": True,
                "message": "Access granted",
                "plan": "professional",  # Would come from database
                "expires_at": "2026-08-05T23:59:59"  # Would come from database
            })
        else:
            return jsonify({"error": "Invalid access code"}), 401
            
    except Exception as e:
        logger.error(f"Error verifying access: {e}")
        return jsonify({"error": "Access verification failed"}), 500

@app.route("/payment-instructions", methods=["GET"])
def payment_instructions():
    """Display payment instructions with account details"""
    return jsonify({
        "success": True,
        "payment_methods": {
            "stripe": {
                "enabled": true,
                "publishable_key": "pk_live_YOUR_STRIPE_KEY",
                "supported_cards": ["Visa", "Mastercard", "RuPay", "Amex"],
                "currencies": ["INR", "USD", "EUR", "GBP", "AED", "SGD"]
            },
            "paypal": {
                "enabled": true,
                "client_id": "YOUR_PAYPAL_CLIENT_ID",
                "supported_countries": 195
            },
            "upi": {
                "enabled": true,
                "apps": ["PhonePe", "Google Pay", "Paytm", "BHIM"]
            }
        },
        "instructions": {
            "step1": "Choose your plan (Starter: ₹3,000, Professional: ₹10,000, Enterprise: ₹25,000)",
            "step2": "Make payment to any of the above methods",
            "step3": "Send payment screenshot with your email to payments@aichatbot.in",
            "step4": "Receive access code within 2 hours",
            "step5": "Use access code to activate your AI chatbot"
        },
        "note": "Please include your email address in payment description for faster processing"
    })

@app.route("/upi-payment", methods=["POST"])
def upi_payment():
    """Handle UPI payment requests"""
    try:
        data = request.get_json()
        plan = data.get('plan', 'starter')
        
        pricing = {
            'starter': 3000,
            'professional': 10000,
            'enterprise': 25000
        }
        
        if plan not in pricing:
            return jsonify({"error": "Invalid plan selected"}), 400
        
        amount = pricing[plan]
        
        # UPI details for payment
        upi_details = {
            "upi_id": "aichatbot@okicici",
            "amount": amount,
            "note": f"AI Chatbot {plan.title()} Plan",
            "customer_name": data.get('name', ''),
            "customer_phone": data.get('phone', ''),
            "transaction_id": f"UPI_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        logger.info(f"UPI payment initiated: {upi_details['transaction_id']}")
        
        return jsonify({
            "success": True,
            "upi_details": upi_details,
            "qr_code_url": f"https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=upi://pay?pa={upi_details['upi_id']}&pn=AI%20Chatbot&am={amount}&cu=INR&tn={upi_details['note']}"
        })
        
    except Exception as e:
        logger.error(f"Error creating UPI payment: {e}")
        return jsonify({"error": "Failed to create UPI payment"}), 500

@app.route("/wise-payment", methods=["POST"])
def wise_payment():
    """Handle Wise (TransferWise) international payment requests"""
    try:
        data = request.get_json()
        plan = data.get('plan', 'starter')
        country = data.get('country', 'IN')
        
        # International pricing in local currencies
        pricing = {
            'IN': {'starter': 3000, 'professional': 10000, 'enterprise': 25000, 'currency': 'INR'},
            'AE': {'starter': 499, 'professional': 1499, 'enterprise': 3499, 'currency': 'AED'},
            'SG': {'starter': 89, 'professional': 269, 'enterprise': 629, 'currency': 'SGD'},
            'GB': {'starter': 49, 'professional': 149, 'enterprise': 349, 'currency': 'GBP'},
            'US': {'starter': 59, 'professional': 179, 'enterprise': 419, 'currency': 'USD'}
        }
        
        if country not in pricing:
            return jsonify({"error": "Country not supported"}), 400
        
        if plan not in ['starter', 'professional', 'enterprise']:
            return jsonify({"error": "Invalid plan selected"}), 400
        
        country_pricing = pricing[country]
        amount = country_pricing[plan]
        currency = country_pricing['currency']
        
        # Wise payment details
        wise_details = {
            "amount": amount,
            "currency": currency,
            "recipient_name": "AI Chatbot Services",
            "recipient_email": "payments@aichatbot.in",
            "wise_account_id": "AI123456789",
            "reference": f"Chatbot_{plan}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "customer_name": data.get('name', ''),
            "customer_email": data.get('email', ''),
            "description": f"AI Chatbot {plan.title()} Plan - {country}",
            "payment_url": f"https://wise.com/pay/{wise_details['reference']}",
            "bank_details": {
                "account_name": "AI Chatbot Services",
                "account_number": "1234567890",
                "ifsc_code": "ICIC0001234",
                "bank_name": "ICICI Bank",
                "branch": "Mumbai Main"
            }
        }
        
        logger.info(f"Wise payment initiated: {wise_details['reference']} for {amount} {currency}")
        
        return jsonify({
            "success": True,
            "payment_method": "wise",
            "details": wise_details,
            "instructions": f"Transfer {amount} {currency} to the provided bank details or use the Wise payment link"
        })
        
    except Exception as e:
        logger.error(f"Error creating Wise payment: {e}")
        return jsonify({"error": "Failed to create Wise payment"}), 500

@app.route("/paypal-payment", methods=["POST"])
def paypal_payment():
    """Handle PayPal international payment requests"""
    try:
        data = request.get_json()
        plan = data.get('plan', 'starter')
        country = data.get('country', 'IN')
        
        # PayPal pricing in local currencies
        pricing = {
            'IN': {'starter': 3000, 'professional': 10000, 'enterprise': 25000, 'currency': 'INR'},
            'AE': {'starter': 499, 'professional': 1499, 'enterprise': 3499, 'currency': 'AED'},
            'SG': {'starter': 89, 'professional': 269, 'enterprise': 629, 'currency': 'SGD'},
            'GB': {'starter': 49, 'professional': 149, 'enterprise': 349, 'currency': 'GBP'},
            'US': {'starter': 59, 'professional': 179, 'enterprise': 419, 'currency': 'USD'}
        }
        
        if country not in pricing:
            return jsonify({"error": "Country not supported"}), 400
        
        if plan not in ['starter', 'professional', 'enterprise']:
            return jsonify({"error": "Invalid plan selected"}), 400
        
        country_pricing = pricing[country]
        amount = country_pricing[plan]
        currency = country_pricing['currency']
        
        # PayPal payment details
        paypal_details = {
            "amount": amount,
            "currency": currency,
            "merchant_id": "AI_CHATBOT_MERCHANT",
            "paypal_email": "payments@aichatbot.in",
            "item_name": f"AI Chatbot {plan.title()} Plan",
            "item_number": f"CHATBOT_{plan.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "custom": f"customer:{data.get('email', '')};plan:{plan};country:{country}",
            "return_url": f"{request.url_root}payment-success",
            "cancel_url": f"{request.url_root}payment-cancelled",
            "notify_url": f"{request.url_root}paypal-ipn",
            "payment_url": f"https://www.paypal.com/cgi-bin/webscr?cmd=_xclick&business={paypal_details['paypal_email']}&item_name={paypal_details['item_name']}&amount={amount}&currency_code={currency}&return={paypal_details['return_url']}&cancel_return={paypal_details['cancel_url']}"
        }
        
        logger.info(f"PayPal payment initiated: {paypal_details['item_number']} for {amount} {currency}")
        
        return jsonify({
            "success": True,
            "payment_method": "paypal",
            "details": paypal_details,
            "redirect_url": paypal_details['payment_url']
        })
        
    except Exception as e:
        logger.error(f"Error creating PayPal payment: {e}")
        return jsonify({"error": "Failed to create PayPal payment"}), 500

@app.route("/stripe-checkout", methods=["POST"])
def stripe_checkout():
    """Create Stripe Checkout session and return URL"""
    try:
        data = request.get_json()
        plan     = data.get('plan', 'starter')
        name     = data.get('name', '')
        email    = data.get('email', '')
        currency = data.get('currency', 'aed')

        # AED pricing (in fils = AED * 100)
        aed_prices = {'starter': 49900, 'professional': 149900, 'enterprise': 349900}
        inr_prices = {'starter': 300000, 'professional': 1000000, 'enterprise': 2500000}

        stripe_key = os.environ.get("STRIPE_SECRET_KEY", "")

        if stripe_key:
            import stripe as stripe_lib
            stripe_lib.api_key = stripe_key
            amount   = aed_prices.get(plan, 49900) if currency == 'aed' else inr_prices.get(plan, 300000)
            cur_code = 'aed' if currency == 'aed' else 'inr'
            session  = stripe_lib.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price_data': {
                        'currency': cur_code,
                        'unit_amount': amount,
                        'product_data': {
                            'name': f'AI Chatbot {plan.title()} Plan',
                            'description': f'Monthly subscription — {plan.title()} plan'
                        },
                    },
                    'quantity': 1,
                }],
                mode='payment',
                customer_email=email,
                metadata={'plan': plan, 'name': name, 'email': email},
                success_url=request.url_root + f'dashboard?access={{CHECKOUT_SESSION_ID}}&plan={plan}&name={name}&email={email}',
                cancel_url=request.url_root + 'uae',
            )
            return jsonify({"success": True, "checkout_url": session.url})
        else:
            # Stripe key not set yet — return empty so frontend falls back
            return jsonify({"success": False, "checkout_url": None})

    except Exception as e:
        logger.error(f"Stripe checkout error: {e}")
        return jsonify({"success": False, "checkout_url": None})

@app.route("/stripe-payment", methods=["POST"])
def stripe_payment():
    """Handle Stripe payment session creation"""
    try:
        data = request.get_json()
        plan = data.get('plan', 'starter')
        name = data.get('name', '')
        email = data.get('email', '')
        amount = data.get('amount', 3000)  # Amount in paise
        currency = data.get('currency', 'inr')
        
        if not plan or not name or not email:
            return jsonify({"error": "Missing required fields"}), 400
        
        # In production, use actual Stripe API
        # For now, simulate session creation
        session_id = f"cs_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(email + plan)}"
        
        # Create checkout session details
        session_data = {
            "session_id": session_id,
            "publishable_key": "pk_test_51234567890abcdef",  # Test key
            "checkout_url": f"https://checkout.stripe.com/pay/{session_id}",
            "amount": amount,
            "currency": currency,
            "plan": plan,
            "customer_email": email,
            "customer_name": name,
            "success_url": f"{request.url_root}payment-success?session_id={session_id}",
            "cancel_url": f"{request.url_root}payment-cancelled"
        }
        
        logger.info(f"Stripe session created: {session_id} for {email}")
        
        return jsonify({
            "success": True,
            "session_id": session_id,
            "checkout_url": session_data["checkout_url"],
            "publishable_key": session_data["publishable_key"]
        })
        
    except Exception as e:
        logger.error(f"Error creating Stripe session: {e}")
        return jsonify({"error": "Failed to create payment session"}), 500

@app.route("/stripe-webhook", methods=["POST"])
def stripe_webhook():
    """Handle Stripe webhook events"""
    try:
        # Verify webhook signature in production
        event = request.get_json()
        
        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            # Payment successful - grant access
            access_code = generate_access_code(
                session['customer_details']['email'], 
                session['metadata']['plan']
            )
            
            logger.info(f"Stripe payment completed: {session['id']} - Access: {access_code}")
            
            return jsonify({"status": "success", "access_code": access_code})
        
        return jsonify({"status": "received"})
        
    except Exception as e:
        logger.error(f"Stripe webhook error: {e}")
        return jsonify({"error": "Webhook processing failed"}), 500

@app.route("/payment-cancelled", methods=["GET", "POST"])
def payment_cancelled():
    """Handle cancelled payments"""
    return jsonify({
        "success": False,
        "message": "Payment was cancelled",
        "redirect_url": "/"
    })

@app.route("/paypal-ipn", methods=["POST"])
def paypal_ipn():
    """Handle PayPal Instant Payment Notification"""
    try:
        # In production, verify IPN data with PayPal
        logger.info("PayPal IPN received")
        return jsonify({"status": "verified"}), 200
    except Exception as e:
        logger.error(f"Error processing PayPal IPN: {e}")
        return jsonify({"status": "error"}), 500

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
    context = get_relevant_context(user_message)
    logger.info("Sending message to Claude (length=%d chars, lang=%s, tz=%s, history=%d, context=%d)", len(user_message), language, timezone, len(history), len(context))

    try:
        reply = get_ai_reply(user_message, language, timezone, history, context)
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

@app.route("/upload", methods=["POST"])
def upload_document():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md', '.csv', '.json'}
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in allowed_extensions:
        return jsonify({"error": f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_extensions)}"}), 400

    try:
        text = extract_text_from_file(file, file.filename)
    except Exception as e:
        logger.error("Extraction failed: %s", e)
        return jsonify({"error": "Failed to extract text from file"}), 500

    if not text.strip():
        return jsonify({"error": "No text could be extracted from the file"}), 400

    chunk_count = add_document_to_knowledge_base(file.filename, text)
    preview = text[:300].replace('\n', ' ')
    return jsonify({
        "success": True,
        "filename": file.filename,
        "chunks": chunk_count,
        "preview": preview
    })

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
        if resp.status_code == 401 or resp.status_code == 429:
            logger.warning("ElevenLabs quota exceeded or unauthorized - client will use browser TTS")
            return jsonify({"error": "quota_exceeded"}), 503
        if resp.status_code != 200:
            logger.error("ElevenLabs error: status=%s, body=%s", resp.status_code, resp.text[:500])
            return jsonify({"error": "ElevenLabs error"}), 503
        return Response(resp.content, mimetype="audio/mpeg")
    except Exception as e:
        logger.error("ElevenLabs error: %s", e)
        return jsonify({"error": "Failed to generate audio"}), 503

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

