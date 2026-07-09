import sqlite3
import os
import hashlib
import uuid
from datetime import datetime, timedelta

DB_PATH = os.environ.get('DATABASE_PATH', 'botifyai.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT,
            website TEXT,
            access_code TEXT UNIQUE NOT NULL,
            plan TEXT DEFAULT 'basic',
            active INTEGER DEFAULT 1,
            message_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            expires_at TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            content TEXT NOT NULL,
            chunk_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (client_id) REFERENCES clients(id) ON DELETE CASCADE
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (client_id) REFERENCES clients(id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()

def generate_access_code(email="", plan="basic"):
    raw = f"{email}{plan}{uuid.uuid4().hex}{datetime.utcnow().timestamp()}"
    code = hashlib.sha256(raw.encode()).hexdigest()[:16].upper()
    return f"BOT-{code}"

def create_client(name, email="", website="", plan="basic", days_valid=365):
    access_code = generate_access_code(email, plan)
    created_at = datetime.utcnow().isoformat()
    expires_at = (datetime.utcnow() + timedelta(days=days_valid)).isoformat() if days_valid else None
    
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO clients (name, email, website, access_code, plan, active, created_at, expires_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (name, email, website, access_code, plan, 1, created_at, expires_at))
    client_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return client_id, access_code

def get_client_by_access_code(access_code):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM clients WHERE access_code = ? AND active = 1', (access_code,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_client(client_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM clients WHERE id = ?', (client_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_clients():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM clients ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def add_document(client_id, filename, content, chunk_count=0):
    created_at = datetime.utcnow().isoformat()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO documents (client_id, filename, content, chunk_count, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (client_id, filename, content, chunk_count, created_at))
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return doc_id

def get_documents(client_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM documents WHERE client_id = ? ORDER BY created_at DESC', (client_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_client_context(client_id, query="", top_k=5):
    """Return relevant context for a client from their documents."""
    docs = get_documents(client_id)
    if not docs:
        return ""
    
    # Simple keyword search across documents
    query_words = set(query.lower().split()) if query else set()
    scored = []
    for doc in docs:
        content = doc['content']
        score = 0
        if query_words:
            content_lower = content.lower()
            for word in query_words:
                if word in content_lower:
                    score += content_lower.count(word)
        scored.append((score, content))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]
    return "\n\n".join([s[1][:2000] for s in top])

def increment_message_count(access_code):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE clients SET message_count = message_count + 1 WHERE access_code = ?', (access_code,))
    conn.commit()
    conn.close()

def deactivate_client(client_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE clients SET active = 0 WHERE id = ?', (client_id,))
    conn.commit()
    conn.close()

def delete_client(client_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM clients WHERE id = ?', (client_id,))
    conn.commit()
    conn.close()
