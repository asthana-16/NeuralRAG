"""
Flask RAG LLM Application
=========================================
Features:
  - User Authentication (Register/Login with CAPTCHA)
  - CRUD for Knowledge Base Documents
  - RAG-powered Q&A via Gemini API
  - Chat History Management
  - Profile & Settings
  - SQLite Database (no external ORM needed)
"""
from dotenv import load_dotenv
load_dotenv()
import os, io, re, json, math, random, string, sqlite3, hashlib, hmac, base64
from datetime import datetime, timedelta
from functools import wraps
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, jsonify, g)

# ──────────────────────────────────────────────────────────
# APP CONFIG
# ──────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'rag-llm-super-secret-2024-xyz')
app.permanent_session_lifetime = timedelta(hours=24)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config.update(
    DB_PATH=os.path.join(BASE_DIR, 'instance', 'app.db'),
    UPLOAD_FOLDER=os.path.join(BASE_DIR, 'uploads'),
    GEMINI_API_KEY=os.environ.get('GEMINI_API_KEY', ''),
    MAX_CONTENT_LENGTH=10 * 1024 * 1024,
)

os.makedirs(os.path.join(BASE_DIR, 'instance'), exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Font path for captcha
# REPLACE KAR YE SE:
_FONT_CANDIDATES = [
    'C:/Windows/Fonts/arialbd.ttf',      # Windows Arial Bold
    'C:/Windows/Fonts/arial.ttf',         # Windows Arial
    'C:/Windows/Fonts/verdana.ttf',       # Windows Verdana
    'C:/Windows/Fonts/tahoma.ttf',        # Windows Tahoma
    'C:/Windows/Fonts/trebucbd.ttf',      # Windows Trebuchet Bold
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',   # Linux
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
]
FONT_PATH = None
for _fp in _FONT_CANDIDATES:
    if os.path.exists(_fp):
        FONT_PATH = _fp
        print(f"✅ Font loaded: {_fp}")
        break
if not FONT_PATH:
    print("⚠️ No font found — captcha will use default tiny font")


# ──────────────────────────────────────────────────────────
# DATABASE
# ──────────────────────────────────────────────────────────
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(app.config['DB_PATH'])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db:
        db.close()

def init_db():
    db = sqlite3.connect(app.config['DB_PATH'])
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT DEFAULT '',
            bio TEXT DEFAULT '',
            avatar_color TEXT DEFAULT '#6c63ff',
            is_active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            category TEXT DEFAULT 'General',
            tags TEXT DEFAULT '',
            word_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT DEFAULT 'New Chat',
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT DEFAULT '[]',
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
        );
    """)
    db.commit()
    db.close()
    print("✅ Database initialized.")


# ──────────────────────────────────────────────────────────
# PASSWORD HELPERS
# ──────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 200_000)
    return base64.b64encode(salt + key).decode()

def check_password(password: str, stored: str) -> bool:
    try:
        raw = base64.b64decode(stored.encode())
        salt, key = raw[:32], raw[32:]
        new_key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 200_000)
        return hmac.compare_digest(key, new_key)
    except Exception:
        return False


# ──────────────────────────────────────────────────────────
# AUTH DECORATOR
# ──────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please sign in to continue.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def get_current_user():
    if 'user_id' not in session:
        return None
    db = get_db()
    return db.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()


# ──────────────────────────────────────────────────────────
# CAPTCHA GENERATOR  (clear, readable)
# ──────────────────────────────────────────────────────────
CAPTCHA_CHARS = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'

def generate_captcha_image() -> tuple:
    """Returns (base64_image_uri, answer_text) — 5 chars packed tightly, easy to read"""
    text = ''.join(random.choices(CAPTCHA_CHARS, k=5))
    W, H = 220, 70

    img = Image.new('RGB', (W, H), (15, 15, 30))
    draw = ImageDraw.Draw(img)

    for _ in range(150):
        x, y = random.randint(0, W - 1), random.randint(0, H - 1)
        g_val = random.randint(25, 40)
        draw.point((x, y), fill=(g_val, g_val, g_val + 10))

    try:
        font = ImageFont.truetype(FONT_PATH, 38)
    except Exception:
        font = ImageFont.load_default()

    palette = ['#a78bfa', '#60a5fa', '#34d399', '#fbbf24', '#fb923c', '#f472b6']
    char_step = 40
    start_x = (W - (char_step * len(text))) // 2

    for i, ch in enumerate(text):
        x = start_x + i * char_step + random.randint(-2, 2)
        y = random.randint(6, 16)
        color = palette[i % len(palette)]
        tmp = Image.new('RGBA', (48, 56), (0, 0, 0, 0))
        tmp_draw = ImageDraw.Draw(tmp)
        tmp_draw.text((4, 2), ch, font=font, fill=color)
        angle = random.randint(-12, 12)
        tmp = tmp.rotate(angle, expand=False, resample=Image.BICUBIC)
        img.paste(tmp, (x, y), tmp)

    x1 = random.randint(0, W // 4)
    y1 = random.randint(10, H - 10)
    x2 = random.randint(3 * W // 4, W)
    y2 = random.randint(10, H - 10)
    draw.line([(x1, y1), (x2, y2)], fill=(45, 45, 65), width=1)

    img = img.filter(ImageFilter.SMOOTH_MORE)
    buf = io.BytesIO()
    img.save(buf, 'PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}", text


# ──────────────────────────────────────────────────────────
# RAG ENGINE
# ──────────────────────────────────────────────────────────
def tokenize(text: str) -> list:
    return re.findall(r'\b[a-z]{2,}\b', text.lower())

def build_tfidf(docs: list) -> tuple:
    corpus = [tokenize(d['title'] + ' ' + d['content']) for d in docs]
    df = {}
    for tokens in corpus:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1
    N = len(docs)
    vocab = {w: i for i, w in enumerate(sorted(df.keys()))}

    vectors = []
    for tokens in corpus:
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vec = [0.0] * len(vocab)
        for t, cnt in tf.items():
            if t in vocab:
                idf = math.log((N + 1) / (df.get(t, 0) + 1)) + 1
                vec[vocab[t]] = (cnt / max(len(tokens), 1)) * idf
        norm = math.sqrt(sum(v*v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vocab, vectors

def cosine_sim(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))

def retrieve_top_k(query: str, docs: list, k: int = 4) -> list:
    if not docs:
        return []
    vocab, vecs = build_tfidf(docs)
    if not vocab:
        return docs[:k]
    q_tokens = tokenize(query)
    q_tf = {}
    for t in q_tokens:
        q_tf[t] = q_tf.get(t, 0) + 1
    q_vec = [0.0] * len(vocab)
    for t, cnt in q_tf.items():
        if t in vocab:
            q_vec[vocab[t]] = cnt / max(len(q_tokens), 1)
    q_norm = math.sqrt(sum(v*v for v in q_vec)) or 1.0
    q_vec = [v / q_norm for v in q_vec]
    scored = [(cosine_sim(q_vec, vecs[i]), docs[i]) for i in range(len(docs))]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:k] if _ > 0.01]

try:
    from google import genai as _genai
    _NEW_SDK = True
except ImportError:
    import google.generativeai as _genai
    _NEW_SDK = False

def call_llm(query: str, context_docs: list, api_key: str) -> str:
    context_text = ""
    for d in context_docs:
        context_text += f"\n\n### {d['title']}\n{d['content'][:800]}"

    prompt = (
        "You are an AI assistant.\n"
        "If the answer exists in the provided context, use it.\n"
        "If not, answer using your general knowledge.\n\n"
        f"Context:\n{context_text}\n\n"
        f"User Question: {query}"
    )

    if not api_key:
        return _mock_response(query, context_docs)

    try:
        if _NEW_SDK:
            # New SDK: google-genai
            client = _genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
        else:
            # Old SDK: google-generativeai (still works, just shows warning)
            _genai.configure(api_key=api_key)
            model = _genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini Error: {str(e)}"

def _mock_response(query: str, docs: list) -> str:
    if docs:
        titles = ', '.join(f'"{d["title"]}"' for d in docs)
        return (f"**[Demo Mode — No API Key Configured]**\n\n"
                f"Your query: *{query}*\n\n"
                f"I found {len(docs)} relevant document(s): {titles}.\n\n"
                f"To get real AI-powered answers, add your **Gemini API key** in ⚙️ Settings.\n\n"
                f"**Sample content from top result:**\n> {docs[0]['content'][:300]}...")
    return (f"**[Demo Mode]**\n\nNo relevant documents found for: *{query}*\n\n"
            f"Add documents to your knowledge base, then ask questions about them!")


# ──────────────────────────────────────────────────────────
# ROUTES: PUBLIC
# ──────────────────────────────────────────────────────────
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/captcha-refresh')
def captcha_refresh():
    img, answer = generate_captcha_image()
    session['captcha'] = answer
    return jsonify({'image': img})


# ──────────────────────────────────────────────────────────
# ROUTES: AUTH
# ──────────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        identifier = request.form.get('identifier', '').strip()
        password   = request.form.get('password', '')
        captcha_in = request.form.get('captcha', '').strip().upper()
        stored     = session.get('captcha', '')

        # Generate new captcha for next attempt
        img, new_answer = generate_captcha_image()
        session['captcha'] = new_answer

        # Check against OLD stored answer
        if captcha_in != stored:
            flash('Incorrect CAPTCHA — please try again.', 'error')
            return render_template('login.html', captcha_img=img)

        db = get_db()
        user = db.execute(
            'SELECT * FROM users WHERE (username = ? OR email = ?) AND is_active = 1',
            (identifier, identifier)
        ).fetchone()

        if not user or not check_password(password, user['password_hash']):
            flash('Invalid credentials.', 'error')
            return render_template('login.html', captcha_img=img)

        session.permanent = True
        session['user_id'] = user['id']
        session['username'] = user['username']
        flash(f'Welcome back, {user["username"]}! 👋', 'success')
        return redirect(url_for('dashboard'))

    # GET request — generate fresh captcha
    img, answer = generate_captcha_image()
    session['captcha'] = answer
    return render_template('login.html', captcha_img=img)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username   = request.form.get('username', '').strip()
        email      = request.form.get('email', '').strip().lower()
        full_name  = request.form.get('full_name', '').strip()
        password   = request.form.get('password', '')
        confirm    = request.form.get('confirm_password', '')
        captcha_in = request.form.get('captcha', '').strip().upper()
        stored     = session.get('captcha', '')

        # Generate new captcha for next attempt
        img, new_answer = generate_captcha_image()
        session['captcha'] = new_answer

        errors = []
        if captcha_in != stored:
            errors.append('Incorrect CAPTCHA — please try again.')
        if not re.match(r'^[a-zA-Z0-9_]{3,20}$', username):
            errors.append('Username: 3–20 chars, letters/numbers/underscore only.')
        if not re.match(r'^[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}$', email):
            errors.append('Enter a valid email address.')
        if len(password) < 8:
            errors.append('Password must be at least 8 characters.')
        if password != confirm:
            errors.append('Passwords do not match.')

        if errors:
            for e in errors:
                flash(e, 'error')
            return render_template('register.html', captcha_img=img,
                                   username=username, email=email, full_name=full_name)

        db = get_db()
        if db.execute('SELECT id FROM users WHERE username=?', (username,)).fetchone():
            flash('Username already taken.', 'error')
            return render_template('register.html', captcha_img=img,
                                   username=username, email=email, full_name=full_name)
        if db.execute('SELECT id FROM users WHERE email=?', (email,)).fetchone():
            flash('Email already registered.', 'error')
            return render_template('register.html', captcha_img=img,
                                   username=username, email=email, full_name=full_name)

        colors = ['#6c63ff', '#f43f5e', '#10b981', '#f59e0b', '#3b82f6', '#8b5cf6']
        db.execute(
            'INSERT INTO users (username, email, full_name, password_hash, avatar_color) VALUES (?,?,?,?,?)',
            (username, email, full_name, hash_password(password), random.choice(colors))
        )
        db.commit()
        flash('Account created! Please sign in.', 'success')
        return redirect(url_for('login'))

    # GET request — generate fresh captcha
    img, answer = generate_captcha_image()
    session['captcha'] = answer
    return render_template('register.html', captcha_img=img)


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been signed out.', 'info')
    return redirect(url_for('login'))


# ──────────────────────────────────────────────────────────
# ROUTES: DASHBOARD
# ──────────────────────────────────────────────────────────
@app.route('/dashboard')
@login_required
def dashboard():
    db = get_db()
    uid = session['user_id']
    user = get_current_user()
    stats = {
        'docs':  db.execute('SELECT COUNT(*) FROM documents WHERE user_id=?', (uid,)).fetchone()[0],
        'chats': db.execute('SELECT COUNT(*) FROM chat_sessions WHERE user_id=?', (uid,)).fetchone()[0],
        'msgs':  db.execute('SELECT COUNT(*) FROM chat_messages WHERE user_id=?', (uid,)).fetchone()[0],
        'words': db.execute('SELECT COALESCE(SUM(word_count),0) FROM documents WHERE user_id=?', (uid,)).fetchone()[0],
    }
    recent_docs = db.execute(
        'SELECT * FROM documents WHERE user_id=? ORDER BY updated_at DESC LIMIT 5', (uid,)
    ).fetchall()
    recent_sessions = db.execute(
        'SELECT s.*, COUNT(m.id) as msg_count FROM chat_sessions s '
        'LEFT JOIN chat_messages m ON s.id=m.session_id '
        'WHERE s.user_id=? GROUP BY s.id ORDER BY s.created_at DESC LIMIT 4', (uid,)
    ).fetchall()
    categories = db.execute(
        'SELECT category, COUNT(*) as cnt FROM documents WHERE user_id=? GROUP BY category', (uid,)
    ).fetchall()
    return render_template('dashboard.html', user=user, stats=stats,
                           recent_docs=recent_docs, recent_sessions=recent_sessions,
                           categories=categories)


# ──────────────────────────────────────────────────────────
# ROUTES: DOCUMENTS (CRUD)
# ──────────────────────────────────────────────────────────
@app.route('/documents')
@login_required
def documents():
    db = get_db()
    uid = session['user_id']
    search   = request.args.get('q', '').strip()
    category = request.args.get('cat', '').strip()
    tag      = request.args.get('tag', '').strip()

    sql = 'SELECT * FROM documents WHERE user_id=?'
    params = [uid]
    if search:
        sql += ' AND (title LIKE ? OR content LIKE ?)'
        params += [f'%{search}%', f'%{search}%']
    if category:
        sql += ' AND category=?'
        params.append(category)
    if tag:
        sql += ' AND tags LIKE ?'
        params.append(f'%{tag}%')
    sql += ' ORDER BY updated_at DESC'

    docs = db.execute(sql, params).fetchall()
    all_cats = db.execute(
        'SELECT DISTINCT category FROM documents WHERE user_id=? ORDER BY category', (uid,)
    ).fetchall()
    all_tags_raw = db.execute('SELECT tags FROM documents WHERE user_id=?', (uid,)).fetchall()
    all_tags = set()
    for row in all_tags_raw:
        for t in (row['tags'] or '').split(','):
            t = t.strip()
            if t:
                all_tags.add(t)

    return render_template('documents.html', docs=docs, all_cats=all_cats,
                           all_tags=sorted(all_tags), search=search,
                           sel_cat=category, sel_tag=tag)

@app.route('/documents/new', methods=['GET', 'POST'])
@login_required
def doc_create():
    if request.method == 'POST':
        title    = request.form.get('title', '').strip()
        content  = request.form.get('content', '').strip()
        category = request.form.get('category', 'General').strip() or 'General'
        tags     = ', '.join(t.strip() for t in request.form.get('tags', '').split(',') if t.strip())

        if not title or not content:
            flash('Title and content are required.', 'error')
            return render_template('doc_form.html', doc=None, action='Create')

        wc = len(content.split())
        db = get_db()
        db.execute(
            'INSERT INTO documents (user_id, title, content, category, tags, word_count) VALUES (?,?,?,?,?,?)',
            (session['user_id'], title, content, category, tags, wc)
        )
        db.commit()
        flash(f'Document "{title}" created successfully!', 'success')
        return redirect(url_for('documents'))
    return render_template('doc_form.html', doc=None, action='Create')

@app.route('/documents/<int:doc_id>')
@login_required
def doc_view(doc_id):
    db = get_db()
    doc = db.execute('SELECT * FROM documents WHERE id=? AND user_id=?',
                     (doc_id, session['user_id'])).fetchone()
    if not doc:
        flash('Document not found.', 'error')
        return redirect(url_for('documents'))
    return render_template('doc_view.html', doc=doc)

@app.route('/documents/<int:doc_id>/edit', methods=['GET', 'POST'])
@login_required
def doc_edit(doc_id):
    db = get_db()
    doc = db.execute('SELECT * FROM documents WHERE id=? AND user_id=?',
                     (doc_id, session['user_id'])).fetchone()
    if not doc:
        flash('Document not found.', 'error')
        return redirect(url_for('documents'))

    if request.method == 'POST':
        title    = request.form.get('title', '').strip()
        content  = request.form.get('content', '').strip()
        category = request.form.get('category', 'General').strip() or 'General'
        tags     = ', '.join(t.strip() for t in request.form.get('tags', '').split(',') if t.strip())

        if not title or not content:
            flash('Title and content are required.', 'error')
            return render_template('doc_form.html', doc=doc, action='Edit')

        wc = len(content.split())
        db.execute(
            'UPDATE documents SET title=?, content=?, category=?, tags=?, word_count=?, updated_at=datetime("now") '
            'WHERE id=? AND user_id=?',
            (title, content, category, tags, wc, doc_id, session['user_id'])
        )
        db.commit()
        flash('Document updated!', 'success')
        return redirect(url_for('doc_view', doc_id=doc_id))

    return render_template('doc_form.html', doc=doc, action='Edit')

@app.route('/documents/<int:doc_id>/delete', methods=['POST'])
@login_required
def doc_delete(doc_id):
    db = get_db()
    r = db.execute('DELETE FROM documents WHERE id=? AND user_id=?',
                   (doc_id, session['user_id']))
    db.commit()
    if r.rowcount:
        flash('Document deleted.', 'success')
    else:
        flash('Document not found.', 'error')
    return redirect(url_for('documents'))


# ──────────────────────────────────────────────────────────
# ROUTES: CHAT (RAG)
# ──────────────────────────────────────────────────────────
@app.route('/chat')
@login_required
def chat_list():
    db = get_db()
    sessions = db.execute(
        'SELECT s.*, COUNT(m.id) as msg_count FROM chat_sessions s '
        'LEFT JOIN chat_messages m ON s.id=m.session_id '
        'WHERE s.user_id=? GROUP BY s.id ORDER BY s.created_at DESC',
        (session['user_id'],)
    ).fetchall()
    return render_template('chat_list.html', sessions=sessions)

@app.route('/chat/new')
@login_required
def chat_new():
    db = get_db()
    r = db.execute('INSERT INTO chat_sessions (user_id, title) VALUES (?,?)',
                   (session['user_id'], 'New Chat'))
    db.commit()
    return redirect(url_for('chat_session', session_id=r.lastrowid))

@app.route('/chat/<int:session_id>')
@login_required
def chat_session(session_id):
    db = get_db()
    chat_sess = db.execute(
        'SELECT * FROM chat_sessions WHERE id=? AND user_id=?',
        (session_id, session['user_id'])
    ).fetchone()
    if not chat_sess:
        flash('Chat session not found.', 'error')
        return redirect(url_for('chat_list'))
    messages = db.execute(
        'SELECT * FROM chat_messages WHERE session_id=? ORDER BY created_at ASC',
        (session_id,)
    ).fetchall()
    doc_count = db.execute('SELECT COUNT(*) FROM documents WHERE user_id=?',
                           (session['user_id'],)).fetchone()[0]
    has_api_key = bool(session.get('gemini_api_key') or app.config['GEMINI_API_KEY'])
    return render_template('chat_session.html', chat_sess=chat_sess,
                           messages=messages, doc_count=doc_count, has_api_key=has_api_key)

@app.route('/chat/<int:session_id>/send', methods=['POST'])
@login_required
def chat_send(session_id):
    db = get_db()
    chat_sess = db.execute(
        'SELECT * FROM chat_sessions WHERE id=? AND user_id=?',
        (session_id, session['user_id'])
    ).fetchone()
    if not chat_sess:
        return jsonify({'error': 'Session not found'}), 404

    data  = request.get_json() or {}
    query = (data.get('query') or '').strip()
    if not query:
        return jsonify({'error': 'Query is empty'}), 400

    db.execute('INSERT INTO chat_messages (session_id, user_id, role, content) VALUES (?,?,?,?)',
               (session_id, session['user_id'], 'user', query))

    msg_count = db.execute('SELECT COUNT(*) FROM chat_messages WHERE session_id=?',
                           (session_id,)).fetchone()[0]
    if msg_count == 1:
        title = query[:50] + ('…' if len(query) > 50 else '')
        db.execute('UPDATE chat_sessions SET title=? WHERE id=?', (title, session_id))

    docs = db.execute('SELECT * FROM documents WHERE user_id=?',
                      (session['user_id'],)).fetchall()
    doc_list = [dict(d) for d in docs]
    relevant = retrieve_top_k(query, doc_list, k=4)

    api_key = session.get('gemini_api_key') or app.config.get('GEMINI_API_KEY') or os.environ.get('GEMINI_API_KEY', '')
    answer  = call_llm(query, relevant, api_key)

    sources = [{'id': d['id'], 'title': d['title']} for d in relevant]
    db.execute(
        'INSERT INTO chat_messages (session_id, user_id, role, content, sources) VALUES (?,?,?,?,?)',
        (session_id, session['user_id'], 'assistant', answer, json.dumps(sources))
    )
    db.commit()

    return jsonify({
        'answer':    answer,
        'sources':   sources,
        'timestamp': datetime.now().strftime('%H:%M')
    })

@app.route('/chat/<int:session_id>/delete', methods=['POST'])
@login_required
def chat_delete(session_id):
    db = get_db()
    db.execute('DELETE FROM chat_sessions WHERE id=? AND user_id=?',
               (session_id, session['user_id']))
    db.commit()
    flash('Chat session deleted.', 'success')
    return redirect(url_for('chat_list'))


# ──────────────────────────────────────────────────────────
# ROUTES: SETTINGS
# ──────────────────────────────────────────────────────────
@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    db   = get_db()
    user = get_current_user()

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'profile':
            full_name    = request.form.get('full_name', '').strip()
            bio          = request.form.get('bio', '').strip()
            avatar_color = request.form.get('avatar_color', '#6c63ff')
            db.execute(
                'UPDATE users SET full_name=?, bio=?, avatar_color=?, updated_at=datetime("now") WHERE id=?',
                (full_name, bio, avatar_color, session['user_id'])
            )
            db.commit()
            flash('Profile updated!', 'success')

        elif action == 'password':
            curr = request.form.get('current_password', '')
            new_ = request.form.get('new_password', '')
            conf = request.form.get('confirm_new', '')
            if not check_password(curr, user['password_hash']):
                flash('Current password is incorrect.', 'error')
            elif len(new_) < 8:
                flash('New password must be at least 8 characters.', 'error')
            elif new_ != conf:
                flash('New passwords do not match.', 'error')
            else:
                db.execute('UPDATE users SET password_hash=? WHERE id=?',
                           (hash_password(new_), session['user_id']))
                db.commit()
                flash('Password changed successfully!', 'success')

        elif action == 'api_key':
            key = request.form.get('api_key', '').strip()
            app.config['GEMINI_API_KEY'] = key
            session['gemini_api_key'] = key
            flash('API key updated!', 'success')

        user = get_current_user()

    has_api = bool(session.get('gemini_api_key') or app.config['GEMINI_API_KEY'])
    return render_template('settings.html', user=user, has_api=has_api)


# ──────────────────────────────────────────────────────────
# DOCUMENT UPLOAD
# ──────────────────────────────────────────────────────────
from PyPDF2 import PdfReader
from docx import Document as DocxDocument

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    return "".join(page.extract_text() or "" for page in reader.pages)

def extract_text_from_docx(path):
    doc = DocxDocument(path)
    return "\n".join([p.text for p in doc.paragraphs])

@app.route('/documents/upload', methods=['GET', 'POST'])
@login_required
def doc_upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            flash("No file selected.", "error")
            return redirect(request.url)

        filename = file.filename.lower()
        if not (filename.endswith('.pdf') or filename.endswith('.docx')):
            flash("Only PDF or DOCX allowed.", "error")
            return redirect(request.url)

        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        content = extract_text_from_pdf(save_path) if filename.endswith('.pdf') else extract_text_from_docx(save_path)

        if not content.strip():
            flash("Could not extract text from file.", "error")
            return redirect(request.url)

        wc = len(content.split())
        db = get_db()
        db.execute(
            'INSERT INTO documents (user_id, title, content, category, tags, word_count) VALUES (?,?,?,?,?,?)',
            (session['user_id'], file.filename, content, "Uploaded File", "", wc)
        )
        db.commit()
        flash("File uploaded and processed successfully!", "success")
        return redirect(url_for('documents'))

    return render_template('doc_upload.html')


# ──────────────────────────────────────────────────────────
# TEMPLATE HELPERS
# ──────────────────────────────────────────────────────────
@app.template_filter('from_json')
def from_json_filter(s):
    try:
        return json.loads(s or '[]')
    except Exception:
        return []

@app.context_processor
def inject_helpers():
    return dict(get_current_user=get_current_user)


# ──────────────────────────────────────────────────────────
# STARTUP
# ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    init_db()
    print("\n" + "=" * 50)
    print("  🚀  Flask RAG LLM App")
    print("=" * 50)
    print("  URL  →  http://127.0.0.1:5000")
    print("  API  →", "✅ Loaded" if app.config['GEMINI_API_KEY'] else "❌ Not set (add to .env)")
    print("=" * 50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
