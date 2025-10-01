# server.py (Memory-optimized)
import os
import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import openai

# --- Config ---
KB_DIR = Path("kb_store")
KB_ITEMS_PATH = KB_DIR / "kb_items.json"
KB_EMB_PATH = KB_DIR / "kb_embeddings.npz"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo").strip()

OLLAMA_TIMEOUT = 120
TOP_K = int(os.environ.get("TOP_K", 3))
MAX_CONTEXT_CHARS = 4000

ALPHA_TEXT = 0.7
BETA_TITLE = 1.2
GAMMA_PRIORITY = 1.0
RECENCY_BASE_YEAR = 2018
RECENCY_SCALE = 0.03

ID_BOOSTS = {"humly": 1.0, "outliar": 0.9, "meliox": 0.3}

WORK_INTENT_PATTERNS = [
    r"\b(work|experience|job|employed|role|intern|position|worked|employment|professional experience)\b",
    r"\b(current role|previous role|where do you work|where did you work)\b",
]

GREETING_PATTERNS = [
    r'^\s*(hi|hello|hey|hiya|yo)\b',
    r'\bhow are you\b',
    r'\bhow\'s it going\b',
    r'\bgood morning\b',
    r'\bgood afternoon\b',
    r'\bgood evening\b',
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Flask init ---
app = Flask(__name__)
CORS(app, origins="*")  # or specify your Vercel domain for more security

# --- OpenAI init ---
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set. OpenAI calls will fail until configured.")
openai.api_key = OPENAI_API_KEY

# --- Load KB and embeddings ---
if not KB_ITEMS_PATH.exists() or not KB_EMB_PATH.exists():
    raise FileNotFoundError("kb_items.json or kb_embeddings.npz not found in kb_store. Run embed.py first.")

with open(KB_ITEMS_PATH, "r", encoding="utf-8") as f:
    kb_items = json.load(f)

npz = np.load(KB_EMB_PATH, allow_pickle=True)
kb_embeddings = np.asarray(npz["embeddings"], dtype=np.float32)
title_embeddings = np.asarray(npz["title_embeddings"], dtype=np.float32)
ids = list(npz["ids"])

# Make embeddings read-only to save memory
kb_embeddings.flags.writeable = False
title_embeddings.flags.writeable = False

# --- Lazy CPU-only embedder + query cache ---
_embedder = None
_query_cache = {}

def get_embedder():
    global _embedder
    if _embedder is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU
        logger.info("Loading SentenceTransformer model on CPU")
        _embedder = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2", device="cpu")
    return _embedder

def get_query_embedding(query: str):
    if query in _query_cache:
        return _query_cache[query]
    embedder = get_embedder()
    q_emb = embedder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    if q_emb.ndim == 2:
        q_emb = q_emb[0]
    q = q_emb / (np.linalg.norm(q_emb) + 1e-12)
    _query_cache[query] = q
    return q

# --- Helper functions ---
def is_greeting(text: str) -> bool:
    if not text:
        return False
    t = text.lower().strip()
    for pat in GREETING_PATTERNS:
        if re.search(pat, t):
            return True
    return len(t.split()) <= 2 and t in {"hey", "hi", "hello", "hiya", "how are you", "how r u"}

def extract_most_recent_year(text: str) -> Optional[int]:
    if not text:
        return None
    years = re.findall(r"(?:19|20)\d{2}", text)
    return max(map(int, years)) if years else None

def is_work_intent(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(re.search(pat, t) for pat in WORK_INTENT_PATTERNS)

def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

kb_embeddings = normalize_rows(kb_embeddings)
title_embeddings = normalize_rows(title_embeddings)

def get_top_k(query: str, k: int = TOP_K):
    q = get_query_embedding(query)
    sim_text = np.dot(kb_embeddings, q)
    sim_title = np.dot(title_embeddings, q)
    combined = ALPHA_TEXT * sim_text + BETA_TITLE * sim_title

    for idx, it in enumerate(kb_items):
        priority = float(it.get("priority", 0.0) or 0.0)
        combined[idx] += GAMMA_PRIORITY * priority
        item_id = it.get("id")
        if item_id in ID_BOOSTS:
            combined[idx] += float(ID_BOOSTS[item_id])
        year = it.get("year") or extract_most_recent_year(it.get("text", ""))
        if year:
            combined[idx] += max(0, int(year) - RECENCY_BASE_YEAR) * RECENCY_SCALE

    idxs = np.argsort(-combined)[:k]
    return idxs, combined[idxs]

# --- OpenAI call helper ---
def call_openai_chat(prompt: str, model: str = OPENAI_MODEL, timeout: int = 60):
    if not openai.api_key:
        raise RuntimeError("OpenAI API key not configured.")
    messages = [
        {"role": "system", "content": "You are an assistant that answers questions about Omar Dalal using ONLY the provided context."},
        {"role": "user", "content": prompt},
    ]
    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=1024,
        temperature=0.0,
    )
    choices = resp.get("choices", [])
    if choices and isinstance(choices, list):
        first = choices[0]
        if "message" in first and isinstance(first["message"], dict) and "content" in first["message"]:
            return first["message"]["content"]
        if "text" in first and isinstance(first["text"], str):
            return first["text"]
    return json.dumps(resp, ensure_ascii=False)

# --- Flask routes ---
@app.route("/", methods=["GET"])
def home():
    html = """
    <!doctype html>
    <html>
      <head><meta charset="utf-8"><title>Backend is running</title></head>
      <body>
        <h1>Flask backend is running</h1>
        <p>This server exposes a POST endpoint <code>/api/query</code> for the chatbot.</p>
      </body>
    </html>
    """
    return Response(html, content_type="text/html; charset=utf-8")

@app.route("/favicon.ico")
def favicon():
    return Response(status=204)

@app.route("/api/query", methods=["GET"])
def api_query_get():
    return jsonify({"error": "POST JSON required"}), 400

@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.get_json() or {}
    question = (data.get("question") or "").strip()
    top_k = int(data.get("top_k", TOP_K))
    if not question:
        return jsonify({"error": "Missing question"}), 400

    if is_greeting(question):
        return jsonify({"answer": "Hi! 👋 I'm an AI chatbot about Omar Dalal. Ask me anything!"})

    work_query = is_work_intent(question)
    retrieval_k = top_k if not work_query else max(top_k, 5)

    try:
        idxs, scores = get_top_k(question, retrieval_k)
        idxs = list(idxs)

        if work_query:
            for fid in ("humly", "outliar"):
                found_index = next((i for i, it in enumerate(kb_items) if it.get("id") == fid), None)
                if found_index is not None and found_index not in idxs:
                    idxs.append(found_index)
            idxs = idxs[:retrieval_k]

        top_items = [kb_items[i] for i in idxs]
        context_parts = [f"{it.get('title', '')}: {it.get('summary','')}\n\n{it.get('text','')}" for it in top_items]
        context_str = "\n\n---\n\n".join(context_parts)[:MAX_CONTEXT_CHARS]

        prompt = f"You are an assistant using ONLY the context below.\n\n{context_str}\n\nUser question: \"{question}\""
        stdout = call_openai_chat(prompt, model=OPENAI_MODEL, timeout=OLLAMA_TIMEOUT).strip()

        return jsonify({"answer": stdout, "context_used": context_parts})
    except Exception as e:
        logger.exception("Unhandled error")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5174))
    app.run(host=host, port=port, debug=False)
