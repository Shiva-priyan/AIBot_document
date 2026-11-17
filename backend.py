import json
import os
import re
import shutil
import tempfile
import time
from collections import defaultdict
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ====================================================================
# SETTINGS  (set in .env)
# ====================================================================
RAG_MODE = "hybrid"             # fixed after evaluation: keyword (BM25) + vector search
TOP_K = int(os.getenv("TOP_K", "4"))
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
MAX_ANALYSIS_CHUNKS = 30        # passages sampled for summary/analysis tasks

EMBEDDING_CONFIG = {
    "openai": {"key_env": "OPENAI_API_KEY", "model": "text-embedding-3-small"},
    "gemini": {"key_env": "GOOGLE_API_KEY", "model": "models/gemini-embedding-001"},
}
if EMBEDDING_PROVIDER not in EMBEDDING_CONFIG:
    raise RuntimeError("EMBEDDING_PROVIDER in .env must be 'openai' or 'gemini'.")

KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GOOGLE_API_KEY"}

# Edit this list to change which models users can pick (model ids change over time)
MODEL_CATALOG = [
    {"label": "GPT-4.1", "id": "gpt-4.1", "provider": "openai"},
    {"label": "GPT-4o", "id": "gpt-4o", "provider": "openai"},
    {"label": "GPT-4o mini", "id": "gpt-4o-mini", "provider": "openai"},
    {"label": "Claude Sonnet 5.5", "id": "claude-sonnet-5-5", "provider": "anthropic"},
    {"label": "Gemini 2.5 Flash", "id": "gemini-2.5-flash", "provider": "gemini"},
]

LANGUAGES = {
    "English": "English",
    "Tamil (தமிழ்)": "Tamil, written in Tamil script",
    "Tanglish": "Tanglish: Tamil written in English (Roman) letters, the casual way "
                "Tamil speakers type Tamil in chat",
}

FAISS_FOLDER = "faiss_index"
MARKER_FILE = os.path.join(FAISS_FOLDER, "provider.txt")
LOG_FILE = os.path.join("logs", "queries.jsonl")
SUPPORTED = {".pdf", ".docx", ".txt", ".md"}

app = FastAPI(title="AI Document Analyser API", version="4.0")

# ====================================================================
# PROMPTS
# ====================================================================
REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """Rewrite the user's latest question as a standalone search query in English.
- Use the chat history to resolve words like "it", "its" or "that scheme".
- If the question is in Tamil or Tanglish, translate it to English.
- Do NOT answer it. Return only the query.

Chat history:
{history}

Latest question: {question}

Standalone English query:"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant that answers questions about the user's documents.
Answer ONLY from the context below. Be clear and well organised, and mention the
document name when it helps. Write your answer in {language_rule}.
If the answer is not in the context, say (in that language) that you cannot find
that information in the uploaded documents.

Chat history:
{history}

Context:
{context}

Question: {question}
"""
)

TASKS = {
    "summary": "Write a concise, meaningful summary of the document in about 150 to 250 "
               "words. Start with one sentence stating what the document is and its purpose.",
    "key_points": "List the 6 to 10 most important key points as short bullet points, "
                  "most important first.",
    "dates": "Extract all important dates, deadlines, time periods and durations. Present "
             "them as a table with columns: Date or period | What it refers to. If there "
             "are none, say so.",
    "structured": "Provide a structured summary with these headings: Purpose, Who it applies "
                  "to, Key provisions, Important numbers (amounts, percentages, quantities), "
                  "Important dates, Responsibilities, Conditions or exclusions. Write "
                  "'Not mentioned' for any heading the text does not cover.",
    "analyse": "Extract key information under these headings: Document type and issuer, "
               "Main subject, Key entities (organisations, schemes, people, places), Amounts "
               "and percentages, Important dates, Important clauses / rules / eligibility "
               "conditions, Obligations or responsibilities. Mention clause numbers when the "
               "text shows them. Write 'Not mentioned' where the text has nothing.",
    "insights": "Give insights under these headings: Summary (3 sentences), Key points "
                "(5 bullets), Things to watch (risks, gaps or conditions that are easy to "
                "miss), Suggested follow-up questions (3 questions the reader could ask next).",
}

TASK_PROMPT = ChatPromptTemplate.from_template(
    """You are a careful document analyst. {instruction}

Use ONLY the document text below. Do not add outside knowledge.
Write the answer in {language_rule}.
{sample_note}

Document text:
{text}
"""
)


# ====================================================================
# REQUEST MODELS
# ====================================================================
class Message(BaseModel):
    role: str
    content: str


class RetrieveRequest(BaseModel):
    question: str
    history: List[Message] = []
    top_k: Optional[int] = None
    model_name: str = ""
    document: str = ""            # empty = search all documents
    language: str = "English"


class GenerateRequest(BaseModel):
    question: str
    context: str
    history: List[Message] = []
    model_name: str = ""
    language: str = "English"


class AnalyzeRequest(BaseModel):
    task: str
    document: str = ""            # empty = all documents
    model_name: str = ""
    language: str = "English"


# ====================================================================
# MODEL / EMBEDDING HELPERS
# ====================================================================
def available_models():
    return [m for m in MODEL_CATALOG if os.getenv(KEY_ENV[m["provider"]])]


def default_model_id() -> str:
    models = available_models()
    if not models:
        raise HTTPException(500, "No chat model available. Add an API key to the .env file.")
    return models[0]["id"]


def get_llm(model_id: str):
    model_id = model_id or default_model_id()
    entry = next((m for m in MODEL_CATALOG if m["id"] == model_id), None)
    if entry is None:
        raise HTTPException(400, f"Unknown model '{model_id}'.")
    key_name = KEY_ENV[entry["provider"]]
    if not os.getenv(key_name):
        raise HTTPException(500, f"{key_name} not found in .env file.")
    if entry["provider"] == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_id, temperature=0)
    if entry["provider"] == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_id, temperature=0)
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model_id, temperature=0)


def get_embeddings():
    cfg = EMBEDDING_CONFIG[EMBEDDING_PROVIDER]
    if not os.getenv(cfg["key_env"]):
        raise HTTPException(500, f"{cfg['key_env']} not found in .env file.")
    if EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=cfg["model"])
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(model=cfg["model"])


# ====================================================================
# KNOWLEDGE BASE HELPERS
# ====================================================================
def index_exists() -> bool:
    return os.path.exists(os.path.join(FAISS_FOLDER, "index.faiss"))


def check_embedding_provider():
    if os.path.exists(MARKER_FILE):
        with open(MARKER_FILE, encoding="utf-8") as f:
            built_with = f.read().strip()
        if built_with != EMBEDDING_PROVIDER:
            raise HTTPException(
                409,
                f"The knowledge base was built with '{built_with}' embeddings, but "
                f"EMBEDDING_PROVIDER is '{EMBEDDING_PROVIDER}'. Remove all documents "
                f"(or delete the faiss_index folder) and upload them again.",
            )


def load_db():
    if not index_exists():
        raise HTTPException(404, "No documents yet. Upload a document first.")
    check_embedding_provider()
    return FAISS.load_local(FAISS_FOLDER, get_embeddings(),
                            allow_dangerous_deserialization=True)


def doc_summary(db) -> dict:
    summary = {}
    for doc in db.docstore._dict.values():
        name = doc.metadata.get("source_label", "Unknown")
        summary[name] = summary.get(name, 0) + 1
    return summary


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_loader(path: str, ext: str):
    if ext == ".pdf":
        return PyPDFLoader(path)
    if ext == ".docx":
        return Docx2txtLoader(path)
    return TextLoader(path, encoding="utf-8")


def save_docs_to_faiss(docs, source_label: str) -> int:
    """Clean -> split -> embed -> store (merging into the existing index)."""
    for d in docs:
        d.page_content = clean_text(d.page_content)
        d.metadata["source_label"] = source_label

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("No readable text found (is it a scanned/image-only file?).")

    embeddings = get_embeddings()
    if index_exists():
        check_embedding_provider()
        db = FAISS.load_local(FAISS_FOLDER, embeddings, allow_dangerous_deserialization=True)
        db.merge_from(FAISS.from_documents(chunks, embeddings))
    else:
        db = FAISS.from_documents(chunks, embeddings)

    db.save_local(FAISS_FOLDER)
    with open(MARKER_FILE, "w", encoding="utf-8") as f:
        f.write(EMBEDDING_PROVIDER)
    return len(chunks)


def format_history(history: List[Message]) -> str:
    if not history:
        return "(no previous messages)"
    return "\n".join(f"{m.role.capitalize()}: {m.content}" for m in history)


# ====================================================================
# RETRIEVAL: hybrid search (BM25 keywords + vector similarity, merged with RRF)
# Takes (db, query, k, llm, flt) and returns [(document, score)]
# flt = {"source_label": name} to search one document, or None for all
# ====================================================================
def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def fk(db, flt):
    """Extra search arguments when searching a single document."""
    return {"filter": flt, "fetch_k": max(db.index.ntotal, 1)} if flt else {}


def retrieve_hybrid(db, q, k, llm, flt):
    """Keyword (BM25) + vector search, merged with Reciprocal Rank Fusion."""
    all_docs = [d for d in db.docstore._dict.values()
                if not flt or d.metadata.get("source_label") == flt["source_label"]]
    if not all_docs:
        return []
    bm25 = BM25Okapi([tokenize(d.page_content) for d in all_docs])
    kw_scores = bm25.get_scores(tokenize(q))
    kw_rank = sorted(range(len(all_docs)), key=lambda i: kw_scores[i], reverse=True)[: k * 3]
    vec_docs = db.similarity_search(q, k=k * 3, **fk(db, flt))

    fused, by_text = defaultdict(float), {}
    for rank, i in enumerate(kw_rank):
        text = all_docs[i].page_content
        by_text[text] = all_docs[i]
        fused[text] += 1 / (60 + rank)
    for rank, d in enumerate(vec_docs):
        by_text[d.page_content] = d
        fused[d.page_content] += 1 / (60 + rank)

    top = sorted(fused, key=fused.get, reverse=True)[:k]
    return [(by_text[t], round(fused[t], 4)) for t in top]      # higher is better


def log_query(entry: dict):
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ====================================================================
# DOCUMENT MANAGEMENT
# ====================================================================
@app.post("/documents")
def upload_documents(files: List[UploadFile] = File(...)):
    existing = set(doc_summary(load_db()).keys()) if index_exists() else set()
    results = []

    for f in files:
        name = f.filename
        ext = os.path.splitext(name)[1].lower()

        if ext not in SUPPORTED:
            results.append({"file": name, "status": "error", "detail": f"Unsupported type {ext}"})
            continue
        if name in existing:
            results.append({"file": name, "status": "skipped",
                            "detail": "Already in the knowledge base."})
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:   # Windows-safe
            tmp.write(f.file.read())
            tmp_path = tmp.name
        try:
            docs = get_loader(tmp_path, ext).load()
            count = save_docs_to_faiss(docs, name)
            existing.add(name)
            results.append({"file": name, "status": "added", "chunks": count})
        except Exception as e:
            results.append({"file": name, "status": "error",
                            "detail": str(getattr(e, "detail", e))})
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return {"results": results}


@app.get("/documents")
def list_documents():
    if not index_exists():
        return {"documents": []}
    summary = doc_summary(load_db())
    return {"documents": [{"name": n, "chunks": c} for n, c in sorted(summary.items())]}


@app.get("/documents/{name}/text")
def document_text(name: str):
    """Full extracted text of one document (used by validate_eval.py)."""
    db = load_db()
    parts = [d.page_content for d in db.docstore._dict.values()
             if d.metadata.get("source_label") == name]
    if not parts:
        raise HTTPException(404, f"'{name}' not found.")
    return {"name": name, "text": "\n".join(parts)}


@app.delete("/documents/{name}")
def delete_document(name: str):
    db = load_db()
    ids = [i for i, d in db.docstore._dict.items() if d.metadata.get("source_label") == name]
    if not ids:
        raise HTTPException(404, f"'{name}' not found.")
    db.delete(ids)
    if len(db.docstore._dict) == 0:
        shutil.rmtree(FAISS_FOLDER, ignore_errors=True)
    else:
        db.save_local(FAISS_FOLDER)
    return {"status": "deleted", "name": name}


# ====================================================================
# CHAT: STEP 1, RETRIEVE
# ====================================================================
@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    db = load_db()
    flt = None
    if req.document:
        if req.document not in doc_summary(db):
            raise HTTPException(404, f"Document '{req.document}' not found.")
        flt = {"source_label": req.document}

    llm = get_llm(req.model_name)
    k = req.top_k or TOP_K
    start = time.time()

    # Rewrite when needed: follow-up questions, Tamil/Tanglish, or non-English letters
    standalone = req.question
    if req.history or req.language != "English" or not req.question.isascii():
        try:
            standalone = (REWRITE_PROMPT | llm | StrOutputParser()).invoke(
                {"history": format_history(req.history), "question": req.question}
            ).strip() or req.question
        except Exception:
            standalone = req.question

    results = retrieve_hybrid(db, standalone, k, llm, flt)
    latency_ms = int((time.time() - start) * 1000)

    chunks = []
    for doc, score in results:
        page = doc.metadata.get("page")
        chunks.append({
            "source": doc.metadata.get("source_label", "Unknown"),
            "page": page + 1 if isinstance(page, int) else "-",
            "score": score,
            "content": doc.page_content,
        })

    log_query({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": RAG_MODE,
        "question": req.question,
        "searched_for": standalone,
        "document": req.document or "ALL",
        "latency_ms": latency_ms,
        "sources": [f"{c['source']} p{c['page']}" for c in chunks],
    })
    return {"standalone_question": standalone, "rag_mode": RAG_MODE,
            "latency_ms": latency_ms, "retrieved_chunks": chunks}


# ====================================================================
# CHAT: STEP 2, GENERATE (STREAMED)
# ====================================================================
@app.post("/generate-stream")
def generate_stream(req: GenerateRequest):
    llm = get_llm(req.model_name)
    chain = ANSWER_PROMPT | llm | StrOutputParser()
    inputs = {
        "history": format_history(req.history),
        "context": req.context,
        "question": req.question,
        "language_rule": LANGUAGES.get(req.language, "English"),
    }

    def token_gen():
        for token in chain.stream(inputs):
            yield token

    return StreamingResponse(token_gen(), media_type="text/plain")


# ====================================================================
# DOCUMENT TASKS: summary, key points, dates, analysis, insights
# ====================================================================
@app.post("/analyze-stream")
def analyze_stream(req: AnalyzeRequest):
    if req.task not in TASKS:
        raise HTTPException(400, f"Unknown task. Use one of: {list(TASKS)}")

    db = load_db()
    docs = [d for d in db.docstore._dict.values()
            if not req.document or d.metadata.get("source_label") == req.document]
    if not docs:
        raise HTTPException(404, "Document not found.")

    sampled = len(docs) > MAX_ANALYSIS_CHUNKS
    if sampled:                                   # evenly spaced sample across the document
        step = len(docs) / MAX_ANALYSIS_CHUNKS
        docs = [docs[int(i * step)] for i in range(MAX_ANALYSIS_CHUNKS)]

    parts = []
    for d in docs:
        page = d.metadata.get("page")
        page = page + 1 if isinstance(page, int) else "-"
        parts.append(f"[{d.metadata.get('source_label')} · page {page}]\n{d.page_content}")

    sample_note = (
        "NOTE: The text below is an evenly spaced sample of a much longer document, so some "
        "details may be missing. Say this briefly at the end of your answer."
        if sampled else ""
    )

    llm = get_llm(req.model_name)
    chain = TASK_PROMPT | llm | StrOutputParser()
    inputs = {
        "instruction": TASKS[req.task],
        "language_rule": LANGUAGES.get(req.language, "English"),
        "sample_note": sample_note,
        "text": "\n\n".join(parts),
    }

    def token_gen():
        for token in chain.stream(inputs):
            yield token

    return StreamingResponse(token_gen(), media_type="text/plain")


# ====================================================================
# UTILITIES
# ====================================================================
@app.get("/status")
def status():
    key_env = EMBEDDING_CONFIG[EMBEDDING_PROVIDER]["key_env"]
    return {
        "embedding_provider": EMBEDDING_PROVIDER,
        "key_name": key_env,
        "api_key_loaded": bool(os.getenv(key_env)),
        "faiss_ready": index_exists(),
        "models": [{"label": m["label"], "id": m["id"]} for m in available_models()],
        "languages": list(LANGUAGES),
        "rag_mode": RAG_MODE,
    }


@app.delete("/reset")
def reset():
    shutil.rmtree(FAISS_FOLDER, ignore_errors=True)
    return {"status": "deleted"}