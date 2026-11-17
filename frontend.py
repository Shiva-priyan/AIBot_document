import os
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"
MAX_HISTORY = 6
ALL = "All documents"            # internal value, shown translated via format_func
UPLOAD_TYPES = ["pdf", "docx", "txt", "md"]
TASK_KEYS = ["summary", "key_points", "dates", "structured", "analyse", "insights"]
CHIP_TASKS = ["summary", "key_points", "dates", "structured"]
CARD_KEYS = [("analyse", "🧠"), ("search", "🔎"), ("insights", "📊"), ("summary", "📝")]

# ====================================================================
# SCREEN TEXT IN EACH LANGUAGE  (the whole screen follows the Language dropdown)
# Have a Tamil speaker review these once; edit any text here freely.
# ====================================================================
UI = {
    "English": {
        "brand": "AI Document<br>Analyser",
        "upload_btn": "⬆️  Upload Document",
        "kb_btn": "🗄️  Add to AI Knowledge",
        "kb_count": "📚 {n} document(s) in the knowledge base",
        "model": "🤖 AI Model",
        "lang_pref": "🌐 Language Preference",
        "banner_title": "AI-Powered Document Analysis",
        "banner_text": "Upload, analyse, build a knowledge base and get insights from your documents.",
        "cards": {
            "analyse": ("Analyse Document", "Extract key information, entities and important clauses."),
            "search": ("Search in Documents", "Ask questions and get contextual answers from your documents."),
            "insights": ("Document Insights", "Generate summary, key points and structured outputs."),
            "summary": ("Document Summary", "Create concise and meaningful summaries."),
        },
        "open": "Open ▸",
        "toast_search": "Type your question in the box below 👇",
        "search_in": "Search in",
        "all_docs": "All documents",
        "clear": "🗑️ Clear Chat",
        "placeholder": "Ask a question about your documents...",
        "try_asking": "Try asking:",
        "chips": {
            "summary": "Summarize this document.",
            "key_points": "List the key points.",
            "dates": "Extract important dates.",
            "structured": "Provide a structured summary.",
        },
        "tasks": {
            "summary": "Document Summary", "key_points": "Key Points",
            "dates": "Important Dates", "structured": "Structured Summary",
            "analyse": "Document Analysis", "insights": "Document Insights",
        },
        "sources": "📚 Sources",
        "searching": "Searching your documents...",
        "need_doc": "👈 Upload a document from the sidebar to start. The cards and questions "
                    "below become active once a document is added.",
        "lang_changed": "Language changed. Re-writing the last answer...",
        "up_title": "Upload Document",
        "up_caption": "PDF, DOCX or TXT. The file is read, split into passages and saved, "
                      "so you can analyse it and ask questions about it.",
        "choose": "Choose files",
        "process": "Process",
        "working": "Reading → splitting → embedding → saving...",
        "kb_title": "AI Knowledge Base",
        "kb_caption": "Add documents the AI can search, or remove ones you no longer need.",
        "add_docs": "Add documents",
        "kb_add": "Add to AI Knowledge",
        "docs_n": "Documents ({n})",
        "no_docs": "No documents yet.",
        "remove_all": "Remove all documents",
    },
    "Tamil (தமிழ்)": {
        "brand": "AI ஆவண<br>பகுப்பாய்வி",
        "upload_btn": "⬆️  ஆவணத்தைப் பதிவேற்று",
        "kb_btn": "🗄️  AI அறிவுத் தளத்தில் சேர்",
        "kb_count": "📚 அறிவுத் தளத்தில் {n} ஆவணம்(கள்)",
        "model": "🤖 AI மாதிரி",
        "lang_pref": "🌐 மொழி விருப்பம்",
        "banner_title": "AI ஆற்றல் கொண்ட ஆவணப் பகுப்பாய்வு",
        "banner_text": "உங்கள் ஆவணங்களைப் பதிவேற்றி, பகுப்பாய்வு செய்து, அறிவுத் தளத்தை "
                       "உருவாக்கி, நுண்ணறிவுகளைப் பெறுங்கள்.",
        "cards": {
            "analyse": ("ஆவணத்தை ஆய்வு செய்", "முக்கிய தகவல்கள், அமைப்புகள் மற்றும் முக்கிய விதிகளை எடுக்கவும்."),
            "search": ("ஆவணங்களில் தேடு", "கேள்விகள் கேட்டு, உங்கள் ஆவணங்களிலிருந்து பதில்களைப் பெறுங்கள்."),
            "insights": ("ஆவண நுண்ணறிவு", "சுருக்கம், முக்கிய புள்ளிகள் மற்றும் கட்டமைக்கப்பட்ட முடிவுகளை உருவாக்கவும்."),
            "summary": ("ஆவணச் சுருக்கம்", "சுருக்கமான, பொருள் நிறைந்த சுருக்கங்களை உருவாக்கவும்."),
        },
        "open": "திற ▸",
        "toast_search": "உங்கள் கேள்வியைக் கீழே உள்ள பெட்டியில் தட்டச்சு செய்யுங்கள் 👇",
        "search_in": "இதில் தேடு",
        "all_docs": "அனைத்து ஆவணங்களும்",
        "clear": "🗑️ உரையாடலை அழி",
        "placeholder": "உங்கள் ஆவணங்கள் பற்றிக் கேளுங்கள்...",
        "try_asking": "முயற்சி செய்யுங்கள்:",
        "chips": {
            "summary": "இந்த ஆவணத்தைச் சுருக்கவும்.",
            "key_points": "முக்கிய புள்ளிகளைப் பட்டியலிடுங்கள்.",
            "dates": "முக்கியமான தேதிகளை எடுக்கவும்.",
            "structured": "கட்டமைக்கப்பட்ட சுருக்கம் தாருங்கள்.",
        },
        "tasks": {
            "summary": "ஆவணச் சுருக்கம்", "key_points": "முக்கிய புள்ளிகள்",
            "dates": "முக்கிய தேதிகள்", "structured": "கட்டமைக்கப்பட்ட சுருக்கம்",
            "analyse": "ஆவண ஆய்வு", "insights": "ஆவண நுண்ணறிவு",
        },
        "sources": "📚 ஆதாரங்கள்",
        "searching": "உங்கள் ஆவணங்களில் தேடுகிறது...",
        "need_doc": "👈 தொடங்க, பக்கப்பட்டியிலிருந்து ஒரு ஆவணத்தைப் பதிவேற்றுங்கள். "
                    "ஆவணம் சேர்த்த பிறகு கீழுள்ள அட்டைகளும் கேள்விகளும் செயல்படும்.",
        "lang_changed": "மொழி மாற்றப்பட்டது. கடைசி பதிலை மீண்டும் எழுதுகிறது...",
        "up_title": "ஆவணத்தைப் பதிவேற்று",
        "up_caption": "PDF, DOCX அல்லது TXT. கோப்பு படிக்கப்பட்டு, பகுதிகளாகப் பிரிக்கப்பட்டு "
                      "சேமிக்கப்படும்; பிறகு அதை ஆய்வு செய்யவும் கேள்விகள் கேட்கவும் முடியும்.",
        "choose": "கோப்புகளைத் தேர்ந்தெடுக்கவும்",
        "process": "செயலாக்கு",
        "working": "படிக்கிறது → பிரிக்கிறது → உட்பொதிக்கிறது → சேமிக்கிறது...",
        "kb_title": "AI அறிவுத் தளம்",
        "kb_caption": "AI தேடக்கூடிய ஆவணங்களைச் சேர்க்கவும், தேவையற்றவற்றை நீக்கவும்.",
        "add_docs": "ஆவணங்களைச் சேர்க்கவும்",
        "kb_add": "AI அறிவுத் தளத்தில் சேர்",
        "docs_n": "ஆவணங்கள் ({n})",
        "no_docs": "இன்னும் ஆவணங்கள் இல்லை.",
        "remove_all": "அனைத்து ஆவணங்களையும் நீக்கு",
    },
    "Tanglish": {
        "brand": "AI Document<br>Analyser",
        "upload_btn": "⬆️  Document Upload Pannu",
        "kb_btn": "🗄️  AI Knowledge-il Serkka",
        "kb_count": "📚 Knowledge base-il {n} document(s)",
        "model": "🤖 AI Model",
        "lang_pref": "🌐 Mozhi Virupam",
        "banner_title": "AI-yaal Document Analysis",
        "banner_text": "Ungal documents-ai upload panni, analyse panni, knowledge base uruvaakki, "
                       "insights petrukkollungal.",
        "cards": {
            "analyse": ("Document-ai Analyse Pannu", "Mukkiyamaana thagavalgal, entities, clauses-ai edukkavum."),
            "search": ("Documents-il Thedu", "Kelvigal kettu, ungal documents-il irundhu badhilgal pettrukkollungal."),
            "insights": ("Document Insights", "Summary, key points, structured output-gal uruvaakkavum."),
            "summary": ("Document Summary", "Surukkamaana, arthamulla summary-gal uruvaakkavum."),
        },
        "open": "Thira ▸",
        "toast_search": "Ungal kelviyai keezhe ulla box-il type pannunga 👇",
        "search_in": "Edhil thedanum",
        "all_docs": "Ella documents-um",
        "clear": "🗑️ Chat-ai Azhi",
        "placeholder": "Ungal documents pathi kelungal...",
        "try_asking": "Try pannunga:",
        "chips": {
            "summary": "Indha document-ai summarize pannunga.",
            "key_points": "Key points-ai list pannunga.",
            "dates": "Mukkiyamaana dates-ai edungal.",
            "structured": "Structured summary kodungal.",
        },
        "tasks": {
            "summary": "Document Summary", "key_points": "Key Points",
            "dates": "Mukkiya Dates", "structured": "Structured Summary",
            "analyse": "Document Analysis", "insights": "Document Insights",
        },
        "sources": "📚 Aadharangal",
        "searching": "Ungal documents-il thedugiradhu...",
        "need_doc": "👈 Aarambikka, sidebar-il irundhu oru document upload pannunga. "
                    "Document serthadhum keezhe ulla cards-um kelvigalum velai seyyum.",
        "lang_changed": "Mozhi maaritru. Kadaisi badhilai meendum ezhudhugiradhu...",
        "up_title": "Document Upload Pannu",
        "up_caption": "PDF, DOCX alladhu TXT. File padikkappattu, pirikkappattu, semikkappadum; "
                      "adhan piragu analyse pannalaam, kelvigal kekkalaam.",
        "choose": "Files-ai therndhedungal",
        "process": "Process Pannu",
        "working": "Padikkiradhu → pirikkiradhu → embed seigiradhu → semikkiradhu...",
        "kb_title": "AI Knowledge Base",
        "kb_caption": "AI thedakkoodiya documents-ai serkkavum, thevaiyillaadhavatrai neekkavum.",
        "add_docs": "Documents serkkavum",
        "kb_add": "AI Knowledge-il Serkka",
        "docs_n": "Documents ({n})",
        "no_docs": "Innum documents illai.",
        "remove_all": "Ella documents-ayum neekku",
    },
}

st.set_page_config(page_title="AI Document Analyser", page_icon="📄", layout="wide")
admin = st.query_params.get("admin") == "1"      # open  http://localhost:8501/?admin=1

# The language widget (key="language") lives in the sidebar, but its value is already in
# session_state at the top of every run, so the whole screen can follow it.
language = st.session_state.get("language", "English")
T = UI.get(language, UI["English"])

CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1250px;}
header[data-testid="stHeader"] {background: transparent;}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {background: linear-gradient(180deg,#0a1235 0%,#101a4d 55%,#1d1458 100%);}
[data-testid="stSidebar"] * {color: #e9edff;}
[data-testid="stSidebar"] hr {border-color: rgba(160,170,255,.25);}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,.09); border: 1px solid rgba(160,170,255,.35); border-radius: 10px;}
[data-testid="stSidebar"] [data-baseweb="select"] svg {fill: #e9edff;}
[data-testid="stSidebar"] .stButton > button {
    border-radius: 12px; padding: .7rem 1rem; font-weight: 600; width: 100%;
    border: 1px solid rgba(160,170,255,.3); background: rgba(255,255,255,.06); color: #fff;}
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(90deg,#7b2ff7,#a14df5); border: none;
    box-shadow: 0 6px 18px rgba(123,47,247,.45);}
.brand {display:flex; align-items:center; gap:.7rem; font-size:1.25rem; line-height:1.25;
    font-weight:700; margin: .2rem 0 1.2rem 0;}
.brand .logo {font-size:2.2rem;}
.side-title {font-weight:600; margin: .6rem 0 .3rem 0;}

/* ---------- banner ---------- */
.banner {background: linear-gradient(100deg,#1536c9 0%,#5b2bd6 55%,#8a3ff0 100%);
    border-radius: 26px; padding: 2.1rem 2.6rem; color: #fff; display: flex;
    justify-content: space-between; align-items: center; margin-bottom: 1.2rem;}
.banner h1 {color:#fff; font-size: 2.2rem; margin: 0 0 .5rem 0; padding: 0;}
.banner p {color:#e2e6ff; font-size: 1.08rem; margin: 0; max-width: 520px;}
.banner .art {font-size: 4.2rem; letter-spacing: .3rem;}

/* ---------- cards ---------- */
[class*="st-key-card_"] {border-radius: 20px; padding: 1.1rem 1.2rem; min-height: 215px;}
.st-key-card_analyse {background: #dff3f3;}
.st-key-card_search {background: #fdebd6;}
.st-key-card_insights {background: #ebe4fb;}
.st-key-card_summary {background: #dff0fb;}
[class*="st-key-card_"] h4 {margin: .3rem 0 .3rem 0; color: #14213d;}
[class*="st-key-card_"] p {color: #33405c; font-size: .93rem;}
.card-icon {font-size: 2rem;}

/* ---------- pill buttons in the main area (readable on any theme) ---------- */
[data-testid="stMain"] .stButton > button {
    border-radius: 999px; font-weight: 600; white-space: normal; height: auto;
    min-height: 2.5rem; line-height: 1.3;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- language just changed? take the last exchange back so it can be rewritten ----
previous_language = st.session_state.get("prev_language", language)
st.session_state.prev_language = language
if previous_language != language and st.session_state.chat_history:
    last = st.session_state.chat_history[-1]
    if last["role"] == "assistant" and last.get("replay"):
        st.session_state.replay_backup = st.session_state.chat_history[-2:]
        del st.session_state.chat_history[-2:]
        st.session_state.replay = last["replay"]


# ====================================================================
# HELPERS
# ====================================================================
def error_detail(resp, default="Request failed"):
    try:
        return resp.json().get("detail", default)
    except Exception:
        return resp.text[:200] or default


def stream_from(path, payload):
    with requests.post(f"{BACKEND_URL}{path}", json=payload, stream=True, timeout=300) as r:
        if r.status_code != 200:
            raise RuntimeError(error_detail(r))
        r.encoding = "utf-8"
        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk


def upload_files(files):
    payload = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
               for f in files]
    try:
        r = requests.post(f"{BACKEND_URL}/documents", files=payload, timeout=900)
        if r.status_code == 200:
            return r.json()["results"]
        return [{"file": "-", "status": "error", "detail": error_detail(r)}]
    except Exception as e:
        return [{"file": "-", "status": "error", "detail": str(e)}]


def render_sources(sources, debug=None):
    if not sources:
        return
    with st.expander(T["sources"]):
        for i, c in enumerate(sources, 1):
            score = f" · score {c['score']}" if admin and c.get("score") is not None else ""
            st.markdown(f"**{i}. {c['source']}** · page {c['page']}{score}")
            st.text(c["content"])
        if admin and debug:
            st.caption(f"🛠 RAG mode: {debug['rag_mode']} · retrieval {debug['latency_ms']} ms · "
                       f"searched for: {debug['standalone_question']}")


def scope_label(name):
    return T["all_docs"] if name == ALL else name


# ====================================================================
# LOAD BACKEND INFO
# ====================================================================
documents, backend_ok, backend_error, status = [], True, None, {}
try:
    status = requests.get(f"{BACKEND_URL}/status", timeout=3).json()
    r = requests.get(f"{BACKEND_URL}/documents", timeout=10)
    if r.status_code == 200:
        documents = r.json()["documents"]
    else:
        backend_error = error_detail(r)
except requests.exceptions.ConnectionError:
    backend_ok = False
except Exception as e:
    backend_error = str(e)

models = status.get("models", [])
languages = status.get("languages", list(UI))
doc_names = [d["name"] for d in documents]
scope_options = [ALL] + doc_names


# ====================================================================
# DIALOGS
# ====================================================================
@st.dialog(T["up_title"])
def upload_dialog():
    st.caption(T["up_caption"])
    files = st.file_uploader(T["choose"], type=UPLOAD_TYPES, accept_multiple_files=True)
    if st.button(T["process"], type="primary", disabled=not files):
        with st.spinner(T["working"]):
            results = upload_files(files)
        st.session_state.upload_results = results
        done = [x["file"] for x in results if x["status"] in ("added", "skipped")]
        if done:
            st.session_state.next_scope = done[0]      # make it the current document
        st.rerun()


@st.dialog(T["kb_title"], width="large")
def knowledge_dialog():
    st.caption(T["kb_caption"])
    files = st.file_uploader(T["add_docs"], type=UPLOAD_TYPES,
                             accept_multiple_files=True, key="kb_files")
    if st.button(T["kb_add"], type="primary", disabled=not files, key="kb_add"):
        with st.spinner(T["working"]):
            st.session_state.upload_results = upload_files(files)
        st.rerun()

    st.divider()
    st.markdown(f"**{T['docs_n'].format(n=len(documents))}**")
    if not documents:
        st.caption(T["no_docs"])
    for d in documents:
        c1, c2 = st.columns([6, 1])
        c1.markdown(f"📄 {d['name']}  \n`{d['chunks']} passages`")
        if c2.button("🗑️", key=f"del_{d['name']}", help=f"Remove {d['name']}"):
            resp = requests.delete(f"{BACKEND_URL}/documents/{quote(d['name'], safe='')}")
            if resp.status_code == 200:
                st.rerun()
            st.error(error_detail(resp))
    if documents and st.button(T["remove_all"], key="kb_reset"):
        requests.delete(f"{BACKEND_URL}/reset")
        st.rerun()


# ====================================================================
# SIDEBAR
# ====================================================================
with st.sidebar:
    st.markdown(f'<div class="brand"><span class="logo">📄</span><div>{T["brand"]}</div></div>',
                unsafe_allow_html=True)

    if st.button(T["upload_btn"], type="primary", disabled=not backend_ok):
        upload_dialog()
    if st.button(T["kb_btn"], disabled=not backend_ok):
        knowledge_dialog()
    st.caption(T["kb_count"].format(n=len(documents)))

    st.divider()
    st.markdown(f'<div class="side-title">{T["model"]}</div>', unsafe_allow_html=True)
    model_label = (st.selectbox("AI Model", [m["label"] for m in models],
                                label_visibility="collapsed") if models else None)
    st.markdown(f'<div class="side-title">{T["lang_pref"]}</div>', unsafe_allow_html=True)
    st.selectbox("Language", languages, key="language", label_visibility="collapsed")

model_id = next((m["id"] for m in models if m["label"] == model_label), "")


# ====================================================================
# MAIN: BANNER + STATUS MESSAGES
# ====================================================================
st.markdown(
    f'<div class="banner"><div><h1>{T["banner_title"]}</h1>'
    f'<p>{T["banner_text"]}</p></div><div class="art">📄🔍</div></div>',
    unsafe_allow_html=True)

if not backend_ok:
    st.error("Backend offline. Start FastAPI on port 8000.")
elif backend_error:
    st.error(f"Backend error: {backend_error}")
elif not status.get("api_key_loaded"):
    st.error(f"{status.get('key_name', 'API key')} missing in .env (needed for embeddings).")
elif not models:
    st.error("No chat model available. Add OPENAI_API_KEY, ANTHROPIC_API_KEY or "
             "GOOGLE_API_KEY to the .env file.")

for res in st.session_state.pop("upload_results", None) or []:
    if res["status"] == "added":
        st.success(f"✅ {res['file']} added ({res['chunks']} passages)")
    elif res["status"] == "skipped":
        st.info(f"⏭️ {res['file']}: {res['detail']}")
    else:
        st.error(f"❌ {res['file']}: {res['detail']}")

# Apply a scope chosen in a dialog BEFORE the scope widget is created
if "next_scope" in st.session_state:
    st.session_state.scope = st.session_state.pop("next_scope")
if st.session_state.get("scope") not in scope_options:
    st.session_state.scope = ALL

ready = backend_ok and not backend_error and bool(documents) and bool(models)
if backend_ok and not backend_error and not documents:
    st.info(T["need_doc"])


# ====================================================================
# MAIN: FEATURE CARDS
# ====================================================================
for col, (key, icon) in zip(st.columns(4), CARD_KEYS):
    title, desc = T["cards"][key]
    with col:
        with st.container(key=f"card_{key}"):
            st.markdown(f"<div class='card-icon'>{icon}</div><h4>{title}</h4><p>{desc}</p>",
                        unsafe_allow_html=True)
            if st.button(T["open"], key=f"btn_{key}", disabled=not ready):
                if key == "search":
                    st.toast(T["toast_search"])
                else:
                    st.session_state.pending = key

st.write("")


# ====================================================================
# MAIN: CHAT PANEL
# ====================================================================
with st.container(border=True):
    head_l, head_r = st.columns([5, 1.2])
    with head_l:
        # constant widget label (a changing label would reset the selection on language change)
        st.markdown(f"**{T['search_in']}**")
        st.selectbox("Search in", scope_options, key="scope", format_func=scope_label,
                     label_visibility="collapsed")
    with head_r:
        st.write("")
        if st.button(T["clear"], use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # "Try asking" questions: shown ABOVE the chat so they are always visible
    chip_cols = st.columns([1, 2, 2, 2, 2])
    chip_cols[0].markdown(f"**{T['try_asking']}**")
    for col, task in zip(chip_cols[1:], CHIP_TASKS):
        if col.button(f"🔍 {T['chips'][task]}", key=f"chip_{task}", disabled=not ready,
                      use_container_width=True):
            st.session_state.pending = task

    history_area = st.container()
    with history_area:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                render_sources(msg.get("sources"), msg.get("debug"))

    question = st.chat_input(T["placeholder"], disabled=not ready)


# ====================================================================
# ACTIONS (run after the widgets so the answer appears in the chat area)
# ====================================================================
scope = st.session_state.scope
doc_filter = "" if scope == ALL else scope


def history_payload():
    return [{"role": m["role"], "content": m["content"][:600]}
            for m in st.session_state.chat_history[-MAX_HISTORY:]]


def run_task(task, doc):
    user_text = f"**{T['tasks'][task]}** · {scope_label(doc or ALL)}"
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(stream_from("/analyze-stream", {
                "task": task, "document": doc,
                "model_name": model_id, "language": language}))
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            return
    st.session_state.chat_history.append({"role": "user", "content": user_text})
    st.session_state.chat_history.append({
        "role": "assistant", "content": answer,
        "replay": {"kind": "task", "task": task, "doc": doc}})


def answer_question(text, doc):
    history = history_payload()                         # memory BEFORE this turn
    with st.chat_message("user"):
        st.markdown(text)
    with st.chat_message("assistant"):
        try:
            with st.spinner(T["searching"]):
                r = requests.post(f"{BACKEND_URL}/retrieve", timeout=180, json={
                    "question": text, "history": history, "model_name": model_id,
                    "document": doc, "language": language})
            if r.status_code != 200:
                st.error(error_detail(r, "Search failed."))
                return
            data = r.json()
            chunks = data["retrieved_chunks"]
            context = "\n\n".join(
                f"[Source: {c['source']}, page {c['page']}]\n{c['content']}" for c in chunks)
            answer = st.write_stream(stream_from("/generate-stream", {
                "question": text, "context": context, "history": history,
                "model_name": model_id, "language": language}))
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            return
        debug = {k: data[k] for k in ("rag_mode", "latency_ms", "standalone_question")}
        render_sources(chunks, debug)
    st.session_state.chat_history.append({"role": "user", "content": text})
    st.session_state.chat_history.append({
        "role": "assistant", "content": answer, "sources": chunks, "debug": debug,
        "replay": {"kind": "question", "text": text, "doc": doc}})


def replay_last(replay):
    """Language changed: write the last answer again in the new language."""
    before = len(st.session_state.chat_history)
    if ready:
        st.toast(T["lang_changed"])
        if replay["kind"] == "task":
            run_task(replay["task"], replay["doc"])
        else:
            answer_question(replay["text"], replay["doc"])
    if len(st.session_state.chat_history) == before:       # failed: keep the old answer
        st.session_state.chat_history.extend(st.session_state.get("replay_backup", []))


pending = st.session_state.pop("pending", None)
replay = st.session_state.pop("replay", None)
if pending or question or replay:
    with history_area:
        if replay and not (pending or question):
            replay_last(replay)
        elif pending:
            run_task(pending, doc_filter)
        else:
            answer_question(question, doc_filter)
st.session_state.pop("replay_backup", None)

if admin and os.path.exists("eval_history.csv"):
    with st.expander("🛠 Evaluation history (admin only)"):
        st.dataframe(pd.read_csv("eval_history.csv"), hide_index=True)