# 📄 AI Document Analyser

A full-stack **Retrieval-Augmented Generation (RAG)** app. Upload documents, build a knowledge base, and get streamed, source-backed answers, summaries, key dates and structured insights. It works in **English, Tamil and Tanglish**.

- **Frontend:** Streamlit
- **Backend:** FastAPI + LangChain
- **Vector store:** FAISS (saved on disk)
- **Retrieval:** hybrid search (BM25 keywords + vector similarity), chosen after testing six RAG modes
- **Models:** OpenAI, Anthropic Claude or Google Gemini. A model appears in the dropdown only if its API key is in `.env`.

---

## ✨ Features

| Area | What you get |
|---|---|
| Documents | Upload PDF, DOCX, TXT or MD. Add or delete files in the AI Knowledge Base manager. |
| Chat | Ask questions and get streamed answers with a **Sources** box (document name and page). |
| One-click tasks | Document Summary, Key Points, Important Dates, Structured Summary, Document Analysis and Insights. |
| Scope | Search all documents, or limit answers to one document. |
| Languages | English, Tamil (தமிழ்) and Tanglish. The whole screen and the answers follow the chosen language. |
| Memory | Follow-up questions such as "what about its eligibility?" are understood from chat history. |
| Admin view | Open `http://localhost:8501/?admin=1` to see retrieval time, scores and evaluation history. |

---

## 🏗️ Architecture

```
                 ┌──────────────────────── Streamlit frontend ────────────────────────┐
  User ───────►  │ Upload • Knowledge base • Cards • Try-asking • Chat • Language      │
                 └───────────────┬───────────────────────────────▲─────────────────────┘
                                 │ REST (JSON / streaming)       │ streamed answer + sources
                 ┌───────────────▼───────────────────────────────┴─────────────────────┐
                 │                         FastAPI backend                              │
                 │                                                                      │
                 │  INGESTION                          QUERY                            │
                 │  Load → Clean → Chunk               Rewrite query (history,          │
                 │  (1000 chars, 200 overlap)          Tamil/Tanglish → English)        │
                 │        │                                   │                         │
                 │        ▼                                   ▼                         │
                 │  Embeddings ──► FAISS index  ◄── Hybrid retrieval (BM25 + vector,    │
                 │  (+ metadata: file, page)         merged with Reciprocal Rank Fusion)│
                 │                                            │                         │
                 │                                   Augmented prompt ──► LLM ──► stream│
                 └──────────────────────────────────────────────────────────────────────┘
```

**Ingestion:** the file is read, cleaned, split into overlapping chunks, turned into embeddings and saved in FAISS with the file name and page number.

**Query:** the question is rewritten as a standalone English query when needed. Hybrid search retrieves the best passages, they are placed into a prompt together with the question, and the LLM streams an answer in the chosen language.

---

## 📁 Project structure

```
GEN AI APP - 1/
├── backend.py            # FastAPI API + RAG pipeline (hybrid retrieval)
├── frontend.py           # Streamlit UI
├── requirements.txt
├── .env                  # your secrets (NOT committed)
├── .env.example          # template with empty values (committed)
├── .gitignore
├── .streamlit/
│   └── config.toml       # light theme
├── backend_full.py       # optional: version with all 6 RAG modes, for testing
├── evaluate.py           # optional: compares the RAG modes
├── questions.json        # optional: your evaluation questions
├── faiss_index/          # created automatically (the knowledge base)
└── logs/                 # created automatically (query log)
```

---

## ⚙️ Setup

You need **Python 3.10 or newer** and at least one API key (OpenAI is required for the default embeddings).

### 1. Get the code
```bash
git clone <your-repo-url>
cd "GEN AI APP - 1"
```

### 2. Create a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the packages
```bash
python -m pip install -r requirements.txt
```

### 4. Create the `.env` file
Copy `.env.example` to `.env` and fill in your keys (see the next section).

### 5. Create the theme file
Create a folder named `.streamlit` with a file `config.toml` inside it:
```toml
[theme]
base = "light"
primaryColor = "#7b2ff7"
```

---

## 🔑 The `.env` file

Create a file named `.env` in the project folder:

```env
# ---- API keys (add only the ones you have) ----
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=

# ---- Embeddings: openai or gemini (Claude has no embeddings) ----
EMBEDDING_PROVIDER=openai

# ---- Number of passages sent to the LLM per question ----
TOP_K=4
```

| Variable | Meaning |
|---|---|
| `OPENAI_API_KEY` | Needed for OpenAI chat models and OpenAI embeddings. |
| `ANTHROPIC_API_KEY` | Optional. Adds Claude to the AI Model dropdown. |
| `GOOGLE_API_KEY` | Optional. Adds Gemini, and is needed if `EMBEDDING_PROVIDER=gemini`. |
| `EMBEDDING_PROVIDER` | `openai` (default) or `gemini`. Changing it later means re-uploading your documents. |
| `TOP_K` | How many passages are retrieved per question. The default is 4. |

> ⚠️ **Never commit `.env`.** It is listed in `.gitignore`. If a key is ever pushed to GitHub, delete it in your provider's dashboard and create a new one.

Your `.gitignore` should contain:
```
.env
faiss_index/
logs/
venv/
__pycache__/
```

---

## ▶️ Run the app

Use **two terminals**, both inside the project folder with the virtual environment activated.

**Terminal 1: backend (FastAPI).** It is started with `uvicorn`, not `python backend.py`:
```bash
python -m uvicorn backend:app --host 127.0.0.1 --port 8000 --reload
```
Check it at <http://127.0.0.1:8000/status>. It should list your models and languages.

**Terminal 2: frontend (Streamlit):**
```bash
python -m streamlit run frontend.py --server.address 127.0.0.1 --server.port 8501
```
Open <http://127.0.0.1:8501>.

### Using the app
1. Click **Upload Document**, choose a PDF, and click **Process**.
2. Pick an **AI Model** and a **Language** in the sidebar.
3. Click a card (for example **Document Summary**), try one of the *Try asking* questions, or type your own question.
4. Open **📚 Sources** under an answer to see which document and page it came from.
5. Use **Add to AI Knowledge** to add more files or delete old ones.

---

## 🧪 Evaluation (choosing the RAG type)

Six retrieval modes were compared on 10 questions about four agriculture-scheme PDFs: `basic`, `mmr`, `multi_query`, `hybrid`, `hyde` and `rerank`.

| Mode | Hit rate | Keyword recall | Avg retrieval time |
|---|---|---|---|
| basic | 7/10 | 0.85 | 525 ms |
| **hybrid (chosen)** | **7/10** | **0.85** | **384 ms** |
| multi_query | 7/10 | 0.85 | 2394 ms |
| hyde | 7/10 | 0.85 | 1612 ms |
| rerank | 7/10 | 0.85 | 1462 ms |
| mmr | 7/10 | 0.80 | 362 ms |

Five modes tied on accuracy, so hybrid was picked as the fastest of the tied modes, and it needs no extra LLM calls. The three failed questions failed in **every** mode. They pointed to data problems (a missing or conflicting document, and a table split across chunks), not to the retrieval algorithm.

**How the score works:** for each question, `evaluate.py` fetches the top passages and checks whether your expected keywords appear in them. *Hit rate* is the share of questions where all keywords were found. *Recall* is the average share of keywords found.

### Run your own evaluation
1. Keep `backend_full.py` (the version with all six modes). Add these to `.env`:
   ```env
   RAG_MODE=basic
   ALLOW_MODE_OVERRIDE=true
   ```
2. Start it in place of `backend.py`:
   ```bash
   python -m uvicorn backend_full:app --host 127.0.0.1 --port 8000
   ```
3. Edit `questions.json`. Each item needs a question and keywords that a correct passage must contain:
   ```json
   [
     {"question": "What is the fee and validity of a seed selling licence?",
      "expected_keywords": ["Rs.50", "3 years"]}
   ]
   ```
4. Run it:
   ```bash
   python evaluate.py "baseline"
   python evaluate.py "topk6" --top-k 6
   python evaluate.py "quick" --modes basic,hybrid
   ```
5. Results are saved to `eval_history.csv` (summary) and `eval_details.csv` (per question). The history also shows in the app at `?admin=1`.

---

## 🔌 API endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/documents` | Upload and index one or more files |
| GET | `/documents` | List documents in the knowledge base |
| DELETE | `/documents/{name}` | Remove one document |
| DELETE | `/reset` | Remove all documents |
| POST | `/retrieve` | Rewrite the query and retrieve the top passages |
| POST | `/generate-stream` | Stream an answer from the retrieved context |
| POST | `/analyze-stream` | Stream a summary, key points, dates or analysis |
| GET | `/status` | Models, languages and index status |

Interactive docs: <http://127.0.0.1:8000/docs>

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| "Backend offline" | Start Terminal 1 first and check <http://127.0.0.1:8000/status>. |
| "No chat model available" | Add at least one API key to `.env`, then restart the backend. |
| "knowledge base was built with another embedding provider" | Remove all documents (Add to AI Knowledge → Remove all) and upload again. |
| A model says "not found" | Model IDs change over time. Edit `MODEL_CATALOG` in `backend.py`. |
| Cards or buttons are not coloured | Use Streamlit 1.40 or newer. |
| Scanned PDF gives "No readable text" | The file is an image. Run OCR on it first. |
| Answer says the information is not in the documents | Check **Sources**, make sure the right document is selected in *Search in*, or try a more specific question. |

---

## ⚠️ Known limits

- For long documents, summaries use up to 30 evenly spaced passages, and the answer says so.
- Tamil and Tanglish questions are translated to English before searching, which works best for English documents.
- If two uploaded files contradict each other, the app cannot tell which one is current. Keep only the latest version.
- Tanglish quality depends on the chosen model.

---

## 🧰 Tech stack

Python · Streamlit · FastAPI · LangChain · FAISS · BM25 · OpenAI / Anthropic / Gemini   