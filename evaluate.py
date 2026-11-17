"""Compare the RAG modes of backend.py on your own questions.

Needs in .env:   ALLOW_MODE_OVERRIDE=true     (restart the backend after changing it)
Backend must be running on port 8000 and at least one document must be uploaded.

Usage:
    python evaluate.py "baseline"
    python evaluate.py "after-new-pdf" --questions questions.json --top-k 4
    python evaluate.py "quick" --modes basic,hybrid --document "my_scheme.pdf"

For every question the script asks the backend for the top passages (no answer is
generated, so it is cheap) and checks whether the expected keywords appear in them.
"""

import argparse
import csv
import json
import os
import sys
import time

import requests

BACKEND = "http://127.0.0.1:8000"
HISTORY_FILE = "eval_history.csv"      # the frontend shows this in ?admin=1
DETAILS_FILE = "eval_details.csv"

SAMPLE_QUESTIONS = [
    {
        "question": "Who is eligible for this scheme?",
        "expected_keywords": ["eligible", "age"],
    },
    {
        "question": "What is the last date to apply?",
        "expected_keywords": ["last date", "2025"],
    },
]


def load_questions(path):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(SAMPLE_QUESTIONS, f, indent=2, ensure_ascii=False)
        print(f"Created a sample {path}. Edit it with questions and keywords from YOUR "
              f"document, then run this script again.")
        sys.exit(0)

    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    for i, item in enumerate(items, 1):
        if "question" not in item or not item.get("expected_keywords"):
            sys.exit(f"{path}: item {i} needs 'question' and a non-empty 'expected_keywords' list.")
    return items


def score_chunks(chunks, keywords):
    """Return (keyword_recall, hit). Recall = share of keywords found in the passages."""
    text = " ".join(c["content"] for c in chunks).lower()
    found = [k for k in keywords if k.lower() in text]
    recall = len(found) / len(keywords)
    return recall, recall == 1.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG modes.")
    parser.add_argument("run_name", help='A label for this run, e.g. "baseline"')
    parser.add_argument("--questions", default="questions.json")
    parser.add_argument("--top-k", type=int, default=None, help="Passages per question")
    parser.add_argument("--modes", default="", help="Comma-separated modes (default: all)")
    parser.add_argument("--document", default="", help="Limit to one document name")
    args = parser.parse_args()

    try:
        status = requests.get(f"{BACKEND}/status", timeout=10).json()
    except requests.exceptions.ConnectionError:
        sys.exit("Backend is not running. Start it first (uvicorn backend:app ...).")

    if not status.get("faiss_ready"):
        sys.exit("No documents in the knowledge base. Upload a document first.")
    if not status.get("models"):
        sys.exit("No chat model available. Add an API key to .env.")

    model_id = status["models"][0]["id"]
    all_modes = status["rag_modes"]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()] or all_modes
    unknown = [m for m in modes if m not in all_modes]
    if unknown:
        sys.exit(f"Unknown mode(s): {unknown}. Choose from: {all_modes}")

    questions = load_questions(args.questions)
    print(f"Run '{args.run_name}': {len(questions)} questions x {len(modes)} modes "
          f"(model for query rewriting: {model_id})\n")

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    summary_rows, detail_rows = [], []

    for mode in modes:
        recalls, hits, latencies, errors = [], 0, [], 0

        for item in questions:
            payload = {
                "question": item["question"],
                "model_name": model_id,
                "document": item.get("document", args.document),
                "rag_mode": mode,
            }
            if args.top_k:
                payload["top_k"] = args.top_k

            try:
                r = requests.post(f"{BACKEND}/retrieve", json=payload, timeout=180)
            except requests.exceptions.RequestException as e:
                print(f"  [{mode}] request failed: {e}")
                errors += 1
                continue

            if r.status_code == 403:
                sys.exit("The backend refused the mode override. Set ALLOW_MODE_OVERRIDE=true "
                         "in .env and restart the backend.")
            if r.status_code != 200:
                try:
                    detail = r.json().get("detail", r.text[:150])
                except Exception:
                    detail = r.text[:150]
                print(f"  [{mode}] '{item['question'][:40]}' failed: {detail}")
                errors += 1
                continue

            data = r.json()
            recall, hit = score_chunks(data["retrieved_chunks"], item["expected_keywords"])
            recalls.append(recall)
            hits += hit
            latencies.append(data["latency_ms"])
            detail_rows.append({
                "run": args.run_name, "time": stamp, "mode": mode,
                "question": item["question"], "keyword_recall": round(recall, 2),
                "hit": int(hit), "latency_ms": data["latency_ms"],
                "sources": "; ".join(f"{c['source']} p{c['page']}"
                                     for c in data["retrieved_chunks"]),
            })

        n = len(recalls)
        summary_rows.append({
            "run": args.run_name, "time": stamp, "mode": mode,
            "questions": len(questions), "answered": n, "errors": errors,
            "hit_rate": round(hits / n, 3) if n else 0,
            "keyword_recall": round(sum(recalls) / n, 3) if n else 0,
            "avg_latency_ms": int(sum(latencies) / n) if n else 0,
        })

    # ---------- print the comparison table ----------
    print(f"{'Mode':<13}{'Hit rate':>10}{'Recall':>10}{'Avg ms':>10}{'Errors':>9}")
    print("-" * 52)
    for row in sorted(summary_rows, key=lambda r: (-r["keyword_recall"], r["avg_latency_ms"])):
        print(f"{row['mode']:<13}{row['hit_rate']:>10.0%}{row['keyword_recall']:>10.0%}"
              f"{row['avg_latency_ms']:>10}{row['errors']:>9}")
    print("\nHit rate = questions where ALL keywords were found in the retrieved passages.")
    print("Recall   = average share of keywords found.  Best mode is at the top.")

    # ---------- save results ----------
    for path, rows in ((HISTORY_FILE, summary_rows), (DETAILS_FILE, detail_rows)):
        if not rows:
            continue
        new_file = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new_file:
                writer.writeheader()
            writer.writerows(rows)
    print(f"\nSaved to {HISTORY_FILE} (summary) and {DETAILS_FILE} (per question).")


if __name__ == "__main__":
    main()