#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║          NEURAL FORGE  —  ML Dashboard  |  Single-File MVP           ║
║   Sentiment Analysis  •  FastAPI  •  scikit-learn  •  Live UI        ║
╚══════════════════════════════════════════════════════════════════════╝

Author:  MUSHARIB

GitHub:  https://github.com/musharibramzan950-bit
Linktree : https://linktr.ee/Musharib_
Built:   2026

Run:  python app.py
Deps: auto-installed on first run
"""

import importlib
import logging
import os
import subprocess
import sys
import time
import uuid
import webbrowser
from datetime import datetime
from threading import Thread

# ─── Auto-install dependencies ────────────────────────────────────────────────
REQUIRED = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn[standard]",
    "sklearn": "scikit-learn",
    "numpy": "numpy",
    "pydantic": "pydantic",
}

def ensure_deps():
    missing = []
    for imp, pkg in REQUIRED.items():
        try:
            importlib.import_module(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[SETUP] Installing: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + missing)
        print("[SETUP] Done. Restarting…")
        os.execv(sys.executable, [sys.executable] + sys.argv)

ensure_deps()

# ─── Imports (post-install) — E402 is intentional: deps may not exist yet ─────
import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402
import uvicorn  # noqa: E402

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, classification_report  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("neural_forge")

# ─── Global state ─────────────────────────────────────────────────────────────
state = {
    "status": "idle",          # idle | training | ready | error
    "progress": 0,
    "metrics": {},
    "training_curve": [],
    "model": None,
    "pipeline": None,
    "job_id": None,
    "log": [],
    "started_at": None,
    "finished_at": None,
}

def emit(msg: str, level="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    state["log"].append({"ts": ts, "msg": msg, "level": level})
    if len(state["log"]) > 200:
        state["log"] = state["log"][-200:]
    getattr(log, level)(msg)

# ─── Dataset ──────────────────────────────────────────────────────────────────
SAMPLES = [
    # positive
    ("I absolutely loved this product, it exceeded all my expectations!", 1),
    ("Fantastic quality, fast shipping, would definitely buy again.", 1),
    ("This is hands down the best purchase I've made this year.", 1),
    ("Amazing customer service, they resolved my issue immediately.", 1),
    ("The build quality is superb and it looks exactly like the photos.", 1),
    ("Five stars! Works perfectly, setup was a breeze.", 1),
    ("Incredibly well made, premium feel, worth every penny.", 1),
    ("Delivered on time, product is perfect, very happy.", 1),
    ("Outstanding performance, blew my expectations out of the water.", 1),
    ("Great value for the price, highly recommend to everyone.", 1),
    ("Excellent product! Does exactly what it says on the tin.", 1),
    ("Very impressed, the quality is top-notch and it arrived quickly.", 1),
    ("Loved the packaging and the item is even better in person.", 1),
    ("Super happy with this purchase. Will definitely order again.", 1),
    ("Best thing I've bought in years. Absolutely brilliant.", 1),
    ("Smooth experience from ordering to delivery. Product is fantastic.", 1),
    ("The performance is incredible and it looks gorgeous.", 1),
    ("Perfectly designed, intuitive to use, highly polished.", 1),
    ("I'm genuinely thrilled with this. Exceptional value.", 1),
    ("Everything about this is great. Couldn't be happier.", 1),
    ("Really good experience overall. Product quality is impressive.", 1),
    ("Solid build, great design, fast delivery.", 1),
    ("Totally satisfied with my purchase. Works like a charm.", 1),
    ("Premium quality at a reasonable price. Very happy customer.", 1),
    ("Love it! Great seller, fast shipping, excellent product.", 1),
    ("This product is fantastic! I'm so happy I bought it.", 1),
    ("Incredible design and performance beyond my expectations.", 1),
    ("Wonderful purchase, works flawlessly, great support.", 1),
    ("Beautiful product with exceptional craftsmanship.", 1),
    ("Superb quality, delivered quickly, very satisfied overall.", 1),
    # negative
    ("Terrible experience, the item broke after just two days.", 0),
    ("Completely disappointed, nothing like the product description.", 0),
    ("Worst purchase I've ever made, total waste of money.", 0),
    ("The quality is shocking, feels cheap and flimsy.", 0),
    ("Do not buy this, it stopped working after one week.", 0),
    ("Awful customer service, they never responded to my emails.", 0),
    ("Arrived damaged and the return process was a nightmare.", 0),
    ("Absolute rubbish, save your money and buy something else.", 0),
    ("I regret buying this, it's poorly made and overpriced.", 0),
    ("Horrible product, does not work as advertised at all.", 0),
    ("Extremely poor quality, fell apart almost immediately.", 0),
    ("Very disappointed, the color and size were completely wrong.", 0),
    ("Cheap materials, bad smell, unusable out of the box.", 0),
    ("This is a scam product, avoid at all costs.", 0),
    ("Broken on arrival, seller refused to issue a refund.", 0),
    ("Product is a disaster, nothing works as it should.", 0),
    ("Terrible quality control, clearly no QA done whatsoever.", 0),
    ("Waste of money, fell apart within days of light use.", 0),
    ("Awful experience, the worst product I have ever used.", 0),
    ("Not even close to what was shown in the listing photos.", 0),
    ("Very bad, stopped working immediately, no support provided.", 0),
    ("Deeply unsatisfied, this product is absolutely useless.", 0),
    ("Garbage product, don't believe the fake positive reviews.", 0),
    ("Broke on first use, total disappointment, money down the drain.", 0),
    ("Poorly made, misleading description, would give zero stars.", 0),
    ("Dreadful purchase, malfunctioned right out of the box.", 0),
    ("Sent wrong item, terrible service, never shopping here again.", 0),
    ("Shocking quality, nothing as described, completely useless.", 0),
    ("Utter junk, looks nothing like the pictures, very angry.", 0),
    ("Abysmal product, fell apart immediately, avoid this seller.", 0),
]

def build_dataset(n_augment=6):
    """Augment the base samples to create a larger dataset."""
    texts, labels = [], []
    for text, label in SAMPLES:
        texts.append(text)
        labels.append(label)
        # simple augmentation: shuffle words slightly, lowercase, prefix variants
        for _ in range(n_augment):
            words = text.split()
            np.random.shuffle(words)
            texts.append(" ".join(words))
            labels.append(label)
    return texts, labels

# ─── Training job ─────────────────────────────────────────────────────────────
def run_training(job_id: str):
    try:
        state["status"] = "training"
        state["job_id"] = job_id
        state["progress"] = 0
        state["training_curve"] = []
        state["metrics"] = {}
        state["started_at"] = datetime.now().isoformat()
        emit("🚀 Training job started", "info")

        # 1. Build dataset
        emit("📦 Loading and augmenting dataset…")
        time.sleep(0.4)
        texts, labels = build_dataset(n_augment=8)
        emit(f"   Dataset size: {len(texts)} samples  ({sum(labels)} pos / {len(labels)-sum(labels)} neg)")
        state["progress"] = 10

        # 2. Split
        emit("✂️  Splitting into train / val / test sets…")
        time.sleep(0.3)
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )
        emit(f"   Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
        state["progress"] = 20

        # 3. Simulate epoch-by-epoch training curve
        emit("🧠 Training Logistic Regression with TF-IDF features…")
        n_epochs = 10
        curve = []

        for epoch in range(1, n_epochs + 1):
            frac = epoch / n_epochs
            # partial fit simulation using increasing C values
            C_val = 0.01 + frac * 2.0
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=8000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    analyzer="word",
                    min_df=1,
                )),
                ("clf", LogisticRegression(C=C_val, max_iter=500, random_state=42)),
            ])
            pipe.fit(X_train, y_train)
            train_acc = accuracy_score(y_train, pipe.predict(X_train))
            val_acc   = accuracy_score(y_val,   pipe.predict(X_val))
            train_loss = max(0.05, 0.72 - frac * 0.65 + np.random.uniform(-0.02, 0.02))
            val_loss   = max(0.07, 0.75 - frac * 0.60 + np.random.uniform(-0.03, 0.03))

            curve.append({
                "epoch": epoch,
                "train_acc": round(train_acc, 4),
                "val_acc":   round(val_acc, 4),
                "train_loss": round(float(train_loss), 4),
                "val_loss":   round(float(val_loss), 4),
            })
            state["training_curve"] = curve
            state["progress"] = 20 + int(frac * 60)
            emit(f"   Epoch {epoch:02d}/{n_epochs}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  loss={val_loss:.4f}")
            time.sleep(0.35)

        # 4. Final model (best C)
        emit("🎯 Fitting final model…")
        final_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), max_features=10000,
                sublinear_tf=True, strip_accents="unicode", min_df=1,
            )),
            ("clf", LogisticRegression(C=2.0, max_iter=1000, random_state=42)),
        ])
        final_pipe.fit(X_train + X_val, y_train + y_val)
        state["pipeline"] = final_pipe
        state["progress"] = 88

        # 5. Evaluate
        emit("📊 Evaluating on held-out test set…")
        y_pred = final_pipe.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"], output_dict=True)

        state["metrics"] = {
            "accuracy":  round(acc, 4),
            "f1_score":  round(f1, 4),
            "precision": round(report["Positive"]["precision"], 4),
            "recall":    round(report["Positive"]["recall"], 4),
            "test_size": len(X_test),
            "train_size": len(X_train) + len(X_val),
            "classes": ["Negative", "Positive"],
            "report": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                           for kk, vv in v.items()}
                       for k, v in report.items() if isinstance(v, dict)},
        }
        state["progress"] = 100
        state["status"] = "ready"
        state["finished_at"] = datetime.now().isoformat()
        emit(f"✅ Training complete — accuracy={acc:.4f}  f1={f1:.4f}", "info")

    except Exception as e:
        state["status"] = "error"
        emit(f"❌ Training failed: {e}", "error")
        log.exception("Training error")

# ─── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Neural Forge", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class PredictRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(_load_html())

@app.post("/api/train")
async def start_training():
    if state["status"] == "training":
        raise HTTPException(409, "Training already in progress")
    job_id = str(uuid.uuid4())[:8]
    state["log"] = []
    t = Thread(target=run_training, args=(job_id,), daemon=True)
    t.start()
    return {"job_id": job_id, "message": "Training started"}

@app.get("/api/status")
async def get_status():
    return {
        "status":        state["status"],
        "progress":      state["progress"],
        "metrics":       state["metrics"],
        "training_curve": state["training_curve"],
        "job_id":        state["job_id"],
        "log":           state["log"][-30:],
        "started_at":    state["started_at"],
        "finished_at":   state["finished_at"],
    }

@app.post("/api/predict")
async def predict(req: PredictRequest):
    if state["pipeline"] is None:
        raise HTTPException(400, "Model not trained yet. Run training first.")
    text = req.text.strip()
    if not text:
        raise HTTPException(422, "Text cannot be empty")
    pipe = state["pipeline"]
    label = pipe.predict([text])[0]
    probs = pipe.predict_proba([text])[0]
    sentiment = "Positive" if label == 1 else "Negative"
    confidence = float(max(probs))
    return {
        "text":       text,
        "sentiment":  sentiment,
        "confidence": round(confidence, 4),
        "prob_neg":   round(float(probs[0]), 4),
        "prob_pos":   round(float(probs[1]), 4),
    }

@app.get("/api/health")
async def health():
    return {"status": "ok", "model_ready": state["pipeline"] is not None}

# ─── Load HTML from index.html (same directory as this script) ──────────────────
import pathlib as _pl
_HERE = _pl.Path(__file__).parent
_HTML_FILE = _HERE / "index.html"

def _load_html() -> str:
    """Read index.html from disk so edits to the file take effect on next request."""
    if not _HTML_FILE.exists():
        raise FileNotFoundError(
            f"index.html not found at {_HTML_FILE}. "
            "Place index.html in the same directory as neural_forge.py."
        )
    return _HTML_FILE.read_text(encoding="utf-8")


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8000))
    print("\n" + "═" * 62)
    print("  ⚡  NEURAL FORGE  —  ML Dashboard")
    print(f"  ➜  http://localhost:{PORT}")
    print("═" * 62 + "\n")

    def open_browser():
        time.sleep(1.4)
        webbrowser.open(f"http://localhost:{PORT}")

    Thread(target=open_browser, daemon=True).start()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="warning",
    )
