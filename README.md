# ml 1 project
⚡ Production-grade sentiment analysis ML dashboard — trains a model, streams live metrics, and runs inference. One Python file. One command.
# ⚡ Neural Forge — Single-File ML Dashboard

> A production-grade sentiment analysis web app that trains a model, streams live metrics, and runs inference — all from one Python file.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikitlearn)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## 🚀 Run It

```bash
python app.py
```

That's it. Dependencies auto-install. Browser opens automatically at `http://localhost:8000`.

---

## 📸 What You Get

A dark, premium SaaS-style dashboard that:

- **Trains a real ML model** in the browser with live epoch-by-epoch progress
- **Streams training curves** — accuracy and loss charts animate in real time
- **Runs live inference** — type any text, get sentiment + confidence score instantly
- **Logs everything** — terminal-style training log updates as the model trains
- **Tracks predictions** — history panel records every inference you run

---

## 🧠 ML Pipeline

| Stage | Detail |
|-------|--------|
| Dataset | 60 hand-labelled product reviews, augmented to ~540 samples |
| Features | TF-IDF vectorizer, bigrams, 10k features, sublinear TF |
| Model | Logistic Regression (C=2.0, max_iter=1000) |
| Split | 68% train / 12% val / 20% test |
| Metrics | Accuracy, F1, Precision, Recall on held-out test set |
| Inference | Predict sentiment + probability for any input text |

Typical results: **~95%+ accuracy**, **~0.95 F1** on the test set.

---

## 🛠 Tech Stack

- **Backend** — FastAPI + Uvicorn
- **ML** — scikit-learn (TF-IDF + Logistic Regression)
- **Frontend** — Vanilla HTML/CSS/JS embedded in Python (no separate files)
- **Charts** — Chart.js (CDN)
- **Fonts** — Syne + IBM Plex Mono (Google Fonts)

---

## 📦 Dependencies

Auto-installed on first run — you don't need to do anything:

```
fastapi
uvicorn[standard]
scikit-learn
numpy
pydantic
```

Requires **Python 3.8+**. No Docker. No virtual env required.

---

## 🗂 Project Structure

```
app.py        ← the entire project (1 file, ~1000 lines)
README.md     ← this file
```

---

## ✨ Features

- **One-command setup** — zero configuration
- **Auto-installs deps** — pip installs missing packages then restarts
- **Cross-platform** — works on macOS, Windows, Linux
- **Background training** — non-blocking async job with polling
- **Live charts** — training curves update every epoch
- **Inference UI** — Cmd/Ctrl+Enter to analyse, probability bars for both classes
- **Error handling** — full try/catch on training and inference endpoints
- **Logging** — structured terminal log streamed to the UI

---

## 🎯 Use Cases

- Portfolio project to showcase ML + full-stack skills
- Starting template for NLP classification apps
- Demo for investors / technical interviews
- Learning reference for FastAPI + scikit-learn integration

---

## 📄 License

MIT — free to use, modify, and distribute.


# ⚡ Neural Forge — Single-File ML Dashboard

> A production-grade sentiment analysis web app that trains a model, streams live metrics, and runs inference — all from one Python file.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikitlearn)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## 🚀 Run It

```bash
python app.py
```

That's it. Dependencies auto-install. Browser opens automatically at `http://localhost:8000`.

---

## 📸 What You Get

A dark, premium SaaS-style dashboard that:

- **Trains a real ML model** in the browser with live epoch-by-epoch progress
- **Streams training curves** — accuracy and loss charts animate in real time
- **Runs live inference** — type any text, get sentiment + confidence score instantly
- **Logs everything** — terminal-style training log updates as the model trains
- **Tracks predictions** — history panel records every inference you run

---

## 🧠 ML Pipeline

| Stage | Detail |
|-------|--------|
| Dataset | 60 hand-labelled product reviews, augmented to ~540 samples |
| Features | TF-IDF vectorizer, bigrams, 10k features, sublinear TF |
| Model | Logistic Regression (C=2.0, max_iter=1000) |
| Split | 68% train / 12% val / 20% test |
| Metrics | Accuracy, F1, Precision, Recall on held-out test set |
| Inference | Predict sentiment + probability for any input text |

Typical results: **~95%+ accuracy**, **~0.95 F1** on the test set.

---

## 🛠 Tech Stack

- **Backend** — FastAPI + Uvicorn
- **ML** — scikit-learn (TF-IDF + Logistic Regression)
- **Frontend** — Vanilla HTML/CSS/JS embedded in Python (no separate files)
- **Charts** — Chart.js (CDN)
- **Fonts** — Syne + IBM Plex Mono (Google Fonts)

---

## 📦 Dependencies

Auto-installed on first run — you don't need to do anything:

```
fastapi
uvicorn[standard]
scikit-learn
numpy
pydantic
```

Requires **Python 3.8+**. No Docker. No virtual env required.

---

## 🗂 Project Structure

```
app.py        ← the entire project (1 file, ~1000 lines)
README.md     ← this file
```

---

## ✨ Features

- **One-command setup** — zero configuration
- **Auto-installs deps** — pip installs missing packages then restarts
- **Cross-platform** — works on macOS, Windows, Linux
- **Background training** — non-blocking async job with polling
- **Live charts** — training curves update every epoch
- **Inference UI** — Cmd/Ctrl+Enter to analyse, probability bars for both classes
- **Error handling** — full try/catch on training and inference endpoints
- **Logging** — structured terminal log streamed to the UI

---

## 🎯 Use Cases

- Portfolio project to showcase ML + full-stack skills
- Starting template for NLP classification apps
- Demo for investors / technical interviews
- Learning reference for FastAPI + scikit-learn integration

---

## 📄 License

MIT — free to use, modify, and distribute.