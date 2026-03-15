# OffBooksAI

> **This era's Anti-AI AI.**

An adversarial fraud detection system where two AI agents fight in real time. One attacks. One learns. The longer they fight, the harder fraud becomes.

---

## What Is This

Traditional fraud detection is reactive. Analysts study an attack, write a rule, push it through compliance, deploy it. That takes **days**. Sophisticated fraudsters adapt in **hours**.

OffBooksAI closes that gap to seconds.

**FraudGPT** — a red-team AI that generates novel money laundering topologies every round. Fan-outs, circular routing, binary tree splits, micro-transfer chains, scatter-gather patterns. It tracks what gets caught and deliberately avoids those patterns. It evolves.

**DefenseAI** — a blue-team AI that monitors every transaction, identifies fraud rings through graph analysis, and **writes its own detection rules** every time something slips through. No human. No deployment. Seconds.

After one hour of continuous battling: **344 attacks, 305 caught, 39 new rules written autonomously.**

---

## Demo

| Battle Mode | Attack Mode |
|---|---|
| Two AIs sparring autonomously | Manual control across 5 difficulty levels |
| Live transaction graph updating in real time | Full AI investigation report on every detection |
| Detection rules strip grows as DefenseAI learns | Rules from Battle Mode carry over here |

Both modes share the same rule engine. Rules DefenseAI writes during a battle are immediately active in Attack Mode. After 39 battles, a Level 5 stealth attack isn't facing 10 rules — it's facing 49.

---

## How It Works

```
Kaggle Fraud Data (10k transactions)
         │
         ▼
  XGBoost Classifier ─────────────────────────────────┐
  (AUC 0.9999, Recall 100%)                           │
                                                       ▼
FraudGPT (GPT-OSS 120B) ──► Transaction Graph ──► RingScout
         ▲                   (D3 Force Graph)      (Suspicion Score)
         │                        │                    │
         │                        ▼                    ▼
         └──── learns ────── DefenseAI ◄──── evaded? write rule
                             (GPT-OSS 120B)
```

**Suspicion scoring:**

$$S = \sum_i w_i \cdot r_i$$

Where $r_i$ are individual rule signals (ML score, PageRank anomaly, velocity, circular flow, etc.) and $w_i$ are their weights. Anything above 70/100 triggers a full investigation.

---

## ML Performance

Trained on 10,000 real credit card fraud transactions from Kaggle. Class imbalance (1.51% fraud rate) handled with SMOTE.

| Metric | Score |
|---|---|
| AUC | 0.9999 |
| Recall | 1.0 (100%) |
| F1 Score | 0.9375 |
| Missed frauds | 0 |

Confusion matrix:

|  | Predicted Legit | Predicted Fraud |
|---|---|---|
| **Actually Legit** | 1966 | 4 |
| **Actually Fraud** | 0 | 30 |

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost + SMOTE (scikit-learn) |
| Graph Engine | NetworkX — PageRank, betweenness centrality, Louvain community detection |
| LLM Agents | GPT-OSS-120B via HuggingFace inference endpoints |
| Backend | FastAPI + async Python |
| Frontend | Vanilla HTML/CSS/JS + D3.js (no build step) |
| Real-time | Server-Sent Events (SSE) for live streaming |
| Training Data | Kaggle Credit Card Fraud Detection dataset |

---

## Installation

### Prerequisites

- Python 3.10+
- The Kaggle credit card fraud dataset (`creditcard.csv`) placed in `data/`

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/offbooksai.git
cd offbooksai
pip install -r requirements.txt
```

### Configure environment

Create a `.env` file in the project root:

```env
# HuggingFace GPT-OSS Endpoints
OPENAI_API_KEY=test
MODEL_NAME=openai/gpt-oss-120b
OPENAI_BASE_URL=https://vjioo4r1vyvcozuj.us-east-2.aws.endpoints.huggingface.cloud/v1
OPENAI_BASE_URL_2=https://qyt7893blb71b5d3.us-east-2.aws.endpoints.huggingface.cloud/v1
```

### Train the model

```bash
python train_model.py
```

This generates `models/fraud_scorer.pkl`, `models/scaler.pkl`, and `models/encoder.pkl`.

### Run

```bash
python main.py
```

Then open `offbooksai.html` directly in your browser. No separate frontend server needed.

```
[Main] Noise loop started.
[Main] ML model ready — AUC: 0.9999
INFO: Uvicorn running on http://0.0.0.0:8000
```

---

## Architecture

### Agents

| Agent | Role | Learns? |
|---|---|---|
| **FraudGPT** | Generates attack topologies via LLM | Yes — avoids caught patterns each round |
| **DefenseAI** | Monitors graph, writes detection rules | Yes — new rule on every evasion |
| **RingScout** | Scores suspicious clusters | Static + injected rules |
| **TransactionScorer** | Per-transaction ML inference | Static (pre-trained) |
| **DataStreamer** | Serves real Kaggle transactions as background noise | N/A |

### API Endpoints

| Endpoint | Description |
|---|---|
| `POST /battle/start` | Start the autonomous battle loop |
| `POST /battle/stop` | Stop the battle loop |
| `GET /battle/state` | Current battle state + agent memory |
| `POST /round` | Launch a single manual attack |
| `GET /graph` | Current transaction graph (nodes + edges) |
| `GET /stats` | Independent attack and battle counters |
| `GET /stream/battle` | SSE stream of live battle events |
| `POST /reset` | Reset all state |

---

## Project Structure

```
offbooksai/
├── main.py                  # FastAPI app + battle loop + all endpoints
├── train_model.py           # XGBoost training script
├── offbooksai.html          # Full frontend (single file)
├── data/
│   └── creditcard.csv       # Kaggle fraud dataset
├── models/
│   ├── fraud_scorer.pkl     # Trained XGBoost model
│   ├── scaler.pkl           # Feature scaler
│   └── encoder.pkl          # Category encoder
├── agents/
│   ├── fraud_gpt.py         # Red team agent
│   ├── defense_ai.py        # Blue team agent
│   └── investigation_agent.py  # Report generator
└── utils/
    ├── llm_client.py        # Dual-endpoint async LLM client
    ├── ring_scout.py        # Graph-based fraud detection engine
    ├── transaction_scorer.py # XGBoost inference wrapper
    └── graph_builder.py     # NetworkX transaction graph
```

---

## The Pitch

> *Banks write fraud rules after the attack happens. We built a system that writes them during. Every evasion is just a rule that hasn't been written yet.*

Every chip in the detection rules strip was written by DefenseAI — named, described, and immediately active. That's not a simulation. After an hour of battling, the system is genuinely harder to fool than when it started.

---

## What's Next

- **Persistence** — store DefenseAI's learned rules across sessions so the system builds permanently over time
- **Live transaction feeds** — connect to real bank APIs instead of replaying Kaggle data
- **Multi-institution learning** — share anonymized rules across a network of banks so a new attack pattern at one institution triggers a rule at all of them within seconds
- **SAR integration** — structured output into Suspicious Activity Report format so DefenseAI files the compliance paperwork automatically

---

## Built At

**GenAI Hackathon 2026** — built in 24 hours.

---

*Every evasion is just a rule that hasn't been written yet.*
