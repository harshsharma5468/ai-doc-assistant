# 🧠 AI Document Assistant

> Production-grade RAG pipeline with LangChain, FAISS/ChromaDB, FastAPI, Streamlit, and full AWS EC2 deployment via Docker.

[![CI/CD](https://github.com/YOUR_ORG/ai-doc-assistant/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/YOUR_ORG/ai-doc-assistant/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-orange.svg)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            AI Document Assistant                             │
├──────────────┬──────────────────────────────────────┬───────────────────────┤
│   Streamlit  │           FastAPI Backend            │    Infrastructure     │
│   Frontend   │                                      │                       │
│              │  ┌──────────────────────────────┐    │  • FAISS / ChromaDB   │
│  • Chat UI   │  │         RAG Pipeline          │    │  • Redis (memory)     │
│  • Doc upload│  │                               │    │  • Prometheus         │
│  • Eval dash │  │  Retrieve → Rerank → Generate │    │  • Grafana            │
│  • Source     │  │                               │    │  • Nginx              │
│    viewer    │  │  LangChain + MMR + CrossEnc   │    │  • Docker Compose     │
│              │  └──────────────────────────────┘    │  • AWS EC2            │
│              │                                      │  • GitHub Actions     │
└──────────────┴──────────────────────────────────────┴───────────────────────┘
```

## Features

| Category | Details |
|---|---|
| **Document Ingestion** | PDF, DOCX, TXT, MD, HTML via LangChain loaders |
| **Chunking** | `RecursiveCharacterTextSplitter` with configurable size & overlap |
| **Embeddings** | OpenAI `text-embedding-3-small` or local Ollama (`nomic-embed-text`) |
| **Vector Store** | FAISS (local, zero config) or ChromaDB (server, persistent) |
| **Retrieval** | MMR (diversity-aware) or similarity search with score threshold |
| **Reranking** | Cross-encoder `ms-marco-MiniLM-L-6-v2` for precision boost |
| **LLM** | OpenAI GPT-4o-mini, Anthropic Claude, or fully local Ollama (Llama3) |
| **Memory** | Windowed conversation buffer per session (Redis-backed) |
| **Evaluation** | RAGAS: answer relevance, faithfulness, context precision/recall |
| **Streaming** | Server-Sent Events (SSE) for real-time token streaming |
| **Observability** | Prometheus metrics + Grafana dashboards + structured JSON logs |
| **CI/CD** | GitHub Actions: lint → test → build → security scan → deploy |
| **Deployment** | Docker Compose on AWS EC2 with blue/green production deploys |

---

## Quick Start (Local)

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- OpenAI API key (or Ollama for free local inference)

### 1. Clone & Configure

```bash
git clone https://github.com/YOUR_ORG/ai-doc-assistant.git
cd ai-doc-assistant
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

### 2. Start with Docker Compose

```bash
cd docker
docker compose up -d
```

Services start on:
- **UI** → http://localhost:8501
- **API** → http://localhost:8000
- **API Docs** → http://localhost:8000/docs
- **Grafana** → http://localhost:3000
- **Prometheus** → http://localhost:9090

### 3. Run Without Docker (Dev Mode)

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

---

## API Reference

### Upload a Document
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@report.pdf" \
  -F "namespace=finance"
```

### Ask a Question
```bash
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the key findings?",
    "namespace": "finance",
    "search_type": "mmr",
    "top_k": 5,
    "include_sources": true
  }'
```

### Stream a Response (SSE)
```bash
curl -N http://localhost:8000/api/v1/chat/stream \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarize the document", "namespace": "finance"}'
```

### Run Evaluation
```bash
curl -X POST http://localhost:8000/api/v1/evaluate/ \
  -H "Content-Type: application/json" \
  -d '{
    "questions": ["What is the main topic?", "What are the conclusions?"],
    "namespace": "finance"
  }'
```

---

## AWS EC2 Deployment

### 1. Launch EC2 Instance
- **AMI**: Amazon Linux 2023 or Ubuntu 22.04
- **Instance Type**: t3.large (recommended) or t3.xlarge for heavy workloads
- **Security Group**: Open ports 22, 80, 443

### 2. Run Setup Script
```bash
# SSH into your EC2 instance
ssh -i your-key.pem ec2-user@YOUR_EC2_IP

# Download and run setup
curl -sSL https://raw.githubusercontent.com/YOUR_ORG/ai-doc-assistant/main/scripts/setup-ec2.sh | sudo bash
```

### 3. Set API Keys & Start
```bash
sudo nano /opt/ai-doc-assistant/.env   # Set OPENAI_API_KEY
sudo systemctl start ai-doc-assistant
sudo systemctl status ai-doc-assistant
```

### 4. CI/CD Setup (GitHub Actions)
Add these secrets to your GitHub repository:
- `EC2_PRIVATE_KEY` — SSH private key for EC2
- `PRODUCTION_HOST` — EC2 public IP/DNS
- `STAGING_HOST` — Staging EC2 IP/DNS
- `SLACK_WEBHOOK_URL` — (optional) deployment notifications

---

## Configuration

All settings are in `backend/app/core/config.py` with env var overrides:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` / `anthropic` / `ollama` |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model name |
| `VECTOR_STORE_TYPE` | `faiss` | `faiss` or `chroma` |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `SIMILARITY_TOP_K` | `5` | Chunks retrieved per query |
| `ENABLE_RERANKING` | `true` | Cross-encoder reranking |
| `RETRIEVER_SEARCH_TYPE` | `mmr` | `mmr` / `similarity` |
| `ENABLE_TRACING` | `true` | Per-query RAGAS evaluation |

---

## Evaluation Metrics (RAGAS)

| Metric | Formula | Ideal |
|---|---|---|
| **Answer Relevance** | Cosine sim of generated question ↔ original question | → 1.0 |
| **Faithfulness** | Fraction of answer claims supported by context | → 1.0 |
| **Context Precision** | Relevant chunks / total retrieved chunks | → 1.0 |
| **Context Recall** | Fraction of ground truth covered by context | → 1.0 |

Run batch evaluation via API or Streamlit dashboard.

---

## Project Structure

```
ai-doc-assistant/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI routers (chat, documents, evaluation)
│   │   ├── core/         # Config, logging, dependencies
│   │   ├── models/       # Pydantic schemas
│   │   ├── services/     # RAG pipeline, vector store, LLM factory, evaluation
│   │   └── main.py       # App factory with middleware
│   ├── tests/            # pytest test suite
│   ├── requirements.txt
│   └── pyproject.toml    # Ruff, mypy, pytest config
├── frontend/
│   ├── app.py            # Streamlit multi-tab UI
│   └── requirements.txt
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   ├── docker-compose.yml
│   ├── nginx.conf
│   └── prometheus.yml
├── scripts/
│   └── setup-ec2.sh      # One-command EC2 provisioning
├── .github/
│   └── workflows/
│       └── ci-cd.yml     # Full CI/CD pipeline
├── .env.example
└── README.md
```

---

## Development

```bash
# Run tests
cd backend && pytest tests/ -v --cov=app

# Lint
ruff check app && ruff format --check app

# Type check
mypy app --ignore-missing-imports
```

---

## License
MIT © Your Organization
