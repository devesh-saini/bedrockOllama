# Docker Knowledge Base

A voice-enabled, conversational RAG (Retrieval-Augmented Generation) application for Docker documentation. Ask questions in plain English — or speak them — and get grounded, accurate answers powered by AWS Bedrock and a locally running Ollama model.

![Docker Knowledge Base]("../devesh-saini/Screenshot 2026-05-07 200916.png")

---

## What it does

Instead of digging through dense Docker documentation, you ask a question and get a focused answer. The app retrieves the most relevant chunks from a knowledge base and generates a response grounded in that content — not model memory alone.

- **Voice in** — speak your query via the microphone (powered by faster-whisper)
- **Streamed response** — answer types out in real time, token by token
- **Voice out** — response is read aloud via AWS Polly
- **Multi-turn conversation** — follow-up questions are understood in context
- **Smart fallback** — if Ollama is down, AWS Bedrock handles generation automatically

---

## Architecture

```
Browser (HTML / CSS / JS)
    │
    ├── GET  /stream/      → Django → is_ollama_running()?
    │                                 ├── Yes → Retrieve (Bedrock) → Generate (Ollama)
    │                                 └── No  → Retrieve & Generate (Bedrock)
    │
    ├── GET  /polly/       → Django → AWS Polly → MP3 audio (TTS)
    │
    └── POST /transcribe/  → Django → faster-whisper → transcript (STT)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Django 5 |
| Knowledge base & retrieval | AWS Bedrock |
| Local LLM | Ollama + Mistral |
| Bedrock fallback | Ollama + Mistral |
| Text to Speech | AWS Polly |
| Speech to Text | faster-whisper (local) |
| Error monitoring | Sentry |
| Frontend | Vanilla HTML, CSS, JS |

---

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com) installed and running locally
- AWS account with:
  - Bedrock Knowledge Base set up
  - Polly access
  - IAM credentials configured
- Mistral model pulled in Ollama:
  ```bash
  ollama pull mistral:latest
  ```

---

## Setup

**1. Clone the repo:**
```bash
git clone https://github.com/devesh-saini/bedrockOllama.git
cd bedrockOllama/DockerRag
```

**2. Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables:**

Copy `.env.example` to `.env` and fill in your values:
```bash
cp .env.example .env
```

```env
DJANGO_SECRET_KEY=your_secret_key
REGION=eu-north-1
BEDROCK_KB_ID=your_knowledge_base_id
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
SENTRY_DSN=your_sentry_dsn
```

**5. Run migrations:**
```bash
python manage.py migrate
```

**6. Start the server:**
```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000`

---

## IAM Permissions Required

```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:Retrieve",
    "bedrock:RetrieveAndGenerate",
    "polly:SynthesizeSpeech"
  ],
  "Resource": "*"
}
```

---

## Features in Detail

**Retrieval-Augmented Generation**
Queries are answered using real documentation chunks retrieved from AWS Bedrock — not hallucinated from model weights. Each response shows an accuracy score and chunk count.

**Ollama Fallback**
On every request, the app checks if Ollama is running. If it is, Mistral handles generation locally at zero cost. If not, Bedrock's  model takes over automatically — no user-facing disruption.

**Streaming Responses**
Responses stream token by token via Server-Sent Events (SSE), giving a real-time feel without page reloads.

**Voice Pipeline**
Speak a query → faster-whisper transcribes locally → answer generated → AWS Polly reads it aloud. Fully local STT, cloud TTS.

**Multi-turn Conversation**
Conversation history is stored in Django sessions. Follow-up questions are passed with full context so the model understands references to previous answers.

---

## Project Structure

---

## Running the RAG pipeline standalone

```bash
cd DockerRag
python DockerRag/rag.py
>>> How do I create a Docker network?
```

