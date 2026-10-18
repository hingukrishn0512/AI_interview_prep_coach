# 🎯 AI Interview Prep Coach

An AI-powered interview preparation assistant built with **LangGraph** and **FastAPI**. You enter a target company and role, **upload your own resume**, and chat with a coach that tailors everything to you: DSA practice, behavioral questions, live company research, and resume-based coaching.

Instead of generating one generic practice question, the coach classifies what you're actually asking for and routes it through a purpose-built workflow.

---

## Features

- **Resume upload per session**: each user uploads their own PDF resume when starting a session. Resume questions ("what should I brush up on?", "how do I explain my chatbot project?", "give me an elevator pitch") are answered from *that* resume only.
- **Intent classification**: a router node reads each message, together with the conversation history, and classifies it into one of five categories: `dsa`, `behavioral`, `company_research`, `resume_gap`, `general_chat`.
- **Smart DSA question generation**: one LLM call extracts both the **topic** (arrays/hashing, trees/graphs, dynamic programming) and the **difficulty** from your message and the chat history, so follow-ups like *"give me a harder one"* continue from where you left off. A single topic goes straight to the matching generator node. Only if the topic is genuinely ambiguous do all three run in parallel, followed by a picker node.
- **Asks before guessing**: bare requests such as "give me a DSA question" or "give me an HR question" get a short clarifying question (topic, difficulty, theme) instead of a random pick.
- **Live company research**: pulls current information with Tavily web search instead of relying on stale training data, then summarizes it as Interview Process / Culture / Recent News. If the search fails, the coach says so honestly instead of crashing.
- **Resume-grounded coaching**: the full resume text is placed in the prompt (a resume is only 1–2 pages), so any question about any project or skill can be answered. There is no embedding API, no vector store, and no retrieval step that might miss something.
- **Conversational memory**: LangGraph's checkpointer keeps each session's history, so the coach can answer questions like "what did I ask you first?" accurately.
- **Resilient LLM calls**: exponential backoff with jitter on rate limits, a fast failure on exhausted quotas, and a clean 503 so the UI can offer a Retry button.
- **Session hygiene**: resumes and chat history are kept in memory only, deleted when the user clicks "New session", and automatically expired after 2 hours. The uploaded PDF is never saved to disk.

---

## Architecture

```
START
  │
  ▼
classifier_node ──┬── "dsa" ──────────► reviwer_node (topic + difficulty)
                  │                          │
                  │                  topic_router_function
                  │            ┌─────────────┼─────────────┐
                  │            ▼             ▼             ▼
                  │    array_hasing_node  trees_graphs_node  dp_node
                  │            └─────────────┼─────────────┘
                  │                          ▼
                  │                     picker_node ──► END
                  │      (one topic: only that node runs;
                  │       ambiguous: all three run, picker chooses)
                  │
                  ├── "company_research" ──► company_reasearch_node ──► END
                  ├── "resume_gap" ────────► resume_gap_node ──► END
                  ├── "behavioral" ────────► behavioral_node ──► END
                  └── "general_chat" ──────► general_chat_node ──► END
```

The DSA generator nodes write into a shared `candidate_questions` key using a custom reducer, so parallel results merge instead of overwriting each other. `requested_topics` is set fresh every turn, so the picker never mixes in questions from earlier turns.

### Request flow

```
Browser (static/index.html)
   │  POST /session  (company, role, resume PDF)  ──► extracts resume text, creates thread_id
   │  POST /chat     (thread_id, message)         ──► runs the LangGraph pipeline
   │  DELETE /session/{thread_id}                 ──► frees resume + chat history
   ▼
FastAPI (main.py)  ──►  LangGraph graph (project_raw.py)  ──►  Groq LLM / Tavily search
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM | Groq (`openai/gpt-oss-120b`) via `langchain-groq` |
| Web search | [Tavily](https://tavily.com/) via `langchain-tavily` |
| Resume parsing | `pypdf` via `PyPDFLoader` |
| Backend API | FastAPI + Uvicorn |
| Frontend | Plain HTML/CSS/JS (served by FastAPI from `static/`) |
| Memory | LangGraph `MemorySaver` checkpointer (in-memory) |

---

## Project Structure

```
.
├── Agent/
│   ├── main.py            # FastAPI app: /session, /chat, /session/{id}
│   ├── project_raw.py     # LangGraph pipeline, nodes, prompts, resume storage
│   ├── static/
│   │   └── index.html     # chat UI (setup screen + chat screen)
│   ├── requirements.txt
│   └── .env               # not committed: API keys
└── README.md
```

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd <repo-folder>/Agent
pip install -r requirements.txt
```

`python-multipart` is required for the resume upload.

### 2. Environment variables

Create `Agent/.env`:

```
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 3. Run

```bash
python main.py
```

Open `http://localhost:8000`, enter a company and role, upload your resume PDF, and start chatting.

Or run it directly with Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/session` | Multipart form: `company_name`, `role`, `resume` (PDF, max 5 MB). Returns `{thread_id}`. |
| `POST` | `/chat` | JSON: `{thread_id, user_input}`. Returns `{final_result, classifier}`. Returns 404 if the session expired, 503 if the LLM provider is rate-limiting. |
| `DELETE` | `/session/{thread_id}` | Deletes the session's resume and chat history. |
| `GET` | `/health` | Health check for deployment platforms. |

---

## Usage

1. Enter your target **company** and **role**, and upload your **resume** (text-based PDF).
2. Chat naturally. Some things to try:
   - *"Give me a medium question on graphs"*
   - *"Give me a harder one"*
   - *"What's their interview process and culture like?"*
   - *"What should I brush up on for this role?"*
   - *"How should I explain my first project in an interview?"*
   - *"Give me an elevator pitch based on my resume"*
   - *"Ask me a behavioral question about handling conflict"*
3. Click **New session** to start over with a different company, role, or resume.

---

## Deployment (Render)

- **Root Directory:** `Agent`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1`
- **Environment variables:** `GROQ_API_KEY`, `TAVILY_API_KEY`
- If the build fails on a very new Python version, set `PYTHON_VERSION` to `3.12.8`.

> **Use a single worker.** Resumes and chat history live in process memory, so multiple workers would not share sessions.

---

## Notes / Limitations

- **Sessions are in-memory.** A server restart (or a free-tier host spinning down) clears all sessions. The app detects this and sends the user back to the upload screen.
- **Scanned PDFs are not supported.** Resumes must contain selectable text. Image-only PDFs are rejected with a clear message.
- **Only resume-related requests read the resume.** Behavioral questions and general chat don't use it yet.
- Classification and topic/difficulty detection are LLM-based, so they are reliable but not 100% deterministic.
- Resume text is capped at about 15,000 characters, which is far more than a normal resume.
- CORS is open (`*`) for local development. Restrict it to your real domain in production.

---

## Acknowledgements

Built while learning LangGraph's conditional routing, parallel fan-out with custom reducers, tool-grounded generation, and per-user session design, as part of a hands-on AI/ML project series.
