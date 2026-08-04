# 🎯 AI Interview Prep Coach

An AI-powered interview preparation assistant built with **LangGraph**, featuring intent classification, parallel question generation, live web search grounding, and resume-aware RAG — wrapped in a **Streamlit** chat interface.

Instead of generating a single generic practice question, this coach classifies what you're actually asking for, and routes you through a purpose-built workflow: DSA practice, behavioral questions, company research, resume gap analysis, or just casual conversation.

---

## Features

- **Intent classification** — a router node reads each message and classifies it into one of five categories (`dsa`, `behavioral`, `company_research`, `resume_gap`, `general_chat`) before deciding what to do next.
- **Parallel DSA question generation** — when you ask for a coding question, three nodes run *in parallel*, each generating a candidate question from a different topic area (arrays/hashing, trees/graphs, dynamic programming). A dedicated picker node then selects the single best candidate based on your stated difficulty and topic preference.
- **Adaptive difficulty** — the coach remembers the difficulty level from your last question, so relative requests like *"give me an easier one"* actually adjust from where you left off, instead of guessing from scratch every time.
- **No-repeat question tracking** — previously asked questions are tracked across the conversation and explicitly avoided when generating new ones, so you don't see the same question twice.
- **Live company research** — pulls real, current information via Tavily web search instead of letting the model guess from stale training data, then summarizes it into Interview Process / Culture / Recent News.
- **Resume-grounded gap analysis** — uses a FAISS + HuggingFace embeddings RAG pipeline over your actual resume PDF to identify real skill gaps for a target role, instead of inventing generic advice.
- **Conversational memory** — powered by LangGraph's checkpointer, so the whole session (difficulty level, asked questions, chat history) persists turn to turn.

---

## Architecture

```
START
  │
  ▼
classifier_node ──┬── "dsa" ─────────► reviwer_node (difficulty detection)
                   │                        │
                   │                        ▼
                   │            ┌───────────┼───────────┐
                   │            ▼            ▼            ▼
                   │   arrays_hashing   trees_graphs      dp
                   │      _node           _node         _node
                   │            └───────────┼───────────┘
                   │                        ▼
                   │                  picker_node ──► END
                   │
                   ├── "company_research" ──► company_reasearch_node ──► END
                   ├── "resume_gap" ─────────► resume_gap_node ──► END
                   ├── "behavioral" ─────────► behavioral_node ──► END
                   └── "general_chat" ───────► general_chat_node ──► END
```

The three DSA nodes (`arrays_hashing`, `trees_graphs`, `dp`) write into a shared `candidate_questions` state key using a custom reducer, so all three survive instead of overwriting each other — this is what makes the fan-out → pick pattern work.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM | Groq (`llama-3.3-70b-versatile`) via `langchain-groq` |
| Web Search | [Tavily](https://tavily.com/) via `langchain-tavily` |
| Resume RAG | FAISS + HuggingFace sentence-transformer embeddings |
| Frontend | Streamlit |
| Memory | LangGraph `MemorySaver` checkpointer |

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

### 2. Set up environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 3. Add your resume

Place your resume PDF in the project, and either:
- update `RESUME_PATH` directly in the script, or
- set it via environment variable:
  ```
  RESUME_PATH=path/to/your_resume.pdf
  ```

### 4. Run the app

```bash
streamlit run interview_coach_app_final.py
```

This opens the app in your browser at `http://localhost:8501`.

---

## Usage

1. Enter your target **company** and **role** in the sidebar.
2. Chat naturally. Some things to try:
   - *"Give me a DSA question, medium level"*
   - *"That was tough, give me an easier one"*
   - *"What's their interview process and culture like?"*
   - *"What should I brush up on for this role?"*
   - *"Ask me a behavioral question"*
   - *"Hey, how's it going?"*
3. Use **Reset conversation** in the sidebar to start a fresh session (clears memory and asked-question history).

---

## Project Structure

```
.
├── interview_coach_app_final.py   # Streamlit app + LangGraph pipeline
├── requirements.txt
├── .env                            # not committed — holds API keys
└── README.md
```

---

## Notes / Limitations

- Difficulty and topic classification are LLM-based, not rule-based — they're generally reliable but not 100% deterministic.
- The FAISS resume index is built once per session and cached; if the resume file is missing or fails to load, resume-gap requests degrade gracefully with a clear message instead of crashing.
- Conversation memory currently uses a single session `thread_id` per app run — this is intended for individual local/demo use.

---

## Acknowledgements

Built while learning LangGraph's conditional routing, parallel fan-out with custom reducers, tool-grounded generation, and RAG — as part of a broader hands-on AI/ML project series.
