import os
import uuid
import logging
import tempfile
import traceback

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from project_raw import (
    create_graph,
    RateLimitExceeded,
    register_resume,
    has_resume,
    clear_resume,
    pop_expired_sessions,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("interview_coach.api")

MAX_RESUME_BYTES = 5 * 1024 * 1024  # 5 MB
SESSION_MAX_AGE_SECONDS = 2 * 60 * 60  # sessions older than 2 hours get cleaned up

# --- build the graph ONCE, when the API server starts up ---
compiled_graph = create_graph()

app = FastAPI(title="AI Interview Prep Coach API")

# The frontend is served by this same app, so CORS is only needed if you host the
# UI somewhere else. Credentials are off because "*" origins + credentials is
# not a valid combination. In production, replace "*" with your real domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- request/response schemas ---

class StartSessionResponse(BaseModel):
    thread_id: str


class ChatRequest(BaseModel):
    thread_id: str
    user_input: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    final_result: str
    classifier: str


# --- helpers ---

def drop_thread(thread_id: str):
    """Forgets everything about a session: its resume and its chat history."""
    clear_resume(thread_id)
    saver = getattr(compiled_graph, "checkpointer", None)
    delete_thread = getattr(saver, "delete_thread", None)
    if delete_thread:
        try:
            delete_thread(thread_id)
        except Exception as e:
            logger.warning("Couldn't delete checkpoints for %s: %s", thread_id, e)


def cleanup_old_sessions():
    for thread_id in pop_expired_sessions(SESSION_MAX_AGE_SECONDS):
        drop_thread(thread_id)
        logger.info("Cleaned up expired session %s", thread_id)


# --- frontend ---
# serves the chat UI at the root URL, and any other files placed in /static
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


# --- endpoints ---

@app.get("/health")
def health_check():
    """Simple endpoint to confirm the server is up."""
    return {"status": "ok"}


@app.post("/session", response_model=StartSessionResponse)
def start_session(
    company_name: str = Form(...),
    role: str = Form(...),
    resume: UploadFile = File(...),
):
    """Creates a new conversation session for a company + role + uploaded resume.
       Call this ONCE at the start of a user's conversation."""

    company_name = company_name.strip()
    role = role.strip()
    if not company_name or not role:
        raise HTTPException(status_code=400, detail="Company and role are required.")

    cleanup_old_sessions()

    # Read one byte past the limit, so oversized files are rejected without
    # loading the whole thing into memory.
    data = resume.file.read(MAX_RESUME_BYTES + 1)
    if len(data) > MAX_RESUME_BYTES:
        raise HTTPException(status_code=413, detail="Resume is too large (max 5 MB).")
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Please upload a valid PDF file.")

    thread_id = str(uuid.uuid4())

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        register_resume(thread_id, tmp_path)
    except ValueError as e:
        logger.error("Resume has no readable text: %s", e)
        raise HTTPException(
            status_code=400,
            detail="Couldn't read text from that PDF. Make sure it isn't a scanned image.",
        )
    except Exception as e:
        logger.error("Resume processing failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Couldn't process your resume. Please try again.",
        )
    finally:
        # the uploaded PDF itself is never kept on disk
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    try:
        # update_state seeds the checkpoint with company_name/role WITHOUT running
        # any graph nodes - it's a direct write to the persisted state for this thread_id
        config = {"configurable": {"thread_id": thread_id}}
        compiled_graph.update_state(
            config,
            {
                "company_name": company_name,
                "role": role,
                "user_input": "",
                "messages": [],
                "classifier": "",
                "difficulty_level": "",
                "topic": "",
                "requested_topics": [],
                "final_result": "",
                "candidate_questions": {},
            },
        )
    except Exception as e:
        logger.error("Couldn't create session state: %s", e)
        drop_thread(thread_id)
        raise HTTPException(status_code=500, detail="Couldn't start the session. Please try again.")

    return StartSessionResponse(thread_id=thread_id)


@app.delete("/session/{thread_id}")
def end_session(thread_id: str):
    """Frees the resume and chat history of a finished session."""
    drop_thread(thread_id)
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    """Sends one user message into an existing session and returns the answer."""

    # Sessions live in memory, so a server restart (or cleanup) loses them.
    # Return a clear 404 so the frontend can send the user back to the setup screen.
    if not has_resume(payload.thread_id):
        raise HTTPException(
            status_code=404,
            detail="This session has expired. Please start a new session and upload your resume again.",
        )

    config = {"configurable": {"thread_id": payload.thread_id}}

    try:
        result = compiled_graph.invoke(
            {"user_input": payload.user_input.strip()},
            config=config,
        )
    except RateLimitExceeded as e:
        # 503 tells the frontend this is transient and worth retrying shortly.
        logger.error("Rate limit exhausted for thread %s: %s", payload.thread_id, e)
        raise HTTPException(
            status_code=503,
            detail="The AI provider is rate-limiting requests right now. Please wait a few seconds and try again.",
        )
    except Exception as e:
        # Log the FULL traceback server-side so the real cause is visible in the logs.
        logger.error("chat() failed for thread %s: %s", payload.thread_id, e)
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while generating the answer. Please try again.",
        )

    return ChatResponse(
        final_result=result["final_result"],
        classifier=result["classifier"],
    )


if __name__ == "__main__":
    import uvicorn
    # Keep a single worker: resumes are stored in this process's memory.
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)