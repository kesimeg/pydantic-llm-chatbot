"""FastAPI application providing chat endpoints with user permission filtering,
in-memory session history, out-of-band confidential payload delivery,
and Human-In-The-Loop (HITL) pause-and-resume.
"""

import uuid
from typing import Dict, List, Tuple
from fastapi import FastAPI, HTTPException, status
from pydantic_ai.messages import ModelMessage

from models import USERS_DATABASE, ChatRequest, ChatResponse, ResumeRequest
from agent_factory import create_agent_for_user

app = FastAPI(
    title="Pydantic AI Permission & HITL Chatbot API",
    description="Chatbot API with permissions, in-memory chat history, confidential RAG, and HITL tools.",
    version="1.2.0",
)

# -----------------------------------------------------------------------------
# In-Memory Storage
# -----------------------------------------------------------------------------
# Maps (user_id, session_id) -> list[ModelMessage]
SESSIONS: Dict[Tuple[str, str], List[ModelMessage]] = {}

# Maps action_id -> dict with action details
PENDING_ACTIONS: Dict[str, dict] = {}


@app.get("/")
async def root():
    return {
        "status": "online",
        "description": "Pydantic AI Chatbot API with permissions, chat history, and HITL tools.",
        "endpoints": {
            "POST /chat": "Send a prompt with user_id and optional session_id",
            "POST /chat/resume": "Resume a paused tool with user's selected option",
            "GET /users": "View mock users and their assigned permissions",
            "GET /sessions/{user_id}": "List active sessions for a user",
            "DELETE /sessions/{user_id}/{session_id}": "Clear chat history for a session",
        },
    }


@app.get("/users")
async def list_users():
    """Lists mock users in the in-memory database and their current permissions."""
    return {
        user_id: {
            "username": user.username,
            "role": user.role,
            "allowed_tools": user.allowed_tools,
            "allowed_topics": user.allowed_topics,
            "allowed_confidential_topics": user.allowed_confidential_topics,
        }
        for user_id, user in USERS_DATABASE.items()
    }


@app.get("/sessions/{user_id}")
async def list_user_sessions(user_id: str):
    """Returns active session IDs and message counts for a user."""
    sessions = []
    for (uid, sid), history in SESSIONS.items():
        if uid == user_id:
            sessions.append({"session_id": sid, "message_count": len(history)})
    return {"user_id": user_id, "sessions": sessions}


@app.delete("/sessions/{user_id}/{session_id}")
async def clear_session(user_id: str, session_id: str):
    """Clears conversation history for a specific user session."""
    key = (user_id, session_id)
    if key in SESSIONS:
        del SESSIONS[key]
        return {"status": "success", "message": f"Session '{session_id}' cleared for user '{user_id}'."}
    return {"status": "not_found", "message": f"No active session '{session_id}' for user '{user_id}'."}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Processes a user message through a dynamically filtered Pydantic AI agent,
    maintaining in-memory multi-turn conversation history.
    """
    user = USERS_DATABASE.get(request.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{request.user_id}' not found. Available mock users: {list(USERS_DATABASE.keys())}",
        )

    # 1. Resolve or generate session_id
    session_id = request.session_id or f"sess-{uuid.uuid4().hex[:6]}"

    # 2. Reset per-request buffers
    user.confidential_deliveries = []
    user.pending_action = None

    # 3. Retrieve chat history for this user & session
    history = SESSIONS.get((user.user_id, session_id), [])

    # 4. Dynamically instantiate agent with user's permitted tools
    agent = create_agent_for_user(user)

    try:
        # Run agent with history and hidden user context
        result = await agent.run(request.message, deps=user, message_history=history)
        
        # Save updated conversation history
        SESSIONS[(user.user_id, session_id)] = result.all_messages()

        # 5. Check if a tool paused for human input (Structured HITL)
        if user.pending_action:
            pending = user.pending_action
            PENDING_ACTIONS[pending.action_id] = {
                "user_id": user.user_id,
                "session_id": session_id,
                "action_type": pending.action_type,
            }
            return ChatResponse(
                user_id=user.user_id,
                session_id=session_id,
                status="needs_action",
                reply=result.output,
                visible_tools=user.allowed_tools,
                confidential_deliveries=list(user.confidential_deliveries),
                action_id=pending.action_id,
                action_prompt=pending.prompt,
                options=pending.options,
            )

        # Standard completed turn
        return ChatResponse(
            user_id=user.user_id,
            session_id=session_id,
            status="completed",
            reply=result.output,
            visible_tools=user.allowed_tools,
            confidential_deliveries=list(user.confidential_deliveries),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {str(e)}",
        )


@app.post("/chat/resume", response_model=ChatResponse)
async def resume_action(request: ResumeRequest):
    """Resumes a paused tool execution with the option selected by the user."""
    user = USERS_DATABASE.get(request.user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    pending_info = PENDING_ACTIONS.get(request.action_id)
    if not pending_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Action ID '{request.action_id}' not found or already completed.",
        )

    # Reset per-request buffers
    user.confidential_deliveries = []
    user.pending_action = None

    # Retrieve history
    history = SESSIONS.get((user.user_id, request.session_id), [])

    # Instantiate agent
    agent = create_agent_for_user(user)

    # Continue agent execution with the user's selection
    resume_prompt = (
        f"The user selected the option '{request.selected_option}' for action '{request.action_id}'. "
        f"Please proceed with the tool operation using this selected option."
    )

    try:
        result = await agent.run(resume_prompt, deps=user, message_history=history)
        SESSIONS[(user.user_id, request.session_id)] = result.all_messages()
        
        # Remove completed action from pending store
        PENDING_ACTIONS.pop(request.action_id, None)

        return ChatResponse(
            user_id=user.user_id,
            session_id=request.session_id,
            status="completed",
            reply=result.output,
            visible_tools=user.allowed_tools,
            confidential_deliveries=list(user.confidential_deliveries),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed resuming agent: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
