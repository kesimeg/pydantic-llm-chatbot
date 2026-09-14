"""FastAPI application providing chat endpoints with user permission filtering
and out-of-band confidential payload delivery.
"""

from fastapi import FastAPI, HTTPException, status
from models import USERS_DATABASE, ChatRequest, ChatResponse
from agent_factory import create_agent_for_user

app = FastAPI(
    title="Pydantic AI Permission-Aware Chatbot API",
    description="API demonstrating per-user dynamic tool filtering, hidden context injection, and confidential out-of-band data delivery.",
    version="1.1.0",
)


@app.get("/")
async def root():
    return {
        "status": "online",
        "description": "Pydantic AI Chatbot API with dynamic tool filtering, hidden context, and confidential payload delivery.",
        "endpoints": {
            "POST /chat": "Send a message with user_id and prompt",
            "GET /users": "View mock users and their assigned permissions",
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


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Processes a user message through a dynamically filtered Pydantic AI agent."""
    # 1. Resolve user and permissions from in-memory store
    user = USERS_DATABASE.get(request.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{request.user_id}' not found. Available mock users: {list(USERS_DATABASE.keys())}",
        )

    # 2. Reset per-request confidential deliveries
    user.confidential_deliveries = []

    # 3. Dynamically instantiate an agent with ONLY the tools visible to this user
    agent = create_agent_for_user(user)

    # 4. Execute agent run:
    # - `deps=user` passes user context directly to tools via RunContext.
    # - The LLM NEVER sees the user_id or internal permissions in prompt/tool arguments.
    # - Confidential documents fetched by confidential_topic_rag are stored in
    #   user.confidential_deliveries while the LLM receives only a redacted receipt.
    try:
        result = await agent.run(request.message, deps=user)
        return ChatResponse(
            user_id=user.user_id,
            reply=result.output,
            visible_tools=user.allowed_tools,
            confidential_deliveries=list(user.confidential_deliveries),
        )
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
