from dataclasses import dataclass, field
from typing import List
from pydantic import BaseModel, Field


@dataclass
class ConfidentialDelivery:
    """Confidential payload delivered directly to the user out-of-band.
    
    The LLM NEVER sees this text in its context window or history.
    """
    topic: str
    content: str


@dataclass
class UserContext:
    """Internal user context passed to tools via Pydantic AI's RunContext.
    
    The LLM NEVER sees this object or its fields. It is injected
    directly by the system at runtime.
    """
    user_id: str
    username: str
    role: str
    allowed_tools: List[str] = field(default_factory=list)
    allowed_topics: List[str] = field(default_factory=list)
    allowed_confidential_topics: List[str] = field(default_factory=list)
    confidential_deliveries: List[ConfidentialDelivery] = field(default_factory=list)


# --- In-Memory Mock User Database ---
# Maps user_id -> UserContext
USERS_DATABASE = {
    "user_alice": UserContext(
        user_id="user_alice",
        username="Alice (Senior Analyst)",
        role="senior_analyst",
        allowed_tools=["topic_info", "database_query", "confidential_topic_rag"],
        allowed_topics=["topic_a", "topic_b"],
        allowed_confidential_topics=["quantum_keys", "payroll_audit"],
    ),
    "user_bob": UserContext(
        user_id="user_bob",
        username="Bob (Junior Analyst)",
        role="junior_analyst",
        allowed_tools=["topic_info", "confidential_topic_rag"],  # Has tool, but restricted topics
        allowed_topics=["topic_b"],                              # Only topic_b
        allowed_confidential_topics=["payroll_audit"],           # Cannot access quantum_keys
    ),
    "user_charlie": UserContext(
        user_id="user_charlie",
        username="Charlie (Guest)",
        role="guest",
        allowed_tools=[],                                        # No tools visible to LLM
        allowed_topics=[],
        allowed_confidential_topics=[],
    ),
}


# --- FastAPI Request & Response Schemas ---
class ChatRequest(BaseModel):
    user_id: str = Field(..., description="ID of the user interacting with the bot", example="user_alice")
    message: str = Field(..., description="User's prompt or question", example="Can you fetch the confidential documentation for quantum_keys?")


class ChatResponse(BaseModel):
    user_id: str
    reply: str
    visible_tools: List[str] = Field(..., description="List of tools the LLM was allowed to see for this user")
    confidential_deliveries: List[ConfidentialDelivery] = Field(
        default_factory=list,
        description="Confidential payloads delivered directly to the user out-of-band, redacted from LLM view",
    )
