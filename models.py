from dataclasses import dataclass, field
from typing import List, Optional
from pydantic import BaseModel, Field


@dataclass
class ConfidentialDelivery:
    """Confidential payload delivered directly to the user out-of-band.
    
    The LLM NEVER sees this text in its context window or history.
    """
    topic: str
    content: str


@dataclass
class PendingAction:
    """Represents a paused tool action awaiting structured human input."""
    action_id: str
    action_type: str  # e.g. "selection" or "confirmation"
    prompt: str
    options: List[str]


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
    pending_action: Optional[PendingAction] = None


# --- In-Memory Mock User Database ---
# Maps user_id -> UserContext
USERS_DATABASE = {
    "user_alice": UserContext(
        user_id="user_alice",
        username="Alice (Senior Analyst / Admin)",
        role="senior_analyst",
        allowed_tools=[
            "topic_info",
            "database_query",
            "confidential_topic_rag",
            "request_report_export",         # Structured HITL (options sent to UI)
            "execute_critical_system_action" # Verbal HITL (LLM asks in conversation)
        ],
        allowed_topics=["topic_a", "topic_b"],
        allowed_confidential_topics=["quantum_keys", "payroll_audit"],
    ),
    "user_bob": UserContext(
        user_id="user_bob",
        username="Bob (Junior Analyst)",
        role="junior_analyst",
        allowed_tools=[
            "topic_info",
            "confidential_topic_rag",
            "request_report_export",         # Bob can export reports
            # Bob CANNOT execute critical system actions or query the database
        ],
        allowed_topics=["topic_b"],
        allowed_confidential_topics=["payroll_audit"],
    ),
    "user_charlie": UserContext(
        user_id="user_charlie",
        username="Charlie (Guest)",
        role="guest",
        allowed_tools=[],                     # No tools visible to LLM
        allowed_topics=[],
        allowed_confidential_topics=[],
    ),
}


# --- FastAPI Request & Response Schemas ---
class ChatRequest(BaseModel):
    user_id: str = Field(..., description="ID of the user interacting with the bot", example="user_alice")
    message: str = Field(..., description="User's prompt or question", example="Can you export the financial report?")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation history tracking", example="session-101")


class ResumeRequest(BaseModel):
    user_id: str = Field(..., description="User ID resuming the action", example="user_alice")
    session_id: str = Field(..., description="Active session ID", example="session-101")
    action_id: str = Field(..., description="ID of the pending action", example="act-8f2b")
    selected_option: str = Field(..., description="The option selected by the user", example="CSV Format")


class ChatResponse(BaseModel):
    user_id: str
    session_id: str
    status: str = Field("completed", description="'completed' or 'needs_action'")
    reply: str
    visible_tools: List[str] = Field(..., description="List of tools the LLM was allowed to see for this user")
    confidential_deliveries: List[ConfidentialDelivery] = Field(
        default_factory=list,
        description="Confidential payloads delivered directly to the user out-of-band",
    )
    action_id: Optional[str] = Field(None, description="Action ID if status is 'needs_action'")
    action_prompt: Optional[str] = Field(None, description="Prompt describing the required human decision")
    options: Optional[List[str]] = Field(None, description="List of options presented to the user")
