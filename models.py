from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


@dataclass
class ConfidentialDelivery:
    """General out-of-band data delivery payload delivered directly to the client.
    
    The LLM NEVER sees this content in its context window, prompt, or message history.
    Any tool can use this to deliver sensitive documents, reports, raw records, etc.
    """
    content: str
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PendingAction:
    """Generic representation of a tool action awaiting human input (buttons, approvals, selections).
    
    Fully adaptable to any tool without tool-specific arguments hardcoded.
    """
    action_id: str
    prompt: str
    options: List[str]
    tool_name: Optional[str] = None
    context_data: Dict[str, Any] = field(default_factory=dict)
    resume_instruction: Optional[str] = None


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
    
    # Generic permissions store (e.g. {"topics": [...], "clearances": [...], ...})
    permissions: Dict[str, Any] = field(default_factory=dict)
    
    # Generic out-of-band delivery buffer (cleared per request)
    confidential_deliveries: List[ConfidentialDelivery] = field(default_factory=list)
    
    # Generic pending action state
    pending_action: Optional[PendingAction] = None
    
    # Generic human selection storage: tool-agnostic
    action_selections: Dict[str, Any] = field(default_factory=dict)
    last_selected_option: Optional[str] = None

    def deliver_confidential(self, content: str, label: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Helper for any tool to deliver data out-of-band without LLM visibility."""
        self.confidential_deliveries.append(
            ConfidentialDelivery(
                content=content,
                label=label,
                metadata=metadata or {},
            )
        )

    def has_permission(self, permission_category: str, item: str) -> bool:
        """Helper to verify if a user has access to a specific permission scope."""
        allowed = self.permissions.get(permission_category, [])
        if isinstance(allowed, list):
            return item.lower() in [str(x).lower() for x in allowed]
        return False


# --- In-Memory Mock User Database ---
# Maps user_id -> UserContext with generic permission scopes
USERS_DATABASE = {
    "user_alice": UserContext(
        user_id="user_alice",
        username="Alice (Senior Analyst / Admin)",
        role="senior_analyst",
        allowed_tools=[
            "topic_info",
            "database_query",
            "confidential_topic_rag",
            "request_report_export",
            "execute_critical_system_action",
        ],
        permissions={
            "topics": ["topic_a", "topic_b"],
            "clearances": ["quantum_keys", "payroll_audit"],
        },
    ),
    "user_bob": UserContext(
        user_id="user_bob",
        username="Bob (Junior Analyst)",
        role="junior_analyst",
        allowed_tools=[
            "topic_info",
            "confidential_topic_rag",
            "request_report_export",
            # Bob lacks database_query and execute_critical_system_action
        ],
        permissions={
            "topics": ["topic_b"],                     # Only topic_b
            "clearances": ["payroll_audit"],            # Lacks quantum_keys
        },
    ),
    "user_charlie": UserContext(
        user_id="user_charlie",
        username="Charlie (Guest)",
        role="guest",
        allowed_tools=[],                               # Zero tools visible to LLM
        permissions={},
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
    selected_option: str = Field(..., description="The option clicked or selected by the user", example="CSV Format")


class ChatResponse(BaseModel):
    user_id: str
    session_id: str
    status: str = Field("completed", description="'completed' or 'needs_action'")
    reply: str
    visible_tools: List[str] = Field(..., description="List of tools the LLM was allowed to see for this user")
    confidential_deliveries: List[ConfidentialDelivery] = Field(
        default_factory=list,
        description="Data payloads delivered directly to the user out-of-band",
    )
    action_id: Optional[str] = Field(None, description="Action ID if status is 'needs_action'")
    action_prompt: Optional[str] = Field(None, description="Prompt describing the required human decision")
    options: Optional[List[str]] = Field(None, description="List of options presented to the user as buttons")
