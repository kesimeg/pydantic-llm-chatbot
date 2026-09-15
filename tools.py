"""Tool definitions with permissions, hidden context, confidential RAG, and Human-in-the-Loop (HITL) tools.
"""

import uuid
from typing import Dict, Optional
from pydantic_ai import RunContext, Tool
from models import UserContext, ConfidentialDelivery, PendingAction

# --- Mock Knowledge Base for General Topics ---
TOPIC_DATA = {
    "topic_a": "Project Falcon specifications: High-security architecture blueprints and quantum cryptography plans.",
    "topic_b": "Public Quarterly Overview: Q3 revenue grew by 15%, customer retention reached 94%, and global expansion is on schedule.",
}

# --- Mock Knowledge Base for Confidential Documentation ---
CONFIDENTIAL_DOCUMENTATION = {
    "quantum_keys": (
        "TOP SECRET // NOFORN\n"
        "Quantum Cryptography Master Key Material:\n"
        "Seed: 9f8a-bc34-1182-de09-55aa\n"
        "Rotational period: 60 minutes.\n"
        "Primary Hardware Security Module: Vault-Zero (Rack 4B)."
    ),
    "payroll_audit": (
        "CONFIDENTIAL // INTERNAL ONLY\n"
        "Executive Compensation & Bonus Allocations Q3:\n"
        "Total Bonus Pool: $1,450,000.\n"
        "CEO Allocation: $400,000; VP Engineering: $250,000."
    ),
}


# ==============================================================================
# Standard & Confidential Tools
# ==============================================================================

async def get_topic_information(ctx: RunContext[UserContext], topic: str) -> str:
    """Retrieve detailed information about a specific topic.
    
    Args:
        topic: The topic identifier to look up (e.g., 'topic_a', 'topic_b').
    """
    normalized_topic = topic.strip().lower()
    user = ctx.deps
    
    # In-Tool Permission Check
    if normalized_topic not in user.allowed_topics:
        return (
            f"Permission Denied: User '{user.username}' is not authorized "
            f"to view information on '{topic}'."
        )

    if normalized_topic in TOPIC_DATA:
        return f"Information for {topic}: {TOPIC_DATA[normalized_topic]}"
    else:
        available = ", ".join(TOPIC_DATA.keys())
        return f"Topic '{topic}' not found. Available topics in knowledge base: {available}"


async def confidential_topic_rag(ctx: RunContext[UserContext], topic: str) -> str:
    """Retrieve and securely dispatch confidential internal documentation directly to the user.
    The content is redacted from the LLM to prevent data exposure.
    
    Args:
        topic: The name of the confidential topic to retrieve (e.g., 'quantum_keys', 'payroll_audit').
    """
    normalized_topic = topic.strip().lower()
    user = ctx.deps
    
    # Permission check for confidential topic
    if normalized_topic not in user.allowed_confidential_topics:
        return (
            f"Access Denied: User '{user.username}' does not have security clearance "
            f"for confidential topic '{topic}'."
        )

    if normalized_topic not in CONFIDENTIAL_DOCUMENTATION:
        available = ", ".join(CONFIDENTIAL_DOCUMENTATION.keys())
        return f"Confidential topic '{topic}' not found. Available topics: {available}"

    # Retrieve and deliver out-of-band directly to user's payload
    full_content = CONFIDENTIAL_DOCUMENTATION[normalized_topic]
    user.confidential_deliveries.append(
        ConfidentialDelivery(topic=topic, content=full_content)
    )

    # Return REDACTED receipt to the LLM
    return (
        f"[REDACTED RECEIPT]: Confidential documentation for '{topic}' has been successfully "
        f"retrieved and dispatched directly to the user's secure side-channel. "
        f"The content is strictly redacted from your model context for security. "
        f"Notify the user that the document has been delivered directly to their secure screen."
    )


async def query_database(ctx: RunContext[UserContext], table_name: str) -> str:
    """Query internal database tables for administrative and technical records.
    
    Args:
        table_name: The name of the table to inspect (e.g., 'users', 'audit_logs', 'transactions').
    """
    user = ctx.deps
    
    if "database_query" not in user.allowed_tools:
        return f"Security Error: User '{user.username}' lacks database querying privileges."

    mock_db = {
        "users": "Table 'users': [ID 1: Alice (Admin), ID 2: Bob (Analyst), ID 3: Charlie (Guest)]",
        "audit_logs": "Table 'audit_logs': [2026-09-15 10:15: User Alice performed backup]",
        "transactions": "Table 'transactions': [TX1092: $5,400 completed, TX1093: $1,200 pending]",
    }
    
    table = table_name.strip().lower()
    if table in mock_db:
        return f"Database query result for {table_name}: {mock_db[table]}"
    return f"Table '{table_name}' not found. Available tables: {', '.join(mock_db.keys())}"


# ==============================================================================
# Human-In-The-Loop (HITL) Tools
# ==============================================================================

async def request_report_export(
    ctx: RunContext[UserContext],
    report_name: str,
    format: Optional[str] = None,
) -> str:
    """Export an analytical report. If 'format' is not provided, the tool pauses and
    presents structured format options directly to the client UI for selection.
    
    Args:
        report_name: The name of the report to export (e.g., 'financial_q3', 'traffic_metrics').
        format: The export format ('Executive Summary', 'Full Raw Logs', 'CSV Format').
    """
    user = ctx.deps
    if "request_report_export" not in user.allowed_tools:
        return f"Security Error: User '{user.username}' lacks permission to export reports."

    valid_formats = ["Executive Summary", "Full Raw Logs", "CSV Format"]

    # If no valid format was provided yet, pause and register pending action for client UI
    if not format or format not in valid_formats:
        action_id = f"act-{uuid.uuid4().hex[:6]}"
        user.pending_action = PendingAction(
            action_id=action_id,
            action_type="selection",
            prompt=f"Please select an export format for report '{report_name}':",
            options=valid_formats,
        )
        return (
            f"ACTION_REQUIRED: The client UI is being presented with format options: {valid_formats}. "
            f"Action ID: {action_id}. Acknowledge to the user that you are waiting for their format selection."
        )

    # When format has been chosen by the user
    mock_data = {
        "Executive Summary": f"=== EXECUTIVE SUMMARY: {report_name.upper()} ===\n• Key Takeaway: Growth rate +18% YoY.\n• Status: Operational.",
        "Full Raw Logs": f"=== RAW LOGS: {report_name.upper()} ===\n[2026-09-15 12:00:01] event=sync status=ok latency=14ms\n[2026-09-15 12:00:02] event=report_calc cpu=42%",
        "CSV Format": f"metric,period,value\nrevenue,q3,450000\nactive_users,q3,12400\nchurn_rate,q3,0.03",
    }
    return f"Report '{report_name}' generated successfully ({format}):\n\n{mock_data.get(format)}"


async def execute_critical_system_action(
    ctx: RunContext[UserContext],
    action_name: str,
    confirmed: bool = False,
) -> str:
    """Execute a critical infrastructure or high-risk administrative action.
    Requires VERBAL human confirmation in natural chat before execution.
    
    Args:
        action_name: The operation to execute (e.g., 'restart_primary_cluster', 'flush_redis_cache').
        confirmed: Set to True ONLY if the user has explicitly confirmed this action verbally in chat history.
    """
    user = ctx.deps
    if "execute_critical_system_action" not in user.allowed_tools:
        return f"Security Error: User '{user.username}' lacks privileges to execute critical system operations."

    if not confirmed:
        return (
            f"VERBAL CONFIRMATION REQUIRED: Action '{action_name}' is a high-risk system operation. "
            f"The user has NOT confirmed it yet. Do NOT execute this action now. "
            f"In your response, verbally ask the user: 'Are you sure you want to proceed with {action_name}?' "
            f"and wait for their verbal response."
        )

    # User confirmed verbally in previous turn
    return (
        f"SUCCESS: Critical system operation '{action_name}' has been executed successfully "
        f"under authorization of user '{user.username}'."
    )


# --- Tool Registry ---
TOOL_REGISTRY: Dict[str, Tool[UserContext]] = {
    "topic_info": Tool(get_topic_information, takes_ctx=True),
    "confidential_topic_rag": Tool(confidential_topic_rag, takes_ctx=True),
    "database_query": Tool(query_database, takes_ctx=True),
    "request_report_export": Tool(request_report_export, takes_ctx=True),
    "execute_critical_system_action": Tool(execute_critical_system_action, takes_ctx=True),
}
