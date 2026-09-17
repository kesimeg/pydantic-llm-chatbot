"""Tool definitions with permissions, hidden context, confidential out-of-band delivery,
and Human-in-the-Loop (HITL) tools.

All mock data dictionaries are encapsulated inside their respective tools for easy replacement.
"""

import uuid
from typing import Dict
from pydantic_ai import RunContext, Tool
from models import UserContext, PendingAction


# ==============================================================================
# Standard & Knowledge Retrieval Tools
# ==============================================================================

async def get_topic_information(ctx: RunContext[UserContext], topic: str) -> str:
    """Retrieve detailed information about a specific topic.
    
    Args:
        topic: The topic identifier to look up (e.g., 'topic_a', 'topic_b').
    """
    # Encapsulated mock data (easily replaced with a vector DB or external service)
    topic_database = {
        "topic_a": "Project Falcon specifications: High-security architecture blueprints and quantum cryptography plans.",
        "topic_b": "Public Quarterly Overview: Q3 revenue grew by 15%, customer retention reached 94%, and global expansion is on schedule.",
    }

    normalized_topic = topic.strip().lower()
    user = ctx.deps
    
    # Generic In-Tool Permission Check
    if not user.has_permission("topics", normalized_topic):
        return (
            f"Permission Denied: User '{user.username}' is not authorized "
            f"to view information on '{topic}'."
        )

    if normalized_topic in topic_database:
        return f"Information for {topic}: {topic_database[normalized_topic]}"
    else:
        available = ", ".join(topic_database.keys())
        return f"Topic '{topic}' not found. Available topics in knowledge base: {available}"


async def confidential_topic_rag(ctx: RunContext[UserContext], topic: str) -> str:
    """Retrieve and securely dispatch confidential internal documentation directly to the user.
    The content is redacted from the LLM to prevent sensitive data exposure.
    
    Args:
        topic: The name of the confidential topic to retrieve (e.g., 'quantum_keys', 'payroll_audit').
    """
    # Encapsulated mock confidential storage
    confidential_store = {
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

    normalized_topic = topic.strip().lower()
    user = ctx.deps
    
    # Generic clearance check
    if not user.has_permission("clearances", normalized_topic):
        return (
            f"Access Denied: User '{user.username}' does not have security clearance "
            f"for confidential topic '{topic}'."
        )

    if normalized_topic not in confidential_store:
        available = ", ".join(confidential_store.keys())
        return f"Confidential topic '{topic}' not found. Available topics: {available}"

    # Retrieve and deliver out-of-band directly to user's payload
    full_content = confidential_store[normalized_topic]
    user.deliver_confidential(
        content=full_content,
        label=f"Confidential Document: {topic}",
        metadata={"topic": topic, "source": "confidential_topic_rag"},
    )

    # Return REDACTED receipt to the LLM (raw text is not in LLM context)
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
    # Encapsulated mock database tables
    mock_database = {
        "users": "Table 'users': [ID 1: Alice (Admin), ID 2: Bob (Analyst), ID 3: Charlie (Guest)]",
        "audit_logs": "Table 'audit_logs': [2026-09-17 10:15: User Alice performed backup]",
        "transactions": "Table 'transactions': [TX1092: $5,400 completed, TX1093: $1,200 pending]",
    }

    user = ctx.deps
    if "database_query" not in user.allowed_tools:
        return f"Security Error: User '{user.username}' lacks database querying privileges."

    table = table_name.strip().lower()
    if table in mock_database:
        return f"Database query result for {table_name}: {mock_database[table]}"
    return f"Table '{table_name}' not found. Available tables: {', '.join(mock_database.keys())}"


# ==============================================================================
# Human-In-The-Loop (HITL) Tools
# ==============================================================================

async def request_report_export(
    ctx: RunContext[UserContext],
    report_name: str,
) -> str:
    """Export an analytical report for a given topic or dataset.
    The LLM triggers this tool with the report name only.
    Format options are presented directly to the human as UI buttons, and the selected
    format is injected in the background without the LLM seeing or choosing the format argument.
    The generated report data is dispatched securely via out-of-band delivery.
    
    Args:
        report_name: The name of the report to export (e.g., 'financial_q3', 'traffic_metrics').
    """
    user = ctx.deps
    if "request_report_export" not in user.allowed_tools:
        return f"Security Error: User '{user.username}' lacks permission to export reports."

    valid_formats = ["Executive Summary", "Full Raw Logs", "CSV Format"]

    # Step 1: If no selection was injected in background, pause and emit button options
    selected_option = user.last_selected_option
    if not selected_option or selected_option not in valid_formats:
        action_id = f"act-{uuid.uuid4().hex[:6]}"
        user.pending_action = PendingAction(
            action_id=action_id,
            prompt=f"Please select an export format for report '{report_name}':",
            options=valid_formats,
            tool_name="request_report_export",
            context_data={"report_name": report_name},
            resume_instruction=(
                f"The human has selected their format option for action '{action_id}'. "
                f"Please call 'request_report_export(report_name=\"{report_name}\")' to complete the report export."
            ),
        )
        return (
            f"PAUSED_FOR_USER_SELECTION: Format options {valid_formats} have been sent directly to the user's interface. "
            f"Action ID: {action_id}. The user will select a format button. "
            f"Acknowledge that you are waiting for the user's selection."
        )

    # Step 2: Human made their selection! (Injected in background into user.last_selected_option)
    fmt = selected_option
    user.last_selected_option = None  # Consume

    # Encapsulated mock report generation data
    mock_reports = {
        "Executive Summary": f"=== EXECUTIVE SUMMARY: {report_name.upper()} ===\n• Key Takeaway: Growth rate +18% YoY.\n• Status: Operational.",
        "Full Raw Logs": f"=== RAW LOGS: {report_name.upper()} ===\n[2026-09-17 10:00:01] event=sync status=ok latency=14ms\n[2026-09-17 10:00:02] event=report_calc cpu=42%",
        "CSV Format": f"metric,period,value\nrevenue,q3,450000\nactive_users,q3,12400\nchurn_rate,q3,0.03",
    }
    report_content = mock_reports.get(fmt, f"Report content for {report_name} in {fmt}.")

    # Out-of-band delivery: report data is delivered directly to the user client via ConfidentialDelivery
    user.deliver_confidential(
        content=report_content,
        label=f"Report Export: {report_name} ({fmt})",
        metadata={"report_name": report_name, "format": fmt, "tool": "request_report_export"},
    )

    # Return a clean confirmation to the LLM without cluttering model context with raw tables
    return (
        f"Report '{report_name}' has been successfully generated in format '{fmt}' "
        f"and delivered directly to the user's secure client interface."
    )


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
