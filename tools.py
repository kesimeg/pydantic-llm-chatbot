"""Tool definitions with in-tool permission checks, hidden context, and out-of-band confidential data delivery.
"""

from typing import Dict
from pydantic_ai import RunContext, Tool
from models import UserContext, ConfidentialDelivery

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


async def get_topic_information(ctx: RunContext[UserContext], topic: str) -> str:
    """Retrieve detailed information about a specific topic.
    
    Args:
        topic: The topic identifier to look up (e.g., 'topic_a', 'topic_b').
    """
    normalized_topic = topic.strip().lower()
    
    # In-Tool Permission Check (LLM is unaware of this validation)
    user = ctx.deps
    if normalized_topic not in user.allowed_topics:
        return (
            f"Permission Denied: User '{user.username}' is not authorized "
            f"to view information on '{topic}'."
        )

    # If authorized, retrieve topic data
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
    
    # 1. Permission check for confidential topic
    if normalized_topic not in user.allowed_confidential_topics:
        return (
            f"Access Denied: User '{user.username}' does not have security clearance "
            f"for confidential topic '{topic}'."
        )

    if normalized_topic not in CONFIDENTIAL_DOCUMENTATION:
        available = ", ".join(CONFIDENTIAL_DOCUMENTATION.keys())
        return f"Confidential topic '{topic}' not found. Available topics: {available}"

    # 2. Retrieve the raw confidential content
    full_content = CONFIDENTIAL_DOCUMENTATION[normalized_topic]

    # 3. Deliver out-of-band directly to user's payload
    # The LLM never sees ctx.deps.confidential_deliveries
    user.confidential_deliveries.append(
        ConfidentialDelivery(topic=topic, content=full_content)
    )

    # 4. Return REDACTED receipt to the LLM
    # The LLM only sees that delivery happened, not the actual secret text.
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
    
    # Secondary in-tool safety check
    if "database_query" not in user.allowed_tools:
        return f"Security Error: User '{user.username}' lacks database querying privileges."

    mock_db = {
        "users": "Table 'users': [ID 1: Alice (Admin), ID 2: Bob (Analyst), ID 3: Charlie (Guest)]",
        "audit_logs": "Table 'audit_logs': [2026-09-09 10:15: User Alice performed backup]",
        "transactions": "Table 'transactions': [TX1092: $5,400 completed, TX1093: $1,200 pending]",
    }
    
    table = table_name.strip().lower()
    if table in mock_db:
        return f"Database query result for {table_name}: {mock_db[table]}"
    return f"Table '{table_name}' not found. Available tables: {', '.join(mock_db.keys())}"


# --- Tool Registry ---
# Maps tool permission identifiers to Tool instances
TOOL_REGISTRY: Dict[str, Tool[UserContext]] = {
    "topic_info": Tool(get_topic_information, takes_ctx=True),
    "confidential_topic_rag": Tool(confidential_topic_rag, takes_ctx=True),
    "database_query": Tool(query_database, takes_ctx=True),
}
