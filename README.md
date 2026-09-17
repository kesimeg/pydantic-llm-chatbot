# Pydantic AI Modular, Permission-Aware Chatbot API

A decoupled, extensible chatbot system built with **Pydantic AI** and **FastAPI**. It enforces dynamic tool filtering, hidden context injection, out-of-band data delivery, in-memory multi-turn chat history, two distinct forms of Human-in-the-Loop (HITL) interaction, and complete conversation history inspection.

---

## Architectural Highlights

### 1. Fully Decoupled & Adaptable Data Models
- **`ConfidentialDelivery`**: A generic out-of-band payload container (`content: str`, optional `label: str`, `metadata: dict`). Any tool can deliver sensitive data (confidential documents, analytical reports, raw database dumps) directly to the user client without exposing tokens to the LLM.
- **`PendingAction`**: Generic representation of a paused tool state (`action_id`, `prompt`, `options`, `tool_name`, `context_data: dict`, `resume_instruction`). Completely tool-agnostic.
- **`UserContext`**: Generic user permissions store (`permissions: Dict[str, Any]`, `last_selected_option`, `action_selections`). No hardcoded tool-specific fields.

### 2. Encapsulated Mock Data Inside Tools
All mock dictionaries (knowledge base topics, confidential documents, database tables, and analytical reports) are encapsulated directly **inside the tool functions** in [tools.py](file:///home/ege/Desktop/Pytorch/Pydantic%20Bot/tools.py). Replacing any mock tool with a real database or external API requires changing only that specific function.

### 3. Out-of-Band Report Delivery
In `request_report_export`, the generated report (CSV, Raw Logs, or Executive Summary) is delivered via `ConfidentialDelivery` directly to the user's client. The LLM receives only a confirmation receipt, ensuring raw metrics and table dumps do not consume or pollute the model's context window.

### 4. Generic Resume Endpoint (`POST /chat/resume`)
The resume endpoint in [main.py](file:///home/ege/Desktop/Pytorch/Pydantic%20Bot/main.py) is completely decoupled from specific tools. When a human selects an option (button click), the choice is injected into `ctx.deps.last_selected_option` in the background and the tool resumes using the metadata stored in `PendingAction`.

### 5. Two Clear HITL Patterns
- **Button Trigger (Background Injected)**: The LLM schema for `request_report_export` only takes `(report_name: str)`. It has **no** format parameter. The tool emits button options to the UI, the user clicks a button, and the choice is injected out-of-band without the LLM ever handling or seeing the format argument.
- **Verbal Confirmation (Conversational)**: `execute_critical_system_action` checks `confirmed: bool = False`. The LLM verbally asks the user for confirmation in natural language, and executes on the next turn after reading user approval from the chat history.

### 6. Message History & Tool Inspection (`.all_messages()`)
- `GET /sessions/{user_id}/{session_id}/history` and `client.print_history()` allow inspecting:
  - Exact tools called and arguments passed.
  - Tool return observations.
  - Model thinking / reasoning tokens (`ThinkingPart`).
  - Raw Pydantic AI message structures.

---

## Mock User Clearances & Tools

Declared in [models.py](file:///home/ege/Desktop/Pytorch/Pydantic%20Bot/models.py):

| User ID | Role | Allowed Tools | Permissions Scope |
| :--- | :--- | :--- | :--- |
| `user_alice` | Senior Analyst | `["topic_info", "database_query", "confidential_topic_rag", "request_report_export", "execute_critical_system_action"]` | Topics: `[topic_a, topic_b]`, Clearances: `[quantum_keys, payroll_audit]` |
| `user_bob` | Junior Analyst | `["topic_info", "confidential_topic_rag", "request_report_export"]` *(No database or critical actions)* | Topics: `[topic_b]`, Clearances: `[payroll_audit]` |
| `user_charlie` | Guest | `[]` *(Zero tools visible to LLM)* | No permissions |

---

## Project Structure

- **`models.py`**: Generic Pydantic schemas (`ChatRequest`, `ResumeRequest`, `ChatResponse`, `PendingAction`, `ConfidentialDelivery`), `UserContext`, and `USERS_DATABASE`.
- **`tools.py`**:
  - `request_report_export`: Button trigger HITL tool with out-of-band `ConfidentialDelivery`.
  - `execute_critical_system_action`: Verbal HITL tool (conversational confirmation).
  - `confidential_topic_rag`: Out-of-band delivery with LLM redaction.
  - `get_topic_information`: In-tool topic clearance checks.
  - `query_database`: Restricted database tool.
- **`agent_factory.py`**: Dynamically builds agents with model configuration and filtered tools.
- **`main.py`**: FastAPI application managing sessions, `/chat`, `/chat/resume`, `/sessions/{user_id}/{session_id}/history`, and session lifecycle.
- **`client.py`**: Reusable Python client class with `print_history()` inspection and interactive chat loop.
- **`client_notebook.ipynb`**: Ready-to-run Jupyter notebook testing all features step-by-step.
- **`test_offline.py`**: Offline test suite using Pydantic AI's `TestModel` to verify all decoupled logic without an LLM.

---

## Setup & Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Offline Tests (No LLM Required)
```bash
python test_offline.py
```

### 3. Start the FastAPI Server
```bash
uvicorn main:app --reload --port 8000
```

### 4. Test in Jupyter Notebook
Open and run [client_notebook.ipynb](file:///home/ege/Desktop/Pytorch/Pydantic%20Bot/client_notebook.ipynb) in Jupyter:
```bash
jupyter notebook client_notebook.ipynb
```

Or test via Python in your terminal / script:
```python
from client import ChatClient

client = ChatClient(base_url="http://127.0.0.1:8000", user_id="user_alice")

# Multi-turn context
client.chat("Hello, my favorite project code name is 'Project Phoenix'.")
client.chat("What was my favorite project code name?")

# Button trigger HITL (format injected in background, report delivered out-of-band)
client.chat("Export the financial_q3 report.")

# Inspect tool calls, arguments, and thinking process!
client.print_history()
```
