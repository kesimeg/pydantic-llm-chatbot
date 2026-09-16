# Pydantic AI Permission-Aware Chatbot API with Human-in-the-Loop (HITL)

A FastAPI-based chatbot system built with **Pydantic AI** enforcing dynamic tool filtering, hidden context injection, out-of-band confidential payload delivery, in-memory multi-turn chat history, two distinct forms of Human-in-the-Loop (HITL) interaction, and full conversation history/tool inspection.

---

## Key Features

1. **Button Trigger HITL (`request_report_export`) with Background Format Injection**:
   - **No Argument Leaking to LLM**: The LLM's tool schema only accepts `report_name: str`. It has **no** `format` parameter.
   - **Button Emission**: When invoked, the tool pauses and emits format buttons (`["Executive Summary", "Full Raw Logs", "CSV Format"]`) to the client UI.
   - **Background Injection**: The human clicks a button on the UI, and the selection is injected into `ctx.deps.selected_format` behind the scenes via `/chat/resume`. The LLM never sees or chooses the format argument.

2. **Verbal HITL (`execute_critical_system_action`)**:
   - High-risk operations (e.g. `restart_primary_cluster`) require conversational confirmation.
   - The tool instructs the LLM to ask the user verbally in chat: *"Are you sure you want to restart the primary cluster?"*.
   - In the subsequent turn, the LLM reads conversation history, detects the user's verbal consent, and re-executes the tool with `confirmed=True`.

3. **Message History & Tool Inspection (`.all_messages()`)**:
   - Endpoint `GET /sessions/{user_id}/{session_id}/history` and method `client.print_history()` allow inspecting:
     - Exact tools called by the model and the arguments passed.
     - Tool outputs and return observations.
     - Model thinking / reasoning tokens (`ThinkingPart`).
     - Multi-turn request and response sequences.

4. **In-Memory Multi-Turn Chat History (`session_id`)**:
   - Conversations are tracked in-memory using `(user_id, session_id)` mapping to Pydantic AI `ModelMessage` history.
   - Users can maintain continuous multi-turn conversations or switch threads by passing a different `session_id`.

5. **Confidential Out-of-Band Delivery (`confidential_topic_rag`)**:
   - Sensitive documentation (e.g. `quantum_keys`) is delivered directly to the user's API response payload.
   - The LLM receives only a **redacted receipt**—raw secrets never touch the model's prompt, context window, or message history.

6. **Dynamic Tool Filtering & Hidden Context (`RunContext[Deps]`)**:
   - The system provisions an `Agent` with **only** the tools the user has permission to view.
   - Clearances and `user_id` are injected into tools via `ctx.deps` without the LLM's knowledge.

---

## Mock User Clearances & Tools

Declared in `models.py`:

| User ID | Role | Allowed Tools | Standard Topics | Confidential Topics |
| :--- | :--- | :--- | :--- | :--- |
| `user_alice` | Senior Analyst | `["topic_info", "database_query", "confidential_topic_rag", "request_report_export", "execute_critical_system_action"]` | `["topic_a", "topic_b"]` | `["quantum_keys", "payroll_audit"]` |
| `user_bob` | Junior Analyst | `["topic_info", "confidential_topic_rag", "request_report_export"]` *(No database or critical actions)* | `["topic_b"]` | `["payroll_audit"]` |
| `user_charlie` | Guest | `[]` *(Zero tools visible)* | `[]` | `[]` |

---

## Project Structure

- **`models.py`**: Pydantic schemas (`ChatRequest`, `ResumeRequest`, `ChatResponse`, `PendingAction`, `ConfidentialDelivery`), `UserContext`, and `USERS_DATABASE`.
- **`tools.py`**:
  - `request_report_export`: Button trigger HITL tool (background format injection).
  - `execute_critical_system_action`: Verbal HITL tool (conversational confirmation).
  - `confidential_topic_rag`: Out-of-band delivery with LLM redaction.
  - `get_topic_information`: In-tool topic clearance checks.
  - `query_database`: Restricted tool for senior analysts.
- **`agent_factory.py`**: Dynamically builds agents with model configuration (`OPENAI_BASE_URL` support) and permitted tools.
- **`main.py`**: FastAPI application managing sessions, `/chat`, `/chat/resume`, `/sessions/{user_id}/{session_id}/history`, and session lifecycle.
- **`client.py`**: Reusable Python client class with `print_history()` inspection and interactive chat loop.
- **`client_notebook.ipynb`**: Ready-to-run Jupyter notebook testing all features step-by-step.
- **`test_offline.py`**: Offline test suite using Pydantic AI's `TestModel` to verify all logic without an LLM.

---

## Setup & Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables (For your deployed model)
```bash
# Example: Local model (vLLM, Ollama, LM Studio, etc.)
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="not-needed"
export OPENAI_MODEL_NAME="your-deployed-model-name"
```

### 3. Run Offline Tests (No LLM Required)
```bash
python test_offline.py
```

### 4. Start the FastAPI Server
```bash
uvicorn main:app --reload --port 8000
```

### 5. Test in Jupyter Notebook
Open and run [client_notebook.ipynb](file:///home/ege/Desktop/Pytorch/Pydantic%20Bot/client_notebook.ipynb) in Jupyter:
```bash
jupyter notebook client_notebook.ipynb
```

Or test via Python:
```python
from client import ChatClient

client = ChatClient(base_url="http://127.0.0.1:8000", user_id="user_alice")

# Multi-turn context
client.chat("Hello, my favorite project code name is 'Project Phoenix'.")
client.chat("What was my favorite project code name?")

# Button trigger HITL (format injected in background)
client.chat("Export the financial_q3 report.")

# Inspect tool calls, arguments, and thinking process!
client.print_history()
```
