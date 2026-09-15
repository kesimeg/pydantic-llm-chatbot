# Pydantic AI Permission-Aware Chatbot API with Human-in-the-Loop (HITL)

A FastAPI-based chatbot system built with **Pydantic AI** enforcing dynamic tool filtering, hidden context injection, out-of-band confidential payload delivery, in-memory multi-turn chat history, and two forms of Human-in-the-Loop (HITL) interaction.

---

## Key Features

1. **In-Memory Multi-Turn Chat History (`session_id`)**:
   - Conversations are tracked in-memory using `(user_id, session_id)` mapping to Pydantic AI `ModelMessage` history.
   - Allows users to maintain continuous multi-turn conversations or switch threads by passing a different `session_id`.

2. **Structured Human-in-the-Loop (`request_report_export`)**:
   - **Pause-and-Resume Pattern**: If the tool is invoked without a chosen format, it pauses execution and sends structured options (`["Executive Summary", "Full Raw Logs", "CSV Format"]`) to the client UI.
   - The client selects an option and posts to `/chat/resume`, allowing the tool to complete its work.

3. **Verbal Human-in-the-Loop (`execute_critical_system_action`)**:
   - High-risk operations (e.g. `restart_primary_cluster`) require verbal confirmation.
   - The tool instructs the LLM to ask the user verbally in chat: *"Are you sure you want to restart the primary cluster?"*.
   - In the subsequent turn, the LLM reads conversation history, detects the user's verbal approval, and executes the action with `confirmed=True`.

4. **Confidential Out-of-Band Delivery (`confidential_topic_rag`)**:
   - Sensitive documentation (e.g. `quantum_keys`) is delivered directly to the user's API response payload.
   - The LLM receives only a **redacted receipt**—raw secrets never touch the model's prompt or context window.

5. **Dynamic Tool Filtering & Hidden Context (`RunContext[Deps]`)**:
   - The system provisions an `Agent` with **only** the tools the user has permission to view.
   - Internal clearances and `user_id` are injected into tools via `ctx.deps` without the LLM's knowledge.

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
  - `request_report_export`: Structured HITL tool (pauses for UI format selection).
  - `execute_critical_system_action`: Verbal HITL tool (requires conversational confirmation).
  - `confidential_topic_rag`: Out-of-band delivery with LLM redaction.
  - `get_topic_information`: In-tool topic clearance checks.
  - `query_database`: Restricted tool for senior analysts.
- **`agent_factory.py`**: Dynamically builds agents with model configuration (`OPENAI_BASE_URL` support) and permitted tools.
- **`main.py`**: FastAPI application managing sessions, `/chat`, `/chat/resume`, and session lifecycle.
- **`client.py`**: Reusable Python client class with support for interactive input loops and HITL resuming.
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

Or test via Python in your terminal / script:
```python
from client import ChatClient

client = ChatClient(base_url="http://127.0.0.1:8000", user_id="user_alice")

# Multi-turn context
client.chat("Hello, my favorite project code name is 'Project Phoenix'.")
client.chat("What was my favorite project code name?")

# Structured HITL
client.chat("Export the financial_q3 report.")

# Verbal HITL
client.chat("Restart the primary cluster.")
client.chat("Yes, I confirm. Please restart the primary cluster.")
```
