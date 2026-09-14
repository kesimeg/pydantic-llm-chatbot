"""Test script to demonstrate the API with different user permission tiers and confidential deliveries.
Run with: python test_client.py
Make sure FastAPI server is running: uvicorn main:app --port 8000
"""

import httpx
import json

BASE_URL = "http://127.0.0.1:8001"

TEST_CASES = [
    {
        "description": "User Alice (Admin): Request Topic A (Authorized topic) and Topic B",
        "user_id": "user_alice",
        "message": "Please give me information about topic_a and topic_b.",
    },
    {
        "description": "User Alice (Admin): Request confidential 'quantum_keys' (Authorized)",
        "user_id": "user_alice",
        "message": "Fetch confidential documentation for quantum_keys.",
    },
    {
        "description": "User Bob (Junior Analyst): Request confidential 'payroll_audit' (Authorized)",
        "user_id": "user_bob",
        "message": "Fetch confidential documentation for payroll_audit.",
    },
    {
        "description": "User Bob (Junior Analyst): Request confidential 'quantum_keys' (Unauthorized)",
        "user_id": "user_bob",
        "message": "Fetch confidential documentation for quantum_keys.",
    },
    {
        "description": "User Bob (Junior Analyst): Request database query (Tool is FILTERED OUT, not visible)",
        "user_id": "user_bob",
        "message": "Please query the database audit_logs table.",
    },
    {
        "description": "User Charlie (Guest): No tools visible",
        "user_id": "user_charlie",
        "message": "Can you query topic_b, quantum_keys, or query the database?",
    },
]


def run_tests():
    print("=" * 75)
    print("Testing Pydantic AI Chatbot API (Permissions + Redacted Confidential RAG)")
    print("=" * 75)

    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        # Check server health
        try:
            health = client.get("/")
            print(f"[✓] Connected to API server: {health.json()['description']}\n")
        except Exception as e:
            print(f"[!] Could not connect to API server at {BASE_URL}.")
            print(f"    Make sure to run: uvicorn main:app --reload\n    Error: {e}")
            return

        for idx, test in enumerate(TEST_CASES, start=1):
            print(f"--- Test #{idx}: {test['description']} ---")
            print(f"User ID: {test['user_id']}")
            print(f"Prompt:  \"{test['message']}\"")

            try:
                response = client.post(
                    "/chat",
                    json={"user_id": test["user_id"], "message": test["message"]},
                )
                if response.status_code == 200:
                    data = response.json()
                    print(f"Visible Tools to LLM: {data['visible_tools']}")
                    print(f"Bot Reply (What LLM answered):\n{data['reply']}")
                    
                    deliveries = data.get("confidential_deliveries", [])
                    if deliveries:
                        print(f"Out-of-band Confidential Payload (Sent to user, hidden from LLM):")
                        for d in deliveries:
                            print(f"  [Topic: {d['topic']}]\n  {d['content']}")
                    else:
                        print("No confidential payload dispatched.")
                    print()
                else:
                    print(f"Error ({response.status_code}): {response.text}\n")
            except Exception as e:
                print(f"Request failed: {e}\n")


if __name__ == "__main__":
    run_tests()
