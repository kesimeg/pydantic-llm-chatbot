"""Reusable Chat Client library for interacting with the Pydantic AI Chatbot API.
Designed to run seamlessly inside Jupyter Notebooks and standard Python scripts.
"""

from typing import Optional, List, Dict, Any
import httpx


class ChatClient:
    """Client for the Pydantic AI Chatbot API.
    
    Supports:
    - Multi-turn chat history via session_id
    - Dynamic user switching
    - Human-In-The-Loop (HITL) pause-and-resume
    - Confidential out-of-band payload display
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        user_id: str = "user_alice",
        session_id: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id
        self.session_id = session_id
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def set_user(self, user_id: str, new_session: bool = True):
        """Switch active user. Optionally starts a new session."""
        self.user_id = user_id
        if new_session:
            self.session_id = None
        print(f"[Client] Switched active user to: '{self.user_id}' (session: {self.session_id or 'new'})")

    def reset_session(self):
        """Resets session ID to start a fresh chat thread."""
        old_sess = self.session_id
        self.session_id = None
        if old_sess:
            try:
                self.client.delete(f"/sessions/{self.user_id}/{old_sess}")
            except Exception:
                pass
        print(f"[Client] Session reset for user '{self.user_id}'. Next message starts fresh conversation.")

    def get_users(self) -> Dict[str, Any]:
        """Fetch available mock users and their permission sets."""
        resp = self.client.get("/users")
        resp.raise_for_status()
        return resp.json()

    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Fetch active session IDs for the current user."""
        resp = self.client.get(f"/sessions/{self.user_id}")
        resp.raise_for_status()
        return resp.json().get("sessions", [])

    def send(self, message: str) -> Dict[str, Any]:
        """Low-level method to send a prompt to the API."""
        payload = {"user_id": self.user_id, "message": message}
        if self.session_id:
            payload["session_id"] = self.session_id

        resp = self.client.post("/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        
        # Keep track of active session_id returned by server
        self.session_id = data.get("session_id")
        return data

    def resume(self, action_id: str, selected_option: str) -> Dict[str, Any]:
        """Resume a paused tool execution with the chosen option."""
        if not self.session_id:
            raise ValueError("No active session_id found to resume.")

        payload = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "action_id": action_id,
            "selected_option": selected_option,
        }
        resp = self.client.post("/chat/resume", json=payload)
        resp.raise_for_status()
        return resp.json()

    def chat(self, message: str, auto_prompt_hitl: bool = True) -> Dict[str, Any]:
        """Convenience method that sends a message and pretty-prints the output.
        If the tool pauses for Human-in-the-Loop input and auto_prompt_hitl is True,
        it prompts the user and automatically resumes.
        """
        print(f"\n👤 [{self.user_id}] (Session: {self.session_id or 'new'}):")
        print(f"   {message}\n")

        data = self.send(message)
        self._print_response(data)

        # Handle Structured HITL pause
        if data.get("status") == "needs_action" and auto_prompt_hitl:
            action_id = data.get("action_id")
            options = data.get("options", [])
            print(f"\n⚙️ [HUMAN DECISION REQUIRED]")
            print(f"   {data.get('action_prompt')}")
            for idx, opt in enumerate(options, 1):
                print(f"   [{idx}] {opt}")

            # Prompt user (works in terminal and Jupyter Notebook)
            choice = input(f"Enter option number (1-{len(options)}) or option name: ").strip()
            selected = None
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                selected = options[int(choice) - 1]
            elif choice in options:
                selected = choice
            else:
                selected = options[0]  # default fallback

            print(f"👉 You selected: '{selected}'\n")
            resume_data = self.resume(action_id, selected)
            self._print_response(resume_data)
            return resume_data

        return data

    def _print_response(self, data: Dict[str, Any]):
        """Helper to format and print server responses."""
        print(f"🤖 Bot [Tools visible: {', '.join(data.get('visible_tools', [])) or 'None'}]:")
        print(f"   {data.get('reply')}\n")

        # Display confidential deliveries if any
        confidential = data.get("confidential_deliveries", [])
        if confidential:
            print("🔒 [CONFIDENTIAL SECURE SIDE-CHANNEL PAYLOAD (Hidden from LLM)]:")
            for item in confidential:
                print(f"   --- Document: {item['topic']} ---")
                for line in item['content'].splitlines():
                    print(f"   | {line}")
                print("   --------------------------------------\n")

    def interactive_loop(self):
        """Starts an interactive chat loop directly in the console or notebook cell."""
        print("=" * 60)
        print(f"Interactive Chat Session started for '{self.user_id}'")
        print("Type 'exit', 'quit', or 'reset' to manage session.")
        print("=" * 60)
        while True:
            try:
                user_msg = input("\nYou: ").strip()
                if not user_msg:
                    continue
                if user_msg.lower() in ["exit", "quit"]:
                    print("Exiting chat session.")
                    break
                if user_msg.lower() == "reset":
                    self.reset_session()
                    continue

                self.chat(user_msg, auto_prompt_hitl=True)
            except KeyboardInterrupt:
                print("\nSession interrupted.")
                break
