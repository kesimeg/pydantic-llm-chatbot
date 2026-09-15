"""Offline test suite using Pydantic AI's TestModel.
Verifies tool filtering, in-tool permissions, confidential redaction,
Structured HITL, Verbal HITL, and multi-turn chat history.
Run with: python test_offline.py
"""

import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from models import USERS_DATABASE, UserContext
from tools import (
    get_topic_information,
    confidential_topic_rag,
    request_report_export,
    execute_critical_system_action,
)
from agent_factory import get_permitted_tools


class MockContext:
    def __init__(self, deps: UserContext):
        self.deps = deps


async def test_tool_visibility_filtering():
    print("\n--- Test 1: Verifying Tool Visibility Filtering ---")
    alice = USERS_DATABASE["user_alice"]
    bob = USERS_DATABASE["user_bob"]
    charlie = USERS_DATABASE["user_charlie"]

    alice_tools = [t.function.__name__ for t in get_permitted_tools(alice)]
    bob_tools = [t.function.__name__ for t in get_permitted_tools(bob)]
    charlie_tools = [t.function.__name__ for t in get_permitted_tools(charlie)]

    print(f"Alice's tools:   {alice_tools}")
    print(f"Bob's tools:     {bob_tools}")
    print(f"Charlie's tools: {charlie_tools}")

    assert "request_report_export" in alice_tools
    assert "execute_critical_system_action" in alice_tools
    assert "database_query" in alice_tools

    assert "request_report_export" in bob_tools
    assert "execute_critical_system_action" not in bob_tools, "Security failure: Bob should NOT see critical system tool"
    assert "database_query" not in bob_tools, "Security failure: Bob should NOT see database query tool"

    assert len(charlie_tools) == 0
    print("[PASS] Tool visibility filtering verified.")


async def test_structured_hitl_tool():
    print("\n--- Test 2: Verifying Structured HITL Tool (Report Export) ---")
    alice = USERS_DATABASE["user_alice"]
    alice.pending_action = None
    ctx = MockContext(deps=alice)

    # 1. Calling tool without format -> pauses and sets pending_action
    res_pause = await request_report_export(ctx, report_name="q3_financial")
    print(f"Tool response without format:\n  {res_pause}")
    assert "ACTION_REQUIRED" in res_pause
    assert alice.pending_action is not None
    assert alice.pending_action.options == ["Executive Summary", "Full Raw Logs", "CSV Format"]
    print(f"Pending Action registered in context: {alice.pending_action}")

    # 2. Resuming tool with format provided
    res_complete = await request_report_export(ctx, report_name="q3_financial", format="CSV Format")
    print(f"\nTool response with format ('CSV Format'):\n  {res_complete}")
    assert "revenue,q3,450000" in res_complete
    print("[PASS] Structured HITL pause-and-resume logic verified.")


async def test_verbal_hitl_tool():
    print("\n--- Test 3: Verifying Verbal HITL Tool (Critical Action) ---")
    alice = USERS_DATABASE["user_alice"]
    ctx = MockContext(deps=alice)

    # 1. Calling tool without verbal confirmation -> asks LLM to query user
    res_unconfirmed = await execute_critical_system_action(ctx, action_name="restart_primary_cluster", confirmed=False)
    print(f"Tool response without confirmation:\n  {res_unconfirmed}")
    assert "VERBAL CONFIRMATION REQUIRED" in res_unconfirmed

    # 2. Calling tool with confirmation = True -> executes
    res_confirmed = await execute_critical_system_action(ctx, action_name="restart_primary_cluster", confirmed=True)
    print(f"\nTool response with confirmation:\n  {res_confirmed}")
    assert "SUCCESS: Critical system operation" in res_confirmed
    print("[PASS] Verbal HITL logic verified.")


async def test_confidential_out_of_band_delivery():
    print("\n--- Test 4: Verifying Confidential Out-of-Band Delivery & Redaction ---")
    alice = USERS_DATABASE["user_alice"]
    alice.confidential_deliveries = []
    ctx = MockContext(deps=alice)

    receipt = await confidential_topic_rag(ctx, "quantum_keys")
    print(f"LLM View (Redacted Receipt):\n  {receipt}")
    print(f"User View (Confidential Deliveries):\n  {alice.confidential_deliveries}")

    assert "[REDACTED RECEIPT]" in receipt
    assert "9f8a-bc34" not in receipt, "Security failure: Secrets leaked to LLM!"
    assert len(alice.confidential_deliveries) == 1
    assert "Quantum Cryptography Master Key Material" in alice.confidential_deliveries[0].content
    print("[PASS] Confidential redaction verified.")


async def test_multi_turn_history():
    print("\n--- Test 5: Verifying Multi-Turn History in Agent ---")
    alice = USERS_DATABASE["user_alice"]
    test_model = TestModel()
    agent = Agent(model=test_model, deps_type=UserContext, tools=get_permitted_tools(alice))

    # Turn 1
    result1 = await agent.run("Hello, my favorite color is emerald blue.", deps=alice)
    history1 = result1.all_messages()
    print(f"Turn 1 completed. Messages in history: {len(history1)}")

    # Turn 2 with history passed
    result2 = await agent.run("What did I say my favorite color was?", deps=alice, message_history=history1)
    history2 = result2.all_messages()
    print(f"Turn 2 completed. Messages in history: {len(history2)}")

    assert len(history2) > len(history1)
    print("[PASS] Multi-turn history accumulation verified.")


async def main():
    print("================================================================")
    print("Running Pydantic AI Comprehensive Test Suite (HITL + History)")
    print("================================================================")
    await test_tool_visibility_filtering()
    await test_structured_hitl_tool()
    await test_verbal_hitl_tool()
    await test_confidential_out_of_band_delivery()
    await test_multi_turn_history()
    print("\nAll offline tests passed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
