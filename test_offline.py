"""Offline test suite using Pydantic AI's TestModel.
Verifies tool filtering, in-tool permissions, confidential redaction,
Generic Button HITL background injection, Verbal HITL, and message history inspection.
Run with: python test_offline.py
"""

import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

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


async def test_button_hitl_background_injection_and_out_of_band():
    print("\n--- Test 2: Verifying Button HITL & Out-of-Band Report Delivery ---")
    alice = USERS_DATABASE["user_alice"]
    alice.pending_action = None
    alice.last_selected_option = None
    alice.confidential_deliveries = []
    ctx = MockContext(deps=alice)

    # 1. Calling tool with report_name only (NO format parameter in signature)
    res_pause = await request_report_export(ctx, report_name="q3_financial")
    print(f"Tool response without button click:\n  {res_pause}")
    assert "PAUSED_FOR_USER_SELECTION" in res_pause
    assert alice.pending_action is not None
    assert alice.pending_action.options == ["Executive Summary", "Full Raw Logs", "CSV Format"]
    print(f"Button options dispatched to UI: {alice.pending_action.options}")

    # 2. Human clicks button: Injected in background via ctx.deps.last_selected_option!
    alice.last_selected_option = "CSV Format"
    res_complete = await request_report_export(ctx, report_name="q3_financial")
    print(f"\nTool response to LLM after button injection:\n  {res_complete}")
    
    # LLM receives a clean confirmation (raw report is not in LLM context)
    assert "successfully generated" in res_complete
    assert "revenue,q3,450000" not in res_complete

    # The user receives the full unredacted report via ConfidentialDelivery!
    assert len(alice.confidential_deliveries) == 1
    report_delivery = alice.confidential_deliveries[0]
    print(f"Out-of-band Delivery to User:\n  Label: {report_delivery.label}\n  Content preview: {report_delivery.content[:45]}...")
    assert "revenue,q3,450000" in report_delivery.content
    print("[PASS] Button HITL background injection and out-of-band delivery verified.")


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


async def test_message_history_inspection():
    print("\n--- Test 5: Verifying Message History Inspection via all_messages() ---")
    alice = USERS_DATABASE["user_alice"]
    
    test_model = TestModel(call_tools=['get_topic_information'])
    agent = Agent(model=test_model, deps_type=UserContext, tools=get_permitted_tools(alice))

    result = await agent.run("Tell me about topic_a", deps=alice)
    messages = result.all_messages()
    print(f"Total messages in history: {len(messages)}")

    parts_found = []
    for msg in messages:
        for part in msg.parts:
            parts_found.append(part.__class__.__name__)
            if isinstance(part, ToolCallPart):
                print(f"Inspected Tool Call: tool='{part.tool_name}', args={part.args}")
            elif isinstance(part, ToolReturnPart):
                print(f"Inspected Tool Return: tool='{part.tool_name}', content='{part.content[:40]}...'")

    assert "ToolCallPart" in parts_found
    assert "ToolReturnPart" in parts_found
    print(f"Parts successfully inspected: {set(parts_found)}")
    print("[PASS] all_messages() inspection verified.")


async def main():
    print("==================================================================")
    print("Running Pydantic AI Comprehensive Test Suite (Decoupled Architecture)")
    print("==================================================================")
    await test_tool_visibility_filtering()
    await test_button_hitl_background_injection_and_out_of_band()
    await test_verbal_hitl_tool()
    await test_confidential_out_of_band_delivery()
    await test_message_history_inspection()
    print("\nAll offline tests passed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
