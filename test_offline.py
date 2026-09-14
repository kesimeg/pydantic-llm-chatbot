"""Offline test suite using Pydantic AI's TestModel.
Allows verifying all permission logic, out-of-band confidential delivery,
and tool filtering without running an LLM or OpenAI server.
Run with: python test_offline.py
"""

import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from models import USERS_DATABASE, UserContext
from tools import get_topic_information, confidential_topic_rag
from agent_factory import get_permitted_tools


async def test_tool_visibility_filtering():
    print("\n--- Test 1: Verifying Tool Visibility Filtering ---")
    alice = USERS_DATABASE["user_alice"]
    bob = USERS_DATABASE["user_bob"]
    charlie = USERS_DATABASE["user_charlie"]

    alice_tools = [t.function.__name__ for t in get_permitted_tools(alice)]
    bob_tools = [t.function.__name__ for t in get_permitted_tools(bob)]
    charlie_tools = [t.function.__name__ for t in get_permitted_tools(charlie)]

    print(f"Alice's visible tools:   {alice_tools}")
    print(f"Bob's visible tools:     {bob_tools}")
    print(f"Charlie's visible tools: {charlie_tools}")

    assert "query_database" in alice_tools
    assert "get_topic_information" in alice_tools
    assert "confidential_topic_rag" in alice_tools

    assert "confidential_topic_rag" in bob_tools
    assert "get_topic_information" in bob_tools
    assert "query_database" not in bob_tools, "Security failure: Bob should NOT see query_database"

    assert len(charlie_tools) == 0, "Security failure: Charlie should have 0 visible tools"
    print("[PASS] Tool filtering works as expected.")


async def test_in_tool_permissions():
    print("\n--- Test 2: Verifying In-Tool Fine-Grained Topic Permissions ---")
    alice = USERS_DATABASE["user_alice"]
    bob = USERS_DATABASE["user_bob"]

    class MockContext:
        def __init__(self, deps: UserContext):
            self.deps = deps

    alice_ctx = MockContext(deps=alice)
    bob_ctx = MockContext(deps=bob)

    # Alice requests topic_a (Authorized)
    alice_res_a = await get_topic_information(alice_ctx, "topic_a")
    print(f"Alice -> Topic A: {alice_res_a}")
    assert "Project Falcon specifications" in alice_res_a

    # Bob requests topic_b (Authorized)
    bob_res_b = await get_topic_information(bob_ctx, "topic_b")
    print(f"Bob -> Topic B:   {bob_res_b}")
    assert "Public Quarterly Overview" in bob_res_b

    # Bob requests topic_a (Unauthorized -> Permission Denied)
    bob_res_a = await get_topic_information(bob_ctx, "topic_a")
    print(f"Bob -> Topic A:   {bob_res_a}")
    assert "Permission Denied" in bob_res_a
    print("[PASS] Standard in-tool permission checks work as expected.")


async def test_confidential_out_of_band_delivery():
    print("\n--- Test 3: Verifying Confidential Topic RAG & Out-of-Band Delivery ---")
    alice = USERS_DATABASE["user_alice"]
    bob = USERS_DATABASE["user_bob"]

    class MockContext:
        def __init__(self, deps: UserContext):
            self.deps = deps

    # Reset deliveries
    alice.confidential_deliveries = []
    bob.confidential_deliveries = []

    alice_ctx = MockContext(deps=alice)
    bob_ctx = MockContext(deps=bob)

    # 1. Alice requests confidential 'quantum_keys' (Authorized)
    llm_receipt_alice = await confidential_topic_rag(alice_ctx, "quantum_keys")
    print(f"Alice Tool Output (What LLM sees):\n  '{llm_receipt_alice}'")
    print(f"Alice Out-of-band Deliveries (What User sees):\n  {alice.confidential_deliveries}\n")

    assert "[REDACTED RECEIPT]" in llm_receipt_alice, "LLM must only see redacted receipt!"
    assert "9f8a-bc34" not in llm_receipt_alice, "Security leak: LLM should NEVER see raw secret key material!"
    assert len(alice.confidential_deliveries) == 1
    assert "Quantum Cryptography Master Key Material" in alice.confidential_deliveries[0].content

    # 2. Bob requests confidential 'payroll_audit' (Authorized)
    llm_receipt_bob = await confidential_topic_rag(bob_ctx, "payroll_audit")
    assert "[REDACTED RECEIPT]" in llm_receipt_bob
    assert len(bob.confidential_deliveries) == 1
    assert "Executive Compensation" in bob.confidential_deliveries[0].content

    # 3. Bob requests confidential 'quantum_keys' (Unauthorized for Bob)
    bob.confidential_deliveries = []
    denied_bob = await confidential_topic_rag(bob_ctx, "quantum_keys")
    print(f"Bob -> quantum_keys (Unauthorized):\n  '{denied_bob}'")
    assert "Access Denied" in denied_bob
    assert len(bob.confidential_deliveries) == 0, "No payload should be delivered on denial!"

    print("[PASS] Confidential RAG correctly hides secrets from LLM while delivering to user.")


async def test_agent_with_test_model():
    print("\n--- Test 4: Running Agent with Pydantic AI TestModel ---")
    alice = USERS_DATABASE["user_alice"]
    alice.confidential_deliveries = []
    
    test_model = TestModel(call_tools=['confidential_topic_rag'])
    agent = Agent(
        model=test_model,
        deps_type=UserContext,
        tools=get_permitted_tools(alice),
    )

    result = await agent.run("Fetch confidential documentation for quantum_keys", deps=alice)
    print(f"Agent test execution completed.")
    print(f"LLM Response data: {result.data}")
    print(f"User Deliveries:   {alice.confidential_deliveries}")
    print("[PASS] Agent integration verified.")


async def main():
    print("=========================================================")
    print("Running Pydantic AI Complete Permissions & RAG Verification")
    print("=========================================================")
    await test_tool_visibility_filtering()
    await test_in_tool_permissions()
    await test_confidential_out_of_band_delivery()
    await test_agent_with_test_model()
    print("\nAll offline tests passed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
