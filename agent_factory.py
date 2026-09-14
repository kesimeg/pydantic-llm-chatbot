"""Agent factory that provisions dynamically filtered agents per user.
"""

import os
from typing import List
from pydantic_ai import Agent, Tool
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.vllm import VLLMProvider


from models import UserContext
from tools import TOOL_REGISTRY


def get_model() -> OpenAIChatModel:
    """Configures the OpenAI-compatible model.
    
    Supports custom deployed models (vLLM, Ollama, TGI, LM Studio, etc.)
    via OPENAI_BASE_URL, or official OpenAI models.
    """
    model_name = "google/gemma-4-E2B-it"
    base_url = "http://localhost:8000/v1"

    return OpenAIChatModel(
    model_name,
    provider=VLLMProvider(base_url=base_url),
    )   


def get_permitted_tools(user: UserContext) -> List[Tool[UserContext]]:
    """Filters tools from the registry based on the user's permissions.
    
    Any tool not explicitly listed in user.allowed_tools is excluded,
    meaning the LLM will NEVER see its function definition or schema.
    """
    permitted = []
    for tool_name in user.allowed_tools:
        if tool_name in TOOL_REGISTRY:
            permitted.append(TOOL_REGISTRY[tool_name])
    return permitted


def create_agent_for_user(user: UserContext) -> Agent[UserContext, str]:
    """Dynamically creates a lightweight Agent instance tailored to the user's permissions.
    
    - The LLM will only receive schemas for tools the user has permission to use.
    - User context is injected as `deps_type=UserContext`.
    """
    model = get_model()
    permitted_tools = get_permitted_tools(user)

    agent = Agent(
        model=model,
        deps_type=UserContext,
        tools=permitted_tools,
        system_prompt=(
            "You are a helpful and secure enterprise assistant.\n"
            "Help the user with their queries using your available tools.\n"
            "- When asked for confidential internal documents or secrets, use the 'confidential_topic_rag' tool.\n"
            "- For confidential tools, you will receive a redacted receipt because the raw text is "
            "transmitted directly to the user's secure display out-of-band.\n"
            "- If a tool or topic is restricted or denied, explain the restriction politely."
        ),
    )

    return agent
