"""
Generic Multi-Agent Orchestrator Framework — with Memory.

Reads agents.yaml to build an orchestrator agent with dynamically
configured subagents. No Python changes needed — all configuration
is in YAML.

Supported subagent types:
  - genie:          Databricks Genie space (structured data via MCP)
  - vector_search:  Databricks Vector Search index (RAG)
  - uc_function:    Unity Catalog SQL function (governed rules)
  - external_mcp:   External MCP server (HTTP)
  - custom_mcp:     Custom MCP server (subprocess / stdio)

Memory:
  - Short-term:  Conversation history via Lakebase (AsyncDatabricksSession)
  - Long-term:   User facts/preferences via Lakebase (direct SQL, OpenAI SDK pattern)

Both memory types are optional — enabled by setting LAKEBASE_INSTANCE_NAME
or LAKEBASE_AUTOSCALING_PROJECT + LAKEBASE_AUTOSCALING_BRANCH in .env.

Built from: databricks/app-templates (agent-openai-agents-sdk-multiagent + short-term-memory)
Long-term memory: OpenAI SDK state-based pattern with Lakebase SQL persistence
"""

import litellm
import logging
import os
from contextlib import AsyncExitStack
from typing import AsyncGenerator

import mlflow
from agents import Agent, Runner, set_default_openai_api, set_default_openai_client
from agents.tracing import set_trace_processors
from databricks_openai import AsyncDatabricksOpenAI
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent_server.config import load_config
from agent_server.tools.genie import build_genie_mcp
from agent_server.tools.vector_search import build_vector_search_mcp
from agent_server.tools.uc_function import build_uc_function_mcp
from agent_server.tools.external_mcp import build_external_mcp
from agent_server.tools.custom_mcp import build_custom_mcp
from agent_server.utils import process_agent_stream_events, set_request_ws_client

# Memory imports
from agent_server.memory.short_term import (
    is_short_term_enabled,
    get_session_id,
    create_session,
    deduplicate_input,
)
from agent_server.memory.long_term import (
    is_long_term_enabled,
    get_user_id,
    set_current_user_id,
    get_memory_tools,
    MEMORY_INSTRUCTIONS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

set_default_openai_client(AsyncDatabricksOpenAI())
set_default_openai_api("chat_completions")
set_trace_processors([])
mlflow.openai.autolog()
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
litellm.suppress_debug_info = True

# ---------------------------------------------------------------------------
# Memory status
# ---------------------------------------------------------------------------

_short_term = is_short_term_enabled()
_long_term = is_long_term_enabled()
print(f"  [Memory] Short-term (conversation history): {'ENABLED' if _short_term else 'DISABLED'}")
print(f"  [Memory] Long-term (user preferences):      {'ENABLED' if _long_term else 'DISABLED'}")

# ---------------------------------------------------------------------------
# Load config + build tools at import time
# ---------------------------------------------------------------------------

_config = load_config()
_orch_config = _config["orchestrator"]
_subagents = _config["subagents"]

# Separate subagents into MCP servers and function tools
_mcp_builders = []
_function_tools = []

for sa in _subagents:
    sa_type = sa["type"]
    sa_name = sa["name"]

    if sa_type == "genie":
        _mcp_builders.append((sa_name, build_genie_mcp, sa))
        print(f"  [Config] Registered MCP subagent: {sa_name} (genie → HTTP MCP)")
    elif sa_type == "vector_search":
        _mcp_builders.append((sa_name, build_vector_search_mcp, sa))
        print(f"  [Config] Registered MCP subagent: {sa_name} (vector_search → stdio MCP)")
    elif sa_type == "uc_function":
        _mcp_builders.append((sa_name, build_uc_function_mcp, sa))
        print(f"  [Config] Registered MCP subagent: {sa_name} (uc_function → stdio MCP)")
    elif sa_type == "external_mcp":
        _mcp_builders.append((sa_name, build_external_mcp, sa))
        print(f"  [Config] Registered MCP subagent: {sa_name} (external_mcp → HTTP MCP)")
    elif sa_type == "custom_mcp":
        _mcp_builders.append((sa_name, build_custom_mcp, sa))
        print(f"  [Config] Registered MCP subagent: {sa_name} (custom_mcp → stdio MCP)")

# Add long-term memory tools if enabled
if _long_term:
    _function_tools.extend(get_memory_tools())
    print(f"  [Config] Added 3 long-term memory tools (get/save/delete)")

print(f"  [Config] Total: {len(_function_tools)} tools + {len(_mcp_builders)} MCP servers")

# ---------------------------------------------------------------------------
# Build orchestrator instructions
# ---------------------------------------------------------------------------


def _build_instructions() -> str:
    base = _orch_config.get("instructions", "")
    company = os.environ.get("COMPANY_NAME", _orch_config.get("name", "Assistant"))

    # Auto-generate tool routing section
    tool_section = "\n\nAVAILABLE TOOLS:\n"
    for i, sa in enumerate(_subagents, 1):
        tool_section += f"\n{i}. **{sa['name']}** ({sa['type']})\n"
        tool_section += f"   {sa['description']}\n"

    # Add memory instructions if enabled
    memory_section = ""
    if _long_term:
        memory_section = f"\n\nMEMORY:\n{MEMORY_INSTRUCTIONS}"

    if base:
        return base + tool_section + memory_section
    else:
        return (
            f"You are {company}, an AI assistant.\n"
            f"Route the user's request to the most appropriate tool.\n"
            f"If a query requires multiple tools, call them sequentially and synthesize.\n"
            f"If you cannot answer confidently, say so.\n"
            + tool_section + memory_section
        )


INSTRUCTIONS = _build_instructions()

# ---------------------------------------------------------------------------
# Orchestrator agent factory
# ---------------------------------------------------------------------------


def create_orchestrator(mcp_servers: list) -> Agent:
    return Agent(
        name=_orch_config.get("name", "Orchestrator"),
        instructions=INSTRUCTIONS,
        model=_orch_config["model"],
        mcp_servers=mcp_servers,
        tools=_function_tools,
    )


# ---------------------------------------------------------------------------
# MLflow Responses API handlers
# ---------------------------------------------------------------------------


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    set_request_ws_client()
    session_id = get_session_id(request)
    if session_id:
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    # Set user_id for long-term memory tools
    user_id = get_user_id(request)
    set_current_user_id(user_id)

    async with AsyncExitStack() as stack:
        # Build MCP servers
        mcp_servers = []
        for name, builder, config in _mcp_builders:
            server = builder(config)
            entered = await stack.enter_async_context(server)
            mcp_servers.append(entered)

        agent = create_orchestrator(mcp_servers)

        # Short-term memory: use session for conversation history
        # Graceful degradation: if Lakebase connection fails, fall back to stateless
        if _short_term:
            try:
                session = create_session(session_id)
                messages = await deduplicate_input(request, session)
                result = await Runner.run(agent, messages, session=session)
                return ResponsesAgentResponse(
                    output=[item.to_input_item() for item in result.new_items],
                    custom_outputs={"session_id": session.session_id, "user_id": user_id},
                )
            except Exception as e:
                logger.warning("Short-term memory unavailable, falling back to stateless: %s", e)

        # Stateless fallback
        messages = [i.model_dump() for i in request.input]
        result = await Runner.run(agent, messages)
        return ResponsesAgentResponse(
            output=[item.to_input_item() for item in result.new_items],
            custom_outputs={"user_id": user_id} if user_id else None,
        )


@stream()
async def stream_handler(request: ResponsesAgentRequest) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    set_request_ws_client()
    session_id = get_session_id(request)
    if session_id:
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    user_id = get_user_id(request)
    set_current_user_id(user_id)

    if _long_term:
        try:
            await get_store()
        except Exception as e:
            logger.warning("Long-term memory store init failed: %s", e)

    async with AsyncExitStack() as stack:
        mcp_servers = []
        for name, builder, config in _mcp_builders:
            server = builder(config)
            entered = await stack.enter_async_context(server)
            mcp_servers.append(entered)

        agent = create_orchestrator(mcp_servers)

        # Short-term memory with graceful degradation
        session = None
        if _short_term:
            try:
                session = create_session(session_id)
                messages = await deduplicate_input(request, session)
            except Exception as e:
                logger.warning("Short-term memory unavailable in stream, falling back: %s", e)
                session = None

        if session is None:
            messages = [i.model_dump() for i in request.input]

        if session:
            result = Runner.run_streamed(agent, input=messages, session=session)
        else:
            result = Runner.run_streamed(agent, input=messages)

        async for event in process_agent_stream_events(result.stream_events()):
            yield event
