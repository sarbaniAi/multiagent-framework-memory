"""
Long-term memory — user facts/preferences via Lakebase + embeddings.

Ported from LangGraph's AsyncDatabricksStore pattern to work with
the OpenAI Agents SDK. Uses Lakebase Postgres with vector embeddings
for semantic search over user memories.

Provides 3 tools the orchestrator agent can call:
  - get_user_memory:    search for relevant stored facts about the user
  - save_user_memory:   persist a fact/preference about the user
  - delete_user_memory: remove a specific memory

Pattern from: app-templates/agent-langgraph-long-term-memory
Adapted for: OpenAI Agents SDK (function_tool instead of LangChain @tool)
"""

import json
import logging
import os
from typing import Optional

from agents import function_tool
from mlflow.types.responses import ResponsesAgentRequest

logger = logging.getLogger(__name__)

# Lakebase config (reuses same instance as short-term memory)
_LAKEBASE_INSTANCE_RAW = os.environ.get("LAKEBASE_INSTANCE_NAME") or None
LAKEBASE_PROJECT = os.environ.get("LAKEBASE_AUTOSCALING_PROJECT") or None
LAKEBASE_BRANCH = os.environ.get("LAKEBASE_AUTOSCALING_BRANCH") or None
EMBEDDING_ENDPOINT = os.environ.get("EMBEDDING_ENDPOINT", "databricks-gte-large-en")
EMBEDDING_DIMS = int(os.environ.get("EMBEDDING_DIMS", "1024"))

# Resolve instance name
from agent_server.memory.short_term import resolve_lakebase_instance_name
LAKEBASE_INSTANCE = resolve_lakebase_instance_name(_LAKEBASE_INSTANCE_RAW) if _LAKEBASE_INSTANCE_RAW else None


def is_long_term_enabled() -> bool:
    """Check if long-term memory is configured."""
    return bool(LAKEBASE_INSTANCE or (LAKEBASE_PROJECT and LAKEBASE_BRANCH))


def get_user_id(request: ResponsesAgentRequest) -> Optional[str]:
    """Extract user_id from request."""
    ci = dict(request.custom_inputs or {})
    if "user_id" in ci:
        return ci["user_id"]
    if request.context and getattr(request.context, "user_id", None):
        return request.context.user_id
    return None


# ---------------------------------------------------------------------------
# Global store — initialized once, used across requests
# We use a module-level store that's created on first use.
# ---------------------------------------------------------------------------

_store = None
_store_initialized = False


async def get_store():
    """Get or create the AsyncDatabricksStore singleton."""
    global _store, _store_initialized
    if _store is None:
        from databricks_langchain import AsyncDatabricksStore
        _store = AsyncDatabricksStore(
            instance_name=LAKEBASE_INSTANCE,
            project=LAKEBASE_PROJECT,
            branch=LAKEBASE_BRANCH,
            embedding_endpoint=EMBEDDING_ENDPOINT,
            embedding_dims=EMBEDDING_DIMS,
        )
    if not _store_initialized:
        await _store.setup()
        _store_initialized = True
    return _store


# ---------------------------------------------------------------------------
# Memory tools for the OpenAI Agents SDK orchestrator
# These are function_tool instances that the agent can call.
# The user_id is passed via a module-level variable set per-request.
# ---------------------------------------------------------------------------

_current_user_id: Optional[str] = None


def set_current_user_id(user_id: Optional[str]):
    """Set the user_id for the current request (called before agent.run)."""
    global _current_user_id
    _current_user_id = user_id


@function_tool
async def get_user_memory(query: str) -> str:
    """Search for relevant information about the user from long-term memory.
    Use this to recall previously stored facts, preferences, or details about the user.
    Call this at the start of conversations to personalize responses."""
    if not _current_user_id:
        return "Memory not available — no user_id provided."
    if not is_long_term_enabled():
        return "Long-term memory not configured."

    store = await get_store()
    namespace = ("user_memories", _current_user_id.replace(".", "-"))
    results = await store.asearch(namespace, query=query, limit=5)

    if not results:
        return "No memories found for this user."

    items = [f"- [{item.key}]: {json.dumps(item.value)}" for item in results]
    return f"Found {len(results)} relevant memories:\n" + "\n".join(items)


@function_tool
async def save_user_memory(memory_key: str, memory_data_json: str) -> str:
    """Save information about the user to long-term memory.
    Use when the user shares preferences, facts, or asks you to remember something.
    memory_key: short identifier (e.g. 'preferred_language', 'role')
    memory_data_json: JSON object with the data to store (e.g. '{"value": "Python"}')"""
    if not _current_user_id:
        return "Cannot save memory — no user_id provided."
    if not is_long_term_enabled():
        return "Long-term memory not configured."

    store = await get_store()
    namespace = ("user_memories", _current_user_id.replace(".", "-"))

    try:
        data = json.loads(memory_data_json)
        if not isinstance(data, dict):
            return f"Failed: memory_data must be a JSON object, not {type(data).__name__}"
        await store.aput(namespace, memory_key, data)
        return f"Saved memory '{memory_key}' for user."
    except json.JSONDecodeError as e:
        return f"Failed to save: invalid JSON — {e}"


@function_tool
async def delete_user_memory(memory_key: str) -> str:
    """Delete a specific memory from the user's long-term memory.
    Use when the user asks you to forget something."""
    if not _current_user_id:
        return "Cannot delete memory — no user_id provided."
    if not is_long_term_enabled():
        return "Long-term memory not configured."

    store = await get_store()
    namespace = ("user_memories", _current_user_id.replace(".", "-"))
    await store.adelete(namespace, memory_key)
    return f"Deleted memory '{memory_key}'."


def get_memory_tools() -> list:
    """Return the memory tools to add to the orchestrator agent."""
    return [get_user_memory, save_user_memory, delete_user_memory]


MEMORY_INSTRUCTIONS = """
You have access to long-term memory tools to remember information about users:
- Use get_user_memory to search for previously saved information about the user
- Use save_user_memory to remember important facts, preferences, or details
- Use delete_user_memory to forget specific information when asked

Always check for relevant memories at the start of a conversation.

Save memories when the user says: "remember that…", "note that…", "from now on…"
Proactively save: preferences, role/expertise, ongoing projects, recurring constraints.
Do NOT save: temporary facts, trivial details, or highly sensitive personal information.
"""
