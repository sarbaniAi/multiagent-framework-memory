"""
Long-term memory — user facts/preferences persisted to a Delta table.

Uses the official OpenAI Agents SDK pattern: state-based memory with
function_tool for save/recall, persisted via Databricks SQL Statement
Execution API to a Unity Catalog Delta table.

No AsyncDatabricksStore, no LangGraph, no direct Postgres connection.
Uses the same WorkspaceClient + SQL warehouse that UC functions use.

Reference: https://developers.openai.com/cookbook/examples/agents_sdk/context_personalization
"""

import json
import logging
import os
from typing import Optional

from agents import function_tool
from databricks.sdk.service.sql import Disposition, StatementState
from mlflow.types.responses import ResponsesAgentRequest

from agent_server.utils import get_request_ws_client

logger = logging.getLogger(__name__)

# Config
WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")
MEMORY_CATALOG = os.environ.get("MEMORY_CATALOG", os.environ.get("UC_CATALOG", ""))
MEMORY_SCHEMA = os.environ.get("MEMORY_SCHEMA", os.environ.get("UC_SCHEMA", ""))
MEMORY_TABLE = os.environ.get("MEMORY_TABLE", "user_long_term_memory")


def _full_table_name() -> str:
    if MEMORY_CATALOG and MEMORY_SCHEMA:
        return f"{MEMORY_CATALOG}.{MEMORY_SCHEMA}.{MEMORY_TABLE}"
    return MEMORY_TABLE


def is_long_term_enabled() -> bool:
    """Check if long-term memory is configured (needs warehouse + catalog/schema)."""
    return bool(WAREHOUSE_ID and MEMORY_CATALOG and MEMORY_SCHEMA)


def get_user_id(request: ResponsesAgentRequest) -> Optional[str]:
    """Extract user_id from request."""
    ci = dict(request.custom_inputs or {})
    if "user_id" in ci:
        return ci["user_id"]
    if request.context and getattr(request.context, "user_id", None):
        return request.context.user_id
    return None


# ---------------------------------------------------------------------------
# SQL execution via Databricks Statement Execution API
# ---------------------------------------------------------------------------

_table_initialized = False


def _exec_sql(sql: str) -> Optional[list]:
    """Execute SQL and return result rows, or None on error."""
    try:
        resp = get_request_ws_client().statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID,
            statement=sql,
            wait_timeout="30s",
            disposition=Disposition.INLINE,
        )
        if resp.status.state != StatementState.SUCCEEDED:
            error = resp.status.error.message if resp.status.error else "unknown"
            logger.error("Memory SQL failed: %s | SQL: %s", error, sql[:200])
            return None
        if resp.result and resp.result.data_array:
            cols = [c.name for c in resp.manifest.schema.columns]
            return [dict(zip(cols, row)) for row in resp.result.data_array]
        return []
    except Exception as e:
        logger.exception("Memory SQL error: %s", e)
        return None


def _ensure_table() -> bool:
    """Create the memory table if it doesn't exist."""
    global _table_initialized
    if _table_initialized:
        return True
    table = _full_table_name()
    result = _exec_sql(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            user_id STRING NOT NULL,
            memory_key STRING NOT NULL,
            memory_value STRING NOT NULL,
            updated_at TIMESTAMP
        )
    """)
    if result is not None:
        _table_initialized = True
        logger.info("Long-term memory table '%s' ready", table)
        return True
    return False


def _sanitize(val: str) -> str:
    """Escape single quotes for SQL."""
    return str(val).replace("'", "''").replace("\\", "\\\\")


# ---------------------------------------------------------------------------
# Memory tools for the OpenAI Agents SDK orchestrator
# ---------------------------------------------------------------------------

_current_user_id: Optional[str] = None


def set_current_user_id(user_id: Optional[str]):
    global _current_user_id
    _current_user_id = user_id


@function_tool
def get_user_memory(query: str) -> str:
    """Recall stored facts and preferences about the user from long-term memory.
    Use this at the start of conversations to personalize responses.
    Pass "all" to retrieve all memories, or a keyword to filter.

    Args:
        query: Keywords to search for, or "all" to retrieve everything.
    """
    if not _current_user_id:
        return "Memory not available — no user_id provided."
    if not is_long_term_enabled():
        return "Long-term memory not configured."
    if not _ensure_table():
        return "Memory table unavailable."

    table = _full_table_name()
    uid = _sanitize(_current_user_id)

    if query.strip().lower() in ("all", "everything", "*", ""):
        rows = _exec_sql(
            f"SELECT memory_key, memory_value, updated_at FROM {table} "
            f"WHERE user_id = '{uid}' ORDER BY updated_at DESC LIMIT 20"
        )
    else:
        q = _sanitize(query)
        rows = _exec_sql(
            f"SELECT memory_key, memory_value, updated_at FROM {table} "
            f"WHERE user_id = '{uid}' AND (memory_key LIKE '%{q}%' OR memory_value LIKE '%{q}%') "
            f"ORDER BY updated_at DESC LIMIT 20"
        )

    if rows is None:
        return "Memory retrieval temporarily unavailable."
    if not rows:
        return "No memories found for this user."

    items = [f"- [{r['memory_key']}]: {r['memory_value']}" for r in rows]
    return f"Found {len(items)} memories:\n" + "\n".join(items)


@function_tool
def save_user_memory(memory_key: str, memory_value: str) -> str:
    """Save a fact or preference about the user to long-term memory.
    Use when the user shares preferences, facts, or asks you to remember something.

    Args:
        memory_key: Short identifier like 'role', 'region', 'preference_language'.
        memory_value: The information to remember as a plain string.
    """
    if not _current_user_id:
        return "Cannot save — no user_id provided."
    if not is_long_term_enabled():
        return "Long-term memory not configured."
    if not _ensure_table():
        return "Memory table unavailable."

    table = _full_table_name()
    uid = _sanitize(_current_user_id)
    key = _sanitize(memory_key)
    val = _sanitize(memory_value)

    # Upsert via MERGE
    result = _exec_sql(f"""
        MERGE INTO {table} AS target
        USING (SELECT '{uid}' AS user_id, '{key}' AS memory_key) AS source
        ON target.user_id = source.user_id AND target.memory_key = source.memory_key
        WHEN MATCHED THEN UPDATE SET memory_value = '{val}', updated_at = CURRENT_TIMESTAMP()
        WHEN NOT MATCHED THEN INSERT (user_id, memory_key, memory_value, updated_at)
            VALUES ('{uid}', '{key}', '{val}', CURRENT_TIMESTAMP())
    """)

    if result is not None:
        logger.info("Saved memory '%s' for user '%s'", memory_key, _current_user_id)
        return f"Saved memory '{memory_key}' for user."
    return "Memory save temporarily unavailable."


@function_tool
def delete_user_memory(memory_key: str) -> str:
    """Delete a specific memory from the user's long-term memory.
    Use when the user asks you to forget something.

    Args:
        memory_key: The key of the memory to delete.
    """
    if not _current_user_id:
        return "Cannot delete — no user_id provided."
    if not is_long_term_enabled():
        return "Long-term memory not configured."
    if not _ensure_table():
        return "Memory table unavailable."

    table = _full_table_name()
    uid = _sanitize(_current_user_id)
    key = _sanitize(memory_key)

    result = _exec_sql(f"DELETE FROM {table} WHERE user_id = '{uid}' AND memory_key = '{key}'")
    if result is not None:
        return f"Deleted memory '{memory_key}'."
    return "Memory deletion temporarily unavailable."


def get_memory_tools() -> list:
    return [get_user_memory, save_user_memory, delete_user_memory]


MEMORY_INSTRUCTIONS = """
You have access to long-term memory tools to remember information about users:
- Use get_user_memory to recall previously saved information (pass "all" to see everything)
- Use save_user_memory to remember important facts, preferences, or details
- Use delete_user_memory to forget specific information when asked

At the START of every new conversation, call get_user_memory("all") to check for stored context.

Save memories when the user says: "remember that...", "note that...", "from now on..."
Proactively save: role, region, team, preferences, ongoing projects, recurring constraints.
Do NOT save: temporary facts, trivial details, or highly sensitive personal information.
"""
