"""
Short-term memory — conversation history via Lakebase.

Uses AsyncDatabricksSession from databricks-openai to persist conversation
turns in Lakebase (Postgres). The session automatically prepends history
on each turn, so the agent has multi-turn context.

Pattern from: app-templates/agent-openai-agents-sdk-short-term-memory
"""

import logging
import os
from typing import Optional

from databricks.sdk import WorkspaceClient
from databricks_openai.agents import AsyncDatabricksSession
from mlflow.types.responses import ResponsesAgentRequest

logger = logging.getLogger(__name__)


def _is_lakebase_hostname(value: str) -> bool:
    return ".database." in value and value.endswith(".com")


def resolve_lakebase_instance_name(
    instance_name: str, workspace_client: Optional[WorkspaceClient] = None
) -> str:
    """Resolve a Lakebase hostname to an instance name if needed."""
    if not _is_lakebase_hostname(instance_name):
        return instance_name

    client = workspace_client or WorkspaceClient()
    try:
        instances = list(client.database.list_database_instances())
    except Exception as exc:
        raise ValueError(
            f"Unable to list database instances to resolve '{instance_name}'."
        ) from exc

    for inst in instances:
        if instance_name in (inst.read_write_dns, inst.read_only_dns):
            if inst.name:
                logger.info(f"Resolved Lakebase hostname '{instance_name}' → '{inst.name}'")
                return inst.name
    raise ValueError(f"No database instance found matching '{instance_name}'")


# Read config from environment
_LAKEBASE_INSTANCE_RAW = os.environ.get("LAKEBASE_INSTANCE_NAME") or None
LAKEBASE_PROJECT = os.environ.get("LAKEBASE_AUTOSCALING_PROJECT") or None
LAKEBASE_BRANCH = os.environ.get("LAKEBASE_AUTOSCALING_BRANCH") or None

# Resolve instance name only for provisioned instances (not autoscaling)
LAKEBASE_INSTANCE = None
if _LAKEBASE_INSTANCE_RAW:
    try:
        LAKEBASE_INSTANCE = resolve_lakebase_instance_name(_LAKEBASE_INSTANCE_RAW)
    except Exception as e:
        logger.warning("Could not resolve Lakebase instance '%s': %s", _LAKEBASE_INSTANCE_RAW, e)
        # Use raw value as fallback
        LAKEBASE_INSTANCE = _LAKEBASE_INSTANCE_RAW


def is_short_term_enabled() -> bool:
    """Check if short-term memory is configured."""
    return bool(LAKEBASE_INSTANCE or (LAKEBASE_PROJECT and LAKEBASE_BRANCH))


def get_session_id(request: ResponsesAgentRequest) -> str:
    """Extract or generate session_id from request."""
    from uuid_utils import uuid7

    ci = dict(request.custom_inputs or {})
    if ci.get("session_id"):
        return str(ci["session_id"])
    if request.context and getattr(request.context, "conversation_id", None):
        return str(request.context.conversation_id)
    return str(uuid7())


def create_session(session_id: str) -> AsyncDatabricksSession:
    """Create a Lakebase-backed session for conversation history."""
    return AsyncDatabricksSession(
        session_id=session_id,
        instance_name=LAKEBASE_INSTANCE,
        project=LAKEBASE_PROJECT,
        branch=LAKEBASE_BRANCH,
    )


async def deduplicate_input(
    request: ResponsesAgentRequest, session: AsyncDatabricksSession
) -> list[dict]:
    """
    Avoid duplicating messages when session already has history.

    If the session covers prior turns, only send the latest message
    since the session will prepend history automatically.
    """
    messages = [i.model_dump() for i in request.input]
    # Normalize assistant content format
    for msg in messages:
        if (
            isinstance(msg, dict)
            and msg.get("type") == "message"
            and msg.get("role") == "assistant"
            and isinstance(msg.get("content"), str)
        ):
            msg["content"] = [{"type": "output_text", "text": msg["content"], "annotations": []}]

    session_items = await session.get_items()
    if len(session_items) >= len(messages) - 1:
        return [messages[-1]]
    return messages
