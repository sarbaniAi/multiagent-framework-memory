#!/usr/bin/env python3
"""
MCP Server for Databricks Unity Catalog SQL Functions.

Runs as a stdio subprocess, exposing a `call_function` tool via MCP protocol.
Launched by the multiagent-framework when a uc_function subagent is configured.

Environment variables (passed by the framework):
  - UC_FUNC_NAME:         Fully qualified function name (catalog.schema.fn_name)
  - UC_FUNC_PARAMS:       JSON array of {name, type} parameter definitions
  - DATABRICKS_WAREHOUSE_ID: SQL warehouse ID
  - DATABRICKS_HOST:      Workspace host URL
  - DATABRICKS_TOKEN:     Auth token
"""

import json
import logging
import os
import re
import sys

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Read config from environment
FUNC_NAME = os.environ.get("UC_FUNC_NAME", "")
TOOL_NAME = os.environ.get("UC_TOOL_NAME", "call_function")
PARAMS = json.loads(os.environ.get("UC_FUNC_PARAMS", "[]"))
WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")

# Validate function name
_VALID_FUNC_NAME = re.compile(r"^[\w]+\.[\w]+\.[\w]+$")
if FUNC_NAME and not _VALID_FUNC_NAME.match(FUNC_NAME):
    raise ValueError(f"Invalid UC function name: {FUNC_NAME}")

# Create FastMCP server
server = FastMCP(
    name=TOOL_NAME,
)

# Lazy WorkspaceClient
_ws = None


def _get_ws():
    global _ws
    if _ws is None:
        from databricks.sdk import WorkspaceClient
        _ws = WorkspaceClient()
    return _ws


def _sanitize_string(val: str) -> str:
    return str(val).replace("'", "''")


def _validate_numeric(val, target_type: str):
    if target_type in ("integer", "int"):
        return int(val)
    elif target_type in ("float", "double"):
        return float(val)
    raise ValueError(f"Unsupported numeric type: {target_type}")


def _validate_boolean(val) -> str:
    s = str(val).strip().lower()
    if s in ("true", "1", "yes"):
        return "true"
    elif s in ("false", "0", "no"):
        return "false"
    raise ValueError(f"Invalid boolean value: {val}")


def _exec_sql(sql: str) -> str:
    from databricks.sdk.service.sql import Disposition, StatementState

    try:
        resp = _get_ws().statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID,
            statement=sql,
            wait_timeout="30s",
            disposition=Disposition.INLINE,
        )
        if resp.status.state != StatementState.SUCCEEDED:
            error_detail = resp.status.error.message if resp.status.error else "unknown"
            logger.error("SQL failed: %s | SQL: %s", error_detail, sql)
            return "Function execution failed. Please try again."
        if resp.result and resp.result.data_array:
            cols = [c.name for c in resp.manifest.schema.columns]
            if len(resp.result.data_array) == 1:
                row = resp.result.data_array[0]
                if len(cols) == 1:
                    return str(row[0])
                return ", ".join(f"{c}={v}" for c, v in zip(cols, row))
            else:
                rows = []
                for row in resp.result.data_array:
                    rows.append(", ".join(f"{c}={v}" for c, v in zip(cols, row)))
                return "\n".join(rows)
        return "No result returned."
    except Exception as e:
        logger.exception("SQL execution error: %s", e)
        return "Function execution encountered an error. Please try again."


def _call_function(arguments: str) -> str:
    """Call the UC SQL function with the given arguments.

    Args:
        arguments: Comma-separated argument values in parameter order.
                   String values should not be quoted.

    Returns:
        The function result.
    """
    if PARAMS:
        parts = [a.strip() for a in arguments.split(",")]
        args = []
        for i, p in enumerate(PARAMS):
            p_type = p.get("type", "string")
            if i < len(parts):
                val = parts[i].strip().strip("'\"")
                if p_type in ("integer", "int", "float", "double"):
                    try:
                        validated = _validate_numeric(val, p_type)
                        args.append(str(validated))
                    except (ValueError, TypeError):
                        return f"Invalid value for '{p['name']}': expected {p_type}."
                elif p_type == "boolean":
                    try:
                        args.append(_validate_boolean(val))
                    except ValueError:
                        return f"Invalid value for '{p['name']}': expected boolean."
                else:
                    args.append(f"'{_sanitize_string(val)}'")
            else:
                args.append("NULL")

        sql = f"SELECT * FROM {FUNC_NAME}({', '.join(args)})"
    else:
        sanitized = _sanitize_string(arguments)
        sql = f"SELECT * FROM {FUNC_NAME}('{sanitized}')"

    result = _exec_sql(sql)
    return f"{FUNC_NAME}({arguments}) = {result}"


# Register tool with unique name from environment
server.tool(name=TOOL_NAME, description=f"Call UC function {FUNC_NAME.split('.')[-1] if FUNC_NAME else 'function'}")(_call_function)


if __name__ == "__main__":
    server.run(transport="stdio")
