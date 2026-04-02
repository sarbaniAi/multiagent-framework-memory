"""Vector Search MCP subagent — runs a stdio MCP server for Vector Search queries.

All-MCP version: Vector Search is exposed as an MCP server (stdio subprocess)
instead of a function_tool, for a unified MCP-based architecture.
"""

import json
import os
import sys
from pathlib import Path

from agents.mcp import MCPServerStdio


def build_vector_search_mcp(config: dict) -> MCPServerStdio:
    """
    Build an MCP server (stdio) that queries a Vector Search index.

    Required config keys:
        index_name:   Full index name (catalog.schema.index)
        description:  When to use this tool

    Optional config keys:
        columns:      List of columns to return (default: ["content", "source"])
        num_results:  Number of results (default: 5)
    """
    index_name = config["index_name"]
    description = config["description"]
    columns = config.get("columns", ["content", "source"])
    num_results = config.get("num_results", 5)

    # Path to the MCP server script
    server_script = str(Path(__file__).parent.parent.parent / "mcp_servers" / "vector_search_server.py")

    # Unique tool name derived from subagent name
    tool_name = f"search_{config['name']}"

    # Pass config via environment variables
    env = {
        **os.environ,
        "VS_INDEX_NAME": index_name,
        "VS_TOOL_NAME": tool_name,
        "VS_COLUMNS": ",".join(columns),
        "VS_NUM_RESULTS": str(num_results),
    }

    return MCPServerStdio(
        params={
            "command": sys.executable,
            "args": [server_script],
            "env": env,
        },
        name=description,
    )
