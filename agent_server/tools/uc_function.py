"""UC Function MCP subagent — runs a stdio MCP server for UC SQL function calls.

All-MCP version: UC Functions are exposed as MCP servers (stdio subprocess)
instead of function_tools, for a unified MCP-based architecture.

Security:
  - Function name validated via regex before passing to subprocess.
  - SQL injection protection handled in the MCP server process.
  - Raw database errors are never exposed to the user.
"""

import json
import os
import re
import sys
from pathlib import Path

from agents.mcp import MCPServerStdio


_VALID_FUNC_NAME = re.compile(r"^[\w]+\.[\w]+\.[\w]+$")


def build_uc_function_mcp(config: dict) -> MCPServerStdio:
    """
    Build an MCP server (stdio) that calls a UC SQL function.

    Required config keys:
        function:     Fully qualified function name (catalog.schema.fn_name)
        description:  When to use this tool

    Optional config keys:
        parameters:   List of {name, type} dicts describing function params.
    """
    func_name = config["function"]
    description = config["description"]
    params = config.get("parameters", [])

    # Validate function name to prevent injection via config
    if not _VALID_FUNC_NAME.match(func_name):
        raise ValueError(
            f"Invalid UC function name '{func_name}'. "
            f"Must be fully qualified: catalog.schema.function_name"
        )

    # Path to the MCP server script
    server_script = str(Path(__file__).parent.parent.parent / "mcp_servers" / "uc_function_server.py")

    # Unique tool name derived from subagent name
    tool_name = f"call_{config['name']}"

    # Pass config via environment variables
    env = {
        **os.environ,
        "UC_FUNC_NAME": func_name,
        "UC_TOOL_NAME": tool_name,
        "UC_FUNC_PARAMS": json.dumps(params),
    }

    return MCPServerStdio(
        params={
            "command": sys.executable,
            "args": [server_script],
            "env": env,
        },
        name=description,
    )
