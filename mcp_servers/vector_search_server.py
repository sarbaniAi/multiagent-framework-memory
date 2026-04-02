#!/usr/bin/env python3
"""
MCP Server for Databricks Vector Search.

Runs as a stdio subprocess, exposing a uniquely-named search tool via MCP protocol.
Launched by the multiagent-framework when a vector_search subagent is configured.

Environment variables (passed by the framework):
  - VS_INDEX_NAME:    Full index name (catalog.schema.index)
  - VS_TOOL_NAME:     Unique tool name (derived from subagent name)
  - VS_COLUMNS:       Comma-separated columns to return (default: content,source)
  - VS_NUM_RESULTS:   Number of results (default: 5)
"""

import logging
import os

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Read config from environment
INDEX_NAME = os.environ.get("VS_INDEX_NAME", "")
TOOL_NAME = os.environ.get("VS_TOOL_NAME", "search")
COLUMNS = os.environ.get("VS_COLUMNS", "content,source").split(",")
NUM_RESULTS = int(os.environ.get("VS_NUM_RESULTS", "5"))

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


def _do_search(query: str) -> str:
    """Execute the vector search query."""
    try:
        result = _get_ws().vector_search_indexes.query_index(
            index_name=INDEX_NAME,
            query_text=query,
            columns=COLUMNS,
            num_results=NUM_RESULTS,
        )
        if not result.result or not result.result.data_array:
            return "No relevant documents found."

        col_names = [c.name for c in result.manifest.columns]
        chunks, sources = [], set()
        for row in result.result.data_array:
            r = dict(zip(col_names, row))
            content_col = COLUMNS[0] if COLUMNS else col_names[0]
            chunks.append(r.get(content_col, str(r)))
            for c in COLUMNS[1:]:
                if r.get(c):
                    sources.add(str(r[c]))

        context = "\n\n---\n\n".join(chunks)
        src = ", ".join(sources) if sources else "knowledge base"
        return f"Sources: {src}\n\n{context}"

    except Exception as e:
        logger.exception("Vector search error for index '%s': %s", INDEX_NAME, e)
        return "Document search is temporarily unavailable. Please try again."


# Register tool with unique name from environment
server.tool(name=TOOL_NAME, description=f"Search documents in {TOOL_NAME}")(_do_search)


if __name__ == "__main__":
    server.run(transport="stdio")
