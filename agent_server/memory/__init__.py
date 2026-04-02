"""
Memory module for the multi-agent framework.

Short-term memory: Conversation history persisted to Lakebase via AsyncDatabricksSession.
Long-term memory:  User facts/preferences stored in Lakebase via Postgres + embedding search.
"""
