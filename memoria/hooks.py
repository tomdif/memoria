"""
Auto-save hooks for Claude Code integration.

When configured as a Claude Code hook, this module automatically
ingests conversation turns into memoria after each assistant response.

Hook config (settings.json):
{
  "hooks": {
    "PostToolUse": [{
      "matcher": ".*",
      "command": "python3 -m memoria.hooks save \"$CLAUDE_TOOL_RESULT\""
    }]
  }
}

Or simpler: use the MCP server and let Claude call memoria_remember explicitly.
"""

from __future__ import annotations

import json
import os
import sys
import time


def save_from_hook():
    """Called by Claude Code hook to auto-ingest content."""
    if len(sys.argv) < 3:
        return

    text = sys.argv[2]
    if not text or len(text.strip()) < 20:
        return  # skip trivial content

    # Use a lightweight path — no embeddings, no LLM, just store + heuristic extract
    from .core import Memoria

    db_path = os.environ.get("MEMORIA_DB", "~/.memoria/memoria.db")
    session_id = os.environ.get("CLAUDE_SESSION_ID", f"auto_{int(time.time())}")

    with Memoria(db_path=db_path, llm_call=None) as m:
        m.remember(text, role="assistant", session_id=session_id)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m memoria.hooks save <text>", file=sys.stderr)
        return

    cmd = sys.argv[1]
    if cmd == "save":
        save_from_hook()
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)


if __name__ == "__main__":
    main()
