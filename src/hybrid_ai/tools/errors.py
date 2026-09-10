"""Error contract for the tools layer.

Every tool raises :class:`ToolError` (never raw exceptions) so LLM
callers, MCP clients and dashboards can rely on ``code`` + ``message``.
"""

from __future__ import annotations


class ToolError(Exception):
    """A tool call failed in a way the caller should report, not crash on."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message}
