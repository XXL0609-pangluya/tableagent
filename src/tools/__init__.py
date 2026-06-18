"""Tool layer: base contract + registry. Concrete tools land in Phase 1."""

from .base import Tool, ToolRegistry, ToolSpec

__all__ = ["Tool", "ToolRegistry", "ToolSpec"]
