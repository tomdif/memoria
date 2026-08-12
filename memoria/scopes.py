"""Memory scopes and deterministic per-scope database paths.

The original Memoria database remains the global scope. Project scopes use
separate SQLite files, which makes accidental cross-project retrieval
impossible even if a caller forgets to pass a filter.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


GLOBAL_SCOPE_ID = "global"


@dataclass(frozen=True)
class MemoryScope:
    """A stable memory namespace."""

    scope_id: str
    kind: str
    label: str
    root: Path | None = None

    @classmethod
    def global_scope(cls) -> "MemoryScope":
        return cls(scope_id=GLOBAL_SCOPE_ID, kind="global", label="Global")

    @classmethod
    def for_path(cls, path: str | Path) -> "MemoryScope":
        start = Path(path).expanduser().resolve()
        if start.is_file():
            start = start.parent

        project_root, identity = _git_project(start)
        if project_root is None:
            project_root = start
            identity = f"directory:{project_root}"

        label = project_root.name or "project"
        slug = re.sub(r"[^a-z0-9]+", "-", label.casefold()).strip("-")
        slug = slug[:40] or "project"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        return cls(
            scope_id=f"project:{slug}:{digest}",
            kind="project",
            label=label,
            root=project_root,
        )


def _git_project(start: Path) -> tuple[Path | None, str]:
    """Return the worktree root and a shared identity across its worktrees."""
    try:
        root_result = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        root = Path(root_result.stdout.strip()).expanduser().resolve()
        common_result = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--git-common-dir"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        common = Path(common_result.stdout.strip()).expanduser()
        if not common.is_absolute():
            common = (start / common).resolve()
        else:
            common = common.resolve()
        return root, f"git:{common}"
    except (FileNotFoundError, subprocess.SubprocessError, OSError, ValueError):
        return None, ""


def resolve_scope(
    value: MemoryScope | str | None = None,
    *,
    cwd: str | Path | None = None,
) -> MemoryScope:
    """Resolve ``global``, ``auto``, or ``project:/path`` into a scope."""
    if isinstance(value, MemoryScope):
        return value
    if value is None or value.strip().casefold() == GLOBAL_SCOPE_ID:
        return MemoryScope.global_scope()

    normalized = value.strip()
    if normalized.casefold() in {"auto", "project"}:
        return MemoryScope.for_path(cwd or Path.cwd())
    if normalized.casefold().startswith("project:"):
        explicit_path = normalized.split(":", 1)[1]
        if not explicit_path:
            raise ValueError("project scope requires a path")
        return MemoryScope.for_path(explicit_path)
    raise ValueError("scope must be 'global', 'auto', 'project', or 'project:/path'")


def scoped_db_path(base_path: str | Path, scope: MemoryScope) -> Path:
    """Map a scope to its SQLite file while preserving the legacy global path."""
    base = Path(base_path).expanduser()
    if scope.kind == "global":
        return base

    filename = scope.scope_id.replace(":", "-") + ".db"
    return base.parent / "scopes" / filename
