"""Installation and diagnostics for agent integrations."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .scopes import resolve_scope, scoped_db_path


def claude_settings_path(*, project: bool = False, cwd: str | Path | None = None) -> Path:
    if project:
        return Path(cwd or Path.cwd()).resolve() / ".claude" / "settings.local.json"
    return Path("~/.claude/settings.json").expanduser()


def codex_hooks_path(*, project: bool = False, cwd: str | Path | None = None) -> Path:
    if project:
        return Path(cwd or Path.cwd()).resolve() / ".codex" / "hooks.json"
    return Path("~/.codex/hooks.json").expanduser()


def _handler(command: str, args: list[str], *, timeout: int, async_: bool = False) -> dict:
    value: dict[str, Any] = {
        "type": "command",
        "command": command,
        "args": args,
        "timeout": timeout,
    }
    if async_:
        value["async"] = True
    return value


def memoria_hook_groups(python_executable: str | None = None) -> dict[str, dict]:
    executable = python_executable or sys.executable
    return {
        "UserPromptSubmit": {
            "hooks": [
                _handler(
                    executable,
                    ["-m", "memoria.hooks", "user-prompt"],
                    timeout=60,
                )
            ]
        },
        "Stop": {
            "hooks": [
                _handler(
                    executable,
                    ["-m", "memoria.hooks", "stop"],
                    timeout=120,
                    async_=True,
                )
            ]
        },
    }


def _command_line(parts: list[str]) -> str:
    """Render the command-string form required by Codex hooks."""
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    return shlex.join(parts)


def codex_hook_groups(python_executable: str | None = None) -> dict[str, dict]:
    executable = python_executable or sys.executable
    return {
        "UserPromptSubmit": {
            "hooks": [
                {
                    "type": "command",
                    "command": _command_line([
                        executable,
                        "-m",
                        "memoria.hooks",
                        "user-prompt",
                        "--agent",
                        "codex",
                    ]),
                    "timeout": 60,
                    "additionalContextLimit": 2500,
                }
            ]
        },
        "Stop": {
            "hooks": [
                {
                    "type": "command",
                    "command": _command_line([
                        executable,
                        "-m",
                        "memoria.hooks",
                        "stop",
                        "--agent",
                        "codex",
                    ]),
                    "timeout": 120,
                    "async": True,
                }
            ]
        },
    }


def _is_memoria_handler(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    command = str(value.get("command") or "")
    args = " ".join(str(arg) for arg in value.get("args", []))
    return "memoria.hooks" in f"{command} {args}"


def _remove_existing_memoria_hooks(settings: dict[str, Any]) -> None:
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        settings["hooks"] = {}
        return

    for event_name, groups in list(hooks.items()):
        if not isinstance(groups, list):
            continue
        retained_groups = []
        for group in groups:
            if not isinstance(group, dict):
                retained_groups.append(group)
                continue
            # Remove the legacy single-command hook shape.
            if _is_memoria_handler(group):
                continue
            handlers = group.get("hooks")
            if isinstance(handlers, list):
                retained_handlers = [
                    handler for handler in handlers if not _is_memoria_handler(handler)
                ]
                if retained_handlers:
                    updated = dict(group)
                    updated["hooks"] = retained_handlers
                    retained_groups.append(updated)
            else:
                retained_groups.append(group)
        if retained_groups:
            hooks[event_name] = retained_groups
        else:
            hooks.pop(event_name, None)


def _read_settings(path: Path, *, label: str = "Claude settings") -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"cannot install into invalid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _write_hook_settings(
    path: Path,
    settings: dict[str, Any],
    groups: dict[str, dict],
) -> dict[str, Any]:
    _remove_existing_memoria_hooks(settings)
    hooks = settings.setdefault("hooks", {})
    for event_name, group in groups.items():
        hooks.setdefault(event_name, []).append(group)

    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_suffix(path.suffix + ".memoria.bak")
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return {
        "installed": True,
        "settings_path": str(path),
        "backup_path": str(backup) if backup.exists() else None,
        "events": ["UserPromptSubmit", "Stop"],
    }


def install_claude_hooks(
    settings_path: str | Path | None = None,
    *,
    project: bool = False,
    cwd: str | Path | None = None,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Install idempotent Claude Code hooks while preserving other settings."""
    path = Path(settings_path).expanduser() if settings_path else claude_settings_path(
        project=project, cwd=cwd
    )
    settings = _read_settings(path)
    return _write_hook_settings(path, settings, memoria_hook_groups(python_executable))


def install_codex_hooks(
    settings_path: str | Path | None = None,
    *,
    project: bool = False,
    cwd: str | Path | None = None,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Install idempotent Codex hooks while preserving other hook config."""
    path = Path(settings_path).expanduser() if settings_path else codex_hooks_path(
        project=project, cwd=cwd
    )
    settings = _read_settings(path, label="Codex hooks config")
    result = _write_hook_settings(path, settings, codex_hook_groups(python_executable))
    result["trust_required"] = True
    return result


def doctor_report(
    *,
    settings_path: str | Path | None = None,
    project: bool = False,
    cwd: str | Path | None = None,
    db_path: str | Path = "~/.memoria/memoria.db",
    integration: str = "claude",
) -> dict[str, Any]:
    """Inspect installation, project scope resolution, and storage paths."""
    if integration not in {"claude", "codex"}:
        raise ValueError(f"unsupported integration: {integration}")
    if settings_path:
        path = Path(settings_path).expanduser()
    elif integration == "codex":
        path = codex_hooks_path(project=project, cwd=cwd)
    else:
        path = claude_settings_path(project=project, cwd=cwd)
    checks: list[dict[str, Any]] = []
    try:
        label = "Codex hooks config" if integration == "codex" else "Claude settings"
        settings = _read_settings(path, label=label)
        hooks = settings.get("hooks", {})
        groups = codex_hook_groups() if integration == "codex" else memoria_hook_groups()
        for event_name in groups:
            installed = any(
                _is_memoria_handler(handler)
                for group in hooks.get(event_name, [])
                if isinstance(group, dict)
                for handler in group.get("hooks", [])
            )
            checks.append({
                "name": f"{integration}_{event_name}",
                "ok": installed,
                "detail": "installed" if installed else "missing",
            })
    except ValueError as exc:
        checks.append({
            "name": f"{integration}_settings",
            "ok": False,
            "detail": str(exc),
        })

    scope = resolve_scope("auto", cwd=cwd or Path.cwd())
    scope_path = scoped_db_path(db_path, scope)
    checks.extend([
        {
            "name": "project_scope",
            "ok": scope.kind == "project",
            "detail": scope.scope_id,
        },
        {
            "name": "scope_storage",
            "ok": True,
            "detail": str(scope_path.expanduser()),
        },
    ])
    return {
        "ok": all(check["ok"] for check in checks),
        "integration": integration,
        "settings_path": str(path),
        "checks": checks,
    }
