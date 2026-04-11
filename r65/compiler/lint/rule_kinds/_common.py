"""Shared helpers for rule kinds."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from r65.compiler.errors import DiagnosticSeverity


def parse_severity(value: Optional[str]) -> DiagnosticSeverity:
    if value is None or value == "warning":
        return DiagnosticSeverity.WARNING
    if value == "error":
        return DiagnosticSeverity.ERROR
    raise ValueError(f"invalid severity: {value!r} (expected 'warning' or 'error')")


def require_key(spec: Dict[str, Any], key: str, kind: str) -> Any:
    if key not in spec:
        raise ValueError(f"rule kind '{kind}' requires `{key}`")
    return spec[key]


def require_list_of_str(value: Any, field_name: str, kind: str) -> List[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(
            f"rule kind '{kind}': `{field_name}` must be a list of strings"
        )
    return list(value)


def optional_list_of_str(
    spec: Dict[str, Any], key: str, kind: str, default: Iterable[str] = ()
) -> List[str]:
    if key not in spec:
        return list(default)
    return require_list_of_str(spec[key], key, kind)


def optional_addr_range(
    spec: Dict[str, Any], key: str, kind: str
) -> Optional[Tuple[int, int]]:
    """Parse a ``{start, end}`` inline table into ``(start, end)`` or ``None``."""
    if key not in spec:
        return None
    raw = spec[key]
    if not isinstance(raw, dict):
        raise ValueError(
            f"rule kind '{kind}': `{key}` must be a table with `start` and `end`"
        )
    if "start" not in raw or "end" not in raw:
        raise ValueError(
            f"rule kind '{kind}': `{key}` must contain both `start` and `end`"
        )
    start = raw["start"]
    end = raw["end"]
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(
            f"rule kind '{kind}': `{key}.start` and `{key}.end` must be integers"
        )
    if start >= end:
        raise ValueError(
            f"rule kind '{kind}': `{key}.start` (0x{start:x}) must be strictly "
            f"less than `{key}.end` (0x{end:x})"
        )
    return (start, end)
