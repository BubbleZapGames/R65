"""
r65-lint.toml config loader.

Discovers ``r65-lint.toml`` by walking up from the source file's directory (or
takes an explicit path via ``--lint-config``), parses it with stdlib
``tomllib``, and produces a :class:`LintConfig` that :func:`run_lint` consumes.

**Built-in rules are opt-in.** Without a config file, or without an
``[lint].enable`` list, no rules run — ``--lint`` is a no-op. A project that
wants to use the built-in style rules must list them explicitly::

    [lint]
    enable = ["L001", "L002", "L003", "L004", "L005", "L006"]

Schema:

    [lint]
    enable  = ["L001", "L002", ...]     # built-in rules to run (required to activate them)
    disable = ["L006"]                   # subtract from the enable set
    deny    = ["L003"]                   # promote matching codes to error (non-zero exit)

    [[rule]]
    code = "C001"
    kind = "reachability_forbidden_access"
    message = "..."
    # kind-specific params

User rule codes must use the ``C`` prefix; built-ins use ``L``. ``[[rule]]``
tables are always enabled — declaring a custom rule implicitly activates it.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Set

from r65.compiler.lint.rule import LintRule


CONFIG_FILENAME = "r65-lint.toml"

_LINT_TABLE_KEYS = {"enable", "disable", "deny"}
_RULE_REQUIRED_KEYS = {"code", "kind", "message"}
_RULE_OPTIONAL_KEYS = {"severity", "hint", "allow_in", "name_pattern"}


class LintConfigError(Exception):
    """Raised when an ``r65-lint.toml`` file is malformed or inconsistent."""

    def __init__(self, message: str, config_path: Optional[Path] = None):
        self.config_path = config_path
        prefix = f"{config_path}: " if config_path else ""
        super().__init__(f"{prefix}{message}")


@dataclass
class LintConfig:
    """Resolved lint configuration for a single compile invocation."""

    enabled_codes: Set[str] = field(default_factory=set)
    denied_codes: Set[str] = field(default_factory=set)
    custom_rules: List[LintRule] = field(default_factory=list)
    raw_rule_specs: List[dict] = field(default_factory=list)
    config_path: Optional[Path] = None

    def is_enabled(self, code: str) -> bool:
        return code in self.enabled_codes

    def is_denied(self, code: str) -> bool:
        return code in self.denied_codes


def default_config() -> LintConfig:
    """Return an empty config — nothing enabled, nothing denied.

    Built-in rules (L001–L006) only run when explicitly listed in a project's
    ``r65-lint.toml`` ``[lint].enable`` list. Without a config file, ``--lint``
    is a no-op.
    """
    return LintConfig(enabled_codes=set(), denied_codes=set())


def discover_config(source_file: Path) -> Optional[Path]:
    """Walk up from the source file's directory looking for ``r65-lint.toml``.

    Returns the first match or ``None`` if the filesystem root is reached.
    """
    if not source_file:
        return None
    try:
        directory = source_file.resolve().parent
    except OSError:
        return None
    while True:
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
        parent = directory.parent
        if parent == directory:
            return None
        directory = parent


def load_config(
    path: Optional[Path] = None,
    source_file: Optional[Path] = None,
    cli_allow: Optional[Iterable[str]] = None,
    cli_deny: Optional[Iterable[str]] = None,
) -> LintConfig:
    """Resolve and load lint config.

    Precedence (applied in order, later overrides earlier):
      1. All built-in rule codes enabled.
      2. ``[lint].enable`` / ``[lint].disable`` from the TOML file.
      3. ``[lint].deny`` from the TOML file.
      4. CLI ``--allow`` removes codes from enabled.
      5. CLI ``--deny`` adds codes to denied.

    Args:
        path: Explicit config path (from ``--lint-config``). If ``None``,
            auto-discovery walks up from ``source_file``.
        source_file: Source file being compiled; used as the starting point
            for auto-discovery.
        cli_allow: Codes passed via ``--allow`` (removes from enabled).
        cli_deny: Codes passed via ``--deny`` (adds to denied).

    Returns:
        A resolved :class:`LintConfig`.

    Raises:
        LintConfigError: On malformed TOML or schema violations.
    """
    config = default_config()

    resolved_path: Optional[Path] = None
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise LintConfigError(
                f"lint config file not found: {p}", config_path=p
            )
        resolved_path = p
    elif source_file is not None:
        resolved_path = discover_config(Path(source_file))

    if resolved_path is not None:
        _apply_toml_file(config, resolved_path)

    if cli_allow:
        for code in cli_allow:
            config.enabled_codes.discard(code)
    if cli_deny:
        for code in cli_deny:
            config.denied_codes.add(code)

    return config


def _apply_toml_file(config: LintConfig, path: Path) -> None:
    try:
        with path.open("rb") as fp:
            data = tomllib.load(fp)
    except tomllib.TOMLDecodeError as e:
        raise LintConfigError(f"TOML parse error: {e}", config_path=path) from e
    except OSError as e:
        raise LintConfigError(f"cannot read config: {e}", config_path=path) from e

    config.config_path = path

    lint_table = data.get("lint")
    if lint_table is not None:
        if not isinstance(lint_table, dict):
            raise LintConfigError(
                "[lint] must be a table", config_path=path
            )
        unknown = set(lint_table) - _LINT_TABLE_KEYS
        if unknown:
            raise LintConfigError(
                f"[lint] has unknown keys: {sorted(unknown)} "
                f"(allowed: {sorted(_LINT_TABLE_KEYS)})",
                config_path=path,
            )
        _apply_lint_table(config, lint_table, path)

    rule_tables = data.get("rule")
    if rule_tables is not None:
        if not isinstance(rule_tables, list):
            raise LintConfigError(
                "[[rule]] must be an array of tables", config_path=path
            )
        for i, raw_rule in enumerate(rule_tables):
            if not isinstance(raw_rule, dict):
                raise LintConfigError(
                    f"[[rule]] entry #{i} is not a table", config_path=path
                )
            _validate_rule_shape(raw_rule, i, path)
            config.raw_rule_specs.append(raw_rule)
            rule = _instantiate_rule(raw_rule, i, path)
            config.custom_rules.append(rule)
            config.enabled_codes.add(rule.code)


def _apply_lint_table(config: LintConfig, table: dict, path: Path) -> None:
    enable = table.get("enable")
    disable = table.get("disable")
    deny = table.get("deny")

    if enable is not None:
        if not _is_list_of_str(enable):
            raise LintConfigError(
                "[lint].enable must be a list of strings", config_path=path
            )
        # Allowlist: start from the given set intersected with what exists.
        requested = set(enable)
        _warn_unknown_builtin_codes(requested, path, "enable")
        config.enabled_codes = requested

    if disable is not None:
        if not _is_list_of_str(disable):
            raise LintConfigError(
                "[lint].disable must be a list of strings", config_path=path
            )
        for code in disable:
            config.enabled_codes.discard(code)

    if deny is not None:
        if not _is_list_of_str(deny):
            raise LintConfigError(
                "[lint].deny must be a list of strings", config_path=path
            )
        config.denied_codes.update(deny)


def _validate_rule_shape(raw: dict, index: int, path: Path) -> None:
    missing = _RULE_REQUIRED_KEYS - set(raw)
    if missing:
        raise LintConfigError(
            f"[[rule]] entry #{index} missing required keys: {sorted(missing)}",
            config_path=path,
        )
    code = raw["code"]
    if not isinstance(code, str) or not code:
        raise LintConfigError(
            f"[[rule]] entry #{index}: `code` must be a non-empty string",
            config_path=path,
        )
    if not code.startswith("C"):
        raise LintConfigError(
            f"[[rule]] entry #{index}: user rule code `{code}` must use the "
            f"`C` prefix (built-in rules reserve `L`)",
            config_path=path,
        )
    if not isinstance(raw["kind"], str):
        raise LintConfigError(
            f"[[rule]] `{code}`: `kind` must be a string", config_path=path
        )
    if not isinstance(raw["message"], str):
        raise LintConfigError(
            f"[[rule]] `{code}`: `message` must be a string", config_path=path
        )
    severity = raw.get("severity")
    if severity is not None and severity not in ("warning", "error"):
        raise LintConfigError(
            f"[[rule]] `{code}`: `severity` must be \"warning\" or \"error\"",
            config_path=path,
        )


def _instantiate_rule(spec: dict, index: int, path: Path) -> LintRule:
    """Look up the rule kind and call its factory with ``spec``."""
    from r65.compiler.lint.rule_kinds import KINDS

    kind = spec["kind"]
    factory = KINDS.get(kind)
    if factory is None:
        raise LintConfigError(
            f"[[rule]] entry #{index} (`{spec['code']}`): unknown kind `{kind}`. "
            f"Available kinds: {sorted(KINDS)}",
            config_path=path,
        )
    try:
        return factory(spec)
    except ValueError as e:
        raise LintConfigError(
            f"[[rule]] entry #{index} (`{spec['code']}`): {e}", config_path=path
        ) from e


def _warn_unknown_builtin_codes(codes: Set[str], path: Path, field_name: str) -> None:
    """Raise if any built-in code (L-prefix) listed in the config doesn't exist."""
    from r65.compiler.lint.rules import BUILTIN_RULES

    known = {cls().code for cls in BUILTIN_RULES}
    unknown = {c for c in codes if c.startswith("L") and c not in known}
    if unknown:
        raise LintConfigError(
            f"[lint].{field_name} references unknown built-in codes: "
            f"{sorted(unknown)} (known: {sorted(known)})",
            config_path=path,
        )


def _is_list_of_str(value) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)
