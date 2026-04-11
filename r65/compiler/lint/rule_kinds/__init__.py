"""
Rule kinds registry.

A rule *kind* is a parameterized analysis template. Users instantiate them
from ``r65-lint.toml`` ``[[rule]]`` tables. Each kind module exposes a
``from_config`` factory; the config loader looks the kind up in
:data:`KINDS` and calls the factory with the parsed spec.
"""

from typing import Any, Callable, Dict

from r65.compiler.lint.rule import LintRule
from r65.compiler.lint.rule_kinds import (
    call_depth_limit,
    enforce_storage_class,
    forbidden_call,
    forbidden_identifier,
    forbidden_instruction,
    forbidden_value_at_register,
    naming_convention,
    reachability_forbidden_access,
    require_attribute,
    zeropage_budget,
)


KindFactory = Callable[[Dict[str, Any]], LintRule]


KINDS: Dict[str, KindFactory] = {
    reachability_forbidden_access.KIND_NAME: reachability_forbidden_access.from_config,
    forbidden_call.KIND_NAME: forbidden_call.from_config,
    forbidden_instruction.KIND_NAME: forbidden_instruction.from_config,
    forbidden_value_at_register.KIND_NAME: forbidden_value_at_register.from_config,
    require_attribute.KIND_NAME: require_attribute.from_config,
    enforce_storage_class.KIND_NAME: enforce_storage_class.from_config,
    forbidden_identifier.KIND_NAME: forbidden_identifier.from_config,
    naming_convention.KIND_NAME: naming_convention.from_config,
    call_depth_limit.KIND_NAME: call_depth_limit.from_config,
    zeropage_budget.KIND_NAME: zeropage_budget.from_config,
}


__all__ = ["KINDS", "KindFactory"]
