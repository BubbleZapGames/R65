# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
CFG (Conditional Compilation) Condition Evaluator.

Evaluates cfg conditions at compile time based on provided configuration.
Supports identifier, comparison, and logical operations (all, any, not).
"""

from typing import Dict, Set, Union
from r65.compiler.frontend.ast import (
    CfgCondition, CfgIdentifier, CfgNot, CfgAny, CfgAll, CfgComparison
)
from r65.compiler.hir.errors import CompilerError as CompilationError


class CfgEvaluator:
    """Evaluates cfg conditions based on compile-time configuration."""
    
    def __init__(self, cfg_features: Set[str], cfg_values: Dict[str, str]):
        """
        Initialize cfg evaluator.
        
        Args:
            cfg_features: Set of enabled feature flags (e.g., {"snes", "debug"})
            cfg_values: Dictionary of key-value pairs (e.g., {"target": "snes"})
        """
        self.cfg_features = cfg_features
        self.cfg_values = cfg_values
    
    def evaluate(self, condition: CfgCondition) -> bool:
        """
        Evaluate a cfg condition.
        
        Args:
            condition: AST node representing the cfg condition
            
        Returns:
            True if condition evaluates to true, False otherwise
            
        Raises:
            CompilationError: If condition contains unknown identifiers or errors
        """
        if isinstance(condition, CfgIdentifier):
            return self._evaluate_identifier(condition)
        elif isinstance(condition, CfgComparison):
            return self._evaluate_comparison(condition)
        elif isinstance(condition, CfgNot):
            return not self.evaluate(condition.condition)
        elif isinstance(condition, CfgAny):
            return any(self.evaluate(cond) for cond in condition.conditions)
        elif isinstance(condition, CfgAll):
            return all(self.evaluate(cond) for cond in condition.conditions)
        else:
            raise CompilationError(f"Unknown cfg condition type: {type(condition)}")
    
    def _evaluate_identifier(self, ident: CfgIdentifier) -> bool:
        """Evaluate a simple identifier condition."""
        return ident.name in self.cfg_features
    
    def _evaluate_comparison(self, comp: CfgComparison) -> bool:
        """Evaluate a key-value comparison."""
        if comp.key not in self.cfg_values:
            raise CompilationError(f"Unknown cfg key: '{comp.key}'")
        
        actual_value = self.cfg_values[comp.key]
        
        if comp.operator == '=':
            return actual_value == comp.value
        elif comp.operator == '!=':
            return actual_value != comp.value
        else:
            raise CompilationError(f"Unknown cfg operator: '{comp.operator}'")
    
    @classmethod
    def from_string_list(cls, cfg_strings: list[str]) -> 'CfgEvaluator':
        """
        Create CfgEvaluator from a list of configuration strings.
        
        Args:
            cfg_strings: List like ["snes", "debug", "target=snes"]
            
        Returns:
            Configured CfgEvaluator instance
        """
        features = set()
        values = {}
        
        for cfg_str in cfg_strings:
            if '=' in cfg_str:
                # Key=value pair
                key, value = cfg_str.split('=', 1)
                values[key.strip()] = value.strip()
            else:
                # Simple feature flag
                features.add(cfg_str.strip())
        
        return cls(features, values)


def parse_cfg_string(cfg_str: str) -> Dict[str, Union[Set[str], Dict[str, str]]]:
    """
    Parse a single cfg string into features and values.
    
    This is a convenience function for testing.
    """
    evaluator = CfgEvaluator.from_string_list([cfg_str])
    return {
        'features': evaluator.cfg_features,
        'values': evaluator.cfg_values
    }