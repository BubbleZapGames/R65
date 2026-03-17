# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Built-in functions for R65 compiler.

Provides definitions and handling for compiler built-in functions
that map directly to 65816 instructions or runtime library calls.
"""

from r65.compiler.builtins.registry import BuiltinRegistry, BuiltinKind

__all__ = [
    'BuiltinRegistry',
    'BuiltinKind',
]
