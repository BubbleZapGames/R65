# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Queries over the HIR's `trait_dispatch_info`, shared by the HIR (lint) and
MIR (analysis) call graphs. This module imports nothing from either IR, so
both can use it without an import cycle.
"""

from typing import Dict, Set, Tuple


def trait_impls_table(trait_dispatch_info) -> Dict[Tuple[str, str], Set[str]]:
    """(declaring trait, method) -> the implementors a `*dyn` dispatch may invoke.

    `trait_dispatch_info` is built by the HIR and carried unchanged by the
    MIR program. A method with no implementor has no entry.
    """
    table: Dict[Tuple[str, str], Set[str]] = {}
    for trait_name, trait_info in (trait_dispatch_info or {}).items():
        implementors = trait_info.get('implementors', [])
        for method_idx, method_name in enumerate(trait_info.get('methods', [])):
            impls = {impl['mangled'][method_idx] for impl in implementors
                     if method_idx < len(impl.get('mangled', []))}
            if impls:
                table[(trait_name, method_name)] = impls
    return table
