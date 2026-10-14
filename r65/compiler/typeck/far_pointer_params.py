# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Near-to-far promotion of pointer parameters.

A `*T` parameter is near: 16 bits, read through the data bank. When any
direct call passes it an address a near pointer cannot reach (`&BUF` on a
`#[ram]` static, a ROM static in another bank or under HiROM; see
PointerValidator._needs_far_pointer), the parameter is retyped to `far *T`
for the whole program. Its other callers then pass a 24-bit pointer too, and
the MIR builder takes the D=S path (indirect long addressing) for the
function's dereferences.

This is the one place type checking changes a declaration: it runs as an
explicit step before any body is checked, so every call and the callee's own
body see the final parameter types, and it returns what it changed.
"""

from dataclasses import dataclass
from typing import List

from r65.compiler.errors import SourceLocation
from r65.compiler.hir import (
    HIRAddressOf, HIRFunctionCall, HIRIdentifier, HIRParameter, HIRProgram,
    HIRTypeCast, SymbolKind,
)
from r65.compiler.hir.nodes import walk_hir
from r65.compiler.hir.types import PointerTypeInfo
from r65.compiler.typeck.pointer_validator import PointerValidator


@dataclass
class FarPromotion:
    """One parameter retyped from `*T` to `far *T`, and the call that forced it."""
    function: str
    parameter: HIRParameter
    call_loc: SourceLocation


def promote_far_pointer_params(program: HIRProgram) -> List[FarPromotion]:
    """Retype every near pointer parameter that some direct call passes a far address.

    Mutates `param.param_type` and its symbol's `var_type`. A call that wrapped
    the address in a cast (`&BUF as *u8`, written to satisfy the near
    parameter) has the cast removed, so the bank byte survives. Returns the
    promotions in the order they were made.
    """
    func_map = {f.name: f for f in program.all_functions()}
    promotions: List[FarPromotion] = []
    promoted_ids = set()

    for func in func_map.values():
        if func.body is None:
            continue
        for expr in walk_hir(func.body):
            if not isinstance(expr, HIRFunctionCall):
                continue
            # Only direct calls to known functions
            if not isinstance(expr.func, HIRIdentifier):
                continue
            if not expr.func.symbol or expr.func.symbol.kind != SymbolKind.FUNCTION:
                continue
            callee = func_map.get(expr.func.symbol.name)
            if callee is None:
                continue

            for i, arg in enumerate(expr.args):
                if i >= len(callee.parameters):
                    break
                param = callee.parameters[i]
                if not isinstance(param.param_type, PointerTypeInfo):
                    continue
                # A parameter declared `far *T` is left alone; one promoted by
                # an earlier call still has its callers' casts stripped below.
                if param.param_type.is_far and id(param) not in promoted_ids:
                    continue
                # Unwrap explicit casts (e.g. &map as *u8) to find the HIRAddressOf
                inner_arg = arg
                has_cast = False
                while isinstance(inner_arg, HIRTypeCast):
                    has_cast = True
                    inner_arg = inner_arg.expr
                if not isinstance(inner_arg, HIRAddressOf):
                    continue
                if not PointerValidator._needs_far_pointer(inner_arg.operand, program.hirom):
                    continue
                if id(param) not in promoted_ids:
                    param.param_type = PointerTypeInfo(
                        is_far=True,
                        pointee_type=param.param_type.pointee_type
                    )
                    if param.symbol:
                        param.symbol.var_type = param.param_type
                    promoted_ids.add(id(param))
                    promotions.append(FarPromotion(callee.name, param, expr.source_loc))
                if has_cast:
                    expr.args[i] = inner_arg

    return promotions
