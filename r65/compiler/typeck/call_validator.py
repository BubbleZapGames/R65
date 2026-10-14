# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Function call validation for R65 type checker.

Handles validation of function calls, method calls, built-in calls,
and function address expressions.
"""

from typing import Callable, Optional
from r65.compiler.hir import (
    HIRFunctionCall, HIRMethodCall, HIRFunctionAddress, HIRFunctionDecl,
    HIRIdentifier, HIRArrayIndex, HIRIntegerLiteral, HIRFieldAccess,
    HIRAddressOf, RegisterBinding,
    SymbolKind, BasicTypeInfo, StructTypeInfo, NeverTypeInfo,
    HIRError,
)
from r65.compiler.hir.types import (
    INTEGER_TYPE_NAMES,
    TypeInfo, FunctionTypeInfo, ArrayTypeInfo, PointerTypeInfo, TraitTypeInfo,
    NewtypeTypeInfo, EnumTypeInfo,
)
from r65.compiler.typeck.errors import TypeCheckError
from r65.compiler.typeck.type_utils import TypeRelations, TypeUtils, widen_implicitly
from r65.compiler.typeck.processor_mode import ProcessorMode


class CallValidator:
    """Validates function and method calls."""

    def __init__(self, symbol_table, relations: TypeRelations,
                 lookup_function_decl_fn: Callable,
                 check_expression_fn: Callable, check_element_base_fn: Callable,
                 get_current_function_fn: Callable):
        """
        Initialize with dependencies.

        Args:
            symbol_table: Symbol table for lookups
            relations: Assignability rules (the TypeChecker's TypeRelations)
            lookup_function_decl_fn: Function to look up function declarations
            check_expression_fn: Function to type check expressions
            check_element_base_fn: Function to type check a method receiver
                (TypeChecker.check_element_base: a whole #[soa] element is legal)
            get_current_function_fn: Function to get current function being checked
        """
        self.symbol_table = symbol_table
        self.relations = relations
        self.lookup_function_decl = lookup_function_decl_fn
        self.check_expression = check_expression_fn
        self.check_element_base = check_element_base_fn
        self.get_current_function = get_current_function_fn

    def _validate_arguments(self, args, params, func_name: str, source_loc, is_indirect: bool = False):
        """
        Validate argument types against parameter types.

        Args:
            args: List of argument expressions
            params: List of parameters (HIRParameter objects) or param types (TypeInfo for indirect)
            func_name: Function name for error messages
            source_loc: Source location for error reporting
            is_indirect: True if params are raw TypeInfo (function pointer call)
        """
        for i, (arg, param) in enumerate(zip(args, params)):
            # Get param type up-front so we can pass it as context_type when
            # checking the argument expression. This enables shift-operand
            # widening (e.g. `oam_size(i << 2, 1)` with i:u8 and param:u16
            # widens `i` to u16, computing the shift in m16 instead of
            # truncating).
            if is_indirect:
                param_type = param
                param_name = None
            else:
                param_type = param.param_type
                param_name = param.name

            arg_type = self.check_expression(arg, context_type=param_type)
            if isinstance(arg_type, NeverTypeInfo):
                continue

            # Directional: an argument is assigned into the parameter, so a
            # newtype parameter accepts its payload but not the reverse, and a
            # far pointer does not narrow into a near parameter.
            if not self.relations.assignable(arg_type, param_type):
                if TypeUtils.drops_bank_byte(arg_type, param_type):
                    hint = f"use an explicit cast: (value as {param_type}) to drop the bank byte"
                elif param_name:
                    hint = f"parameter '{param_name}' expects type {param_type}"
                else:
                    hint = f"use cast if needed: (value as {param_type})"
                raise TypeCheckError(
                    f"argument {i + 1} to '{func_name}' has wrong type: expected {param_type}, found {arg_type}",
                    source_loc=arg.source_loc if hasattr(arg, 'source_loc') else source_loc,
                    hint=hint
                )
            # A stack or variable-bound parameter is zero-extended by the
            # call's codegen; a register one is not, so it takes the cast.
            if not is_indirect and isinstance(param.binding, RegisterBinding):
                args[i] = widen_implicitly(arg, arg_type, param_type)

    def check_function_call(self, expr: HIRFunctionCall, *, clone_sugar_ok: bool = False) -> TypeInfo:
        """
        Type check function call.

        Handles direct, indirect (function pointer), built-in and method-sugar
        (`recv.m()`) calls. `clone_sugar_ok` permits `a.clone()`; only
        TypeChecker._check_copy_value sets it, for a let initializer or the
        value assigned to an aggregate place.
        """
        # Special-case format! string-segment dispatch (emitted by the
        # built-in format! macro for {s} specifiers). Rewrite to either
        # strcpy(buf, val) or val.to_string(buf) based on val's type.
        if expr.builtin_name == '__fmt_str':
            return self._check_fmt_str_call(expr)

        # Check if this is a built-in function call
        if expr.builtin_name:
            return self._check_builtin_call(expr)

        # Check if this is a method call (receiver.method(args))
        if isinstance(expr.func, HIRFieldAccess):
            # Clone lang-item: .clone() / .clone_from() on aggregates. For a manual
            # (user-bodied) clone_from this returns None and falls through to normal
            # static method resolution below.
            if expr.func.field_name in ('clone', 'clone_from'):
                clone_result = self._try_clone_call(expr, clone_sugar_ok)
                if clone_result is not None:
                    return clone_result
            method_result = self._try_method_call(expr)
            if method_result is not None:
                return method_result

        # Handle direct call vs indirect call. An associated function
        # (`Q10::from_int(5)`) resolves to a METHOD symbol but has no receiver,
        # so it takes the ordinary direct-call path.
        if (isinstance(expr.func, HIRIdentifier)
                and expr.func.symbol.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD)):
            # Direct call to a function
            func_symbol = expr.func.symbol
            func_decl = self.lookup_function_decl(func_symbol.name, expr.source_loc)

            # Check argument count
            if len(expr.args) != len(func_decl.parameters):
                param_names = [p.name for p in func_decl.parameters]
                raise TypeCheckError(
                    f"function '{func_symbol.name}' expects {len(func_decl.parameters)} argument(s), got {len(expr.args)}",
                    source_loc=expr.source_loc,
                    hint=f"parameters: {', '.join(param_names)}" if param_names else None
                )

            # Type check each argument
            self._validate_arguments(expr.args, func_decl.parameters, func_symbol.name, expr.source_loc)

            # Check bank compatibility (near functions can't call near functions in different banks)
            self._check_call_bank_compatibility(func_symbol.name, func_decl, expr.source_loc)

            # Set return type
            if func_decl.return_type:
                expr.expr_type = func_decl.return_type
            else:
                expr.expr_type = BasicTypeInfo('void')

        else:
            # Indirect call (function pointer)
            func_type = self.check_expression(expr.func)

            if not isinstance(func_type, FunctionTypeInfo):
                raise TypeCheckError(
                    f"cannot call expression of type {func_type}",
                    source_loc=expr.source_loc,
                    hint="only functions and function pointers can be called"
                )

            # Check argument count
            if len(expr.args) != len(func_type.param_types):
                raise TypeCheckError(
                    f"function pointer expects {len(func_type.param_types)} argument(s), got {len(expr.args)}",
                    source_loc=expr.source_loc
                )

            # Type check each argument
            self._validate_arguments(expr.args, func_type.param_types, "function pointer", expr.source_loc, is_indirect=True)

            # Set return type
            if func_type.return_type:
                expr.expr_type = func_type.return_type
            else:
                expr.expr_type = BasicTypeInfo('void')

        return expr.expr_type

    @staticmethod
    def _soa_index_receiver(field_access, expr, struct_name):
        """The element-index expression for an `#[soa]` method receiver.

        An `#[soa]` method takes `self` as an element index, so the receiver
        must *be* an element. `POOL[i].m()` passes `i`; `PLAYER.m()` and
        `self.m()` reach here already rewritten to that shape by
        `ExpressionBuilder._soa_rewrite_base`, so the one arm covers all three.

        `POOL[i].field.m()` is the fourth: a tagged enum that is a field of an
        `#[soa]` element is still reached by the outer array's index, and its
        planes are the pool's nested ones. The index is the same `i`.
        """
        from r65.compiler.hir.nodes import HIRArrayIndex, HIRFieldAccess

        base = field_access.base
        if (isinstance(base, HIRFieldAccess)
                and isinstance(getattr(base, 'base', None), HIRArrayIndex)):
            return base.base.index

        if not isinstance(field_access.base, HIRArrayIndex):
            raise TypeCheckError(
                f"method '{field_access.field_name}' on #[soa] struct "
                f"'{struct_name}' needs an element",
                source_loc=expr.source_loc,
                hint=f"call it on one element of the pool, e.g. "
                     f"`POOL[i].{field_access.field_name}()`",
            )
        return field_access.base.index

    def _try_clone_call(self, expr: HIRFunctionCall, clone_sugar_ok: bool) -> Optional[TypeInfo]:
        """Resolve a Clone lang-item call: `a.clone()` or `dst.clone_from(&src)`.

        Returns the result type if handled, or None to defer to normal method
        resolution (the manual/user-bodied `clone_from` case).
        """
        from r65.compiler.hir.types import ArrayTypeInfo, StructTypeInfo, PointerTypeInfo, BasicTypeInfo

        field_access = expr.func
        method_name = field_access.field_name

        # `POOL[i].clone()` is never legal (an #[soa] struct cannot impl
        # Clone). Checking the receiver as an element base only lets it
        # type-check, so the "does not implement Clone" error below is reported
        # instead of check_array_index's whole-element error.
        try:
            receiver_type = self.check_element_base(field_access.base)
        except TypeCheckError:
            return None

        # Identify the aggregate type being cloned.
        receiver_is_pointer = isinstance(receiver_type, PointerTypeInfo)
        agg_type = receiver_type.pointee_type if receiver_is_pointer else receiver_type
        if not isinstance(agg_type, (StructTypeInfo, ArrayTypeInfo)):
            return None  # not clonable; let the normal path report it

        is_array = isinstance(agg_type, ArrayTypeInfo)

        # Classify intrinsic (array / auto-struct) vs manual (user clone_from body).
        if is_array:
            kind = 'intrinsic'
        else:
            marker = self.symbol_table.lookup(f"{agg_type.name}$Clone")
            if marker is None:
                raise TypeCheckError(
                    f"type '{agg_type.name}' does not implement Clone",
                    source_loc=expr.source_loc,
                    hint=f"add `impl Clone for {agg_type.name} {{}}` (empty body = bitwise copy)"
                )
            kind = 'intrinsic' if marker.type_info.get('clone_auto') else 'call'

        # Runtime-pointer operands need an indirect copy (deferred); the place-based
        # path requires a static address for both ends.
        if receiver_is_pointer:
            raise TypeCheckError(
                "clone through a runtime pointer is not yet supported",
                source_loc=expr.source_loc,
                hint="clone named aggregate variables/statics, e.g. `dst.clone_from(&src)`"
            )

        if method_name == 'clone_from':
            if len(expr.args) != 1:
                raise TypeCheckError("clone_from takes exactly 1 argument (the source)",
                                     source_loc=expr.source_loc)
            self.check_expression(expr.args[0])
            if kind == 'call':
                return None  # manual: resolve as an ordinary static method call
            expr.clone_info = {'kind': 'intrinsic', 'agg_type': agg_type, 'method': 'clone_from'}
            expr.expr_type = BasicTypeInfo('void')
            return expr.expr_type

        # method_name == 'clone' (0-arg sugar)
        if len(expr.args) != 0:
            raise TypeCheckError("clone takes no arguments", source_loc=expr.source_loc)
        if not clone_sugar_ok:
            raise TypeCheckError(
                "`.clone()` is only allowed as the direct initializer of a let or "
                "assignment to an aggregate place",
                source_loc=expr.source_loc
            )
        if kind == 'call':
            raise TypeCheckError(
                f"`let x = a.clone()` is not yet supported for '{agg_type.name}' "
                f"(it has a custom Clone impl)",
                source_loc=expr.source_loc,
                hint=f"call `x.clone_from(&a)` explicitly"
            )
        expr.clone_info = {'kind': 'intrinsic', 'agg_type': agg_type, 'method': 'clone'}
        expr.expr_type = agg_type
        return expr.expr_type

    def _try_method_call(self, expr: HIRFunctionCall) -> Optional[TypeInfo]:
        """
        Try to handle expr as a method call (receiver.method(args)).

        Returns the result type if this is a valid method call,
        or None if it's not a method call (should fall through to field access).
        """
        field_access = expr.func
        method_name = field_access.field_name

        # Get the type of the receiver (e.g., PLAYER has type Player). In
        # `POOL[i].m()` the receiver is an #[soa] element, legal exactly here.
        try:
            receiver_type = self.check_element_base(field_access.base)
        except TypeCheckError:
            return None

        # Determine struct name from receiver type
        struct_name = None
        receiver_is_pointer = False
        pointer_is_far = False

        # A receiver whose `self` rides in A rather than being addressed: a
        # newtype's payload, or an enum's discriminant. Both are register-sized
        # values with nothing to point at.
        receiver_by_value = False

        if isinstance(receiver_type, StructTypeInfo):
            struct_name = receiver_type.name
        elif isinstance(receiver_type, NewtypeTypeInfo):
            # Method lookup is keyed on the nominal name.
            struct_name = receiver_type.newtype_name
            receiver_by_value = True
        elif isinstance(receiver_type, EnumTypeInfo):
            struct_name = receiver_type.name
            # A fieldless enum is a value and rides in A. A tagged one has no
            # contiguous element, so its receiver is the element index instead —
            # the `self_is_index` path below, shared with #[soa].
            receiver_by_value = not receiver_type.definition.is_tagged
        elif isinstance(receiver_type, PointerTypeInfo):
            if isinstance(receiver_type.pointee_type, StructTypeInfo):
                struct_name = receiver_type.pointee_type.name
                receiver_is_pointer = True
                pointer_is_far = receiver_type.is_far
            elif isinstance(receiver_type.pointee_type, TraitTypeInfo):
                # Trait pointer dispatch: *Drawable.method(args)
                return self._try_trait_method_call(expr, receiver_type, method_name)

        if not struct_name:
            return None

        # Look up method symbol using StructName.method_name key
        method_key = f"{struct_name}.{method_name}"
        method_symbol = self.symbol_table.lookup(method_key)

        if not method_symbol or method_symbol.kind != SymbolKind.METHOD:
            return None

        # Found a method! Get the mangled function name
        method_info = method_symbol.type_info
        if not method_info or not isinstance(method_info, dict):
            return None

        mangled_name = method_info.get('mangled_name')
        method_self_is_far = method_info.get('method_self_is_far', False)

        # Look up the actual function declaration
        func_decl = self.lookup_function_decl(mangled_name, expr.source_loc)
        if not func_decl:
            raise TypeCheckError(
                f"method '{method_name}' not found for struct '{struct_name}'",
                source_loc=expr.source_loc
            )

        # Check self pointer compatibility. Near/far is driven by the method's
        # own self pointer (`far *self` vs `*self`), independently of whether
        # the method itself is called with JSL or JSR.
        expected_far_self = method_self_is_far

        # If receiver is a pointer, check far/near compatibility
        if receiver_is_pointer:
            if expected_far_self and not pointer_is_far:
                raise TypeCheckError(
                    f"method '{method_name}' expects far pointer but got near pointer",
                    source_loc=expr.source_loc,
                    hint=f"method '{method_name}' declares 'far *self'"
                )
            elif not expected_far_self and pointer_is_far:
                raise TypeCheckError(
                    f"method '{method_name}' expects near pointer but got far pointer",
                    source_loc=expr.source_loc,
                    hint=f"declare the method's self as 'far *self' to accept a far pointer"
                )

        # Transform: receiver.method(args) -> mangled_name(&receiver, args)
        # Create address-of expression for the receiver (unless already a pointer)
        self_param = func_decl.parameters[0] if func_decl.parameters else None
        self_is_index = (
            self_param is not None
            and isinstance(self_param.binding, RegisterBinding)
            and self_param.binding.register_name == 'X'
            and not receiver_is_pointer and not receiver_by_value
        )

        if receiver_is_pointer:
            self_arg = field_access.base
        elif receiver_by_value:
            # By-value self: pass the value itself. Taking its address is exactly
            # what a newtype (or an enum's discriminant) exists to avoid.
            self_arg = field_access.base
        elif self_is_index:
            # #[soa]: `self` is the element index, so pass the index rather than
            # an address the element does not have.
            self_arg = self._soa_index_receiver(field_access, expr, struct_name)
        else:
            self_arg = HIRAddressOf(
                operand=field_access.base,
                source_loc=field_access.base.source_loc if hasattr(field_access.base, 'source_loc') else None
            )
            # Set the type of the address-of expression
            self_arg.expr_type = PointerTypeInfo(
                pointee_type=receiver_type,
                is_far=expected_far_self
            )

        # Check argument count (method has self as first param)
        expected_args = len(func_decl.parameters) - 1  # Exclude self
        if len(expr.args) != expected_args:
            param_names = [p.name for p in func_decl.parameters[1:]]
            raise TypeCheckError(
                f"method '{method_name}' expects {expected_args} argument(s), got {len(expr.args)}",
                source_loc=expr.source_loc,
                hint=f"parameters: {', '.join(param_names)}" if param_names else None
            )

        # Type check self argument
        self_param = func_decl.parameters[0]
        if not self.relations.assignable(self_arg.expr_type, self_param.param_type):
            raise TypeCheckError(
                f"self argument has wrong type: expected {self_param.param_type}, found {self_arg.expr_type}",
                source_loc=expr.source_loc
            )
        self_arg = widen_implicitly(self_arg, self_arg.expr_type, self_param.param_type)

        # Type check remaining arguments
        self._validate_arguments(expr.args, func_decl.parameters[1:], method_name, expr.source_loc)

        # Check bank compatibility
        self._check_call_bank_compatibility(mangled_name, func_decl, expr.source_loc)

        # Transform the HIR node for code generation
        # Store the mangled name and self argument in the expression
        expr.method_call_info = {
            'mangled_name': mangled_name,
            'self_arg': self_arg,
            'func_decl': func_decl
        }

        # Set return type
        if func_decl.return_type:
            expr.expr_type = func_decl.return_type
        else:
            expr.expr_type = BasicTypeInfo('void')

        return expr.expr_type

    def _try_trait_method_call(self, expr: HIRFunctionCall, receiver_type: PointerTypeInfo, method_name: str) -> Optional[TypeInfo]:
        """Handle method call on trait pointer (dynamic dispatch)."""
        trait_type = receiver_type.pointee_type  # TraitTypeInfo
        trait_name = trait_type.name

        # Built-in type_id() method on trait pointers
        if method_name == 'type_id':
            if len(expr.args) != 0:
                raise TypeCheckError(
                    f"type_id() takes no arguments, got {len(expr.args)}",
                    source_loc=expr.source_loc
                )
            expr.expr_type = BasicTypeInfo('u8')
            field_access = expr.func
            expr.method_call_info = {
                'is_type_id': True,
                'self_arg': field_access.base,
            }
            return expr.expr_type

        # Look up trait definition from symbol table
        trait_symbol = self.symbol_table.lookup(trait_name)
        if not trait_symbol or trait_symbol.kind != SymbolKind.TRAIT:
            raise TypeCheckError(
                f"'{trait_name}' is not a trait",
                source_loc=expr.source_loc
            )

        trait_def = trait_symbol.definition
        if not trait_def:
            raise TypeCheckError(
                f"trait '{trait_name}' has no definition",
                source_loc=expr.source_loc
            )

        # Find the method in the trait or any of its (transitive) supertraits.
        # Resolution returns the DECLARING trait so dispatch goes through that
        # trait's own jump table (inherited methods reuse the supertrait's table).
        found = self._find_trait_method(trait_name, method_name)
        if found is None:
            available = self._all_trait_method_names(trait_name)
            raise TypeCheckError(
                f"trait '{trait_name}' has no method '{method_name}'",
                source_loc=expr.source_loc,
                hint=f"available methods: {', '.join(available)}"
            )
        declaring_trait, method_index, trait_method = found

        # Validate argument count (excluding self)
        expected_args = len(trait_method.params)
        if len(expr.args) != expected_args:
            raise TypeCheckError(
                f"trait method '{trait_name}::{method_name}' expects {expected_args} argument(s), got {len(expr.args)}",
                source_loc=expr.source_loc
            )

        # The same checks as any call: directional assignability (a newtype
        # takes its payload in, never the reverse), the parameter as context
        # for literals, and widening into a register-bound parameter.
        self._validate_arguments(expr.args, trait_method.params, method_name, expr.source_loc)

        # Build self argument (the trait pointer itself)
        field_access = expr.func
        self_arg = field_access.base

        # Determine if trait methods are far
        trait_is_far = trait_method.is_far

        # Store dispatch info for codegen. Dispatch through the DECLARING trait's
        # table so inherited supertrait methods reuse that trait's wrapper.
        expr.method_call_info = {
            'is_trait_dispatch': True,
            'trait_name': declaring_trait,
            'method_name': method_name,
            'method_index': method_index,
            'self_arg': self_arg,
            'trait_is_far': trait_is_far
        }

        # Set return type from trait method definition
        if trait_method.return_type:
            expr.expr_type = trait_method.return_type
        else:
            expr.expr_type = BasicTypeInfo('void')

        return expr.expr_type

    def _find_trait_method(self, trait_name, method_name, _seen=None):
        """Search a trait and its transitive supertraits for a method.

        Returns (declaring_trait_name, method_index_within_declaring_trait, method)
        or None. No-shadowing validation guarantees the first match is unique.
        """
        if _seen is None:
            _seen = set()
        if trait_name in _seen:
            return None
        _seen.add(trait_name)

        sym = self.symbol_table.lookup(trait_name)
        if not sym or sym.kind != SymbolKind.TRAIT or not sym.definition:
            return None
        trait_def = sym.definition

        for i, m in enumerate(trait_def.methods):
            if m.name == method_name:
                return (trait_name, i, m)

        for supertrait in getattr(trait_def, 'supertraits', []):
            found = self._find_trait_method(supertrait, method_name, _seen)
            if found is not None:
                return found
        return None

    def _all_trait_method_names(self, trait_name, _seen=None):
        """All method names declared by a trait and its transitive supertraits."""
        if _seen is None:
            _seen = set()
        if trait_name in _seen:
            return []
        _seen.add(trait_name)
        sym = self.symbol_table.lookup(trait_name)
        if not sym or sym.kind != SymbolKind.TRAIT or not sym.definition:
            return []
        names = [m.name for m in sym.definition.methods]
        for supertrait in getattr(sym.definition, 'supertraits', []):
            names.extend(self._all_trait_method_names(supertrait, _seen))
        return names

    def _check_fmt_str_call(self, expr: HIRFunctionCall) -> TypeInfo:
        """
        Type check and rewrite an internal __fmt_str(buf, val) call emitted by
        the format! macro for {s} specifiers.

        Dispatches based on val's type:
          - *u8 / far *u8 / [u8; N] -> rewrite to strcpy(buf, val)
          - struct (or pointer-to-struct) implementing ToString
              -> rewrite to val.to_string(buf)
          - otherwise -> type error

        Both branches return u16.
        """
        if len(expr.args) != 2:
            raise TypeCheckError(
                f"__fmt_str expects 2 arguments, got {len(expr.args)}",
                source_loc=expr.source_loc
            )
        buf_arg, val_arg = expr.args
        self.check_expression(buf_arg)
        val_type = self.check_expression(val_arg)

        # Detect u8 string-like types
        is_u8_string = False
        if isinstance(val_type, PointerTypeInfo):
            pt = val_type.pointee_type
            if isinstance(pt, BasicTypeInfo) and pt.name == 'u8':
                is_u8_string = True
        elif isinstance(val_type, ArrayTypeInfo):
            et = val_type.element_type
            if isinstance(et, BasicTypeInfo) and et.name == 'u8':
                is_u8_string = True

        # Detect struct (or pointer-to-struct), or a newtype.
        #
        # Resolution here is by name, not by trait dispatch, so a newtype works
        # the same way even though it cannot implement ToString (the trait takes
        # `*self`, and newtype methods take `self` by value). Its inherent `to_string` is found by
        # the same lookup and called statically -- which is what a newtype wants
        # anyway: no vtable, and `self` arrives by value.
        struct_name: Optional[str] = None
        is_newtype = False
        if isinstance(val_type, StructTypeInfo):
            struct_name = val_type.name
        elif isinstance(val_type, NewtypeTypeInfo):
            struct_name = val_type.newtype_name
            is_newtype = True
        elif (isinstance(val_type, PointerTypeInfo)
              and isinstance(val_type.pointee_type, StructTypeInfo)):
            struct_name = val_type.pointee_type.name

        if is_u8_string:
            strcpy_sym = self.symbol_table.lookup('strcpy')
            if not strcpy_sym or strcpy_sym.kind != SymbolKind.FUNCTION:
                raise TypeCheckError(
                    "format {s} for byte strings needs strcpy",
                    source_loc=expr.source_loc,
                    hint='include "string.r65" or use a ToString-implementing type'
                )
            expr.func = HIRIdentifier(
                name='strcpy', symbol=strcpy_sym, source_loc=expr.source_loc
            )
            expr.builtin_name = None
            return self.check_function_call(expr)

        if struct_name is not None:
            method_sym = self.symbol_table.lookup(f'{struct_name}.to_string')
            if not method_sym or method_sym.kind != SymbolKind.METHOD:
                if is_newtype:
                    # A newtype cannot implement ToString (it takes `*self`),
                    # so point at the inherent method it can have instead --
                    # taking `self` by value, as all newtype methods do.
                    raise TypeCheckError(
                        f"format {{s}}: newtype '{struct_name}' has no 'to_string' method",
                        source_loc=expr.source_loc,
                        hint=f"add: impl {struct_name} {{ fn to_string(self, buf: far *u8) -> u16 {{...}} }}"
                    )
                raise TypeCheckError(
                    f"format {{s}}: type '{struct_name}' does not implement ToString",
                    source_loc=expr.source_loc,
                    hint=f"add: impl ToString for {struct_name} {{ fn to_string(far *self, buf: far *u8) -> u16 {{...}} }}"
                )
            field_access = HIRFieldAccess(
                base=val_arg,
                field_name='to_string',
                source_loc=expr.source_loc,
            )
            expr.func = field_access
            expr.args = [buf_arg]
            expr.builtin_name = None
            return self.check_function_call(expr)

        raise TypeCheckError(
            f"format {{s}} requires a u8 string or a type implementing ToString, got {val_type}",
            source_loc=val_arg.source_loc if hasattr(val_arg, 'source_loc') else expr.source_loc
        )

    def _check_builtin_call(self, expr: HIRFunctionCall) -> TypeInfo:
        """Type check built-in function call."""
        from r65.compiler.builtins import BuiltinRegistry, BuiltinKind

        builtin = BuiltinRegistry.get_builtin(expr.builtin_name)
        if not builtin:
            raise TypeCheckError(
                f"Unknown built-in function: {expr.builtin_name}",
                source_loc=expr.source_loc
            )

        most = builtin.max_param_count if builtin.max_param_count is not None else builtin.param_count
        if not builtin.param_count <= len(expr.args) <= most:
            expected = (f"{most}" if most == builtin.param_count
                        else f"{builtin.param_count} to {most}")
            raise TypeCheckError(
                f"{expr.builtin_name}() takes {expected} argument(s), got {len(expr.args)}",
                source_loc=expr.source_loc
            )

        # Type check arguments
        for arg in expr.args:
            # For size_of, don't type-check type identifiers
            if expr.builtin_name == "size_of":
                if isinstance(arg, (HIRIdentifier, HIRArrayIndex)):
                    continue
            self.check_expression(arg)

        # Handle const built-ins (like size_of)
        if builtin.kind is BuiltinKind.TYPE_INFO:
            if expr.builtin_name == "size_of":
                return self._check_size_of_builtin(expr)
            else:
                raise TypeCheckError(
                    f"Unknown type info built-in: {expr.builtin_name}",
                    source_loc=expr.source_loc
                )

        # Const math builtins - should be folded away, but set types for safety
        if builtin.kind is BuiltinKind.CONST_MATH:
            expr.expr_type = BasicTypeInfo(builtin.return_type)
            return expr.expr_type

        # Set return type
        if builtin.returns_value:
            if expr.args:
                first_arg_type = expr.args[0].expr_type
                if isinstance(first_arg_type, BasicTypeInfo) and first_arg_type.name in INTEGER_TYPE_NAMES:
                    expr.expr_type = first_arg_type
                else:
                    expr.expr_type = BasicTypeInfo('u8')
            else:
                expr.expr_type = BasicTypeInfo('u8')
        else:
            expr.expr_type = BasicTypeInfo('void')

        return expr.expr_type

    def check_method_call(self, expr: HIRMethodCall) -> TypeInfo:
        """Type check method call (e.g., value.rotate_left(3) or array.len())."""
        # Type check receiver
        receiver_type = self.check_expression(expr.receiver)

        # Handle bank_byte() on far pointers
        if expr.method_name == 'bank_byte':
            if not isinstance(receiver_type, PointerTypeInfo) or not receiver_type.is_far:
                raise TypeCheckError(
                    f"bank_byte() can only be called on far pointers, found {receiver_type}",
                    source_loc=expr.source_loc,
                    hint="bank_byte() extracts the bank byte from a far pointer (e.g., far_ptr.bank_byte())"
                )
            if len(expr.args) != 0:
                raise TypeCheckError(
                    f"bank_byte() takes no arguments, got {len(expr.args)}",
                    source_loc=expr.source_loc
                )
            expr.expr_type = BasicTypeInfo('u8')
            return expr.expr_type

        # Handle type_id() on trait pointers
        if expr.method_name == 'type_id':
            if isinstance(receiver_type, PointerTypeInfo) and isinstance(receiver_type.pointee_type, TraitTypeInfo):
                if len(expr.args) != 0:
                    raise TypeCheckError(
                        f"type_id() takes no arguments, got {len(expr.args)}",
                        source_loc=expr.source_loc
                    )
                expr.expr_type = BasicTypeInfo('u8')
                return expr.expr_type
            else:
                raise TypeCheckError(
                    f"type_id() can only be called on trait pointers, found {receiver_type}",
                    source_loc=expr.source_loc
                )

        # Handle len() method on arrays
        if expr.method_name == 'len':
            # Validate receiver is an array type
            if not isinstance(receiver_type, ArrayTypeInfo):
                raise TypeCheckError(
                    f"method 'len' requires array type, found {receiver_type}",
                    source_loc=expr.source_loc,
                    hint="len() only works on arrays"
                )

            # Validate no arguments
            if len(expr.args) != 0:
                raise TypeCheckError(
                    f"len() takes no arguments, got {len(expr.args)}",
                    source_loc=expr.source_loc,
                    hint="example: array.len()"
                )

            # Return type is u16 for arrays (can hold sizes up to 65535)
            expr.expr_type = BasicTypeInfo('u16')
            return expr.expr_type

        # Handle rotate methods on integers
        # Validate receiver is an integer type
        if not isinstance(receiver_type, BasicTypeInfo) or receiver_type.name not in INTEGER_TYPE_NAMES:
            raise TypeCheckError(
                f"method '{expr.method_name}' requires integer type, found {receiver_type}",
                source_loc=expr.source_loc,
                hint="rotate methods only work on u8, i8, u16, or i16"
            )

        # Validate method name
        if expr.method_name not in ['rotate_left', 'rotate_right']:
            raise TypeCheckError(
                f"unknown method '{expr.method_name}' for type {receiver_type}",
                source_loc=expr.source_loc,
                hint="available methods: rotate_left, rotate_right"
            )

        # Type check argument (rotation count)
        if len(expr.args) != 1:
            raise TypeCheckError(
                f"{expr.method_name}() takes exactly 1 argument, got {len(expr.args)}",
                source_loc=expr.source_loc,
                hint="example: value.rotate_left(1)"
            )

        count_arg = expr.args[0]
        self.check_expression(count_arg)

        # Validate count is an integer literal
        if not isinstance(count_arg, HIRIntegerLiteral):
            raise TypeCheckError(
                f"{expr.method_name}() count must be a constant",
                source_loc=count_arg.source_loc,
                hint="rotation count must be a compile-time constant (1-8)"
            )

        # Validate count is in range 1-8
        count_value = count_arg.value
        if not (1 <= count_value <= 8):
            raise TypeCheckError(
                f"{expr.method_name}() count must be between 1 and 8, got {count_value}",
                source_loc=count_arg.source_loc,
                hint="valid rotation counts: 1, 2, 3, 4, 5, 6, 7, 8"
            )

        expr.expr_type = receiver_type
        return expr.expr_type

    def check_function_address(self, expr: HIRFunctionAddress) -> TypeInfo:
        """Type check function address expression."""
        func_symbol = expr.symbol
        if not func_symbol:
            raise TypeCheckError(
                f"Function '{expr.function_name}' not resolved",
                source_loc=expr.source_loc
            )

        func_decl = self.lookup_function_decl(func_symbol.name, expr.source_loc)

        param_types = [param.param_type for param in func_decl.parameters]
        func_type = FunctionTypeInfo(
            is_far=func_decl.is_far,
            param_types=param_types,
            return_type=func_decl.return_type
        )

        expr.expr_type = func_type
        return func_type

    def _check_call_bank_compatibility(self, func_name: str, func_decl: HIRFunctionDecl, source_loc):
        """
        Check bank compatibility between caller and callee.

        A non-far function cannot call another non-far function in a different bank,
        because JSR only uses a 16-bit address and cannot cross bank boundaries.
        """
        # If callee is a far function, any caller can call it (JSL handles cross-bank)
        if func_decl.is_far:
            return

        # Get caller function
        caller_func = self.get_current_function()
        if caller_func is None:
            return  # No caller context (shouldn't happen in normal flow)

        # Get bank numbers (default to 0 if not specified)
        caller_bank = 0
        if caller_func.bank_attr:
            caller_bank = caller_func.bank_attr.bank_number

        callee_bank = 0
        if func_decl.bank_attr:
            callee_bank = func_decl.bank_attr.bank_number

        # If banks differ and callee is not far, this is an error
        if caller_bank != callee_bank:
            raise TypeCheckError(
                f"cannot call near function '{func_name}' from bank {caller_bank}: "
                f"'function {func_name}' is in bank {callee_bank}",
                source_loc=source_loc,
                hint=f"near functions use JSR which cannot cross bank boundaries; "
                     f"declare '{func_name}' as 'far fn' to allow cross-bank calls"
            )

    def _check_size_of_builtin(self, expr: HIRFunctionCall) -> TypeInfo:
        """Type check size_of built-in function call."""
        if len(expr.args) != 1:
            raise TypeCheckError(
                "size_of expects exactly 1 argument",
                source_loc=expr.source_loc
            )

        arg = expr.args[0]

        if isinstance(arg, HIRIdentifier):
            type_name = arg.name

            # Basic types
            if type_name in ['u8', 'i8', 'bool']:
                expr.evaluated_size = 1
                expr.expr_type = BasicTypeInfo('u8')
                return expr.expr_type
            elif type_name in ['u16', 'i16']:
                expr.evaluated_size = 2
                expr.expr_type = BasicTypeInfo('u8')
                return expr.expr_type

            # Look up struct or enum
            symbol = self.symbol_table.lookup(type_name)
            if symbol is None:
                raise TypeCheckError(
                    f"Unknown type: {type_name}",
                    source_loc=expr.source_loc
                )

            if symbol.kind is SymbolKind.STRUCT:
                struct_type = StructTypeInfo(name=type_name, definition=symbol.definition)
                try:
                    from r65.compiler.hir.unified_type_utils import get_unified_type_size
                    size = get_unified_type_size(struct_type, self.symbol_table)
                    expr.evaluated_size = size
                    expr.expr_type = BasicTypeInfo('u8') if size <= 255 else BasicTypeInfo('u16')
                    return expr.expr_type
                except HIRError:
                    raise TypeCheckError(
                        f"Cannot determine size of struct '{type_name}'",
                        source_loc=expr.source_loc
                    )

            elif symbol.kind is SymbolKind.ENUM:
                expr.evaluated_size = 1
                expr.expr_type = BasicTypeInfo('u8')
                return expr.expr_type

            else:
                raise TypeCheckError(
                    f"'{type_name}' is not a valid type for size_of",
                    source_loc=arg.source_loc
                )

        elif isinstance(arg, HIRArrayIndex):
            element_type = arg.expr_type
            if not isinstance(element_type, ArrayTypeInfo):
                raise TypeCheckError(
                    "size_of expects a type identifier or array type",
                    source_loc=arg.source_loc
                )

            try:
                from r65.compiler.hir.unified_type_utils import get_unified_type_size
                element_size = get_unified_type_size(element_type.element_type, self.symbol_table)
                array_size = element_size * element_type.size
                expr.evaluated_size = array_size
                expr.expr_type = BasicTypeInfo('u8') if array_size <= 255 else BasicTypeInfo('u16')
                return expr.expr_type
            except HIRError:
                raise TypeCheckError(
                    "Cannot determine size of array type",
                    source_loc=arg.source_loc
                )

        else:
            raise TypeCheckError(
                "size_of expects a type identifier (like u8, Player, etc.), not a value expression",
                source_loc=arg.source_loc
            )
