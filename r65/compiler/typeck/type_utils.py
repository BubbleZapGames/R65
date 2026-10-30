# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Type comparison and compatibility utilities for R65."""

from typing import Optional, Tuple
from r65.compiler.hir import (
    TypeInfo, BasicTypeInfo, ArrayTypeInfo, PointerTypeInfo,
    FunctionTypeInfo, StructTypeInfo, EnumTypeInfo,
    NeverTypeInfo
)
from r65.compiler.hir.types import TraitTypeInfo, NewtypeTypeInfo, strip_newtype, INTEGER_TYPE_NAMES


# Valid ranges for integer types
TYPE_RANGES = {
    'u8': (0, 255),
    'i8': (-128, 127),
    'u16': (0, 65535),
    'i16': (-32768, 32767),
}


def function_pointer_type(func_decl, source_loc=None) -> FunctionTypeInfo:
    """The `fn(...)` type a function has as a value.

    A call through a function pointer passes every argument on the stack,
    so a function that binds a parameter to a register or a variable cannot
    be one: the type does not record where the callee looks.
    """
    from r65.compiler.hir.nodes import RegisterBinding
    from r65.compiler.typeck.errors import TypeCheckError
    for param in func_decl.parameters:
        if param.binding is not None:
            where = (param.binding.register_name if isinstance(param.binding, RegisterBinding)
                     else param.binding.variable_name)
            raise TypeCheckError(
                f"function '{func_decl.name}' binds parameter '{param.name}' to "
                f"'{where}' and cannot be used as a function pointer; a call "
                f"through one passes arguments on the stack",
                source_loc=source_loc)
    return FunctionTypeInfo(
        is_far=func_decl.is_far,
        param_types=[param.param_type for param in func_decl.parameters],
        return_type=func_decl.return_type,
    )

def soa_struct_of(type_info) -> Optional[str]:
    """Name of the planed type *type_info* is, or is an array/pointer to.

    A *planed* type is a `#[soa]` struct or a tagged enum: both store their
    bytes as one array per plane, so neither has a contiguous element to copy,
    address, or hold in a local. Every site that refuses one asks this question,
    so both kinds are answered here rather than at each caller.

    Keyed on the *type* rather than on a static's flag: `#[soa]` marks the
    struct, so a local, a parameter, a struct field and a pointer are all
    reachable without any static involved. Reads through `StructTypeInfo.symbol`
    so it answers the same whether `definition` is still the AST node (pass 1)
    or already the HIR node (pass 2).
    """
    from r65.compiler.hir.types import (ArrayTypeInfo, PointerTypeInfo,
                                        StructTypeInfo, EnumTypeInfo)

    seen = 0
    while seen < 8:  # bounded: guards against a cyclic type graph
        seen += 1
        type_info = strip_newtype(type_info)
        if isinstance(type_info, ArrayTypeInfo):
            type_info = type_info.element_type
            continue
        if isinstance(type_info, PointerTypeInfo):
            type_info = type_info.pointee_type
            continue
        break

    if not isinstance(type_info, (StructTypeInfo, EnumTypeInfo)):
        return None
    symbol = getattr(type_info, 'symbol', None)
    definition = getattr(symbol, 'definition', None) if symbol else None
    if definition is None:
        definition = getattr(type_info, 'definition', None)
    if isinstance(type_info, EnumTypeInfo):
        # Asked of the variants rather than of `HIREnumDecl.is_tagged`, because
        # this runs during HIR building too, where an enum declared below the
        # code referring to it still has its AST node here. Both nodes carry
        # `payload_type` per variant, so the question has one answer either way.
        variants = getattr(definition, 'variants', None) or []
        tagged = any(getattr(v, 'payload_type', None) is not None for v in variants)
        return type_info.name if tagged else None
    return type_info.name if getattr(definition, 'is_soa', False) else None


def soa_static_of(expr) -> Optional[object]:
    """The `HIRStaticDecl` behind *expr* if it names a `#[soa]` static, else None.

    Used by the diagnostics that want to name the pool. An SoA array is stored
    as one plane per field, so it has no contiguous base address: everything
    needing one — `&arr`, `&arr[i]`, a clone, a whole-element value — is
    refused, and each refusal site asks this question. `arr[i].field` is the
    one shape that works, and it never reaches here with the array alone.
    """
    from r65.compiler.hir.nodes import HIRIdentifier

    if not isinstance(expr, HIRIdentifier):
        return None
    symbol = getattr(expr, 'symbol', None)
    if symbol is None:
        return None
    definition = getattr(symbol, 'definition', None)
    return definition if getattr(definition, 'is_soa', False) else None


def soa_static_of_index(expr) -> Optional[object]:
    """The `HIRStaticDecl` behind `ARR[i]` if `ARR` is a `#[soa]` static."""
    from r65.compiler.hir.nodes import HIRArrayIndex

    if not isinstance(expr, HIRArrayIndex):
        return None
    return soa_static_of(expr.array)


def widen_implicitly(expr, src_type, dst_type):
    """*expr*, converted when a one-byte value is stored into a two-byte place.

    `assignable` lets any integer flow into any other, but a u8 loaded in
    m8 and stored after `REP #$20` carries whatever B held as its high byte. So
    every assignment-shaped site wraps the value in the same `HIRTypeCast` an
    explicit `as` builds: u8 zero-extends, i8 sign-extends. A literal is only
    retyped. Anything else comes back unchanged.

    A bare register read is left alone: `A` is typed u8 by default, but it is
    the hardware value, and the implicit `A` return of a `-> u16` function
    whose `asm!` left a 16-bit result relies on all of it reaching the caller.
    """
    from r65.compiler.hir import HIRTypeCast, HIRIntegerLiteral, HIRRegister

    if isinstance(expr, HIRRegister):
        return expr
    src = strip_newtype(src_type)
    dst = strip_newtype(dst_type)
    if isinstance(src, EnumTypeInfo):
        narrow = src.size_bytes == 1
    else:
        narrow = isinstance(src, BasicTypeInfo) and src.name in ('u8', 'i8')
    if not (narrow and isinstance(dst, BasicTypeInfo) and dst.name in ('u16', 'i16')):
        return expr
    if isinstance(expr, HIRIntegerLiteral):
        expr.expr_type = dst
        return expr
    cast = HIRTypeCast(expr=expr, target_type=dst, source_loc=getattr(expr, 'source_loc', None))
    cast.expr_type = dst
    return cast


def get_type_range(type_name: str) -> Optional[Tuple[int, int]]:
    """Get the valid (min, max) range for an integer type, or None if unknown."""
    return TYPE_RANGES.get(type_name)


def value_fits_type(value: int, type_name: str) -> bool:
    """Check if an integer value fits within the range of a type."""
    range_info = TYPE_RANGES.get(type_name)
    if range_info is None:
        return True  # Unknown type, assume it fits
    min_val, max_val = range_info
    return min_val <= value <= max_val


class TypeUtils:
    """Utilities for type comparison and compatibility."""

    @staticmethod
    def _pointee_types_compatible(t1: TypeInfo, t2: TypeInfo) -> bool:
        """
        Check if pointee types are compatible for pointer assignment.

        This allows *[T; N] to be compatible with *T (array pointer coerces
        to element pointer).
        """
        # Exact match
        if TypeUtils.types_equal(t1, t2):
            return True

        # *[T; N] can match *T (array pointer coerces to element pointer)
        if isinstance(t1, ArrayTypeInfo) and not isinstance(t2, ArrayTypeInfo):
            return TypeUtils.types_equal(t1.element_type, t2)
        if isinstance(t2, ArrayTypeInfo) and not isinstance(t1, ArrayTypeInfo):
            return TypeUtils.types_equal(t2.element_type, t1)

        return False

    @staticmethod
    def types_equal(t1: TypeInfo, t2: TypeInfo) -> bool:
        """Type identity, except `*[T; N]` equals `*T` (array-pointer coercion)."""
        if isinstance(t1, PointerTypeInfo) and isinstance(t2, PointerTypeInfo):
            if t1.is_far != t2.is_far:
                return False
            # Allow *[T; N] to match *T via array pointer coercion
            return TypeUtils._pointee_types_compatible(t1.pointee_type, t2.pointee_type)

        if type(t1) != type(t2):
            return False

        if isinstance(t1, BasicTypeInfo):
            return t1.name == t2.name

        elif isinstance(t1, ArrayTypeInfo):
            return (t1.size == t2.size and
                    TypeUtils.types_equal(t1.element_type, t2.element_type))

        elif isinstance(t1, FunctionTypeInfo):
            if t1.is_far != t2.is_far:
                return False
            if len(t1.param_types) != len(t2.param_types):
                return False
            for p1, p2 in zip(t1.param_types, t2.param_types):
                if not TypeUtils.types_equal(p1, p2):
                    return False
            if t1.return_type is None and t2.return_type is None:
                return True
            if t1.return_type is None or t2.return_type is None:
                return False
            return TypeUtils.types_equal(t1.return_type, t2.return_type)

        elif isinstance(t1, NewtypeTypeInfo):
            # Nominal identity, so this reads `newtype_name` rather than `name`.
            return t1.newtype_name == t2.newtype_name

        elif isinstance(t1, (StructTypeInfo, EnumTypeInfo, TraitTypeInfo)):
            return t1.name == t2.name

        elif isinstance(t1, NeverTypeInfo):
            return True

        return False

    @staticmethod
    def is_integer_type(t: TypeInfo) -> bool:
        """Check if type is an integer type."""
        return (isinstance(t, BasicTypeInfo) and
                t.name in INTEGER_TYPE_NAMES)

    @staticmethod
    def drops_bank_byte(src: TypeInfo, dst: TypeInfo) -> bool:
        """True if storing `src` into `dst` would narrow a far pointer to near."""
        return (isinstance(src, PointerTypeInfo) and isinstance(dst, PointerTypeInfo)
                and src.is_far and not dst.is_far)

    @staticmethod
    def is_boolean_type(t: TypeInfo) -> bool:
        """Check if type is boolean."""
        return isinstance(t, BasicTypeInfo) and t.name == 'bool'

    @staticmethod
    def get_integer_size(t: TypeInfo) -> Optional[int]:
        """Get size in bytes of integer type."""
        t = strip_newtype(t)
        if isinstance(t, BasicTypeInfo):
            if t.name in ['u8', 'i8', 'bool']:
                return 1
            elif t.name in ['u16', 'i16']:
                return 2
        return None

    @staticmethod
    def is_signed(t: TypeInfo) -> bool:
        """Check if integer type is signed. Asks about the machine representation,
        so a newtype answers for its payload."""
        t = strip_newtype(t)
        return isinstance(t, BasicTypeInfo) and t.name in ['i8', 'i16']

    @staticmethod
    def is_aggregate_type(t: TypeInfo) -> bool:
        """Check if type is an aggregate (array or struct) that cannot be passed by value.

        A newtype is never an aggregate — that is the point of giving it its own
        TypeInfo rather than reusing StructTypeInfo.
        """
        return isinstance(t, (ArrayTypeInfo, StructTypeInfo))

    @staticmethod
    def can_cast(from_type: TypeInfo, to_type: TypeInfo) -> bool:
        """
        Check if explicit cast is valid.

        Allowed casts:
        - Integer to integer (any size/signedness)
        - Integer/bool to bool
        - Pointer to pointer (any combination)
        - Enum to integer
        - Integer to enum
        """
        # Integer <-> Integer
        if TypeUtils.is_integer_type(from_type) and TypeUtils.is_integer_type(to_type):
            return True

        # Bool <-> Integer
        if TypeUtils.is_boolean_type(from_type) and TypeUtils.is_integer_type(to_type):
            return True
        if TypeUtils.is_integer_type(from_type) and TypeUtils.is_boolean_type(to_type):
            return True

        # Pointer <-> Pointer
        if isinstance(from_type, PointerTypeInfo) and isinstance(to_type, PointerTypeInfo):
            return True

        # Pointer <-> Integer (needed for DMA address setup and low-level programming)
        if isinstance(from_type, PointerTypeInfo) and TypeUtils.is_integer_type(to_type):
            return True
        if TypeUtils.is_integer_type(from_type) and isinstance(to_type, PointerTypeInfo):
            return True

        # Enum <-> Integer
        if isinstance(from_type, EnumTypeInfo) and TypeUtils.is_integer_type(to_type):
            return True
        if TypeUtils.is_integer_type(from_type) and isinstance(to_type, EnumTypeInfo):
            return True

        # Newtype: `as` is the explicit escape hatch in both directions. A newtype
        # casts exactly where its payload does — `t as u8`, `n as TileId`.
        if isinstance(from_type, NewtypeTypeInfo) or isinstance(to_type, NewtypeTypeInfo):
            from_payload = strip_newtype(from_type)
            to_payload = strip_newtype(to_type)
            # Wrapping or unwrapping a payload of the same type is a pure retype.
            # Needed for payloads `can_cast` has no widening rule for, such as an
            # enum: `Facing(Direction::East)` casts Direction to Direction.
            if TypeUtils.types_equal(from_payload, to_payload):
                return True
            return TypeUtils.can_cast(from_payload, to_payload)

        return False


class TypeRelations:
    """Assignability and compatibility between types.

    These relations depend on the program being checked: whether a struct
    implements a trait, and which traits extend which, come from its symbol
    table. TypeChecker owns one instance and hands it to the validators.
    """

    def __init__(self, symbol_table):
        self.symbol_table = symbol_table

    def types_compatible(self, t1: TypeInfo, t2: TypeInfo) -> bool:
        """Symmetric compatibility: either type is assignable to the other.

        For sites with no destination: comparison operands and the arms of an
        `if` or `loop` expression. A newtype pairs only with itself here, since
        the transparent-in direction is a property of a write. A site that
        stores into a place uses `assignable`.
        """
        if isinstance(t1, NewtypeTypeInfo) or isinstance(t2, NewtypeTypeInfo):
            return TypeUtils.types_equal(t1, t2)
        return self.assignable(t1, t2) or self.assignable(t2, t1)

    def assignable(self, src: TypeInfo, dst: TypeInfo) -> bool:
        """Check if a value of type `src` may be stored into a place of type `dst`.

        The one directional rule behind every write: let, assignment, argument,
        `self`, return and array element. It accepts:

        - exact matches;
        - any integer into any integer (narrowing truncates, as in `let`);
        - an enum into any integer and any integer into an enum;
        - a payload into its newtype, never a newtype back out:

            let t: TileId = 5;   // ok    - a u8 flows into the newtype
            let n: u8 = t;       // error - a TileId does not flow back out

        - a pointer whose pointee matches, `*[T; N]` and `*T` coercing either
          way; near widens to far, but far never narrows to near, which would
          drop the bank byte (`as *T` says so explicitly);
        - `*Struct` into `*dyn Trait` when the struct implements the trait, and
          `*dyn Sub` into `*dyn Super` (a no-op upcast: same object, same
          TypeId), both at the same far/near width.
        """
        if isinstance(dst, NewtypeTypeInfo) or isinstance(src, NewtypeTypeInfo):
            if TypeUtils.types_equal(src, dst):
                return True
            if isinstance(dst, NewtypeTypeInfo) and not isinstance(src, NewtypeTypeInfo):
                return self.assignable(src, dst.inner)
            return False

        if TypeUtils.types_equal(src, dst):
            return True

        if isinstance(src, PointerTypeInfo) and isinstance(dst, PointerTypeInfo):
            if TypeUtils.drops_bank_byte(src, dst):
                return False
            if TypeUtils._pointee_types_compatible(src.pointee_type, dst.pointee_type):
                return True
            if src.is_far == dst.is_far and isinstance(dst.pointee_type, TraitTypeInfo):
                if isinstance(src.pointee_type, StructTypeInfo):
                    return self._struct_implements_trait(
                        src.pointee_type.name, dst.pointee_type.name)
                if isinstance(src.pointee_type, TraitTypeInfo):
                    return self._trait_extends(
                        src.pointee_type.name, dst.pointee_type.name)
            return False

        if TypeUtils.is_integer_type(src) and TypeUtils.is_integer_type(dst):
            return True

        # An enum is an integer, so it pairs with any integer type.
        if isinstance(src, EnumTypeInfo) and TypeUtils.is_integer_type(dst):
            return True
        if isinstance(dst, EnumTypeInfo) and TypeUtils.is_integer_type(src):
            return True

        return False

    def _struct_implements_trait(self, struct_name: str, trait_name: str) -> bool:
        """Check if a struct implements a trait by looking for dispatch symbols.

        This uses a naming convention check: the HIR builder registers
        'TraitName.method_name.StructName' dispatch symbols for each trait impl.
        We only need to check if any such symbol exists.
        """
        trait_symbol = self.symbol_table.lookup(trait_name)
        if trait_symbol and hasattr(trait_symbol.definition, 'methods'):
            trait_def = trait_symbol.definition
            if trait_def.methods:
                first_method = trait_def.methods[0].name
                dispatch_key = f"{trait_name}.{first_method}.{struct_name}"
                return self.symbol_table.lookup(dispatch_key) is not None
        return False

    def _trait_extends(self, sub_name: str, super_name: str, _seen=None) -> bool:
        """True if `sub_name` (transitively) lists `super_name` as a supertrait."""
        if _seen is None:
            _seen = set()
        if sub_name in _seen:
            return False
        _seen.add(sub_name)
        sym = self.symbol_table.lookup(sub_name)
        defn = sym.definition if sym else None
        supers = getattr(defn, 'supertraits', None) or []
        if super_name in supers:
            return True
        return any(self._trait_extends(s, super_name, _seen) for s in supers)
