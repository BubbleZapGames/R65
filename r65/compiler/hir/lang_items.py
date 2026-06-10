# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Compiler lang-item traits: operator overloading + Clone.

These trait names are *known to the compiler*. They may be implemented via
`impl <Name> for T` WITHOUT a source `trait` declaration, and they are
**static-dispatch only**: never used as `*dyn`, never inject a `TypeId`, and they
do not change struct layout. The right-hand operand type of an operator (and the
source type of a clone) comes from the impl, not a trait signature — which is how
they sidestep R65's lack of generics / associated types / a `Self` type.

See docs/operator-overloading.md for the full design.
"""

# --- Clone -----------------------------------------------------------------
CLONE_TRAIT = "Clone"
CLONE_METHOD = "clone_from"

# --- Tier A: compound-assignment operators ---------------------------------
# operator -> (trait name, method name)
COMPOUND_ASSIGN_OPS = {
    "+=":  ("AddAssign",    "add_assign"),
    "-=":  ("SubAssign",    "sub_assign"),
    "*=":  ("MulAssign",    "mul_assign"),
    "/=":  ("DivAssign",    "div_assign"),
    "%=":  ("RemAssign",    "rem_assign"),
    "&=":  ("BitAndAssign", "bitand_assign"),
    "|=":  ("BitOrAssign",  "bitor_assign"),
    "^=":  ("BitXorAssign", "bitxor_assign"),
    "<<=": ("ShlAssign",    "shl_assign"),
    ">>=": ("ShrAssign",    "shr_assign"),
}

# --- Tier B: comparison operators ------------------------------------------
# Base binary operator -> (trait, method), used to desugar `a OP= b` on aggregates.
# Keyed by the base operator (CompoundAssignment.operator), not the `OP=` spelling.
BINOP_ASSIGN = {
    "+":  ("AddAssign",    "add_assign"),
    "-":  ("SubAssign",    "sub_assign"),
    "*":  ("MulAssign",    "mul_assign"),
    "/":  ("DivAssign",    "div_assign"),
    "%":  ("RemAssign",    "rem_assign"),
    "&":  ("BitAndAssign", "bitand_assign"),
    "|":  ("BitOrAssign",  "bitor_assign"),
    "^":  ("BitXorAssign", "bitxor_assign"),
    "<<": ("ShlAssign",    "shl_assign"),
    ">>": ("ShrAssign",    "shr_assign"),
}

EQ_TRAIT = "PartialEq"
EQ_METHOD = "eq"      # eq(*self, rhs) -> bool        ; powers ==, !=
ORD_TRAIT = "PartialOrd"
ORD_METHOD = "cmp"    # cmp(*self, rhs) -> i8 (sign)  ; powers <, <=, >, >=

OPERATOR_TRAITS = (
    frozenset(trait for trait, _ in COMPOUND_ASSIGN_OPS.values())
    | {EQ_TRAIT, ORD_TRAIT}
)

# Every compiler-known trait that may be impl'd without a source declaration.
LANG_ITEM_TRAITS = frozenset({CLONE_TRAIT}) | OPERATOR_TRAITS


def is_lang_item_trait(name: str) -> bool:
    """True if `name` is a compiler lang-item trait (Clone or an operator trait)."""
    return name in LANG_ITEM_TRAITS
