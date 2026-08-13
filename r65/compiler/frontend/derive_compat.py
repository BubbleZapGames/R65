# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Workaround hints for `#[derive(...)]`, which R65 does not support.

R65 has no derive macros, but most of what Rust derives has a direct equivalent:
`Clone` is compiler-generated from an empty `impl`, the comparison operators come
from the `PartialEq`/`PartialOrd` lang items, and printing goes through
`ToString`. This module turns a derive list into per-trait guidance so the error
tells the author what to write instead of just what they cannot write.

Lives in `frontend` because both the parser (types, which take no attributes in
the grammar) and the HIR attribute processor (functions and statics, which do)
need it, and `hir` may import `frontend` but not the reverse.
"""

# Trait name -> replacement line. `{T}` is the type being derived on.
_EQUIVALENTS = {
    'Clone': "impl Clone for {T} {{}}   // empty body = compiler-generated bitwise copy",
    'Copy': (
        "no Copy — aggregates are pass-by-reference in R65; "
        "use `impl Clone for {T} {{}}` and an explicit `.clone_from(&src)`"
    ),
    'PartialEq': "impl PartialEq for {T} {{ fn eq(*self, other: *{T}) -> bool {{ ... }} }}",
    'Eq': "impl PartialEq for {T} {{ fn eq(*self, other: *{T}) -> bool {{ ... }} }}",
    'PartialOrd': (
        "impl PartialOrd for {T} {{ fn cmp(*self, other: *{T}) -> i8 {{ ... }} }}"
        "   // sign of self - other"
    ),
    'Ord': (
        "impl PartialOrd for {T} {{ fn cmp(*self, other: *{T}) -> i8 {{ ... }} }}"
        "   // sign of self - other"
    ),
    'Debug': (
        "no Debug — implement ToString and print with `format!(\"{{s}}\", &value)`: "
        "impl ToString for {T} {{ fn to_string(far *self, buf: far *u8) -> u16 {{ ... }} }}"
    ),
    'Display': (
        "no Display — implement ToString and print with `format!(\"{{s}}\", &value)`: "
        "impl ToString for {T} {{ fn to_string(far *self, buf: far *u8) -> u16 {{ ... }} }}"
    ),
    'Default': (
        "no Default — initialize explicitly, e.g. "
        "`static mut VALUE: {T} = {T} {{ ... }};` (SNES RAM is unpredictable at power-on)"
    ),
    'Hash': "no Hash — write a hashing free function over the fields",
}

_PLACEHOLDER = "YourType"


def derive_hint(traits, type_name=None) -> str:
    """Build a hint mapping each derived trait to its R65 equivalent.

    Args:
        traits: Iterable of trait names from the derive list. May be empty when
            the names could not be recovered from the source.
        type_name: The type being derived on, if known.

    Returns:
        A hint string, multi-line when there is per-trait guidance to give.
    """
    target = type_name or _PLACEHOLDER
    names = [t for t in (traits or []) if t]

    if not names:
        return (
            "R65 has no derive macros — write the trait `impl` explicitly. "
            "Clone is the one trait the compiler generates: `impl Clone for "
            f"{target} {{}}`"
        )

    lines = ["R65 has no derive macros; write these instead:"]
    for name in names:
        template = _EQUIVALENTS.get(name)
        replacement = (
            template.format(T=target) if template
            else f"no R65 equivalent for '{name}' — implement the behavior with a free function"
        )
        lines.append(f"    {name}: {replacement}")
    return "\n".join(lines)
