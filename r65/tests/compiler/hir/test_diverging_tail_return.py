# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""No implicit return after a function tail that never falls through.

A body not ending in `return` gets an implicit `return A`. Appended after a
`loop` with no break out, or an if/else or match whose branches all return,
it was unreachable — and still type-checked: A's u8 is no `-> bool`, so

    fn f() -> bool { loop { if K == 0 { return true; } } }

failed with "returning 'u8' from a function declared '-> bool'". A trailing
all-returning match was worse off: the trailing-expression rule made it
`return <match>`, a match with no value ("returning 'void'").
"""

import pytest

from r65.compiler.main import compile_string


def _compile(fn: str) -> str:
    return compile_string(f"""
#[lowram] static mut K: u8;
#[lowram] static mut OUT: bool;
{fn}
#[entry]
fn main() {{ OUT = f(); }}
""", "t.r65")


@pytest.mark.parametrize("fn", [
    "fn f() -> bool { loop { if K == 0 { return true; } if K == 1 { return false; } } }",
    "fn f() -> bool { if K == 0 { return true; } else { return false; } }",
    "fn f() -> bool { match K { 0 => { return true; }, _ => { return false; } }; }",
    "fn f() -> bool { match K { 0 => return true, _ => { K = 1; return false; } } }",
])
def test_diverging_tail_needs_no_return(fn):
    _compile(fn)


def test_loop_with_break_still_needs_its_return():
    """A break leaves the loop, so the tail falls through: still an error."""
    from r65.compiler.errors import TypeCheckError
    with pytest.raises(TypeCheckError, match="returning 'u8'"):
        _compile("fn f() -> bool { loop { if K == 0 { break; } return true; } }")
