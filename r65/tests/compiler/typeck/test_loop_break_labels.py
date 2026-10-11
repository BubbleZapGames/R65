# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""A loop expression's value can come from a break anywhere inside it.

Its type comes from the valued breaks that leave it. The collector used to
search only `if` statements and plain blocks, skipping nested loops whole, so
a `break 'outer v` in an inner loop — or any break in a match arm — was
invisible, and a loop whose only valued breaks sat there was rejected with
"loop expression must have at least one break with a value".
"""

import pytest

from r65.compiler.main import compile_string
from r65.compiler.errors import TypeCheckError


def _compile(body: str) -> str:
    return compile_string(f"""
#[lowram] static mut K: u8;
#[lowram] static mut OUT: u8;
#[entry]
fn main() {{
    let v: u8 = {body};
    OUT = v;
}}
""", "t.r65")


@pytest.mark.parametrize("body", [
    "'l: loop { loop { break 'l 5; } }",
    "'l: loop { for i in 0..4 { if K == i { break 'l i; } } }",
    "'l: loop { while K < 3 { break 'l 4; } }",
    "'l: loop { match K { 0 => { break 'l 1; }, _ => { break 'l 2; } }; }",
    "loop { match K { 0 => { break 1; }, _ => { break 2; } }; }",
])
def test_break_inside_nested_construct_counts(body):
    _compile(body)


def test_unlabeled_break_in_inner_loop_leaves_only_that_loop():
    with pytest.raises(TypeCheckError, match="at least one break with a value"):
        _compile("'l: loop { loop { break 5; } }")


def test_labeled_break_from_inner_loop_needs_a_value():
    with pytest.raises(TypeCheckError, match="must have a value"):
        _compile("'l: loop { loop { break 'l; } break 'l 1; }")


def test_inner_loop_with_same_label_shadows_the_outer():
    with pytest.raises(TypeCheckError, match="at least one break with a value"):
        _compile("'l: loop { 'l: loop { break 'l 5; } }")
