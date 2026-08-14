# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Method macros used for their value: `receiver.name!(args)` in expression position.

A method macro used to expand only in statement position, so anything reading its
result died in the HIR builder with `Unknown expression type: MethodMacro`. It now
expands wherever an expression is allowed, which makes a block-expression body
(`{ let t = self.0; t + t }`) yield a value.

A body made of statements has nothing to yield, and is rejected at the call site.
"""

import pytest
from r65.compiler.frontend import parse
from r65.compiler.frontend.macros import expand_macros
from r65.compiler.frontend.preprocessor import preprocess
from r65.compiler.hir import HIRBuilder
from r65.compiler.typeck import TypeChecker
from r65.compiler.errors import MacroError


def build_and_check(source: str):
    program = expand_macros(preprocess(parse(source, "test.r65"), "test.r65"))
    hir_prog = HIRBuilder(source_file="test.r65").build_program(program)
    TypeChecker(hir_prog).check()
    return hir_prog


DECL = """
struct W(u8);

impl W {
    macro_rules! doubled() { { let t: u8 = self.0; t + t } }
    macro_rules! plus($n:expr) { { let u: u8 = self.0; u + $n } }
    macro_rules! bump() { { self = W(self.0 + 1); } }
}

#[zeropage(0x10)]
static mut V: W;
#[zeropage(0x11)]
static mut OUT: u8;
"""


def in_main(body: str) -> str:
    return DECL + "\n#[entry]\nfn main() { V = W(5); " + body + " }"


class TestExpressionPositions:
    """Everywhere an expression is allowed."""

    @pytest.mark.parametrize("body", [
        "OUT = V.doubled!();",
        "let x: u8 = V.doubled!(); OUT = x;",
        "OUT = V.plus!(3);",
        "OUT = V.doubled!() + 1;",
        "OUT = V.doubled!() - V.plus!(1);",
        "if V.doubled!() > 9 { OUT = 1; } else { OUT = 2; }",
        "OUT = 0; while OUT < V.doubled!() { OUT = OUT + 1; }",
        "OUT = V.plus!(V.doubled!());",
    ])
    def test_position(self, body):
        build_and_check(in_main(body))

    def test_as_a_call_argument(self):
        src = (DECL + "\nfn idf(n: u8) -> u8 { return n; }\n"
               "#[entry]\nfn main() { V = W(5); OUT = idf(V.doubled!()); }")
        build_and_check(src)

    def test_as_an_array_index(self):
        src = (DECL + "\n#[ram]\nstatic mut ARR: [u8; 16];\n"
               "#[entry]\nfn main() { V = W(5); OUT = ARR[V.doubled!()]; }")
        build_and_check(src)

    def test_as_a_return_value(self):
        src = (DECL + "\nfn get() -> u8 { return V.doubled!(); }\n"
               "#[entry]\nfn main() { V = W(5); OUT = get(); }")
        build_and_check(src)


class TestStatementPositionUnchanged:
    """The pre-existing shape must keep working; expression support is additive."""

    def test_statement_form(self):
        build_and_check(in_main("V.bump!();"))

    def test_statement_macro_still_selects_arms(self):
        src = ("struct C { x: u8 }\n"
               "impl C {\n"
               "    macro_rules! emit {\n"
               "        ($v:literal) => { self.x = $v; };\n"
               "        ($r:reg)     => { self.x = $r; };\n"
               "    }\n"
               "}\n"
               "#[ram]\nstatic mut CON: C;\n"
               "#[entry]\nfn main() { CON.emit!(5); CON.emit!(X); }")
        build_and_check(src)


class TestStatementBodyRejected:
    """A body with no value cannot be used for one."""

    def test_rejected_with_a_message_naming_the_macro(self):
        with pytest.raises(MacroError, match="'bump' does not produce a value"):
            build_and_check(in_main("OUT = V.bump!();"))

    def test_rejected_in_a_let(self):
        with pytest.raises(MacroError, match="does not produce a value"):
            build_and_check(in_main("let x: u8 = V.bump!();"))


class TestMacrosNestedInBlockExpressions:
    """`_expand_expression` did not recurse into block or if expressions, so any
    macro inside one survived to the HIR builder — free macros included."""

    def test_free_macro_in_a_block_expression(self):
        src = ("macro_rules! inc($x:expr) { $x + 1 }\n"
               "#[zeropage(0x10)]\nstatic mut V: u8;\n"
               "#[zeropage(0x11)]\nstatic mut OUT: u8;\n"
               "#[entry]\nfn main() { V = 5; OUT = { let t: u8 = V; inc!(t) }; }")
        build_and_check(src)

    def test_free_macro_in_an_if_expression(self):
        src = ("macro_rules! inc($x:expr) { $x + 1 }\n"
               "#[zeropage(0x10)]\nstatic mut V: u8;\n"
               "#[zeropage(0x11)]\nstatic mut OUT: u8;\n"
               "#[entry]\nfn main() { V = 5; OUT = if V > 1 { inc!(V) } else { V }; }")
        build_and_check(src)

    def test_free_macro_in_a_block_statement_position(self):
        src = ("macro_rules! setit($x:expr) { OUT = $x; }\n"
               "#[zeropage(0x11)]\nstatic mut OUT: u8;\n"
               "#[entry]\nfn main() { OUT = { setit!(7); OUT }; }")
        build_and_check(src)
