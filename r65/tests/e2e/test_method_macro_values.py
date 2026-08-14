# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for method macros used as values.

The language-level tests confirm these compile; these confirm the expansion is
substituted in the right place and evaluates to the right number.
"""

from r65.tests.e2e import ExpectedState

DECL = """
    struct W(u8);
    impl W {
        macro_rules! doubled() { { let t: u8 = self.0; t + t } }
        macro_rules! plus($n:expr) { { let u: u8 = self.0; u + $n } }
        macro_rules! bump() { { self = W(self.0 + 1); } }
    }
    #[zeropage(0x10)] static mut V: W;
    #[zeropage(0x11)] static mut OUT: u8;
    #[ram] static mut ARR: [u8; 16];
    fn idf(n: u8) -> u8 { return n; }
"""


def program(body: str) -> str:
    return DECL + "\n#[entry]\nfn main() { V = W(5); " + body + " }"


class TestMethodMacroValues:
    def test_assignment(self, e2e):
        r = e2e.run(program("OUT = V.doubled!();"),
                    ExpectedState(memory={0x7E0011: 10}))
        assert r.success, f"Failures: {r.failures}"

    def test_with_argument(self, e2e):
        r = e2e.run(program("OUT = V.plus!(3);"),
                    ExpectedState(memory={0x7E0011: 8}))
        assert r.success, f"Failures: {r.failures}"

    def test_as_a_binary_operand(self, e2e):
        r = e2e.run(program("OUT = V.doubled!() + 1;"),
                    ExpectedState(memory={0x7E0011: 11}))
        assert r.success, f"Failures: {r.failures}"

    def test_two_in_one_expression(self, e2e):
        r = e2e.run(program("OUT = V.doubled!() + V.plus!(1);"),
                    ExpectedState(memory={0x7E0011: 16}))
        assert r.success, f"Failures: {r.failures}"

    def test_nested(self, e2e):
        """`plus!` taking `doubled!()` as its argument: 5 + 10."""
        r = e2e.run(program("OUT = V.plus!(V.doubled!());"),
                    ExpectedState(memory={0x7E0011: 15}))
        assert r.success, f"Failures: {r.failures}"

    def test_as_a_call_argument(self, e2e):
        r = e2e.run(program("OUT = idf(V.doubled!());"),
                    ExpectedState(memory={0x7E0011: 10}))
        assert r.success, f"Failures: {r.failures}"

    def test_as_an_array_index(self, e2e):
        r = e2e.run(program("ARR[10] = 0xAB; OUT = ARR[V.doubled!()];"),
                    ExpectedState(memory={0x7E0011: 0xAB}))
        assert r.success, f"Failures: {r.failures}"

    def test_as_a_condition(self, e2e):
        r = e2e.run(program("if V.doubled!() > 9 { OUT = 1; } else { OUT = 2; }"),
                    ExpectedState(memory={0x7E0011: 1}))
        assert r.success, f"Failures: {r.failures}"

    def test_in_a_loop_condition(self, e2e):
        """Re-expanded each iteration, so the receiver is re-read every time."""
        r = e2e.run(program("OUT = 0; while OUT < V.doubled!() { OUT = OUT + 1; }"),
                    ExpectedState(memory={0x7E0011: 10}))
        assert r.success, f"Failures: {r.failures}"

    def test_statement_form_still_mutates(self, e2e):
        r = e2e.run(program("V.bump!(); OUT = V.0;"),
                    ExpectedState(memory={0x7E0011: 6}))
        assert r.success, f"Failures: {r.failures}"

    def test_free_macro_in_a_block_expression(self, e2e):
        """Was silently unexpanded before `_expand_expression` recursed into blocks."""
        r = e2e.run('''
            macro_rules! inc($x:expr) { $x + 1 }
            #[zeropage(0x10)] static mut V: u8;
            #[zeropage(0x11)] static mut OUT: u8;
            #[entry]
            fn main() { V = 5; OUT = { let t: u8 = V; inc!(t) }; }
        ''', ExpectedState(memory={0x7E0011: 6}))
        assert r.success, f"Failures: {r.failures}"
