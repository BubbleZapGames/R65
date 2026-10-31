# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
`&&` and `||` as a value, not only as a condition.

In an `if` they lowered to short-circuit branches; as a value they reached
instruction selection as a BinaryOp with no lowering. They now take the
condition's branches, each setting 1 or 0.
"""

from r65.tests.e2e import ExpectedState


def test_logical_ops_as_values(e2e):
    result = e2e.run('''
        #[zeropage(0x10)]
        static mut P: u8;
        #[zeropage(0x11)]
        static mut Q: u8;
        #[zeropage(0x20)]
        static mut OUT: [u8; 6];

        #[entry]
        fn main() {
            P = 0;
            Q = 5;
            let a: bool = P != 0 || Q == 5;     // right decides: true
            let b: bool = P == 0 || Q == 9;     // short-circuits: true
            let c: bool = P == 0 && Q == 9;     // right decides: false
            let d: bool = P != 0 && Q == 5;     // short-circuits: false
            OUT[0] = a as u8;
            OUT[1] = b as u8;
            OUT[2] = c as u8;
            OUT[3] = d as u8;
            // Q is read only past the short-circuit; its load is not reused
            OUT[4] = Q;
            OUT[5] = (P == 0 || Q == 1) as u8;
        }
    ''', ExpectedState(memory={0x20: [1, 1, 0, 0, 5, 1]}))
    assert result.success, f"Failures: {result.failures}"
