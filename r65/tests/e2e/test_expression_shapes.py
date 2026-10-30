# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Values of the expression shapes in tests/compiler/codegen/test_expression_shapes.py,
checked on the hardware: `!` on a bool; a subtract computed as
`left + ~right + 1` because its right operand was already in A; a left
operand read by the ALU instruction itself (optimize/memory_operand.py);
constant chains combined (_lower_constant_chain); an inlined bool result
threaded straight to its branch (optimize/jump_thread.py).
"""

from r65.tests.e2e import ExpectedState


class TestExpressionShapes:

    def test_logical_not_and_reverse_subtract(self, e2e):
        result = e2e.run('''
            #[zeropage(0xC0)] static mut F: bool;
            #[zeropage(0xC1)] static mut G: u8;
            #[zeropage(0xC2)] static mut H: u8;
            #[zeropage(0xC4)] static mut V: u16;
            #[lowram(0x0300)] static mut OUT: [u8; 8];
            #[lowram(0x0310)] static mut OUT16: [u16; 2];

            #[inline(never)]
            fn rsub(x @ A: u8) -> u8 { return 5 - x; }

            #[inline(never)]
            fn rsub16() -> u16 { return 0x80 - (V << 4); }

            #[inline(never)]
            fn borrow(x @ A: u8) -> u8 {
                let d: u8 = 0x10 - (x + G);
                if d < 0x20 { return 1; }
                return 0;
            }

            #[entry]
            fn main() {
                F = false;
                F = !F;
                OUT[0] = F as u8;      // 1
                F = !F;
                OUT[1] = F as u8;      // 0
                G = 0; H = 0;
                OUT[2] = rsub(0);      // 5
                OUT[3] = rsub(9);      // 5 - 9 = 0xFC
                G = 0x30;
                OUT[4] = borrow(0x0F); // 0x10 - 0x3F = 0xD1, not < 0x20
                G = 0x01;
                OUT[5] = borrow(0x02); // 0x10 - 0x03 = 0x0D, < 0x20
                V = 0x0003;
                OUT16[0] = rsub16();   // 0x80 - 0x30 = 0x50
                V = 0x0010;
                OUT16[1] = rsub16();   // 0x80 - 0x100 = 0xFF80
            }
        ''', ExpectedState(memory={
            0x7E0300: [1, 0, 5, 0xFC, 0, 1],
            0x7E0310: [0x50, 0x00, 0x80, 0xFF],
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_left_operand_read_by_the_alu_instruction(self, e2e):
        result = e2e.run('''
            #[zeropage(0xC0)] static mut G: u8;
            #[zeropage(0xC1)] static mut H: u8;
            #[zeropage(0xC2)] static mut K: u8;
            #[lowram(0x0300)] static mut EXT: [u8; 4];
            #[lowram(0x0310)] static mut OUT: [u8; 3];
            #[zeropage(0xC4)] static mut W: u16;
            #[lowram(0x0320)] static mut OUT16: u16;

            #[inline(never)]
            fn clear(i @ X: u16, m @ A: u8) { EXT[i] = EXT[i] & ~m; }

            #[inline(never)]
            fn kgh() -> u8 { return K - (G + H); }

            #[inline(never)]
            fn chain(x @ A: u8) -> u8 { return x + 8 - 2; }

            #[inline(never)]
            fn rchain() -> u16 { return 3 - (W + 5); }

            #[entry]
            fn main() {
                EXT[0] = 0xFF; EXT[1] = 0xFF; EXT[2] = 0xFF; EXT[3] = 0xFF;
                clear(2, 0x24);
                clear(1, 0x81);
                K = 0x10; G = 0x20; H = 0x05;
                OUT[0] = kgh();     // 0x10 - 0x25 = 0xEB
                K = 0x80; G = 0x01; H = 0x02;
                OUT[1] = kgh();     // 0x7D
                OUT[2] = chain(250);  // wraps: 250 + 6 = 0x00
                W = 0x0010;
                OUT16 = rchain();     // 3 - 0x15 = 0xFFEE
            }
        ''', ExpectedState(memory={
            0x7E0300: [0xFF, 0x7E, 0xDB, 0xFF],
            0x7E0310: [0xEB, 0x7D, 0x00],
            0x7E0320: [0xEE, 0xFF],
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_inlined_bool_threaded_to_its_branch(self, e2e):
        result = e2e.run('''
            #[zeropage(0xC0)] static mut G: u8;
            #[lowram(0x0300)] static mut OUT: [u8; 6];

            #[inline]
            fn is3() -> bool { return G == 3; }

            #[inline(never)]
            fn pick(i @ X: u16) {
                if !is3() { OUT[i] = 1; } else { OUT[i] = 2; }
            }

            #[inline(never)]
            fn twice(i @ X: u16) {
                let b = is3();
                if b { OUT[i] = 7; }
                if b { G = 0; }
            }

            #[entry]
            fn main() {
                G = 3; pick(0);       // 2
                G = 4; pick(1);       // 1
                G = 3; twice(2);      // OUT[2] = 7, G = 0
                OUT[3] = G;           // 0
                G = 5; twice(4);      // OUT[4] untouched
                OUT[5] = G;           // 5
            }
        ''', ExpectedState(memory={
            0x7E0300: [2, 1, 7, 0],
            0x7E0305: [5],
        }))
        assert result.success, f"Failures: {result.failures}"
