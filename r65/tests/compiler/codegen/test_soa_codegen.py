# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Codegen assertions for `#[soa]` — the reasons the feature exists.

The runtime behaviour is covered in `r65/tests/e2e/test_soa.py`. These pin
the *shape* of the emitted code: the stride multiply that SoA removes, and
the index-`self` method body that motivates putting `self` in X.

See docs/soa.md.
"""

import pytest

from r65.compiler.main import compile_string


# 18 bytes: past the shift-and-add decomposition table, so the AoS form has to
# call mul16 to scale the index.
BIG_FIELDS = ", ".join(f"f{n}: u8" for n in range(17))


def asm_for(source: str) -> str:
    return compile_string(source, filename='<test>')


class TestStrideMultiplyRemoved:
    def test_aos_18_byte_struct_calls_mul16(self):
        """Baseline: without #[soa] the stride is a runtime multiply."""
        asm = asm_for(f'''
            include!("{__import__('pathlib').Path(__file__).parent.parent.parent.parent.parent
                       / 'stdlib' / 'math.r65'}")
            struct E {{ {BIG_FIELDS}, hp: u8 }}
            #[lowram] static mut POOL: [E; 8];
            #[zeropage] static mut i: u16;
            #[entry] fn main() {{ POOL[i].hp = 7; }}
        ''')
        assert 'mul16' in asm

    def test_soa_18_byte_struct_does_not(self):
        """#[soa] scales by the field, which is 1 byte — no multiply at all."""
        asm = asm_for(f'''
            #[soa] struct E {{ {BIG_FIELDS}, hp: u8 }}
            #[lowram] static mut POOL: [E; 8];
            #[zeropage] static mut i: u16;
            #[entry] fn main() {{ POOL[i].hp = 7; }}
        ''')
        assert 'mul16' not in asm
        assert 'POOL__hp' in asm

    def test_soa_u8_pool_needs_no_shift(self):
        """An all-u8 pool strides by 1, so the index goes straight into X."""
        asm = asm_for('''
            #[soa] struct Enemy { kind: u8, hp: u8 }
            #[zeropage(0x40)] static mut POOL: [Enemy; 8];
            #[zeropage] static mut i: u16;
            #[entry] fn main() { POOL[i].hp = 7; }
        ''')
        body = asm[asm.index('main:'):]
        body = body[:body.index('__SCMP')]
        assert 'ASL' not in body, body


class TestIndexSelfMethod:
    def test_method_body_is_direct_indexed_access(self):
        """`self.x` is one `,X` access — no pointer, no frame, no setup."""
        asm = asm_for('''
            #[soa] struct Enemy { kind: u8, hp: u8, x: u8, y: u8 }
            impl Enemy {
                fn step(self) { self.x = self.x + 1; self.y = self.y + 2; }
            }
            #[zeropage(0x40)] static mut POOL: [Enemy; 8];
            #[zeropage] static mut i: u16;
            #[zeropage] static mut j: u16;
            #[entry] fn main() { POOL[i].step(); POOL[j].step(); }
        ''')
        body = asm[asm.index('Enemy__step:'):]
        body = body[:body.index('RTS')]
        # The plane bases: x at 0x40+2*8 = 0x50, y at 0x58.
        # `self.x = self.x + 1` is one read-modify-write on the plane.
        assert 'INC $50,X' in body, body
        assert 'STA $58,X' in body, body
        # No stack frame: the receiver is a register-resident index.
        assert 'PHX' not in body and ',S' not in body, body

    def test_caller_passes_the_index_in_x(self):
        asm = asm_for('''
            #[soa] struct Enemy { hp: u8, pad: u8 }
            impl Enemy {
                fn a(self) { self.hp = 1; }
                fn b(self) { self.hp = 2; }
            }
            #[zeropage(0x40)] static mut POOL: [Enemy; 8];
            #[zeropage] static mut i: u16;
            #[entry] fn main() { POOL[i].a(); POOL[i].b(); }
        ''')
        assert 'TAX' in asm


class TestReceiverIndexWidening:
    """A `u8` index must be zero-extended before it reaches the 16-bit X.

    X is always 16-bit in R65, so a `u8` index has to be widened. The self
    argument was validated with a bare `assignable` rather than checked in
    the parameter's context the way every other argument is, so no widening
    was inserted and codegen emitted a 16-bit `TAX` over a stale high byte.

    The inlined path happened to widen, which is why this hid — the body
    below is large enough to force a real call.
    """

    BODY = ("self.a = t; self.b = t ^ 0x5A; self.c = t + 3; "
            "self.d = t | 0x10; t = t ^ 0x33; asm!(\"NOP\");")

    @pytest.mark.xfail(
        reason="implicit u8->u16 widening is not performed anywhere except the "
               "stack-push path: `o = i` (u16 = u8) and any implicit narrow "
               "argument to an @X/@Y param both leave the stale B register as "
               "the high byte. Pre-existing and general — not specific to "
               "#[soa]. Confirmed on the emulator: `B = 0x77; o = i;` writes "
               "0x7705.",
        strict=True)
    def test_u8_index_is_zero_extended_on_a_real_call(self):
        asm = asm_for(f'''
            #[soa] struct E {{ a: u8, b: u8, c: u8, d: u8 }}
            #[zeropage] static mut t: u8;
            impl E {{ fn m(self) {{ {self.BODY} }} }}
            #[zeropage(0x40)] static mut P: [E; 4];
            #[zeropage] static mut i: u8;
            #[zeropage] static mut j: u8;
            #[entry] fn main() {{ P[i].m(); P[j].m(); }}
        ''')
        body = asm[asm.index('main:'):]
        body = body[:body.index('__SCMP')]
        assert 'JSR E__m' in body, "expected a real call, not an inline: " + body
        assert 'AND #$FF' in body, "u8 index reached X unwidened:\n" + body

    def test_u16_index_needs_no_widening(self):
        asm = asm_for(f'''
            #[soa] struct E {{ a: u8, b: u8, c: u8, d: u8 }}
            #[zeropage] static mut t: u8;
            impl E {{ fn m(self) {{ {self.BODY} }} }}
            #[zeropage(0x40)] static mut P: [E; 4];
            #[zeropage] static mut i: u16;
            #[zeropage] static mut j: u16;
            #[entry] fn main() {{ P[i].m(); P[j].m(); }}
        ''')
        body = asm[asm.index('main:'):]
        body = body[:body.index('__SCMP')]
        assert 'AND #$FF' not in body, "spurious widening for a u16 index:\n" + body


class TestInlinedMethodOnConstantElement:
    """An #[inline(always)] index method called on a constant element folds
    to absolute addressing (optimize/const_index.py): the method shared by
    every element costs what hand-written per-element code does."""

    SRC = '''
        #[soa] struct Air {{ x: u16, hp: u8, landed: u8 }}
        #[zeropage(0x40)] static mut P: [Air; 2];
        impl Air {{
            #[inline(always)]
            fn step(self) {{
                if self.landed == 0 {{
                    self.x = self.x + 1;
                    self.hp = self.hp - 1;
                }}
            }}
        }}
        #[entry] fn main() {{ {calls} }}
    '''

    def body(self, calls):
        asm = asm_for(self.SRC.format(calls=calls))
        return asm[asm.index('main:'):]

    def test_constant_element_reads_no_x(self):
        body = self.body('P[1].step();')
        assert 'Air__step' not in body, "expected the call inlined:\n" + body
        assert ',X' not in body, "a constant element still indexed by X:\n" + body
        assert 'LDX' not in body and 'TAX' not in body, "dead X load left:\n" + body

    def test_constant_element_lands_at_its_offset(self):
        body = self.body('P[1].step();')
        # x is P__x (u16) at $40, element 1 at $42; landed (u8) at $46 + 1.
        assert '$42' in body, body
        assert '$47' in body, body

    def test_variable_element_still_indexes(self):
        asm = asm_for(self.SRC.format(calls='P[i].step();').replace(
            '#[entry]', '#[zeropage] static mut i: u16;\n        #[entry]'))
        body = asm[asm.index('main:'):]
        assert ',X' in body, "a variable element lost its index:\n" + body
