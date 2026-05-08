# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end test for accumulator mode at the back-edge of an inlined loop.

Bug: when a `far fn` containing a loop with mixed-mode body — m16 ops in
the body (e.g. `AND #$8000`) and a u8 counter at the tail — gets inlined
(via `#[inline(always)]` or implicit -O2 inlining of small/called-once
functions), the loop's back-edge does not restore the accumulator to m16
before re-entering the body.

The non-inlined version emits a `REP #$20  ; REQUIRED: restore m16 mode
for block` between the tail's `SEP #$20` and the back-edge `BRA`. The
inlined version drops that REP, leaving the CPU in m8 at the loop top.
At runtime the m16-encoded `AND #$8000` (3 bytes: 29 00 80) decodes as
m8 `AND #$00` (2 bytes) followed by `BRA` opcode + offset, sending PC
into operand bytes — typically a BRK-storm that wraps the stack and
crashes the program.

Concretely: this is what made `classickong.r65` get stuck on the
splash screen at -O2 (U32::mod / U32::div get implicitly inlined into
put_score, and the inlined inner loops don't restore m16 at the back-edge).
"""

from r65.tests.e2e import ExpectedState


class TestInlinedLoopModeBackEdge:
    """Mode tracking on loop back-edges after a function is inlined."""

    def test_inline_always_loop_with_m16_body_and_u8_counter(self, e2e):
        """Inlined loop with m16 body and u8 counter must keep m16 at back-edge.

        `loopy` shifts its u16 input left by 4 with a 1-bit OR fold.
        - m16 ops in the body: `AND #$8000`, `x | 1`, `x + x`.
        - m8 op at the tail: `count--` on a u8 counter, then `count == 0`.

        Trace for input=0x1234:
          start: x=0x1234, count=4
          i=1: x&0x8000=0; x=x*2=0x2468; count=3
          i=2: x&0x8000=0; x=0x48D0; count=2
          i=3: x&0x8000=0; x=0x91A0; count=1
          i=4: x&0x8000!=0; x|=1 -> 0x91A1; x=x*2=0x12342 -> 0x2342 (u16); count=0; break
        Expected: 0x2342.

        The compiler-bug case fails before reaching the assert: WLA-DX
        rejects the `AND #$8000` (asm in m8 mode after a 'mode-fix' SEP),
        which surfaces as `result.success == False` with a CompilationError.
        """
        result = e2e.run('''
            #[zeropage(0x10)]
            static mut RESULT: u16;

            #[inline(always)]
            far fn loopy(input @ A: u16) -> u16 {
                let mut x: u16 = input;
                let mut count: u8 = 4;
                loop {
                    if (x & 0x8000) != 0 {
                        x = x | 1;
                    }
                    x = x + x;
                    count--;
                    if count == 0 { break; }
                }
                return x;
            }

            #[entry]
            far fn main() {
                let result @ A: u16 = loopy(0x1234);
                RESULT = A;
            }
        ''', ExpectedState(memory={
            0x000010: 0x42,  # RESULT low byte
            0x000011: 0x23,  # RESULT high byte
        }))
        assert result.success, f"Failures: {result.failures}, error: {result.error}"

    def test_inline_always_loop_with_xor_body(self, e2e):
        """Same back-edge mode bug, exercised through XOR (EOR #$8000).

        Variant of the AND test using `^ 0x8000`. The body still has m16
        immediates plus a u8 counter at the tail, so the inlined loop's
        back-edge has the same mode-mismatch potential.
        """
        result = e2e.run('''
            #[zeropage(0x14)]
            static mut RESULT: u16;

            #[inline(always)]
            far fn xorshift(input @ A: u16) -> u16 {
                let mut x: u16 = input;
                let mut count: u8 = 4;
                loop {
                    x = x ^ 0x8000;
                    x = x + x;
                    count--;
                    if count == 0 { break; }
                }
                return x;
            }

            #[entry]
            far fn main() {
                // 0x0123 ^ 0x8000 = 0x8123, x*2 = 0x10246 -> 0x0246 (u16)
                // 0x0246 ^ 0x8000 = 0x8246, x*2 = 0x048C
                // 0x048C ^ 0x8000 = 0x848C, x*2 = 0x10918 -> 0x0918
                // 0x0918 ^ 0x8000 = 0x8918, x*2 = 0x11230 -> 0x1230
                let result @ A: u16 = xorshift(0x0123);
                RESULT = A;
            }
        ''', ExpectedState(memory={
            0x000014: 0x30,  # 0x1230 low
            0x000015: 0x12,  # 0x1230 high
        }))
        assert result.success, f"Failures: {result.failures}, error: {result.error}"
