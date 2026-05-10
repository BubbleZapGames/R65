# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end smoke tests for type aliases. Alias resolution is covered by
compiler/typeck/test_type_aliases.py; this file just verifies the codegen
path end-to-end through one pointer alias and one fn-pointer alias.
"""

from r65.tests.e2e import ExpectedState


class TestTypeAliasesE2E:
    def test_pointer_alias_runtime(self, e2e):
        """Write through pointer alias produces correct memory."""
        result = e2e.run('''
            struct Sprite { x: u8, y: u8 }

            type SpritePtr = *Sprite;

            #[zeropage(0x10)]
            static mut SPR: Sprite;

            fn set_pos(ptr: SpritePtr, xval @ A: u8) {
                ptr.x = xval;
            }

            #[entry]
            fn main() {
                set_pos(&SPR, 42);
            }
        ''', ExpectedState(memory={0x7E0010: 42}))
        assert result.success, f"Failures: {result.failures}"

    def test_fn_pointer_alias_runtime(self, e2e):
        """Indirect call via fn-pointer alias evaluates correctly."""
        result = e2e.run('''
            #[zeropage(0x02, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x04, register)]
            static mut SCRATCH1: u16;

            type Callback = fn() -> u8;

            #[zeropage(0x10)]
            static mut CB: Callback;

            fn get_answer() -> u8 {
                asm!("NOP");
                return 42;
            }

            #[entry]
            fn main() {
                CB = get_answer;
                A = CB();
            }
        ''', ExpectedState(A=42))
        assert result.success, f"Failures: {result.failures}"
