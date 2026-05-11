# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Regression test for -O2 implicit-inlining mode-tracking bug.

When `BlockCloner.clone_blocks()` failed to copy `entry_mode`/`exit_mode`
from source to cloned blocks, the inlined body downstream of a mode-mixed
join was emitted as m8 even though the original code required m16. A u16
stack-param load became a u8 load and the high byte was silently truncated.

Discovered in classickong.r65 (set_rect inlined into full_title_screen).
"""

from r65.tests.e2e import ExpectedState


class TestInlineModePreservation:
    """The inliner must preserve per-block mode info when cloning."""

    def test_u16_stack_param_survives_inlining(self, e2e):
        """A u16 stack param of an inlined function must round-trip the full
        16-bit value. Pre-fix, the high byte was lost: $1580 → $0080.

        The function has nested loops and an array write to force the
        codegen pattern (per-block mode tracking through a loop join)
        that exposes the bug.
        """
        source = '''
            #[lowram(0x300)]
            static mut TBL: [u16; 256];

            far fn set_rect(off: u16, wdt: u8, hgt: u8, tile: u16, swdt: u8) {
                let mut row: u8 = 0;
                let mut current_tile: u16 = tile;
                let mut pos: u16 = off;
                loop {
                    let mut col: u8 = 0;
                    loop {
                        TBL[pos + (col as u16)] = current_tile + (col as u16);
                        col++;
                        if col >= wdt { break; }
                    }
                    pos = pos + 32;
                    current_tile = current_tile + (swdt as u16);
                    row++;
                    if row >= hgt { break; }
                }
            }

            #[entry]
            fn main() {
                set_rect(0, 4, 2, 0x1580, 25);
            }
        '''
        # Pre-fix: TBL[0]=$0080 (high byte lost), TBL[1]=$0081, etc.
        # Post-fix: TBL[0]=$1580, TBL[1]=$1581, etc.
        result = e2e.run(source, ExpectedState(
            memory={
                0x7E0300: [0x80, 0x15],  # TBL[0] = $1580 (LE)
                0x7E0302: [0x81, 0x15],  # TBL[1] = $1581
                0x7E0304: [0x82, 0x15],  # TBL[2] = $1582
                0x7E0306: [0x83, 0x15],  # TBL[3] = $1583
            },
        ), max_instructions=100_000, extra_args=["-O2"])
        assert result.success, f"-O2 inlining corrupted u16 param: {result.failures}"

    def test_explicit_u8_as_u16_inlined(self, e2e):
        """Smoke test: an `as u16` cast at the call site survives inlining.

        Distinguished from the implicit-widening case below — here the cast
        is in source.
        """
        source = '''
            #[lowram(0x300)]
            static mut MAP: [u8; 4096];
            #[lowram(0x1400)]
            static mut player_y: u8;
            #[lowram(0x1401)]
            static mut OUT: u8;

            far fn test_map(x: u8, y: u16) -> u8 {
                let offset: u16 = ((y as u16) << 5) + ((x as u16) >> 3);
                return MAP[offset];
            }

            #[entry]
            fn main() {
                MAP[6721] = 0x7E;
                player_y = 10;
                let x: u8 = 8;
                let y_arg: u8 = player_y + 200;
                OUT = test_map(x, y_arg as u16);
            }
        '''
        result = e2e.run(source, ExpectedState(
            memory={0x7E1401: 0x7E},
        ), max_instructions=100_000, extra_args=["-O2"])
        assert result.success, f"-O2 inline missed explicit u8→u16 cast: {result.failures}"

    def test_u8_arg_to_u16_param_zero_extended_when_inlined(self, e2e):
        """When inlining widens a u8 arg into a u16 param IMPLICITLY (no
        explicit `as u16` at the call site), the inliner must insert a
        TypeConvert so the high byte gets zero-extended.

        Pattern from classickong.r65:
            far fn test_map(x: u8, y: u16) -> u8 { ... y << 5 ... }
            // Caller passes a u8 expression directly to the u16 param —
            // the compiler must widen, and that widening must survive
            // inlining.
            test_map(player_x + 7, player_y + 15);
        """
        source = '''
            #[lowram(0x300)]
            static mut MAP: [u8; 4096];
            #[lowram(0x1400)]
            static mut player_y: u8;
            #[lowram(0x1401)]
            static mut OUT: u8;

            far fn test_map(x: u8, y: u16) -> u8 {
                let offset: u16 = ((y as u16) << 5) + ((x as u16) >> 3);
                return MAP[offset];
            }

            #[entry]
            fn main() {
                // y_arg = player_y + 15 (u8 + u8 = u8, implicitly widened to u16)
                // For player_y=195: y_arg = 210 (no u8 overflow).
                // offset = (210 << 5) + (8 >> 3) = 6720 + 1 = 6721 = $1A41.
                MAP[6721] = 0x7E;
                player_y = 195;
                // No `as u16` here — exactly the classickong pattern.
                OUT = test_map(8, player_y + 15);
            }
        '''
        result = e2e.run(source, ExpectedState(
            memory={0x7E1401: 0x7E},
        ), max_instructions=100_000, extra_args=["-O2"])
        assert result.success, f"-O2 inline missed implicit u8→u16 widening: {result.failures}"

    def test_address_of_rom_data_carries_symbol_through_inlining(self):
        """The inliner's `_remap_vreg` must propagate `.symbol`
        (set by the expression lowerer on address-of-ROM-data vregs)
        from the source vreg to the cloned vreg. Otherwise the
        near→far pointer codegen at type_conversion_select.py
        falls back to bank $00 instead of `:rom_label`.

        Asserted at the MIR level (the runtime symptom in classickong
        was "PRESS START" rendered from a wild far-ptr read; the
        underlying cause is checked here directly).
        """
        from r65.compiler.optimize.inline import BlockCloner
        from r65.compiler.mir.nodes import (
            VirtualRegister, MIRFunction, BasicBlock,
        )
        from r65.compiler.hir.types import BasicTypeInfo
        from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator

        # Build a callee whose entry block uses a vreg carrying .symbol
        # (mimicking what expression.py:909 does for `&ROM_STATIC`).
        u8 = BasicTypeInfo('u8')
        callee_alloc = VirtualRegisterAllocator()
        addr_vreg = callee_alloc.alloc(u8, "addr_of_MSG")

        class FakeSymbol:
            name = "MSG"
            rom_label = "__MSG_data"

        addr_vreg.symbol = FakeSymbol()

        callee = MIRFunction(name="callee", vreg_allocator=callee_alloc)
        callee.blocks[0] = BasicBlock(block_id=0, instructions=[])
        callee.entry_block_id = 0

        caller_alloc = VirtualRegisterAllocator()
        caller = MIRFunction(name="caller", vreg_allocator=caller_alloc)

        cloner = BlockCloner(caller, callee)
        remapped = cloner._remap_vreg(addr_vreg)

        assert remapped is not addr_vreg, "should be a fresh vreg"
        assert hasattr(remapped, 'symbol'), (
            "remapped vreg dropped `.symbol` — far-pointer codegen will fall "
            "back to bank $00. See _remap_vreg in inline.py."
        )
        assert remapped.symbol is addr_vreg.symbol
        assert remapped.symbol.rom_label == "__MSG_data"
