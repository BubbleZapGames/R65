# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end tests for far-self trait dispatch chain coalescing.

Verifies that the `analyze_trait_dispatch_chains` pass correctly detects
runs of TraitDispatch on the same far-self vreg and emits the PHB/PLB
DBR bracket once around the run instead of per-call.

Tests inspect assembly output (counting PHB/PLB inside the chain region)
plus runtime behavior to ensure correctness is preserved.
"""

import re
import pytest

from r65.compiler.main import compile_string
from r65.tests.e2e import ExpectedState


# ---------------------------------------------------------------------------
# Helpers for assembly inspection
# ---------------------------------------------------------------------------

def _count_in_function(asm: str, fn_name: str, mnemonic: str) -> int:
    """Count occurrences of ``mnemonic`` within the assembly between the
    label ``fn_name`` and the next top-level label. Returns 0 if the
    function isn't found.
    """
    lines = asm.split('\n')
    in_fn = False
    count = 0
    fn_label_re = re.compile(rf'^\s*{re.escape(fn_name)}:')
    next_label_re = re.compile(r'^\s*\.?[A-Za-z_][A-Za-z_0-9]*:\s*(;.*)?$')
    for line in lines:
        if not in_fn:
            if fn_label_re.match(line):
                in_fn = True
            continue
        # Stop at the next top-level label (skip local labels starting with ".")
        m = next_label_re.match(line)
        if m and not line.lstrip().startswith('.') and not fn_label_re.match(line):
            break
        # Match the mnemonic as a whole word
        if re.search(rf'\b{re.escape(mnemonic)}\b', line):
            count += 1
    return count


# Common preamble for SNES e2e tests
_PREAMBLE = '''
#[zeropage(0x10, register)]
static mut SCRATCH0: u8;
#[zeropage(0x12, register)]
static mut SCRATCH1: u16;

#[lowram]
static mut RESULT: u8;
'''


def _compile_snes(source: str) -> str:
    """Compile with SNES cfg, returning the .asm string."""
    return compile_string(source, "test.r65", cfg_options=['snes'])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFarSelfChainCoalescing:
    """Verify the chain pass elides redundant PHB/PLB brackets."""

    def test_chain_far_dbr_y_leaf(self):
        """Three back-to-back leaf trait calls on the same far self.

        All impls are leaf (only read self via DBR:Y), so the chain pass
        should fire and emit ONE PHB / ONE PLB around the whole run.
        """
        source = _PREAMBLE + '''
            struct Player { x: u8, y: u8, z: u8 }

            trait HasPos {
                far fn get_x(far *self) -> u8;
                far fn get_y(far *self) -> u8;
            }

            impl HasPos for Player {
                far fn get_x(far *self) -> u8 { return self.x; }
                far fn get_y(far *self) -> u8 { return self.y; }
            }

            #[ram]
            static mut PLAYER: Player = Player { x: 1, y: 2, z: 3 };

            #[entry]
            fn main() {
                let p: far *dyn HasPos = &PLAYER;
                let a: u8 = p.get_x();
                let b: u8 = p.get_x();
                let c: u8 = p.get_x();
                RESULT = a + b + c;
            }
        '''
        asm = _compile_snes(source)
        # Inside main, three same-self chained dispatches.
        # The chain bracket: exactly ONE PHB (chain start) and TWO PLB
        # (one for the DBR-set PHA/PLB at chain start, one for the
        # chain-end PLB). Without coalescing we'd see 3 PHB and 6 PLB.
        phb = _count_in_function(asm, 'main', 'PHB')
        plb = _count_in_function(asm, 'main', 'PLB')
        assert phb == 1, f"Expected 1 PHB in main (chain coalesced), got {phb}\n{asm}"
        assert plb == 2, f"Expected 2 PLB in main (1 set-DBR + 1 chain-end), got {plb}\n{asm}"

    def test_chain_solo_unchanged(self):
        """Single trait dispatch — no chain, full SOLO bracket as before."""
        source = _PREAMBLE + '''
            struct Player { x: u8 }

            trait HasX { far fn get_x(far *self) -> u8; }
            impl HasX for Player {
                far fn get_x(far *self) -> u8 { return self.x; }
            }

            #[ram]
            static mut PLAYER: Player = Player { x: 7 };

            #[entry]
            fn main() {
                let p: far *dyn HasX = &PLAYER;
                RESULT = p.get_x();
            }
        '''
        asm = _compile_snes(source)
        # Solo dispatch: 1 PHB (chain bracket save) + 2 PLB (set-DBR + restore)
        assert _count_in_function(asm, 'main', 'PHB') == 1
        assert _count_in_function(asm, 'main', 'PLB') == 2

    def test_chain_broken_by_ram_access(self):
        """A RAM access between trait calls forces two solo brackets.

        Writing to `RESULT` (RAM at $7E0200) is a DBR-relative absolute
        store; under DBR-set-to-self's-bank it would write to the wrong
        bank. The chain pass must reject this run.
        """
        source = _PREAMBLE + '''
            #[lowram]
            static mut TMP: u8;

            struct Player { x: u8 }

            trait HasX { far fn get_x(far *self) -> u8; }
            impl HasX for Player {
                far fn get_x(far *self) -> u8 { return self.x; }
            }

            #[ram]
            static mut PLAYER: Player = Player { x: 9 };

            #[entry]
            fn main() {
                let p: far *dyn HasX = &PLAYER;
                let a: u8 = p.get_x();
                TMP = 5;
                let b: u8 = p.get_x();
                RESULT = a + b + TMP;
            }
        '''
        asm = _compile_snes(source)
        # Two separate dispatches; each emits its own PHB/PLB.
        # The exact PHB/PLB count includes 2 from the dispatches plus any
        # extra; verify >= 2 of each (two separate brackets, not coalesced).
        phb = _count_in_function(asm, 'main', 'PHB')
        plb = _count_in_function(asm, 'main', 'PLB')
        assert phb >= 2, f"Expected >= 2 PHB (uncoalesced), got {phb}"
        assert plb >= 2, f"Expected >= 2 PLB (uncoalesced), got {plb}"

    def test_chain_broken_by_other_self(self):
        """Three dispatches on different selves never coalesce."""
        source = _PREAMBLE + '''
            struct Player { x: u8 }

            trait HasX { far fn get_x(far *self) -> u8; }
            impl HasX for Player {
                far fn get_x(far *self) -> u8 { return self.x; }
            }

            #[ram]
            static mut OBJ_A: Player = Player { x: 1 };
            #[ram]
            static mut OBJ_B: Player = Player { x: 2 };

            #[entry]
            fn main() {
                let pa: far *dyn HasX = &OBJ_A;
                let pb: far *dyn HasX = &OBJ_B;
                let xa: u8 = pa.get_x();
                let xb: u8 = pb.get_x();
                let xa2: u8 = pa.get_x();
                RESULT = xa + xb + xa2;
            }
        '''
        asm = _compile_snes(source)
        # 3 separate dispatches with mixed selves — each is solo.
        phb = _count_in_function(asm, 'main', 'PHB')
        plb = _count_in_function(asm, 'main', 'PLB')
        assert phb >= 3, f"Expected >= 3 PHB (mixed selves uncoalesced), got {phb}"
        assert plb >= 3, f"Expected >= 3 PLB (mixed selves uncoalesced), got {plb}"

    def test_chain_runtime_correctness(self, e2e):
        """Chain a method that mutates self.x with a method that reads it.

        Runs the resulting code on the emulator and verifies the final
        state. This catches any soundness bug introduced by the chain pass.
        """
        result = e2e.run(_PREAMBLE + '''
            struct Counter { v: u8 }

            trait Tickable {
                far fn tick(far *self);
                far fn read(far *self) -> u8;
            }

            impl Tickable for Counter {
                far fn tick(far *self) {
                    self.v = self.v + 1;
                }
                far fn read(far *self) -> u8 {
                    return self.v;
                }
            }

            #[ram]
            static mut C: Counter = Counter { v: 10 };

            #[entry]
            fn main() {
                let p: far *dyn Tickable = &C;
                p.tick();
                p.tick();
                p.tick();
                RESULT = p.read();
            }
        ''', ExpectedState(
            memory={0x7E0200: 13}
        ))
        assert result.success, f"Failures: {result.failures}"

    def test_chain_broken_by_call(self, e2e):
        """A regular function call between chain members breaks the chain.

        The pass must reject the chain because the called helper may run
        with the caller's DBR assumption — but in this v1 we conservatively
        treat any non-leaf interior call as breaking the chain.
        """
        result = e2e.run(_PREAMBLE + '''
            #[lowram]
            static mut LOG: u8 = 0;

            struct Counter { v: u8 }

            trait Tickable {
                far fn tick(far *self);
                far fn read(far *self) -> u8;
            }

            impl Tickable for Counter {
                far fn tick(far *self) {
                    self.v = self.v + 1;
                }
                far fn read(far *self) -> u8 {
                    return self.v;
                }
            }

            fn note() {
                LOG = LOG + 1;
            }

            #[ram]
            static mut C: Counter = Counter { v: 0 };

            #[entry]
            fn main() {
                let p: far *dyn Tickable = &C;
                p.tick();
                note();
                p.tick();
                RESULT = p.read();
            }
        ''', ExpectedState(
            memory={0x7E0200: 2}
        ))
        assert result.success, f"Failures: {result.failures}"
