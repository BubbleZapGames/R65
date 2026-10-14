# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Each #[interrupt(...)] vector reaches its own handler.

The native vector table once pointed COP at the NMI handler and BRK and
ABORT at the empty one, whatever the program declared: a `BRK` ran
nothing and a `COP` ran the NMI handler.
"""

from r65.tests.e2e import ExpectedState


def test_brk_and_cop_run_their_handlers(e2e):
    result = e2e.run('''
        #[zeropage(0x60)] static mut HITS: [u8; 4];

        #[interrupt(brk)] fn on_brk() { HITS[0] = 0xB0; }
        #[interrupt(cop)] fn on_cop() { HITS[1] = 0xC0; }
        #[interrupt(nmi)] fn on_nmi() { HITS[3] = 0xEE; }

        #[entry]
        fn main() {
            HITS[0] = 0; HITS[1] = 0; HITS[2] = 0; HITS[3] = 0;
            asm!("BRK $00");
            asm!("COP $00");
            HITS[2] = 0x42;     // both returned here
        }
    ''', ExpectedState(memory={0x7E0060: [0xB0, 0xC0, 0x42, 0x00]}))
    assert result.success, f"Failures: {result.failures} {result.error}"
