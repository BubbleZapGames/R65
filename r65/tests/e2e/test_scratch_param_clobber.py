# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end test for scratch param clobbering local variables.

Bug: A local variable in function B can be allocated to a scratch DP address
(e.g. $00) that is also used by callees of B as a scratch parameter. When B
calls a function that takes scratch params, the param setup overwrites the
local's scratch address.

Root cause: analyze_scratch_params() (Default ABI) did not set
_global_scratch_param_addrs on functions, so function_gen.py could not
reserve those addresses. The FixedStack ABI path did this correctly.

Repro: function `outer` has a local `flag` set to a computed value, then
calls `inner` which takes a scratch param at the same DP address. After
the call, `flag` must retain its original value, not the inner param.
"""

from r65.tests.e2e import ExpectedState


class TestScratchParamClobberLocal:
    """Test that local variables are not clobbered by callee scratch params."""

    def test_local_survives_callee_scratch_param(self, e2e):
        """Local variable must not share a scratch address with callee params.

        The callee `helper` takes STACK params (no @ register binding) which
        get promoted to scratch by the scratch param analysis. The caller
        `outer` has a local that must not be allocated to the same scratch
        address.
        """
        result = e2e.run('''
            #[zeropage(0x00, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x01, register)]
            static mut SCRATCH1: u8;
            #[zeropage(0x02, register)]
            static mut SCRATCH2: u8;
            #[zeropage(0x03, register)]
            static mut SCRATCH3: u8;

            #[lowram]
            static mut result_flag: u8;
            #[lowram]
            static mut result_val: u8;

            // Stack params get promoted to scratch by scratch_params.py.
            // The scratch addresses used here must be reserved globally
            // so that callers cannot allocate locals to those addresses.
            fn helper(a: u8, b: u8) -> u8 {
                return a + b;
            }

            // This function has a local `flag` that must survive calls.
            // The slot allocator must NOT place `flag` at a scratch address
            // used by `helper`'s promoted params.
            fn outer(count @ A: u8) {
                let flag: u8 = 42;
                let mut val: u8 = 0;
                let mut i: u8 = 0;

                loop {
                    if i >= count {
                        break;
                    }
                    // helper's stack params get promoted to scratch $00/$01.
                    // If flag is also at $00, it gets clobbered here.
                    val = helper(i, 10);
                    i++;
                }

                result_flag = flag;
                result_val = val;
            }

            #[entry]
            fn main() {
                outer(3);
            }
        ''', ExpectedState(memory={
            # flag must still be 42, not clobbered by helper's scratch params
            0x7E0200: 42,
            # val = helper(2, 10) = 12
            0x7E0201: 12,
        }))
        assert result.success, f"Failures: {result.failures}"

    def test_local_survives_multiple_scratch_callees(self, e2e):
        """Local survives multiple callee scratch params across calls."""
        result = e2e.run('''
            #[zeropage(0x00, register)]
            static mut SCRATCH0: u8;
            #[zeropage(0x01, register)]
            static mut SCRATCH1: u8;
            #[zeropage(0x02, register)]
            static mut SCRATCH2: u8;
            #[zeropage(0x03, register)]
            static mut SCRATCH3: u8;

            #[lowram]
            static mut out_fire: u8;
            #[lowram]
            static mut out_count: u8;

            // Both callees have stack params that get scratch-promoted
            fn check_map(x: u8, y: u8) -> u8 {
                if x + y > 100 {
                    return 1;
                }
                return 0;
            }

            fn draw(tile: u8, px: u8) {
                // Side-effectful (prevents dead code elimination)
                out_count = out_count + 1;
            }

            fn process() {
                let mut fire: u8 = 0;

                // Call 1: check_map scratch params may overlap fire's address
                let floor: u8 = check_map(50, 80);
                if floor != 0 {
                    fire = 1;
                }

                // Call 2: draw scratch params may also overlap fire
                draw(10, 20);

                // fire must be 1 (50+80=130 > 100), not clobbered by draw()
                out_fire = fire;
            }

            #[entry]
            fn main() {
                process();
            }
        ''', ExpectedState(memory={
            # fire = 1 (50+80 > 100)
            0x7E0200: 1,
            # out_count = 1 (draw called once)
            0x7E0201: 1,
        }))
        assert result.success, f"Failures: {result.failures}"
