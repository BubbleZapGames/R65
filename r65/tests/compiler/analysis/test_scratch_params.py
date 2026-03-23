# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Tests for Default ABI scratch parameter global reservation.

Bug: analyze_scratch_params() (Default ABI) did not set
_global_scratch_param_addrs on functions, so function_gen.py couldn't
reserve those DP addresses. This allowed the register allocator to place
local variables at the same addresses used by callee scratch params.

The fix mirrors fixedstack_params.py which correctly collects and stores
the global set.
"""

import re
import pytest
from r65.compiler.main import compile_string


class TestScratchParamGlobalReservation:
    """Test that scratch param addresses are globally reserved."""

    SCRATCHES = """
        #[zeropage(0x00, register)]
        static mut SCRATCH0: u8;
        #[zeropage(0x01, register)]
        static mut SCRATCH1: u8;
        #[zeropage(0x02, register)]
        static mut SCRATCH2: u8;
        #[zeropage(0x03, register)]
        static mut SCRATCH3: u8;
    """

    def _find_scratch_param_addrs(self, asm: str) -> set:
        """Extract scratch param addresses from assembly comments."""
        addrs = set()
        for line in asm.split('\n'):
            m = re.search(r'Scratch param \$([0-9A-Fa-f]{2})', line)
            if m:
                addrs.add(int(m.group(1), 16))
        return addrs

    def test_callee_scratch_params_exist(self):
        """Verify that far fn callees get scratch-promoted params."""
        source = self.SCRATCHES + """
        far fn helper(a: u8, b: u8) -> u8 {
            return a + b;
        }

        #[entry]
        fn main() {
            helper(5, 10);
        }
        """
        asm = compile_string(source, "test.r65")
        addrs = self._find_scratch_param_addrs(asm)
        assert len(addrs) > 0, \
            "Expected scratch param promotion for helper's stack params"

    def test_caller_locals_avoid_callee_scratch(self):
        """Assembly must not store non-param locals at callee scratch addresses.

        This is the core regression test. Without _global_scratch_param_addrs,
        the register allocator may place caller locals at DP addresses that
        callees use for scratch params, causing silent clobbering.
        """
        source = self.SCRATCHES + """
        far fn helper(a: u8, b: u8) -> u8 {
            return a + b;
        }

        #[lowram]
        static mut result: u8;

        fn outer(n @ A: u8) {
            let fire: u8 = 99;
            let val: u8 = helper(n, 10);
            result = fire + val;
        }

        #[entry]
        fn main() {
            outer(5);
        }
        """
        asm = compile_string(source, "test.r65")
        helper_scratch = self._find_scratch_param_addrs(asm)
        if not helper_scratch:
            pytest.skip("No scratch params in assembly")

        # In outer's body, any bare STA $XX (without "Scratch param" comment)
        # must NOT target a helper scratch address
        in_outer = False
        for line in asm.split('\n'):
            if line.strip() == 'outer:':
                in_outer = True
                continue
            if in_outer and line.strip() in ('RTS', 'RTL'):
                break
            if in_outer:
                m = re.match(r'\s+STA \$([0-9A-Fa-f]{2})\s*$', line)
                if m:
                    addr = int(m.group(1), 16)
                    if addr in helper_scratch:
                        assert False, (
                            f"Local stored at callee scratch addr ${addr:02X}. "
                            f"Line: {line.strip()}"
                        )
