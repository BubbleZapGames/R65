# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
End-to-end test for the stack-usage compile-time budget check.

The analyzer (step 9/9) computes the worst-case stack high-water mark
from the (acyclic) call graph and refuses to write the ROM when it
exceeds the declared `#[stack(start, end)]` region. This test exercises
both the success path and the overflow diagnostic on a real source
program.
"""

import pytest

from r65.compiler.errors import CodegenError
from r65.compiler.main import compile_string


class TestStackBudget:
    """Static stack-usage check at compile time."""

    # Multiple call sites for each helper defeat the single-call inliner,
    # so each call site really lowers to a JSR and pushes a return address.
    _CALL_CHAIN_SOURCE = '''
        {stack_attr}
        #[lowram] static mut sink: u8;

        fn f2(a: u8, b: u8) -> u8 {{
            return a + b;
        }}

        fn f1(a: u8, b: u8) -> u8 {{
            return f2(a, b) + f2(b, a);
        }}

        #[entry]
        fn main() {{
            sink = f1(1, 2) + f1(3, 4);
        }}
    '''

    def test_overflow_raises_codegen_error(self):
        """A call chain exceeding #[stack(...)] capacity must fail compilation.

        With a 4-byte declared region the chain `main → f1 → f2` cannot
        fit even just the two JSR return addresses, plus any frame each
        function reserves.
        """
        source = self._CALL_CHAIN_SOURCE.format(
            stack_attr='#[stack(0x01FC, 0x01FF)]\nstatic _STACK: u8 = 0;'
        )
        with pytest.raises(CodegenError) as exc:
            compile_string(source, cfg_options=['snes'])
        msg = str(exc.value)
        assert "Stack overflow" in msg
        # Deepest chain should mention the entry function by name.
        assert "main" in msg
        # Hint should suggest widening the region.
        assert "#[stack" in msg

    def test_in_budget_compiles_cleanly(self):
        """The same shape with a generous stack region compiles."""
        source = self._CALL_CHAIN_SOURCE.format(
            stack_attr='#[stack(0x0100, 0x01FF)]\nstatic _STACK: u8 = 0;'
        )
        # Should not raise.
        asm = compile_string(source, cfg_options=['snes'])
        assert asm  # got something
