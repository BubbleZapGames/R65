# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Under D = S, a direct page operand is reached with absolute addressing.

The far-pointer D = S strategy points D at the stack, so `LDA $10` would read
the frame. A zeropage static was forced absolute, but a scratch slot was
converted to a memory location below $100 and resolved straight back to
direct page.
"""

import pytest

from r65.compiler.codegen import (
    AssemblyEmitter, InstructionSelector, RegisterAllocator, MemoryAllocator,
)
from r65.compiler.codegen.opcodes import Opcode
from r65.compiler.codegen.register_alloc import PhysicalLocation, LocationKind
from r65.compiler.mir.nodes import FarPtrStrategy


class _DEqualsS:
    has_far_ptr_stack_params = True
    far_ptr_strategy = FarPtrStrategy.D_EQUALS_S


@pytest.mark.parametrize("location", [
    PhysicalLocation(kind=LocationKind.SCRATCH, scratch_addr=0x10, size=1),
    PhysicalLocation(kind=LocationKind.MEMORY, memory_addr=0x10, size=1),
], ids=["scratch", "zeropage"])
def test_direct_page_operand_is_absolute(location):
    selector = InstructionSelector(AssemblyEmitter(), RegisterAllocator(), MemoryAllocator())
    selector.current_function = _DEqualsS()
    opcode, _operand = selector._get_opcode_for_location('LDA', location)
    assert opcode == Opcode.LDA_ABSOLUTE
