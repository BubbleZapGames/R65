"""Test that X = X + 1 also generates INX (not just X++)"""
from r65.compiler.codegen.memory_alloc import MemoryAllocator
from r65.compiler.codegen.register_alloc import RegisterAllocator, ScratchRegisterPool
from r65.compiler.codegen.instruction_select import InstructionSelector
from r65.compiler.codegen.emitter import AssemblyEmitter
from r65.compiler.mir.nodes import BinaryOp, HardwareRegister, Immediate
from r65.compiler.hir.types import BasicTypeInfo


def test_direct_assignment_x_increment():
    """Test that X = X + 1 (direct assignment) generates INX."""
    emitter = AssemblyEmitter()
    mem_alloc = MemoryAllocator()
    pool = ScratchRegisterPool()
    reg_alloc = RegisterAllocator(scratch_pool=pool)
    selector = InstructionSelector(emitter, reg_alloc, mem_alloc)

    # Create X = X + 1 instruction (direct assignment, not X++)
    x_reg = HardwareRegister('X')
    instr = BinaryOp(
        dest=x_reg,
        left=x_reg,
        op='+',
        right=Immediate(1),
        type_info=BasicTypeInfo('u8')
    )

    selector.select_binary_op(instr)
    asm = emitter.to_string()

    assert 'INX' in asm, f"X = X + 1 should generate INX, got: {asm}"
    assert 'TXA' not in asm, f"X = X + 1 should not use TXA, got: {asm}"
    assert 'ADC' not in asm, f"X = X + 1 should not use ADC, got: {asm}"

    print("✓ X = X + 1 generates INX (direct assignment)")


if __name__ == '__main__':
    test_direct_assignment_x_increment()
    print("\n✅ Direct assignment optimization test passed!")
