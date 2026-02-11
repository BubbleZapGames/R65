#!/usr/bin/env python3
"""Test function pointer implementation."""

from r65.compiler.codegen import ProgramCodeGenerator
from r65.compiler.mir.nodes import (
    MIRProgram,
    MIRFunction,
    BasicBlock,
    Move, Return, Call,
    VirtualRegister,
    Immediate,
    FunctionPointer,
    HardwareRegister,
    Argument,
    ArgumentMechanism,
)
from r65.compiler.hir.types import FunctionTypeInfo, BasicTypeInfo
from r65.compiler.hir.nodes import HIRParameter
from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator


def test_near_function_pointer_assignment():
    """Test assigning a near function to a function pointer variable."""

    vreg_alloc = VirtualRegisterAllocator()

    # Create target function
    target_block = BasicBlock(block_id=0)
    target_block.instructions = [
        Move(
            dest=HardwareRegister('A'),
            source=Immediate(42),
            type_info=BasicTypeInfo('u8')
        ),
        Return(values=[HardwareRegister('A')])
    ]

    target_func = MIRFunction(
        name="target_function",
        parameters=[],
        return_type=BasicTypeInfo('u8'),
        blocks={0: target_block},
        entry_block_id=0,
        is_far=False,  # Near function
        vreg_allocator=vreg_alloc
    )

    # Create main function that stores function address
    main_block = BasicBlock(block_id=0)

    # Create function pointer type
    func_ptr_type = FunctionTypeInfo(
        is_far=False,
        param_types=[],
        return_type=BasicTypeInfo('u8')
    )

    # Virtual register to hold function pointer
    ptr_vreg = vreg_alloc.alloc(func_ptr_type, "handler")

    main_block.instructions = [
        # let handler: fn() -> u8 = target_function;
        Move(
            dest=ptr_vreg,
            source=FunctionPointer(function_name="target_function"),
            type_info=func_ptr_type
        ),
        # Use the function pointer so it's not dead code
        Return(values=[ptr_vreg])
    ]

    main_func = MIRFunction(
        name="main",
        parameters=[],
        return_type=None,
        blocks={0: main_block},
        entry_block_id=0,
        is_far=False,
        vreg_allocator=vreg_alloc
    )

    program = MIRProgram(functions=[main_func, target_func])

    # Generate code
    codegen = ProgramCodeGenerator()
    asm_output = codegen.generate(program)

    # Verify function address loading (near = 2 bytes)
    assert '#<target_function' in asm_output, "Should load low byte of function address"
    assert '#>target_function' in asm_output, "Should load high byte of function address"
    assert '#^target_function' not in asm_output, "Near pointer should not include bank byte"

    print("✓ Near function pointer assignment test passed")


def test_far_function_pointer_assignment():
    """Test assigning a far function to a far function pointer variable."""

    vreg_alloc = VirtualRegisterAllocator()

    # Create target far function
    target_block = BasicBlock(block_id=0)
    target_block.instructions = [
        Return(values=[])
    ]

    from r65.compiler.hir.attributes import BankAttribute
    target_func = MIRFunction(
        name="far_target",
        parameters=[],
        return_type=None,
        blocks={0: target_block},
        entry_block_id=0,
        is_far=True,  # Far function
        mode_attr=None,  # databank=none (default)
        bank_attr=BankAttribute(
            name='bank',
            bank_number=2
        ),
        vreg_allocator=vreg_alloc
    )

    # Create main function
    main_block = BasicBlock(block_id=0)

    # Create far function pointer type
    func_ptr_type = FunctionTypeInfo(
        is_far=True,  # Far pointer
        param_types=[],
        return_type=None
    )

    ptr_vreg = vreg_alloc.alloc(func_ptr_type, "far_handler")

    main_block.instructions = [
        # let far_handler: far fn() = far_target;
        Move(
            dest=ptr_vreg,
            source=FunctionPointer(function_name="far_target"),
            type_info=func_ptr_type
        ),
        # Use the function pointer so it's not dead code
        Return(values=[ptr_vreg])
    ]

    main_func = MIRFunction(
        name="main",
        parameters=[],
        return_type=None,
        blocks={0: main_block},
        entry_block_id=0,
        is_far=False,
        vreg_allocator=vreg_alloc
    )

    program = MIRProgram(functions=[main_func, target_func])

    # Generate code
    codegen = ProgramCodeGenerator()
    asm_output = codegen.generate(program)

    # Verify function address loading (far = 3 bytes)
    assert '#<far_target' in asm_output, "Should load low byte"
    assert '#>far_target' in asm_output, "Should load high byte"
    assert '#:far_target' in asm_output, "Should load bank byte for far pointer"

    print("✓ Far function pointer assignment test passed")


def test_near_indirect_call():
    """Test indirect call through near function pointer."""

    vreg_alloc = VirtualRegisterAllocator()

    # Create callee
    callee_block = BasicBlock(block_id=0)
    callee_block.instructions = [
        Move(
            dest=HardwareRegister('A'),
            source=Immediate(99),
            type_info=BasicTypeInfo('u8')
        ),
        Return(values=[HardwareRegister('A')])
    ]

    callee_func = MIRFunction(
        name="callee",
        parameters=[],
        return_type=BasicTypeInfo('u8'),
        blocks={0: callee_block},
        entry_block_id=0,
        is_far=False,
        vreg_allocator=vreg_alloc
    )

    # Create caller with indirect call
    caller_block = BasicBlock(block_id=0)

    func_ptr_type = FunctionTypeInfo(
        is_far=False,
        param_types=[],
        return_type=BasicTypeInfo('u8')
    )

    ptr_vreg = vreg_alloc.alloc(func_ptr_type, "callback")

    caller_block.instructions = [
        # Load function pointer
        Move(
            dest=ptr_vreg,
            source=FunctionPointer(function_name="callee"),
            type_info=func_ptr_type
        ),
        # Indirect call through function pointer
        Call(
            function=ptr_vreg,  # VirtualRegister, not string!
            args=[],
            returns=[],
            is_far=False,
            bank_attr=None
        ),
        Return(values=[])
    ]

    caller_func = MIRFunction(
        name="caller",
        parameters=[],
        return_type=None,
        blocks={0: caller_block},
        entry_block_id=0,
        is_far=False,
        vreg_allocator=vreg_alloc
    )

    program = MIRProgram(functions=[caller_func, callee_func])

    # Generate code
    codegen = ProgramCodeGenerator()
    asm_output = codegen.generate(program)

    # Verify trampoline generation
    assert 'PHA' in asm_output, "Trampoline should push address bytes"
    assert 'RTS' in asm_output, "Near trampoline should use RTS"

    # Find the caller function in output
    caller_start = asm_output.find('caller:')
    assert caller_start != -1, "caller function should be in output"

    # Verify trampoline sequence (push high, push low, SEC/SBC adjust, RTS)
    caller_section = asm_output[caller_start:caller_start+1000]

    # Should have multiple PHA (2 for near call: high byte, low byte)
    pha_count = caller_section.count('PHA')
    assert pha_count >= 2, f"Near trampoline should push 2 bytes, found {pha_count} PHA"

    # Verify address-1 adjustment (SEC/SBC sequence before RTS)
    assert 'SEC' in caller_section, "Trampoline should have SEC for address-1 adjustment"
    assert 'SBC' in caller_section, "Trampoline should have SBC for address-1 adjustment"

    # SEC should come after the last PHA and before RTS
    last_pha_pos = caller_section.rfind('PHA')
    sec_pos = caller_section.find('SEC')
    rts_pos = caller_section.find('RTS')
    assert last_pha_pos < sec_pos < rts_pos, \
        "SEC should be between last PHA and RTS"

    print("✓ Near indirect call test passed")


def test_far_indirect_call():
    """Test indirect call through far function pointer."""

    vreg_alloc = VirtualRegisterAllocator()

    # Create far callee
    callee_block = BasicBlock(block_id=0)
    callee_block.instructions = [
        Return(values=[])
    ]

    from r65.compiler.hir.attributes import BankAttribute
    callee_func = MIRFunction(
        name="far_callee",
        parameters=[],
        return_type=None,
        blocks={0: callee_block},
        entry_block_id=0,
        is_far=True,
        mode_attr=None,  # databank=none (default)
        bank_attr=BankAttribute(
            name='bank',
            bank_number=3
        ),
        vreg_allocator=vreg_alloc
    )

    # Create caller with far indirect call
    caller_block = BasicBlock(block_id=0)

    func_ptr_type = FunctionTypeInfo(
        is_far=True,  # Far pointer
        param_types=[],
        return_type=None
    )

    ptr_vreg = vreg_alloc.alloc(func_ptr_type, "far_callback")

    caller_block.instructions = [
        # Load far function pointer
        Move(
            dest=ptr_vreg,
            source=FunctionPointer(function_name="far_callee"),
            type_info=func_ptr_type
        ),
        # Far indirect call
        Call(
            function=ptr_vreg,  # VirtualRegister
            args=[],
            returns=[],
            is_far=True,  # Far call
            bank_attr=None
        ),
        Return(values=[])
    ]

    caller_func = MIRFunction(
        name="caller",
        parameters=[],
        return_type=None,
        blocks={0: caller_block},
        entry_block_id=0,
        is_far=False,
        vreg_allocator=vreg_alloc
    )

    program = MIRProgram(functions=[caller_func, callee_func])

    # Generate code
    codegen = ProgramCodeGenerator()
    asm_output = codegen.generate(program)

    # Verify far trampoline generation
    assert 'PHA' in asm_output, "Far trampoline should push address bytes"
    assert 'RTL' in asm_output, "Far trampoline should use RTL"

    # Find caller
    caller_start = asm_output.find('caller:')
    assert caller_start != -1, "caller function should be in output"
    caller_section = asm_output[caller_start:caller_start+1000]

    # Should have 3 PHA for far call (bank, high, low)
    pha_count = caller_section.count('PHA')
    assert pha_count >= 3, f"Far trampoline should push 3 bytes, found {pha_count} PHA"

    # Verify address-1 adjustment (SEC/SBC sequence before RTL)
    assert 'SEC' in caller_section, "Far trampoline should have SEC for address-1 adjustment"
    assert 'SBC' in caller_section, "Far trampoline should have SBC for address-1 adjustment"

    # SEC should come after the last PHA and before RTL
    last_pha_pos = caller_section.rfind('PHA')
    sec_pos = caller_section.find('SEC')
    rtl_pos = caller_section.find('RTL')
    assert last_pha_pos < sec_pos < rtl_pos, \
        "SEC should be between last PHA and RTL"

    # Far trampoline should have 3 SBC instructions (low, high, bank)
    sbc_count = caller_section.count('SBC')
    assert sbc_count >= 3, f"Far trampoline should have 3 SBC instructions, found {sbc_count}"

    print("✓ Far indirect call test passed")


def test_function_pointer_with_arguments():
    """Test function pointer call with stack arguments."""

    vreg_alloc = VirtualRegisterAllocator()

    # Create callee that takes arguments
    callee_block = BasicBlock(block_id=0)
    callee_block.instructions = [
        # Just return
        Return(values=[])
    ]

    # Define stack parameters (binding=None means stack parameter)
    callee_params = [
        HIRParameter(name="a", param_type=BasicTypeInfo('u8'), binding=None),
        HIRParameter(name="b", param_type=BasicTypeInfo('u8'), binding=None),
    ]

    callee_func = MIRFunction(
        name="process",
        parameters=callee_params,  # Stack parameters for callee cleanup
        return_type=None,
        blocks={0: callee_block},
        entry_block_id=0,
        is_far=False,
        vreg_allocator=vreg_alloc
    )

    # Create caller
    caller_block = BasicBlock(block_id=0)

    func_ptr_type = FunctionTypeInfo(
        is_far=False,
        param_types=[BasicTypeInfo('u8'), BasicTypeInfo('u8')],
        return_type=None
    )

    ptr_vreg = vreg_alloc.alloc(func_ptr_type, "handler")

    caller_block.instructions = [
        # Load function pointer
        Move(
            dest=ptr_vreg,
            source=FunctionPointer(function_name="process"),
            type_info=func_ptr_type
        ),
        # Indirect call with arguments
        Call(
            function=ptr_vreg,
            args=[
                Argument(value=Immediate(10), mechanism=ArgumentMechanism.STACK, location=None),
                Argument(value=Immediate(20), mechanism=ArgumentMechanism.STACK, location=None),
            ],
            returns=[],
            is_far=False,
            bank_attr=None
        ),
        Return(values=[])
    ]

    caller_func = MIRFunction(
        name="caller",
        parameters=[],
        return_type=None,
        blocks={0: caller_block},
        entry_block_id=0,
        is_far=False,
        vreg_allocator=vreg_alloc
    )

    program = MIRProgram(functions=[caller_func, callee_func])

    # Generate code
    codegen = ProgramCodeGenerator()
    asm_output = codegen.generate(program)

    # Verify argument pushing
    assert 'LDA #$0A' in asm_output or 'LDA #$14' in asm_output, \
        "Should load argument values"

    # Verify callee cleanup (callee adjusts SP before returning)
    # The callee (process) should have stack adjustment code: PLX, TSC, ADC, TCS, PHX
    assert 'PLX' in asm_output and 'TSC' in asm_output, "Callee should clean up stack arguments"

    # Verify trampoline has SEC/SBC address-1 adjustment
    caller_start = asm_output.find('caller:')
    assert caller_start != -1
    caller_section = asm_output[caller_start:caller_start+2000]
    assert 'SEC' in caller_section, "Trampoline should have SEC for address-1 adjustment"
    assert 'RTS' in caller_section, "Near trampoline should use RTS"

    print("✓ Function pointer with arguments test passed")


def test_state_machine_example():
    """Test state machine pattern with function pointers (from CLAUDE.md)."""

    vreg_alloc = VirtualRegisterAllocator()

    # Create handler functions
    menu_block = BasicBlock(block_id=0)
    menu_block.instructions = [
        Move(dest=HardwareRegister('A'), source=Immediate(0), type_info=BasicTypeInfo('u8')),
        Return(values=[HardwareRegister('A')])
    ]

    menu_func = MIRFunction(
        name="menu_handler",
        parameters=[],
        return_type=BasicTypeInfo('u8'),
        blocks={0: menu_block},
        entry_block_id=0,
        is_far=False,
        vreg_allocator=vreg_alloc
    )

    game_block = BasicBlock(block_id=0)
    game_block.instructions = [
        Move(dest=HardwareRegister('A'), source=Immediate(1), type_info=BasicTypeInfo('u8')),
        Return(values=[HardwareRegister('A')])
    ]

    game_func = MIRFunction(
        name="game_handler",
        parameters=[],
        return_type=BasicTypeInfo('u8'),
        blocks={0: game_block},
        entry_block_id=0,
        is_far=False,
        vreg_allocator=vreg_alloc
    )

    # Create update function that calls through pointer
    update_block = BasicBlock(block_id=0)

    func_ptr_type = FunctionTypeInfo(
        is_far=False,
        param_types=[],
        return_type=BasicTypeInfo('u8')
    )

    handler_vreg = vreg_alloc.alloc(func_ptr_type, "current_handler")
    result_vreg = vreg_alloc.alloc(BasicTypeInfo('u8'), "action")

    update_block.instructions = [
        # Load initial handler (menu_handler)
        Move(
            dest=handler_vreg,
            source=FunctionPointer(function_name="menu_handler"),
            type_info=func_ptr_type
        ),
        # Call through function pointer
        Call(
            function=handler_vreg,
            args=[],
            returns=[result_vreg],
            is_far=False,
            bank_attr=None
        ),
        Return(values=[])
    ]

    update_func = MIRFunction(
        name="update",
        parameters=[],
        return_type=None,
        blocks={0: update_block},
        entry_block_id=0,
        is_far=False,
        vreg_allocator=vreg_alloc
    )

    program = MIRProgram(functions=[update_func, menu_func, game_func])

    # Generate code
    codegen = ProgramCodeGenerator()
    asm_output = codegen.generate(program)

    # Verify all functions are generated
    assert 'menu_handler:' in asm_output, "menu_handler should be generated"
    assert 'game_handler:' in asm_output, "game_handler should be generated"
    assert 'update:' in asm_output, "update should be generated"

    # Verify function pointer operations
    assert '#<menu_handler' in asm_output, "Should load menu_handler address"
    assert 'RTS' in asm_output, "Should have trampoline RTS for indirect call"

    print("✓ State machine example test passed")


def run_all_tests():
    """Run all function pointer tests."""
    print("\n=== Running Function Pointer Tests ===\n")

    test_near_function_pointer_assignment()
    test_far_function_pointer_assignment()
    test_near_indirect_call()
    test_far_indirect_call()
    test_function_pointer_with_arguments()
    test_state_machine_example()

    print("\n=== All Function Pointer Tests Passed ===\n")


if __name__ == '__main__':
    run_all_tests()
