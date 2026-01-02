# Code Generation Phase 4: Instruction Selection

**Status**: ✅ COMPLETE
**Date**: 2026-01-01
**LOC**: ~680 lines

## Overview

Phase 4 implements instruction selection - the core of code generation. This phase converts MIR (Mid-level Intermediate Representation) instructions into actual 65816 assembly mnemonics using the WLA-DX assembler syntax.

## Purpose

The InstructionSelector bridges the gap between high-level MIR operations and low-level 65816 machine code:

- **Input**: MIR instructions (Load, Store, BinaryOp, etc.)
- **Output**: 65816 assembly mnemonics (LDA, STA, ADC, etc.)
- **Uses**: RegisterAllocator to resolve virtual register locations
- **Uses**: MemoryAllocator to resolve memory addresses

## Architecture

### InstructionSelector Class

```python
class InstructionSelector:
    def __init__(self, emitter, register_allocator, memory_allocator):
        self.emitter = emitter
        self.reg_alloc = register_allocator
        self.mem_alloc = memory_allocator

    def select_instruction(instr: MIRInstruction):
        """Main dispatch - converts MIR instruction to assembly"""
```

**Key Responsibilities**:
1. Pattern matching on MIR instruction types
2. Resolving virtual register locations
3. Generating appropriate 65816 instructions
4. Handling 8-bit vs 16-bit operations
5. Emitting assembly with proper addressing modes

## Instruction Mapping

### 1. Memory Operations

#### Move (Assignment)

**MIR**: `dest = source`

**8-bit Example**:
```
Move: %0 = #42

Generated:
    LDA #$2A        ; Load immediate
    STA $16         ; Store to scratch register
```

**16-bit Example**:
```
Move: %0 = #$1234

Generated:
    LDA #$34        ; Low byte
    STA $16
    LDA #$12        ; High byte
    STA $17
```

#### Load

**MIR**: `dest = *source`

Loads value from memory location into virtual register.

#### Store

**MIR**: `*dest = source`

Stores value from virtual register to memory location.

### 2. Arithmetic Operations

#### Addition

**MIR**: `dest = left + right`

**Example with Immediate**:
```
BinaryOp: %2 = %0 + #5

Generated:
    LDA $16         ; Load left operand
    CLC             ; Clear carry
    ADC #$05        ; Add immediate
    STA $17         ; Store result
```

**16-bit Addition**:
```
BinaryOp: %1 = %0 + #$0100

Generated:
    LDA $18         ; Low byte of left
    CLC
    ADC #$00        ; Low byte of immediate
    STA $16         ; Low byte of result
    LDA $19         ; High byte of left
    ADC #$01        ; High byte of immediate (with carry)
    STA $17         ; High byte of result
```

#### Subtraction

**MIR**: `dest = left - right`

**Example**:
```
BinaryOp: %2 = %0 - #10

Generated:
    LDA $16         ; Load left operand
    SEC             ; Set carry (required for SBC)
    SBC #$0A        ; Subtract immediate
    STA $17         ; Store result
```

#### Bitwise Operations

**AND**: `%2 = %0 & #$0F`
```
    LDA $16
    AND #$0F
    STA $17
```

**OR**: `%2 = %0 | #$80`
```
    LDA $16
    ORA #$80
    STA $17
```

**XOR**: `%2 = %0 ^ #$FF`
```
    LDA $16
    EOR #$FF
    STA $17
```

#### Shift Operations

**Left Shift**: `%2 = %0 << 3`
```
    LDA $16
    ASL A           ; Shift left 1 bit
    ASL A           ; Shift left 1 bit
    ASL A           ; Shift left 1 bit
    STA $17
```

**Right Shift**: `%2 = %0 >> 2`
```
    LDA $16
    LSR A           ; Logical shift right
    LSR A
    STA $17
```

### 3. Control Flow

#### Unconditional Jump

**MIR**: `Jump(target=5)`

**Generated**:
```
    JMP __L5
```

#### Conditional Branch

**MIR**: `CondBranch(cond, true=10, false=20, comparison='!=')`

**Generated**:
```
    LDA $16                 ; Load condition
    BEQ __L20               ; Branch if zero (false branch)
    JMP __L10               ; Jump to true branch
```

**Comparison Types**:
- `!=` - Branch if not equal to zero
- `==` - Branch if equal to zero
- Other comparisons: future enhancement

#### Return

**MIR**: `Return(values=[])`

**Generated**:
```
    RTS                     ; Return from subroutine
```

### 4. Mode Control

#### Set Processor Flags (SEP)

**MIR**: `SetMode(mask=0x30, is_set=True)`

**Generated**:
```
    SEP #$30                ; Set M and X flags (8-bit mode)
```

#### Reset Processor Flags (REP)

**MIR**: `SetMode(mask=0x30, is_set=False)`

**Generated**:
```
    REP #$30                ; Reset M and X flags (16-bit mode)
```

**Common Masks**:
- `0x20` - M flag (accumulator size)
- `0x10` - X flag (index register size)
- `0x30` - Both M and X flags

### 5. Register Save/Restore

#### Save Registers

**MIR**: `SaveRegister(register='A')`

**Generated**:
```
    PHA                     ; Push A
    PHX                     ; Push X
    PHY                     ; Push Y
    PHP                     ; Push processor status
    PHD                     ; Push direct page
    PHB                     ; Push data bank
```

#### Restore Registers

**MIR**: `RestoreRegister(register='A')`

**Generated**:
```
    PLA                     ; Pull A
    PLX                     ; Pull X
    PLY                     ; Pull Y
    PLP                     ; Pull processor status
    PLD                     ; Pull direct page
    PLB                     ; Pull data bank
```

**Note**: Restore order must be reverse of save order!

### 6. Function Calls

#### Near Call (Same Bank)

**MIR**: `Call(function="helper", is_far=False)`

**Generated**:
```
    JSR helper              ; Jump to subroutine (RTS expected)
```

#### Far Call (Cross-Bank)

**MIR**: `Call(function="sound_engine", is_far=True)`

**Generated**:
```
    JSL sound_engine        ; Jump subroutine long (RTL expected)
```

## Helper Methods

### Location Resolution

```python
def _get_operand_location(operand) -> PhysicalLocation:
    """Resolve operand to physical location"""
    if isinstance(operand, VirtualRegister):
        return self.reg_alloc.get_location(operand)
    elif isinstance(operand, HardwareRegister):
        return self.reg_alloc.get_hw_location(operand)
    elif isinstance(operand, MemoryLocation):
        alloc = self.mem_alloc.get_allocation(operand.symbol)
        return PhysicalLocation(memory_addr=alloc.address)
```

### Operand Formatting

```python
def _format_operand(location) -> str:
    """Format location as assembly operand"""
    if location.kind == SCRATCH:
        return f"${location.scratch_addr:02X}"  # $16
    elif location.kind == MEMORY:
        if location.memory_addr < 0x100:
            return f"${location.memory_addr:02X}"  # $20 (zero-page)
        else:
            return f"${location.memory_addr:04X}"  # $2000 (absolute)
```

### Location Offsetting

```python
def _offset_location(location, offset) -> PhysicalLocation:
    """Create new location at byte offset (for 16-bit operations)"""
    # For u16: access both low byte and high byte (offset +1)
    return PhysicalLocation(
        kind=location.kind,
        address=location.address + offset,
        size=1
    )
```

### Type Checking

```python
def _is_16bit(type_info) -> bool:
    """Check if type is 16-bit"""
    return type_info.name in ('u16', 'i16')
```

## Test Coverage

### Test File: `test_instruction_select.py`

**Test 1: Memory Operations**
- Move with immediate: `%0 = #42`
- Move register to register: `%1 = %0`
- Verifies correct LDA/STA generation
- ✅ PASSED

**Test 2: Arithmetic Operations**
- Addition (register + register)
- Addition (register + immediate)
- Subtraction (register - immediate)
- Bitwise AND, OR, XOR
- Shift left and right
- ✅ PASSED

**Test 3: Control Flow**
- Unconditional jump: `JMP __L5`
- Conditional branch: `BEQ/BNE + JMP`
- Return: `RTS`
- ✅ PASSED

**Test 4: Mode Control**
- SEP #$30 (8-bit mode)
- REP #$30 (16-bit mode)
- ✅ PASSED

**Test 5: Register Save/Restore**
- Push A, X, Y
- Pull Y, X, A (reverse order)
- ✅ PASSED

**Test 6: 16-bit Operations**
- 16-bit move: handles low and high bytes
- 16-bit addition: propagates carry
- ✅ PASSED

**All Tests**: ✅ PASSED

## Generated Assembly Examples

### Example 1: Simple Arithmetic

**MIR**:
```
%0 = #10
%1 = #20
%2 = %0 + %1
```

**Generated Assembly**:
```asm
    LDA #$0A
    STA $16             ; %0 = 10

    LDA #$14
    STA $17             ; %1 = 20

    LDA $16
    CLC
    ADC $17
    STA $18             ; %2 = %0 + %1
```

### Example 2: Conditional Logic

**MIR**:
```
%0 = COUNTER
if %0 != 0:
    jump block_10
else:
    jump block_20
```

**Generated Assembly**:
```asm
    LDA COUNTER
    STA $16             ; Load into virtual register

    LDA $16
    BEQ __L20           ; Branch if zero
    JMP __L10           ; Jump to true branch

__L20:
    ; false branch code

__L10:
    ; true branch code
```

### Example 3: 16-bit Arithmetic

**MIR**:
```
%0 = #$1234
%1 = %0 + #$0100
```

**Generated Assembly**:
```asm
    LDA #$34
    STA $16             ; Low byte of %0
    LDA #$12
    STA $17             ; High byte of %0

    LDA $16
    CLC
    ADC #$00            ; Low byte + $00
    STA $18
    LDA $17
    ADC #$01            ; High byte + $01 (with carry)
    STA $19             ; Result: %1 = $1334
```

## Design Decisions

1. **Pattern Matching Dispatch**: Single `select_instruction()` method with isinstance checks. Simple and extensible.

2. **Separate Methods per Instruction**: Each MIR instruction type has dedicated selection method. Clear and maintainable.

3. **Helper Method Decomposition**: Common patterns (add, sub, load) extracted to helpers. Reduces code duplication.

4. **16-bit Byte-by-Byte**: Handle 16-bit operations as separate low/high byte operations. Matches 65816 architecture.

5. **Location Abstraction**: Use PhysicalLocation to abstract scratch vs memory vs stack. Simplifies addressing mode selection.

6. **Immediate Detection**: Check `isinstance(operand, Immediate)` to generate immediate addressing mode.

7. **Carry Flag Management**: Always emit CLC before ADC, SEC before SBC. Prevents carry pollution.

8. **Shift Implementation**: Emit multiple ASL/LSR for constant shifts. Simple but inefficient (future: optimize).

## Known Limitations

1. **Function Call Arguments**: Call instruction implemented but argument passing not yet complete. Requires calling convention implementation.

2. **Stack Operands**: Stack-based virtual registers not yet implemented. Requires TSX/TXS manipulation.

3. **Comparison Operators**: Only `==` and `!=` comparisons implemented. Need `<`, `>`, `<=`, `>=` with proper CMP usage.

4. **Unary Operations**: Logical NOT implemented naively. Could be optimized.

5. **Address Modes**: Currently only immediate and direct/absolute. Missing indexed, indirect modes.

6. **Register Pressure**: No spill code generation yet. Assumes enough scratches.

## Integration Points

### With RegisterAllocator

```python
# Resolve virtual register to physical location
dest_loc = self.reg_alloc.get_location(vreg)

# Format as operand
operand_str = self._format_operand(dest_loc)

# Emit instruction
self.emitter.emit_instruction("STA", operand_str)
```

### With MemoryAllocator

```python
# Resolve memory location
alloc = self.mem_alloc.get_allocation(symbol)

# Create physical location
location = PhysicalLocation(
    kind=LocationKind.MEMORY,
    memory_addr=alloc.address
)
```

### With AssemblyEmitter

```python
# Emit single instruction
self.emitter.emit_instruction("LDA", "#$42", "Load value")

# Emit label
self.emitter.emit_label("__L5")

# Emit blank line
self.emitter.emit_blank_line()
```

## Files Created/Modified

**Created**:
- `r65/compiler/codegen/instruction_select.py` (~680 lines)
- `test_instruction_select.py` (~460 lines)

**Modified**:
- `r65/compiler/codegen/__init__.py` (added InstructionSelector export)

**Total**: ~680 LOC for Phase 4 implementation

## Next Steps

**Phase 5 will implement Addressing Mode Selection**:
- Choose optimal addressing modes (immediate, zero-page, absolute, indexed)
- Optimize instruction sequences
- Handle special cases (zero-page vs absolute)

**Phase 6 will implement Function Code Generation**:
- Process MIR functions and basic blocks
- Generate function prologues/epilogues
- Emit block labels
- Call InstructionSelector for each instruction

**Phase 7 will implement Program Assembly Generation**:
- Orchestrate all phases together
- Generate complete assembly files
- Add metadata and exports

---

**Phase 4 Status**: ✅ COMPLETE
**All Tests**: ✅ PASSING
**Ready for Phase 5**: ✅ YES
