# Code Generation Phase 6: Function Code Generation

**Status**: ✅ COMPLETE
**Date**: 2026-01-02
**LOC**: ~370 lines

## Overview

Phase 6 implements complete function code generation from MIR (Mid-level Intermediate Representation) to WLA-DX assembly. This phase orchestrates all previous phases (memory allocation, register allocation, instruction selection, addressing mode selection) to generate complete, commented function bodies with headers, labels, and control flow.

## Purpose

The FunctionCodeGenerator is the top-level orchestrator that:

- Generates function headers with metadata and documentation
- Emits function labels and basic block labels
- Orders basic blocks for optimal control flow
- Allocates virtual registers to physical locations
- Generates assembly instructions for each MIR operation
- Handles function entry/exit points

## Architecture

### FunctionCodeGenerator Class

```python
class FunctionCodeGenerator:
    """
    Generates complete assembly functions from MIR.

    Orchestrates function-level code generation including:
    - Function headers with metadata
    - Basic block ordering and labels
    - Instruction selection
    - Register allocation
    """

    def __init__(self, emitter: AssemblyEmitter, memory_allocator: MemoryAllocator):
        self.emitter = emitter
        self.mem_alloc = memory_allocator
```

### Generation Pipeline

```python
def generate_function(mir_func: MIRFunction):
    # 1. Setup register allocator for this function
    scratch_pool = self._create_scratch_pool()
    reg_alloc = RegisterAllocator(scratch_pool=scratch_pool)

    # 2. Allocate all virtual registers in function
    self._allocate_function_registers(mir_func, reg_alloc)

    # 3. Create instruction selector
    instr_selector = InstructionSelector(self.emitter, reg_alloc, self.mem_alloc)

    # 4. Emit function header comment
    self.emit_function_header(mir_func)

    # 5. Emit function label
    self.emitter.emit_label(mir_func.name)

    # 6. Generate basic blocks
    block_order = self._compute_block_order(mir_func)

    for block_id in block_order:
        block = mir_func.blocks[block_id]

        # Emit block label (except entry block)
        if block_id != mir_func.entry_block_id:
            self.emitter.emit_label(f"__L{block_id}")

        # Emit instructions in block
        for instr in block.instructions:
            instr_selector.select_instruction(instr)

    # 7. Blank line after function
    self.emitter.emit_blank_line()
```

## Function Headers

### Header Format

```asm
; ----------------------------------------------------------------------------
; function_name
;
; Parameters:
;   param1: type1
;   param2: type2
;
; Returns: return_type
;
; Mode: m8, x8
; Preserves: X, Y
; Entry: true
; Far: true (JSL/RTL)
; ----------------------------------------------------------------------------
function_name:
    ; ... function body ...
```

### Header Generation

```python
def emit_function_header(self, mir_func: MIRFunction):
    """Emit function header comment with metadata."""
    divider = "-" * 76
    self.emitter.emit_comment(divider)

    # Function name
    self.emitter.emit_comment(f"{mir_func.name}")
    self.emitter.emit_comment("")

    # Parameters
    if mir_func.parameters:
        self.emitter.emit_comment("Parameters:")
        for param in mir_func.parameters:
            param_desc = f"  {param.name}: {param.param_type}"
            self.emitter.emit_comment(param_desc)
        self.emitter.emit_comment("")

    # Return type
    if mir_func.return_type:
        self.emitter.emit_comment(f"Returns: {mir_func.return_type}")
        self.emitter.emit_comment("")

    # Attributes
    if mir_func.mode_attr:
        mode_str = f"Mode: {mir_func.mode_attr}"
        self.emitter.emit_comment(mode_str)

    if mir_func.preserves_attr:
        preserves = ", ".join(mir_func.preserves_attr.registers)
        self.emitter.emit_comment(f"Preserves: {preserves}")

    if mir_func.is_entry:
        self.emitter.emit_comment("Entry: true")

    if mir_func.is_far:
        self.emitter.emit_comment("Far: true (JSL/RTL)")

    # Closing divider
    self.emitter.emit_comment(divider)
```

## Basic Block Ordering

### DFS Traversal

```python
def _compute_block_order(self, mir_func: MIRFunction) -> List[int]:
    """
    Compute optimal ordering of basic blocks.

    Uses DFS traversal from entry block.
    Future optimization: arrange blocks to minimize jumps.
    """
    visited: Set[int] = set()
    order: List[int] = []

    def visit(block_id: int):
        if block_id in visited:
            return

        visited.add(block_id)
        order.append(block_id)

        # Visit successors
        block = mir_func.blocks.get(block_id)
        if block:
            for successor_id in block.successors:
                visit(successor_id)

    # Start from entry block
    visit(mir_func.entry_block_id)

    # Visit any unreachable blocks (shouldn't happen, but be safe)
    for block_id in mir_func.blocks.keys():
        if block_id not in visited:
            visit(block_id)

    return order
```

**Optimization Opportunity**: Future versions could optimize block ordering to maximize fall-through and minimize jump instructions.

## Register Allocation

### Per-Function Allocation

```python
def _allocate_function_registers(self, mir_func: MIRFunction, reg_alloc: RegisterAllocator):
    """Allocate all virtual registers in function."""
    # Collect all virtual registers used in function
    vregs = set()

    for block in mir_func.blocks.values():
        for instr in block.instructions:
            # Extract virtual registers from instruction
            vregs.update(self._extract_vregs_from_instruction(instr))

    # Allocate all at once
    reg_alloc.allocate_all(list(vregs))
```

### Virtual Register Extraction

```python
def _extract_vregs_from_instruction(self, instr) -> Set:
    """
    Extract virtual registers from instruction.

    Simplified implementation using introspection.
    Production version would use visitor pattern.
    """
    from r65.compiler.mir.nodes import VirtualRegister
    vregs = set()

    # Check all attributes of instruction for VirtualRegisters
    for attr_name in dir(instr):
        if attr_name.startswith('_'):
            continue

        attr = getattr(instr, attr_name)

        if isinstance(attr, VirtualRegister):
            vregs.add(attr)
        elif isinstance(attr, list):
            for item in attr:
                if isinstance(item, VirtualRegister):
                    vregs.add(item)

    return vregs
```

## Scratch Register Pool

### Default Pool Configuration

```python
def _create_scratch_pool(self) -> ScratchRegisterPool:
    """
    Create scratch register pool.

    Default configuration (future: configurable via attributes).
    """
    pool = ScratchRegisterPool()

    # Default scratch registers
    pool.add_scratch(0x16, 1, "SCRATCH0")  # 1-byte scratch at $16
    pool.add_scratch(0x17, 1, "SCRATCH1")  # 1-byte scratch at $17
    pool.add_scratch(0x18, 2, "SCRATCH2")  # 2-byte scratch at $18-$19
    pool.add_scratch(0x1A, 2, "SCRATCH3")  # 2-byte scratch at $1A-$1B

    return pool
```

**Note**: Future enhancement will allow functions to specify custom scratch pools via attributes.

## Label Generation

### Function Labels

```python
self.emitter.emit_label(mir_func.name)
```

Generates:
```asm
add:
```

### Basic Block Labels

```python
if block_id != mir_func.entry_block_id:
    self.emitter.emit_label(f"__L{block_id}")
```

Generates:
```asm
__L1:
__L2:
```

**Convention**: Entry block uses function name as label; subsequent blocks use `__L{id}` format.

## Integration with Previous Phases

### Phase Integration

```python
# Phase 3: Register Allocation
reg_alloc = RegisterAllocator(scratch_pool=scratch_pool)
reg_alloc.allocate_all(vregs)

# Phase 4: Instruction Selection
instr_selector = InstructionSelector(self.emitter, reg_alloc, self.mem_alloc)
instr_selector.select_instruction(instr)

# Phase 5: Addressing Mode Selection (used internally by InstructionSelector)
# InstructionSelector uses AddressingModeSelector automatically
```

## Example Generations

### Example 1: Simple Arithmetic Function

**MIR Input**:
```python
# Function: add(a: u8, b: u8) -> u8
#   let sum @ %0 = a + b
#   return sum

entry_block:
    %0 = Move #10 : u8        # a = 10
    %1 = Move #20 : u8        # b = 20
    %2 = BinaryOp %0 + %1 : u8
    Return %2
```

**Generated Assembly**:
```asm
; ----------------------------------------------------------------------------
; add
;
; Returns: u8
;
; ----------------------------------------------------------------------------
add:
    LDA #$0A                    ; Load 10
    STA $16                     ; Store to SCRATCH0 (%0)
    LDA #$14                    ; Load 20
    STA $17                     ; Store to SCRATCH1 (%1)
    LDA $16                     ; Load %0
    CLC
    ADC $17                     ; Add %1
    STA $18                     ; Store result to %2
    RTS

```

**Analysis**:
- Virtual registers %0, %1, %2 allocated to scratch locations $16, $17, $18
- Clean initialization, arithmetic, and return
- Total: 8 instructions

### Example 2: Function with Conditional Branches

**MIR Input**:
```python
# Function: max(a: u8, b: u8) -> u8
#   if a > b:
#       return a
#   else:
#       return b

Block 0 (entry):
    %0 = Move #10 : u8        # a = 10
    %1 = Move #5 : u8         # b = 5
    %2 = Move %0 : u8         # cond = a
    CondBranch %2 != 0 -> Block 1, Block 2

Block 1:
    Return %0

Block 2:
    Return %1
```

**Generated Assembly**:
```asm
; ----------------------------------------------------------------------------
; max
;
; Returns: u8
;
; ----------------------------------------------------------------------------
max:
    LDA #$0A                    ; a = 10
    STA $16
    LDA #$05                    ; b = 5
    STA $17
    LDA $16                     ; cond = a
    STA $18
    LDA $18                     ; Load condition
    BEQ __L2                    ; Branch if zero
    JMP __L1
__L1:
    RTS
__L2:
    RTS

```

**Analysis**:
- Three basic blocks (entry, then-branch, else-branch)
- Block labels emitted for non-entry blocks
- Conditional branch to __L2 if zero, otherwise fall through to __L1
- Each branch has its own return

### Example 3: Function with Metadata

**MIR Input**:
```python
# Entry point function with far calling convention
# #[entry]
# far fn test_function() -> u16

Function(
    name="test_function",
    return_type=u16,
    is_entry=True,
    is_far=True,
)

Block 0:
    Return
```

**Generated Assembly**:
```asm
; ----------------------------------------------------------------------------
; test_function
;
; Returns: u16
;
; Entry: true
; Far: true (JSL/RTL)
; ----------------------------------------------------------------------------
test_function:
    RTS

```

**Analysis**:
- Header shows entry point marker
- Far calling convention documented (JSL/RTL vs JSR/RTS)
- Return type shown in header
- Empty function body with just return

## ProgramFunctionGenerator

### Program-Level Generation

```python
class ProgramFunctionGenerator:
    """Generates all functions in a program."""

    def __init__(self, emitter: AssemblyEmitter, memory_allocator: MemoryAllocator):
        self.emitter = emitter
        self.mem_alloc = memory_allocator
        self.func_gen = FunctionCodeGenerator(emitter, memory_allocator)

    def generate_all_functions(self, mir_program):
        """Generate all functions in MIR program."""
        # Emit section header
        self.emitter.emit_section_header("Functions")

        # Generate each function
        for mir_func in mir_program.functions:
            self.func_gen.generate_function(mir_func)

        # Blank line after all functions
        self.emitter.emit_blank_line()
```

## Test Coverage

### Test File: `test_function_gen.py`

**Test 1: Simple Function Generation**
- Creates MIR function with simple arithmetic
- Verifies function label, LDA/STA/ADC instructions, RTS
- ✅ PASSED

**Test 2: Function with Branches**
- Creates MIR function with conditional control flow
- Verifies function label, block labels (__L1, __L2), BEQ/JMP instructions
- ✅ PASSED

**Test 3: Function Header Generation**
- Creates MIR function with metadata (is_entry, is_far)
- Verifies header contains function name, return type, entry marker, far marker
- ✅ PASSED

**All Tests**: ✅ PASSED

## Known Limitations

1. **No Prologue/Epilogue**: Function prologue and epilogue generation marked as TODO
   - Future: Stack frame setup/teardown
   - Future: Register preservation implementation
   - Future: Mode transition wrappers

2. **Simple Block Ordering**: Uses DFS traversal without optimization
   - Future: Optimize for fall-through to minimize jumps
   - Future: Hot path analysis for better cache locality

3. **Introspection-Based Register Extraction**: Virtual register collection uses introspection
   - Future: Visitor pattern for cleaner MIR traversal
   - Future: Cached register analysis per function

4. **Hardcoded Scratch Pool**: Scratch registers hardcoded in `_create_scratch_pool()`
   - Future: Allow function attributes to specify scratch pool
   - Future: Global configuration for scratch register ranges

5. **No Stack Frame Management**: Functions don't manage stack frames yet
   - Future: Detect when stack frame needed (locals, spills)
   - Future: Generate TSC/TCS for stack operations

## Files Created/Modified

**Created**:
- `r65/compiler/codegen/function_gen.py` (~370 lines)
- `test_function_gen.py` (~430 lines)
- `docs/codegen-phase6-function-generation.md` (this file)

**Modified**:
- `r65/compiler/codegen/__init__.py` (added FunctionCodeGenerator, ProgramFunctionGenerator exports)

**Total**: ~370 LOC for Phase 6 implementation

## Design Decisions

1. **Per-Function Register Allocation**: Each function gets fresh register allocator
   - Enables per-function scratch pool customization
   - Simplifies allocation (no cross-function interference)

2. **Header-First Generation**: Function header emitted before body
   - Provides documentation at function entry
   - Matches hand-written assembly conventions

3. **Entry Block Special Case**: Entry block uses function label, not `__L0`
   - Cleaner assembly output
   - Standard calling convention

4. **Orchestration Pattern**: FunctionCodeGenerator coordinates other phases
   - Single responsibility: function structure
   - Delegates instruction generation to InstructionSelector
   - Delegates register allocation to RegisterAllocator

5. **DFS Block Ordering**: Simple depth-first traversal for now
   - Correct for all CFGs
   - Optimization deferred to future (premature optimization avoided)

## Integration Points

### Input: MIRFunction

```python
from r65.compiler.mir import MIRFunction

mir_func = MIRFunction(
    name="example",
    parameters=[],
    return_type=BasicTypeInfo('u8'),
    blocks={0: entry_block},
    entry_block_id=0,
    exit_block_ids=[0],
    # ... metadata ...
)
```

### Output: Assembly via Emitter

```python
from r65.compiler.codegen import AssemblyEmitter, FunctionCodeGenerator, MemoryAllocator

emitter = AssemblyEmitter()
mem_alloc = MemoryAllocator()
func_gen = FunctionCodeGenerator(emitter, mem_alloc)

func_gen.generate_function(mir_func)

assembly = emitter.to_string()
```

## Next Steps

**Phase 7: Program Assembly Generation**
- Integrate all phases into complete program generation
- Generate assembly sections (defines, constants, functions, vectors)
- Emit WLA-DX directives (.BANK, .ORGA, .SECTION)
- Handle multiple banks and ROM layout
- Generate interrupt vector table

---

**Phase 6 Status**: ✅ COMPLETE
**All Tests**: ✅ PASSING
**Ready for Phase 7**: ✅ YES
