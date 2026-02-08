"""
MIR node types: instructions, operands, basic blocks, and CFG structures.

Provides a 3-address-code style IR for code generation.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Any, Union
from enum import Enum


# ============================================================================
# Operand Types
# ============================================================================

@dataclass
class VirtualRegister:
    """
    Virtual register (unlimited during MIR, mapped to scratch/stack during codegen).

    Virtual registers are placeholders that will be allocated to:
    - Scratch registers (designated zero-page locations)
    - Stack locations
    during code generation.
    """
    id: int
    type_info: Any  # TypeInfo from HIR
    hint: Optional[str] = None  # Optional name hint for debugging
    register_hint: Optional[str] = None  # Optional hardware register hint ('X', 'Y') for loop variables

    def __repr__(self):
        if self.hint:
            return f"%{self.id}:{self.hint}"
        return f"%{self.id}"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, VirtualRegister) and self.id == other.id


@dataclass
class HardwareRegister:
    """
    Reference to a 65816 hardware register.

    Hardware registers: A, X, Y, B, STATUS, D, DBR, S
    (PBR is read-only and handled separately)
    (B is only valid in m8 mode)
    """
    name: str  # 'A', 'X', 'Y', 'B', 'STATUS', 'D', 'DBR', 'S'

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, HardwareRegister) and self.name == other.name


@dataclass
class Immediate:
    """Immediate value (constant)."""
    value: int

    def __repr__(self):
        return f"#{self.value}"


@dataclass
class FunctionPointer:
    """
    Function pointer operand.

    Represents a reference to a function's address. Used when loading
    function addresses into registers for indirect calls or function
    pointer assignments.
    """
    function_name: str  # Name of the function being referenced

    def __repr__(self):
        return f"&{self.function_name}"


@dataclass
class MemoryLocation:
    """
    Memory location with storage type and address.

    Used for loads/stores from/to static variables.
    """
    storage_type: str  # 'zeropage', 'ram', 'rom', 'hw'
    address: Optional[int]  # Explicit address or None (auto-allocated)
    symbol: Any  # Symbol from symbol table
    is_volatile: bool = False  # True for #[hw] variables
    index_register: Optional[str] = None  # 'X' or 'Y' for indexed addressing (e.g., LDA $20,X)

    def __repr__(self):
        if self.address is not None:
            indexed = f",{self.index_register}" if self.index_register else ""
            return f"[{self.storage_type}:${self.address:04X}{indexed}]"
        indexed = f",{self.index_register}" if self.index_register else ""
        return f"[{self.storage_type}:{self.symbol.name}{indexed}]"


# ============================================================================
# Base Instruction Class
# ============================================================================

@dataclass
class MIRInstruction:
    """Base class for all MIR instructions."""
    source_loc: Optional[Any] = field(default=None, kw_only=True)  # Source location for debugging

    def __repr__(self):
        return f"{self.__class__.__name__}"


# ============================================================================
# Memory Operations
# ============================================================================

@dataclass
class Load(MIRInstruction):
    """
    Load from memory into register.

    dest = *source
    """
    dest: Union[VirtualRegister, HardwareRegister]
    source: MemoryLocation
    type_info: Any  # TypeInfo for size/sign extension

    def __repr__(self):
        return f"{self.dest} = Load {self.source} : {self.type_info}"


@dataclass
class Store(MIRInstruction):
    """
    Store from virtual register to memory.

    *dest = source
    """
    source: Union[VirtualRegister, HardwareRegister, Immediate]
    dest: MemoryLocation
    type_info: Any  # TypeInfo for size

    def __repr__(self):
        return f"Store {self.source} -> {self.dest} : {self.type_info}"


@dataclass
class LoadIndirect(MIRInstruction):
    """
    Load from memory through pointer (indirect addressing).

    dest = *ptr  or  dest = *(ptr + offset)

    For 65816:
    - near pointers use (zp) or (zp),Y addressing
    - far pointers use [zp] or [zp],Y addressing
    - offset is loaded into Y for (zp),Y or [zp],Y addressing
    """
    dest: VirtualRegister
    pointer: VirtualRegister  # Points to memory location holding the address
    is_far: bool  # True for far pointer (long indirect), False for near
    index_register: Optional[str] = None  # 'Y' for indexed indirect
    type_info: Any = None  # TypeInfo for size/sign extension
    offset: int = 0  # Constant offset for field access (ptr->field)

    def __repr__(self):
        far_str = "[" if self.is_far else "("
        close_str = "]" if self.is_far else ")"
        index_str = f",{self.index_register}" if self.index_register else ""
        offset_str = f"+{self.offset}" if self.offset else ""
        return f"{self.dest} = LoadIndirect {far_str}{self.pointer}{offset_str}{close_str}{index_str} : {self.type_info}"


@dataclass
class StoreIndirect(MIRInstruction):
    """
    Store to memory through pointer (indirect addressing).

    *ptr = source  or  *(ptr + offset) = source

    For 65816:
    - near pointers use (zp) or (zp),Y addressing
    - far pointers use [zp] or [zp],Y addressing
    - offset is loaded into Y for (zp),Y or [zp],Y addressing
    """
    source: Union[VirtualRegister, HardwareRegister, Immediate]
    pointer: VirtualRegister  # Points to memory location holding the address
    is_far: bool  # True for far pointer (long indirect), False for near
    index_register: Optional[str] = None  # 'Y' for indexed indirect
    type_info: Any = None  # TypeInfo for size
    offset: int = 0  # Constant offset for field access (ptr->field)

    def __repr__(self):
        far_str = "[" if self.is_far else "("
        close_str = "]" if self.is_far else ")"
        index_str = f",{self.index_register}" if self.index_register else ""
        return f"StoreIndirect {self.source} -> {far_str}{self.pointer}{close_str}{index_str} : {self.type_info}"


@dataclass
class MemoryFill(MIRInstruction):
    """
    Fill a memory region with a constant value using a loop.

    For array fill expressions like [0; 256].

    Generates:
        LDA #fill_value
        LDX #count
    .loop:
        STA dest,X
        DEX
        BNE .loop
    """
    dest: MemoryLocation  # Base address to fill
    fill_value: int       # Value to fill with (constant)
    count: int            # Number of elements
    element_size: int     # Size of each element (1 or 2 bytes)

    def __repr__(self):
        return f"MemoryFill {self.dest} with #{self.fill_value} x {self.count} ({self.element_size}B each)"


@dataclass
class ROMDataRef:
    """
    Reference to ROM data section for array literals.

    Used by BlockCopy to reference ROM data by label.
    """
    label: str            # Label for the ROM data section
    data: List[int]       # Raw bytes to store in ROM
    element_size: int     # Size of each element (for display)

    def __repr__(self):
        return f"ROMDataRef({self.label}, {len(self.data)} bytes)"


@dataclass
class BlockCopy(MIRInstruction):
    """
    Copy a block of data from ROM to RAM using MVN/MVP.

    For array literal expressions like [1, 2, 3, 4].

    Generates (using MVN for forward copy):
        LDA #count-1          ; A = byte count - 1
        LDX #<source_addr     ; X = source low word
        LDY #<dest_addr       ; Y = dest low word
        MVN #src_bank, #dst_bank

    Note: MVN uses banks as operands, source/dest addresses in X/Y,
    and count-1 in A. Copies A+1 bytes.
    """
    dest: MemoryLocation   # Destination address in RAM
    rom_data: ROMDataRef   # Reference to ROM data section
    count: int             # Number of bytes to copy

    def __repr__(self):
        return f"BlockCopy {self.rom_data.label} -> {self.dest} ({self.count} bytes)"


@dataclass
class Move(MIRInstruction):
    """
    Move between registers or load immediate.

    dest = source

    persist_16bit_mode: If True and dest is A with u16 type, keep A in m16 mode
    after the move (don't emit trailing SEP #$20). Used for `let x @ A : u16 = expr;`
    where the binding should keep A in 16-bit mode for its scope.
    """
    dest: Union[VirtualRegister, HardwareRegister]
    source: Union[VirtualRegister, HardwareRegister, Immediate]
    type_info: Any  # TypeInfo for size
    persist_16bit_mode: bool = False  # Keep m16 mode after 16-bit load to A

    def __repr__(self):
        persist = " [persist_m16]" if self.persist_16bit_mode else ""
        return f"{self.dest} = Move {self.source} : {self.type_info}{persist}"


@dataclass
class TypeConvert(MIRInstruction):
    """
    Type conversion (cast) instruction.

    dest = (target_type)source

    Handles:
    - Widening: u8→u16 (zero-extend), i8→i16 (sign-extend)
    - Narrowing: u16→u8 (truncate)
    - Reinterpret: u8↔i8 (same bits, different interpretation)
    - Boolean: any→bool (0=false, ≠0=true), bool→int (normalize)
    """
    dest: Union[VirtualRegister, HardwareRegister]
    source: Union[VirtualRegister, HardwareRegister, Immediate]
    source_type: Any  # TypeInfo for source
    target_type: Any  # TypeInfo for destination

    def __repr__(self):
        return f"{self.dest} = TypeConvert {self.source} from {self.source_type} to {self.target_type}"


@dataclass
class ToBool(MIRInstruction):
    """
    Convert value to boolean (branchless).

    dest = (source != 0) ? 1 : 0

    Uses branchless CMP #1 / LDA #0 / ADC #0 sequence.
    Result is 0 for false, 1 for true.
    """
    dest: Union[VirtualRegister, HardwareRegister]
    source: Union[VirtualRegister, HardwareRegister, Immediate]
    source_type: Any  # TypeInfo for source (u8/u16)

    def __repr__(self):
        return f"{self.dest} = ToBool {self.source}"


# ============================================================================
# Arithmetic and Logical Operations
# ============================================================================

@dataclass
class BinaryOp(MIRInstruction):
    """
    Binary operation.

    dest = left op right

    Operands can be registers, immediates, or memory locations.
    Using MemoryLocation directly (instead of loading into a vreg first)
    avoids clobbering A when A is already used as the left operand.
    """
    dest: Union[VirtualRegister, HardwareRegister]
    left: Union[VirtualRegister, HardwareRegister, Immediate, MemoryLocation]
    right: Union[VirtualRegister, HardwareRegister, Immediate, MemoryLocation]
    op: str  # '+', '-', '*', '/', '%', '&', '|', '^', '<<', '>>'
    type_info: Any  # TypeInfo determines 8-bit vs 16-bit operation

    def __repr__(self):
        return f"{self.dest} = {self.left} {self.op} {self.right} : {self.type_info}"


@dataclass
class UnaryOp(MIRInstruction):
    """
    Unary operation.

    dest = op operand
    """
    dest: Union[VirtualRegister, HardwareRegister]
    operand: Union[VirtualRegister, HardwareRegister]
    op: str  # '!', '~', '-'
    type_info: Any  # TypeInfo

    def __repr__(self):
        return f"{self.dest} = {self.op}{self.operand} : {self.type_info}"


@dataclass
class Compare(MIRInstruction):
    """
    Comparison operation (sets flags for conditional branch).

    compare left, right
    """
    left: Union[VirtualRegister, HardwareRegister, Immediate]
    right: Union[VirtualRegister, HardwareRegister, Immediate]
    comparison: str  # '==', '!=', '<', '<=', '>', '>='
    type_info: Any  # TypeInfo for comparison size

    def __repr__(self):
        return f"Compare {self.left} {self.comparison} {self.right} : {self.type_info}"


@dataclass
class BitTest(MIRInstruction):
    """
    Bit test operation using BIT instruction.

    Tests specific bits in memory without modifying accumulator.
    Sets flags: N = bit 7, V = bit 6, Z = (A & value) == 0

    Used for optimizing:
    - Boolean flag tests
    - Bit 7 tests (sign bit)
    - Bit 6 tests (overflow bit)
    - Hardware register polling
    """
    value: Union[VirtualRegister, HardwareRegister, MemoryLocation]  # Value to test
    test_bit: int  # Bit number to test (6 or 7, or -1 for Z flag test)
    type_info: Any  # TypeInfo

    def __repr__(self):
        if self.test_bit == -1:
            return f"BitTest {self.value} (Z flag)"
        return f"BitTest {self.value} bit {self.test_bit}"


@dataclass
class Rotate(MIRInstruction):
    """
    Rotate operation (ROL/ROR instruction).

    Rotates bits left or right through carry flag.
    - ROL: shifts left, bit 7 → carry, carry → bit 0
    - ROR: shifts right, bit 0 → carry, carry → bit 7

    count is compile-time constant (1-8).
    """
    dest: Union[VirtualRegister, HardwareRegister]
    source: Union[VirtualRegister, HardwareRegister]
    direction: str  # 'left' or 'right'
    count: int  # Number of rotations (1-8)
    type_info: Any  # TypeInfo

    def __repr__(self):
        return f"{self.dest} = Rotate{self.direction.capitalize()} {self.source} by {self.count}"


# ============================================================================
# Control Flow
# ============================================================================

@dataclass
class Jump(MIRInstruction):
    """
    Unconditional jump to target block.

    jmp target
    """
    target: int  # Block ID

    def __repr__(self):
        return f"Jump -> Block {self.target}"


@dataclass
class CondBranch(MIRInstruction):
    """
    Conditional branch based on condition.

    if condition then true_target else false_target
    """
    condition: Union[VirtualRegister, HardwareRegister]
    true_target: int  # Block ID if condition is true/non-zero
    false_target: int  # Block ID if condition is false/zero
    comparison: str = '!='  # Comparison type ('==', '!=', etc.)

    def __repr__(self):
        return f"CondBranch {self.condition} {self.comparison} 0 ? Block {self.true_target} : Block {self.false_target}"


@dataclass
class StatusFlagTest(MIRInstruction):
    """
    Test a STATUS flag for conditional branching.

    For branchable flags (Carry, Zero, Overflow, Negative), this is a no-op
    since the branch instruction directly tests the flag.

    For non-branchable flags (Irq, Decimal, Index, Accumulator), this emits
    PHP; PLA; AND #mask to prepare for the branch.
    """
    flag_name: str        # "Carry", "Zero", etc.
    bit_position: int     # 0-7
    bit_mask: int         # 0x01, 0x02, etc.

    def __repr__(self):
        return f"StatusFlagTest {self.flag_name}"


@dataclass
class StatusFlagSet(MIRInstruction):
    """
    Set or clear a STATUS flag.

    Generates: SEC/CLC, SEI/CLI, SED/CLD, SEP/REP depending on flag.
    """
    flag_name: str
    value: bool  # True = set flag, False = clear flag

    def __repr__(self):
        action = "Set" if self.value else "Clear"
        return f"StatusFlag{action} {self.flag_name}"


@dataclass
class StatusFlagRead(MIRInstruction):
    """
    Read a STATUS flag into a virtual register as boolean (0/1).

    For reads like: let x = STATUS.Carry;
    Generates: PHP; PLA; AND #mask; (normalize to 0/1)
    """
    dest: VirtualRegister
    flag_name: str
    bit_mask: int

    def __repr__(self):
        return f"{self.dest} = StatusFlagRead {self.flag_name}"


@dataclass
class JumpTable(MIRInstruction):
    """
    Jump table for efficient dense integer pattern matching.

    Computes index = (scrutinee - base_value), bounds checks, then jumps to targets[index].
    Falls through to default_target if out of bounds.
    """
    scrutinee: Union[VirtualRegister, HardwareRegister]  # Value to switch on
    base_value: int  # Minimum value in the range (subtracted from scrutinee)
    targets: List[int]  # Block IDs indexed by (scrutinee - base_value)
    default_target: int  # Block ID for out-of-range or missing entries
    type_info: Any  # TypeInfo for scrutinee

    def __repr__(self):
        return f"JumpTable {self.scrutinee} (base={self.base_value}, size={len(self.targets)}) -> {self.targets}, default -> Block {self.default_target}"


@dataclass
class Return(MIRInstruction):
    """
    Return from function with values.

    return values
    """
    values: List[Union[VirtualRegister, HardwareRegister, MemoryLocation]] = field(default_factory=list)

    def __repr__(self):
        if self.values:
            values_str = ', '.join(str(v) for v in self.values)
            return f"Return {values_str}"
        return "Return"


@dataclass
class ReturnFromInterrupt(MIRInstruction):
    """
    Return from interrupt handler (RTI instruction).

    Used only for interrupt handlers. Restores all CPU state including
    processor status register (mode bits) and program counter.
    """
    def __repr__(self):
        return "RTI"


# ============================================================================
# Function Calls
# ============================================================================

class ArgumentMechanism(Enum):
    """Argument passing mechanism."""
    STACK = "stack"          # Pushed on stack
    REGISTER = "register"    # Passed in hardware register
    VARIABLE = "variable"    # Passed via memory location


@dataclass
class Argument:
    """
    Function call argument with passing mechanism.
    """
    value: Union[VirtualRegister, HardwareRegister, Immediate]
    mechanism: ArgumentMechanism
    location: Optional[Union[HardwareRegister, Any]] = None  # Register or Symbol for VARIABLE
    param_type: Optional[Any] = None  # Parameter type from function signature (for correct stack push size)

    def __repr__(self):
        if self.mechanism == ArgumentMechanism.STACK:
            return f"{self.value}:STACK"
        elif self.mechanism == ArgumentMechanism.REGISTER:
            return f"{self.value}:REG({self.location})"
        else:  # VARIABLE
            return f"{self.value}:VAR({self.location})"


@dataclass
class Call(MIRInstruction):
    """
    Function call.

    returns = call function(args)
    """
    function: Union[str, VirtualRegister]  # Function name or function pointer
    args: List[Argument] = field(default_factory=list)
    returns: List[VirtualRegister] = field(default_factory=list)
    is_far: bool = False  # True for JSL/RTL, False for JSR/RTS
    mode_attr: Optional[Any] = None  # ModeAttribute from callee (for caller-managed DBR)
    bank_attr: Optional[Any] = None  # BankAttribute from callee (for bank number)
    builtin_name: Optional[str] = None  # Set if this is a built-in function call

    # Callee's inferred modes for cross-mode call handling
    callee_entry_m_mode: Optional[Any] = None  # ModeState: callee's expected entry mode
    callee_exit_m_mode: Optional[Any] = None   # ModeState: callee's exit mode (return type)

    # Callee's return type (for B register return detection)
    callee_return_type: Optional[Any] = None  # TypeInfo: callee's return type

    # Callee's preserved registers (for caller-save optimization)
    preserves_attr: Optional[Any] = None  # PreservesAttribute from callee

    def __repr__(self):
        args_str = ', '.join(str(arg) for arg in self.args)
        if self.returns:
            returns_str = ', '.join(str(r) for r in self.returns)
            return f"{returns_str} = Call {self.function}({args_str})"
        return f"Call {self.function}({args_str})"


# ============================================================================
# Special Instructions
# ============================================================================

@dataclass
class SetMode(MIRInstruction):
    """
    Set processor mode (SEP/REP instruction).

    SEP/REP #mask
    """
    mask: int  # Bit mask for SEP/REP
    is_set: bool  # True = SEP (set bits), False = REP (reset bits)

    def __repr__(self):
        op = "SEP" if self.is_set else "REP"
        return f"{op} #${self.mask:02X}"


@dataclass
class Push(MIRInstruction):
    """
    Push register onto stack.

    Used for mode transition wrappers and register preservation.
    Generates PHP (Push Processor Status) for STATUS register,
    or PHA/PHX/PHY for other registers.
    """
    register: HardwareRegister

    def __repr__(self):
        reg_name = self.register.name
        if reg_name == 'STATUS':
            return "PHP  ; Push STATUS"
        elif reg_name == 'A':
            return "PHA  ; Push A"
        elif reg_name == 'X':
            return "PHX  ; Push X"
        elif reg_name == 'Y':
            return "PHY  ; Push Y"
        elif reg_name == 'D':
            return "PHD  ; Push D"
        elif reg_name == 'DBR':
            return "PHB  ; Push DBR"
        else:
            return f"Push {reg_name}"


@dataclass
class Pull(MIRInstruction):
    """
    Pull register from stack.

    Used for mode transition wrappers and register preservation.
    Generates PLP (Pull Processor Status) for STATUS register,
    or PLA/PLX/PLY for other registers.
    """
    register: HardwareRegister

    def __repr__(self):
        reg_name = self.register.name
        if reg_name == 'STATUS':
            return "PLP  ; Pull STATUS"
        elif reg_name == 'A':
            return "PLA  ; Pull A"
        elif reg_name == 'X':
            return "PLX  ; Pull X"
        elif reg_name == 'Y':
            return "PLY  ; Pull Y"
        elif reg_name == 'D':
            return "PLD  ; Pull D"
        elif reg_name == 'DBR':
            return "PLB  ; Pull DBR"
        else:
            return f"Pull {reg_name}"


@dataclass
class SaveRegister(MIRInstruction):
    """
    Save hardware register to virtual register.

    save_location = register (for preservation)
    """
    register: HardwareRegister
    save_location: VirtualRegister

    def __repr__(self):
        return f"{self.save_location} = Save {self.register}"


@dataclass
class RestoreRegister(MIRInstruction):
    """
    Restore hardware register from virtual register.

    register = save_location (for preservation)
    """
    register: HardwareRegister
    save_location: VirtualRegister

    def __repr__(self):
        return f"{self.register} = Restore {self.save_location}"


@dataclass
class InlineAsm(MIRInstruction):
    """
    Inline assembly instruction(s).

    Emits raw assembly instructions verbatim. The compiler assumes all
    registers may be clobbered after inline assembly.

    asm!("NOP", "NOP", ...)
    """
    instructions: List[str]  # List of assembly instruction strings

    def __repr__(self):
        return f"InlineAsm({', '.join(self.instructions)})"


# ============================================================================
# CFG Structures
# ============================================================================

@dataclass
class BasicBlock:
    """
    Basic block in control flow graph.

    A basic block is a sequence of instructions with:
    - Single entry point (first instruction)
    - Single exit point (last instruction)
    - No branches except at the end
    """
    block_id: int
    instructions: List[MIRInstruction] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)  # Block IDs
    successors: List[int] = field(default_factory=list)    # Block IDs

    # Mode tracking (for processor mode analysis)
    entry_mode: Optional[Any] = None  # ProcessorMode
    exit_mode: Optional[Any] = None   # ProcessorMode

    # Live register analysis (for optimization)
    live_in: Set[Union[VirtualRegister, HardwareRegister]] = field(default_factory=set)
    live_out: Set[Union[VirtualRegister, HardwareRegister]] = field(default_factory=set)

    def __repr__(self):
        return f"Block {self.block_id} (preds: {self.predecessors}, succs: {self.successors})"


@dataclass
class MIRFunction:
    """
    MIR representation of a function.

    Contains:
    - Control flow graph (basic blocks)
    - Virtual register allocator
    - Register aliasing tracker
    - Metadata from HIR
    """
    name: str
    parameters: List[Any] = field(default_factory=list)  # HIRParameter list
    return_type: Optional[Any] = None  # TypeInfo
    blocks: Dict[int, BasicBlock] = field(default_factory=dict)  # Block ID → BasicBlock
    entry_block_id: int = 0
    exit_block_ids: List[int] = field(default_factory=list)

    # Metadata from HIR (preserved for code generation)
    mode_attr: Optional[Any] = None         # ModeAttribute (databank only)
    preserves_attr: Optional[Any] = None    # PreservesAttribute
    bank_attr: Optional[Any] = None         # BankAttribute
    interrupt_attr: Optional[Any] = None    # InterruptAttribute
    inline_attr: Optional[Any] = None       # InlineAttribute
    is_entry: bool = False
    is_far: bool = False

    # Inferred entry mode (M8 or M16, always X16)
    # Set based on A parameter type: u16 @ A -> M16, otherwise M8
    entry_m_mode: Optional[Any] = None      # ModeState (M8 or M16)

    # Inferred exit mode (M8 or M16, always X16)
    # Set based on return type: u16/i16 -> M16, otherwise M8
    # Determines mode at function exit (for return value)
    exit_m_mode: Optional[Any] = None       # ModeState (M8 or M16)

    # Far pointer stack parameter tracking
    # True if any stack parameters are far pointers (need D=S prologue)
    has_far_ptr_stack_params: bool = False
    # Set of parameter indices that are far pointers on stack
    far_ptr_param_indices: Set[int] = field(default_factory=set)

    # Source location for debugging (from HIR)
    source_loc: Optional[Any] = None  # SourceLocation

    # Virtual register allocator
    vreg_allocator: Optional[Any] = None  # VirtualRegisterAllocator

    # Register aliasing tracker
    alias_tracker: Optional[Any] = None  # RegisterAliasTracker

    # Stack parameter tracking for prologue generation
    # Maps parameter index to stack offset (from SP after return address)
    stack_param_offsets: Dict[int, int] = field(default_factory=dict)
    # Maps parameter index to allocated virtual register
    param_to_vreg: Dict[int, 'VirtualRegister'] = field(default_factory=dict)

    # Codegen-populated stack usage (for stack depth analysis)
    codegen_frame_size: int = 0           # Local variable frame bytes (from slot allocator)
    codegen_prologue_bytes: int = 0       # Bytes pushed by prologue (preserves, DBR, etc.)

    def __repr__(self):
        return f"MIRFunction({self.name}, {len(self.blocks)} blocks)"


@dataclass
class MIRProgram:
    """
    Complete MIR program.

    Contains:
    - All function MIRs
    - Static declarations (preserved from HIR)
    - Symbol table
    - Global attributes (e.g., stack)
    - ROM data sections for array literal initialization
    """
    functions: List[MIRFunction] = field(default_factory=list)
    statics: List[Any] = field(default_factory=list)    # HIRStaticDecl list
    constants: List[Any] = field(default_factory=list)  # HIRConstDecl list
    structs: List[Any] = field(default_factory=list)    # HIRStructDecl list
    enums: List[Any] = field(default_factory=list)      # HIREnumDecl list
    symbol_table: Optional[Any] = None  # SymbolTable from HIR
    stack_attr: Optional[Any] = None    # StackAttribute from #[stack(...)]
    snesrom_config: Optional[Any] = None  # SnesRomConfig from #[snesrom(...)]
    rom_data_sections: List['ROMDataRef'] = field(default_factory=list)  # Array literal data

    def __repr__(self):
        return f"MIRProgram({len(self.functions)} functions, {len(self.statics)} statics)"
