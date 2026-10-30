# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
MIR node types: instructions, operands, basic blocks, and CFG structures.

Provides a 3-address-code style IR for code generation.
"""

from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Optional, List, Dict, Set, Any, Union, Tuple, NamedTuple, FrozenSet
from enum import Enum
from r65.compiler.hir.status_flags import STATUS_FLAGS


# ============================================================================
# Operand Types
# ============================================================================

@dataclass
class VirtualRegister:
    """
    Virtual register (unlimited during MIR, mapped to scratch/stack during codegen).

    Virtual registers are placeholders allocated in codegen to a hardware
    register (A coalescence, X/Y loop/hint promotion), a scratch zero-page
    slot, or a stack slot.
    """
    id: int
    type_info: Any  # TypeInfo from HIR
    hint: Optional[str] = None  # Optional name hint for debugging
    register_hint: Optional[str] = None  # Optional hardware register hint ('X', 'Y') for loop variables
    # The static whose address this vreg holds (`&ARR[i]`, `&S.field`), so a
    # later `as far *T` can take that static's bank; None otherwise.
    symbol: Any = field(default=None, compare=False, repr=False)

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
    # The `let x @ R` alias this operand reads or defines, or None for a bare
    # register. Not part of equality: a tagged A is still A everywhere else.
    # Read by mir/register_alias_check.py.
    alias: Any = field(default=None, compare=False, repr=False)

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, HardwareRegister) and self.name == other.name


@dataclass
class Immediate:
    """Immediate value (constant), or with `symbol` set, the address of that
    static or function plus `value` bytes (resolved in codegen)."""
    value: int
    symbol: Any = field(default=None, compare=False, repr=False)

    @property
    def symbol_offset(self) -> int:
        """Byte offset from `symbol`'s address: the value itself."""
        return self.value

    def __repr__(self):
        if self.symbol is not None:
            return f"#{self.symbol.name}+{self.value}"
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
class LabelRef:
    """
    Reference to a ROM label address (string literals, etc.).

    Used when loading the address of compiler-generated ROM data
    (e.g., inline string literals) into a pointer variable.
    """
    label_name: str

    def __repr__(self):
        return f"&{self.label_name}"


@dataclass
class MemoryLocation:
    """
    Memory location with storage type and address.

    Used for loads/stores from/to static variables.
    """
    storage_type: str  # 'zeropage', 'ram', 'rom', 'hw'
    address: Optional[int]  # Explicit address or None (auto-allocated)
    symbol: Any  # Symbol from symbol table
    index_register: Optional[str] = None  # 'X' or 'Y' for indexed addressing (e.g., LDA $20,X)
    offset: int = 0  # Byte offset from base (for array/struct element access when address is None)

    @property
    def is_volatile(self) -> bool:
        """A #[hw] variable: every access goes to the hardware."""
        return self.storage_type == 'hw'

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
    # Comment attached to the first asm instruction emitted from this MIR
    # instruction. Used by the inliner to mark inlined regions.
    comment: Optional[str] = field(default=None, kw_only=True)

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
class ModifyMemory(MIRInstruction):
    """
    Read-modify-write of a static or plane element, in place.

    *location = *location ± 1   (op INC / DEC)
    *location = *location << 1  (op ASL)
    *location = *location >> 1  (op LSR, unsigned only)

    Emitted as that instruction on memory, which leaves A untouched and sets
    N/Z from the result. Produced by `optimize/memory_incdec.py` from
    `Load m; m op 1; Store m`; only for locations the instruction can address
    (direct page or absolute, optionally `,X`). When `sets_branch_flags` is
    True the block's `== 0` / `!= 0` test of the new value was folded away
    and the CondBranch that follows reads the Z flag this instruction leaves.
    """
    location: MemoryLocation
    op: str  # 'INC', 'DEC', 'ASL' or 'LSR'
    type_info: Any  # TypeInfo for size (1 or 2 bytes)
    sets_branch_flags: bool = False

    def __repr__(self):
        return f"{self.op} {self.location} : {self.type_info}"


def _indirect_target(instr) -> str:
    """`(ptr+offset),Y` / `[ptr+offset],Y` for a LoadIndirect, StoreIndirect
    or CompareIndirect."""
    open_, close = ("[", "]") if instr.is_far else ("(", ")")
    offset = f"+{instr.offset}" if instr.offset else ""
    index = f",{instr.index_register}" if instr.index_register else ""
    return f"{open_}{instr.pointer}{offset}{close}{index}"


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
        return f"{self.dest} = LoadIndirect {_indirect_target(self)} : {self.type_info}"


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
        return f"StoreIndirect {self.source} -> {_indirect_target(self)} : {self.type_info}"


@dataclass
class MemoryFill(MIRInstruction):
    """
    Fill a memory region with a constant value.

    For array fill expressions like [0; 256].

    Codegen picks the strategy by value and size: MVN self-copy for zero
    fills over 32 bytes, otherwise a STZ or LDA/STA indexed loop (see
    InstructionSelector.select_memory_fill).
    """
    dest: MemoryLocation  # Base address to fill
    fill_value: int       # Value to fill with (constant)
    count: int            # Number of elements
    element_size: int     # Size of each element (1 or 2 bytes)

    def __repr__(self):
        return f"MemoryFill {self.dest} with #{self.fill_value} x {self.count} ({self.element_size}B each)"


@dataclass(frozen=True)
class SymbolByte:
    """
    One byte of another symbol's address, resolved by the assembler.

    Lets a static initializer hold `&OTHER_STATIC` — the address is not known
    until link time, so the byte is emitted as a WLA-DX operator on the target
    label (`<label` low, `>label` high, `:label` bank) instead of a literal.
    """
    label: str            # Target data label (e.g. "__AREA0_TEXT_data")
    part: str             # 'low' | 'high' | 'bank'

    # WLA-DX prefix operators for each byte of a 24-bit address.
    PREFIX = {'low': '<', 'high': '>', 'bank': ':'}

    def render(self) -> str:
        return f"{self.PREFIX[self.part]}{self.label}"


@dataclass
class ROMDataRef:
    """
    Reference to ROM data section for array literals.

    Used by BlockCopy to reference ROM data by label.
    """
    label: str                          # Label for the ROM data section
    data: List[Union[int, SymbolByte]]  # Raw bytes, or link-time symbol bytes
    element_size: int                   # Size of each element (for display)

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
    and count-1 in A. Copies A+1 bytes. Bracketed by PHB/PLB: MVN leaves
    DBR at the destination bank.
    """
    dest: MemoryLocation   # Destination address in RAM
    rom_data: ROMDataRef   # Reference to ROM data section
    count: int             # Number of bytes to copy

    def __repr__(self):
        return f"BlockCopy {self.rom_data.label} -> {self.dest} ({self.count} bytes)"


# At/below this byte count, codegen unrolls an AggregateCopy into A-only LDA/STA
# moves; above it, an MVN block move is used (which clobbers A/X/Y). Shared by the
# codegen selector and the loop-register-promotion clobber analysis so the two
# agree on when a clone is X/Y-safe.
AGGREGATE_COPY_UNROLL_MAX = 4


@dataclass
class AggregateCopy(MIRInstruction):
    """
    Copy a fixed-size block of bytes from one memory location to another (RAM->RAM).

    Emitted for Clone: `dst.clone_from(&src)` and `let c = a.clone()` on auto/array
    aggregates. Both ends are static addresses (clone operands are force-promoted).
    count <= AGGREGATE_COPY_UNROLL_MAX unrolls to A-only LDA/STA; larger uses MVN.

    The MVN form copies one byte at a time, ascending, so a copy whose dest
    is k bytes above an overlapping src repeats its first k bytes through the
    region: a run-time `[v; N]` local is element 0 replicated this way
    (MIRBuilder._emit_runtime_fill). The unrolled form moves words and must
    not overlap.
    """
    dest: MemoryLocation   # destination address
    src: MemoryLocation    # source address
    count: int             # number of bytes to copy

    def __repr__(self):
        return f"AggregateCopy {self.src} -> {self.dest} ({self.count} bytes)"


@dataclass
class Move(MIRInstruction):
    """
    Move between registers or load immediate.

    dest = source
    """
    dest: Union[VirtualRegister, HardwareRegister]
    source: Union[VirtualRegister, HardwareRegister, Immediate]
    type_info: Any  # TypeInfo for size

    def __repr__(self):
        return f"{self.dest} = Move {self.source} : {self.type_info}"


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
    # A zero-extend whose high byte nothing reads (its one reader shifts
    # left by 8 or more): codegen may leave the high byte as it is.
    high_undefined: bool = False

    def __repr__(self):
        return f"{self.dest} = TypeConvert {self.source} from {self.source_type} to {self.target_type}"


@dataclass
class BankByte(MIRInstruction):
    """
    Extract bank byte (byte 2) from a far pointer.

    dest = bank_byte(source)

    Source must be a far pointer (3 bytes). Result is u8.
    """
    dest: Union[VirtualRegister, HardwareRegister]
    source: Union[VirtualRegister, HardwareRegister, Immediate]

    def __repr__(self):
        return f"{self.dest} = BankByte {self.source}"


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
    # A u16 shift by 8-15 whose XBA mask clears only bits that are already 0
    # or that no reader looks at (MIR known-bits): codegen leaves them.
    mask_free: bool = False

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
class CompareIndirect(MIRInstruction):
    """
    Comparison against a byte read through a pointer: `CMP [ptr],Y` /
    `CMP (ptr),Y` / `CMP (d,S),Y`, flags for a conditional branch as Compare
    sets them.

    compare left, ptr[Y]

    What a `LoadIndirect` read only by Compares becomes (optimize/
    indirect_compare.py): the byte is read by the CMP itself, so the load no
    longer puts it in A between `left`'s definition and its use. Y holds the
    index, set by a Move the MIR shows, as for an indexed LoadIndirect.
    """
    left: Union[VirtualRegister, HardwareRegister, Immediate]
    pointer: VirtualRegister  # Points to memory location holding the address
    is_far: bool  # True for far pointer (long indirect), False for near
    comparison: str  # '==', '!=', '<', '<=', '>', '>='
    type_info: Any  # TypeInfo for comparison size
    index_register: Optional[str] = 'Y'
    offset: int = 0

    def __repr__(self):
        return (f"CompareIndirect {self.left} {self.comparison} "
                f"{_indirect_target(self)} : {self.type_info}")


@dataclass
class BitTest(MIRInstruction):
    """
    Bit test: sets N from the value's top bit and V from the one below it.

    `test_bit` is 7 or 6 (tested in m8, so the low byte of a wider value), or
    15 or 14 (tested in m16). A V bit is only tested on a MemoryLocation —
    BIT on memory is the one instruction that loads V from a value. Used for
    sign tests and hardware register polling (`BIT $4212`).
    """
    value: Union[VirtualRegister, HardwareRegister, MemoryLocation]  # Value to test
    test_bit: int  # 7, 6, 15 or 14
    type_info: Any  # TypeInfo

    def __repr__(self):
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

    @property
    def bit_mask(self) -> int:
        return STATUS_FLAGS[self.flag_name].bit_mask

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

    @property
    def bit_mask(self) -> int:
        return STATUS_FLAGS[self.flag_name].bit_mask

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
class LookupTable(MIRInstruction):
    """
    Value lookup table for pure constant match expressions.
    Loads result = table[scrutinee - base_value] via indexed ROM read.
    Block terminator — branches to merge_target after lookup.
    """
    dest: VirtualRegister                           # Result register
    scrutinee: Union[VirtualRegister, HardwareRegister]  # Value to index into table
    base_value: int                                 # Minimum value in the range
    values: List[int]                               # Result values indexed by (scrutinee - base_value)
    default_value: int                              # Result for out-of-bounds
    merge_target: int                               # Block ID to branch to after lookup
    type_info: Any                                  # Result type info

    def __repr__(self):
        return f"LookupTable {self.dest} = table[{self.scrutinee} - {self.base_value}] (size={len(self.values)}, default={self.default_value}) -> Block {self.merge_target}"


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

class FarPtrStrategy(Enum):
    """Strategy for accessing far pointer stack parameters."""
    D_EQUALS_S = "d_equals_s"  # PHD/TSC/TCD: D=S enables [dp],Y indirect long
    SET_DBR = "set_dbr"        # PHB/LDA/PHA/PLB: set DBR to ptr bank, use (d,S),Y


class ChainRole(Enum):
    """Position of a TraitDispatch within a same-self chain.

    Chain coalescing detects runs of TraitDispatch instructions on the same
    far-self vreg and emits the PHB/PLB DBR bracket once around the run
    rather than once per call. ChainRole tells codegen which side of the
    bracket (if any) each dispatch should emit.

    SOLO (default): emit both PHB and PLB.
    """
    SOLO = "solo"      # Standalone dispatch — emit PHB and PLB
    START = "start"    # First in chain — emit PHB + DBR set, no PLB
    MIDDLE = "middle"  # Interior — emit neither PHB nor PLB
    END = "end"        # Last in chain — emit PLB only, no PHB/DBR set


class ArgumentMechanism(Enum):
    """Argument passing mechanism."""
    STACK = "stack"          # Pushed on stack
    REGISTER = "register"    # Passed in hardware register
    VARIABLE = "variable"    # Passed via memory location
    SCRATCH_PARAM = "scratch_param"  # Passed via scratch (zero-page) register
    SELF_Y = "self_y"        # Self pointer passed in Y register, bank in DBR


@dataclass
class Argument:
    """
    Function call argument with passing mechanism.
    """
    value: Union[VirtualRegister, HardwareRegister, Immediate]
    mechanism: ArgumentMechanism
    location: Optional[Union[HardwareRegister, Any]] = None  # Register; MemoryLocation or Symbol for VARIABLE
    param_type: Optional[Any] = None  # Parameter type from function signature (for correct stack push size)
    scratch_addr: Optional[int] = None  # Zero-page address for SCRATCH_PARAM mechanism

    def __repr__(self):
        if self.mechanism == ArgumentMechanism.STACK:
            return f"{self.value}:STACK"
        elif self.mechanism == ArgumentMechanism.REGISTER:
            return f"{self.value}:REG({self.location})"
        elif self.mechanism == ArgumentMechanism.SCRATCH_PARAM:
            return f"{self.value}:SCRATCH(${self.scratch_addr:04X})"
        elif self.mechanism == ArgumentMechanism.SELF_Y:
            return f"{self.value}:SELF_Y"
        else:  # VARIABLE
            return f"{self.value}:VAR({self.location})"


@dataclass(kw_only=True)
class CallBase(MIRInstruction):
    """What every call carries, whether a `Call` or a `TraitDispatch`."""
    args: List[Argument] = field(default_factory=list)
    returns: List[VirtualRegister] = field(default_factory=list)
    is_far: bool = False  # True for JSL/RTL, False for JSR/RTS
    callee_return_type: Optional[Any] = None  # TypeInfo: callee's return type


@dataclass
class Call(CallBase):
    """
    Function call.

    returns = call function(args)
    """
    # Function name, function pointer, or a static holding a far one (bank 0;
    # see optimize/memory_operand.py)
    function: Union[str, VirtualRegister, MemoryLocation]
    mode_attr: Optional[Any] = None  # ModeAttribute from callee (for caller-managed DBR)
    bank_attr: Optional[Any] = None  # BankAttribute from callee (for bank number)
    builtin_name: Optional[str] = None  # Set if this is a built-in function call

    # Callee's inferred modes for cross-mode call handling
    callee_entry_m_mode: Optional[Any] = None  # ModeState: callee's expected entry mode
    callee_exit_m_mode: Optional[Any] = None   # ModeState: callee's exit mode (return type)

    # Callee's preserved registers (for caller-save optimization)
    preserves_attr: Optional[Any] = None  # PreservesAttribute from callee
    clobbers_attr: Optional[Any] = None   # ClobbersAttribute from callee
    callee_is_extern: bool = False        # `extern fn`: #[clobbers] is its whole contract

    # Pascal ABI: result space bytes caller must push (0 for void)
    pascal_result_bytes: int = 0

    def __repr__(self):
        args_str = ', '.join(str(arg) for arg in self.args)
        if self.returns:
            returns_str = ', '.join(str(r) for r in self.returns)
            return f"{returns_str} = Call {self.function}({args_str})"
        return f"Call {self.function}({args_str})"


@dataclass
class TraitDispatch(CallBase):
    """
    Trait method dispatch via jump table.

    At runtime: loads TypeId from self_ptr offset 0, indexes into jump table,
    jumps to concrete method implementation.
    """
    trait_name: str = ""
    method_name: str = ""
    method_index: int = 0
    self_ptr: Union[VirtualRegister, None] = None  # Pointer to trait object
    self_is_far: bool = False  # True if self pointer is far (24-bit)
    # Position of this dispatch within a same-self chain. Set by the
    # analysis pass `analyze_trait_dispatch_chains` (analysis/trait_chain_coalesce.py).
    # SOLO (default): emit both PHB and PLB.
    self_chain_role: 'ChainRole' = field(
        default_factory=lambda: ChainRole.SOLO
    )
    # On a MIDDLE/END member: the id() of the member before it, and the
    # blocks from that member's to this one's, in path order. Y may still
    # hold self from that member's dispatch; codegen skips the Y load when
    # the code emitted in between bears that out (call_select).
    self_y_after: Optional[int] = None
    self_y_path: Tuple[int, ...] = ()

    # The method's modes, as on Call. The wrapper enters every method in m8
    # whatever its caller's width; the exit follows the return type.
    callee_entry_m_mode: Optional[Any] = None
    callee_exit_m_mode: Optional[Any] = None

    def __repr__(self):
        args_str = ', '.join(str(arg) for arg in self.args)
        return f"TraitDispatch {self.trait_name}::{self.method_name}({args_str})"


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
# Operand registry — single source of truth for "which fields of each MIR node
# are register operands". Every operand walker (liveness use/def, loop-promotion
# vreg collect/replace, the inliner's HW-read scan) derives from this table, so
# adding a node or field is a one-line change here instead of an edit to every
# walker. The registry self-test (tests/compiler/mir) asserts coverage of all
# MIRInstruction subclasses so a new node can't silently fall out.
# ============================================================================

_REG_TYPES = (VirtualRegister, HardwareRegister)


class OperandRole(Enum):
    READ = "read"      # operand is consumed (a use)
    WRITE = "write"    # operand is produced (a def)


class OperandContainer(Enum):
    SCALAR = "scalar"      # getattr(instr, field) is one operand
    LIST = "list"          # getattr(instr, field) is a list of operands
    ARG_LIST = "arg_list"  # list of Argument; the operand is arg.value


class OperandSpec(NamedTuple):
    field: str
    role: OperandRole
    container: OperandContainer
    accepts_hr: bool  # True => the slot's type union includes HardwareRegister
    # True => codegen legally takes an Immediate here (`LDA #imm`, `ADC #imm`,
    # ...), so the inliner and the loop unroller may substitute a constant for
    # the register. A pointer slot, a switch scrutinee, a branch condition or a
    # return value is register/DP-typed at codegen, where a constant silently
    # corrupts output.
    accepts_imm: bool = False


_R, _W = OperandRole.READ, OperandRole.WRITE
_SCALAR = OperandContainer.SCALAR


def _is_reg_operand(v, accepts_hr):
    """A value counts as a register operand if it is a VirtualRegister, or a
    HardwareRegister in a slot whose type union admits one (accepts_hr)."""
    return isinstance(v, VirtualRegister) or (accepts_hr and isinstance(v, HardwareRegister))


def _read_vr(f): return OperandSpec(f, _R, _SCALAR, False)
def _read_rr(f): return OperandSpec(f, _R, _SCALAR, True)
def _read_ri(f): return OperandSpec(f, _R, _SCALAR, True, accepts_imm=True)
def _write_vr(f): return OperandSpec(f, _W, _SCALAR, False)
def _write_rr(f): return OperandSpec(f, _W, _SCALAR, True)


# node type -> its register-operand fields. accepts_hr reflects the field's
# declared type union (e.g. Load.source/LoadIndirect.pointer/Call.function/
# TraitDispatch.self_ptr never hold a HardwareRegister). Nodes whose only
# register field is a HardwareRegister flyweight (Push/Pull.register) or that
# have no register operands map to (); they are listed so the coverage self-test
# treats them as intentional, not forgotten.
OPERAND_SPECS: Dict[type, Tuple[OperandSpec, ...]] = {
    Load:           (_write_rr('dest'), _read_vr('source')),
    Store:          (_read_ri('source'),),
    ModifyMemory:   (),  # location is a MemoryLocation; liveness_uses adds its index register, as for Load/Store
    LoadIndirect:   (_write_vr('dest'), _read_vr('pointer')),
    StoreIndirect:  (_read_ri('source'), _read_vr('pointer')),
    Move:           (_write_rr('dest'), _read_ri('source')),
    TypeConvert:    (_write_rr('dest'), _read_ri('source')),
    BankByte:       (_write_rr('dest'), _read_ri('source')),
    ToBool:         (_write_rr('dest'), _read_ri('source')),
    Rotate:         (_write_rr('dest'), _read_ri('source')),
    BinaryOp:       (_write_rr('dest'), _read_ri('left'), _read_ri('right')),
    UnaryOp:        (_write_rr('dest'), _read_ri('operand')),
    Compare:        (_read_ri('left'), _read_ri('right')),
    CompareIndirect: (_read_rr('left'), _read_vr('pointer')),
    BitTest:        (_read_rr('value'),),  # BIT #imm sets only Z, never N or V
    CondBranch:     (_read_rr('condition'),),
    JumpTable:      (_read_rr('scrutinee'),),
    LookupTable:    (_write_vr('dest'), _read_rr('scrutinee')),
    StatusFlagRead: (_write_rr('dest'),),
    Return:         (OperandSpec('values', _R, OperandContainer.LIST, True),),
    Call:           (OperandSpec('returns', _W, OperandContainer.LIST, False),
                     OperandSpec('function', _R, _SCALAR, False),
                     OperandSpec('args', _R, OperandContainer.ARG_LIST, True)),
    TraitDispatch:  (OperandSpec('returns', _W, OperandContainer.LIST, False),
                     OperandSpec('self_ptr', _R, _SCALAR, False),
                     OperandSpec('args', _R, OperandContainer.ARG_LIST, True)),
    # No register operands (constants / block-ids / HR flyweights only):
    Jump:           (),
    StatusFlagTest: (),
    StatusFlagSet:  (),
    SetMode:        (),
    ReturnFromInterrupt: (),
    MemoryFill:     (),
    BlockCopy:      (),
    AggregateCopy:  (),  # dest/src are MemoryLocation, not register operands
    InlineAsm:      (),
    Push:           (),
    Pull:           (),
}


def iter_operands(instr, *, role=None, accepts_hr=None):
    """Yield (spec, value) for each VirtualRegister/HardwareRegister operand of
    instr, optionally filtered by role (READ/WRITE) and accepts_hr.

    Non-register slot values (None, Immediate, MemoryLocation, str) are skipped.
    Unknown node types yield nothing (callers decide whether that's expected).
    """
    specs = OPERAND_SPECS.get(type(instr))
    if specs is None:
        return
    for spec in specs:
        if role is not None and spec.role is not role:
            continue
        if accepts_hr is not None and spec.accepts_hr is not accepts_hr:
            continue
        if spec.container is _SCALAR:
            v = getattr(instr, spec.field)
            if _is_reg_operand(v, spec.accepts_hr):
                yield spec, v
        elif spec.container is OperandContainer.LIST:
            for v in getattr(instr, spec.field):
                if _is_reg_operand(v, spec.accepts_hr):
                    yield spec, v
        else:  # ARG_LIST
            for arg in getattr(instr, spec.field):
                if _is_reg_operand(arg.value, spec.accepts_hr):
                    yield spec, arg.value


def map_operands(instr, fn, *, role=None):
    """In place, apply fn to every register-operand slot of instr, writing the
    result back, optionally only the slots of one role (READ/WRITE). fn
    receives the raw slot value (possibly None/Immediate/str) and must return
    it unchanged for non-register values. Unknown node types are left
    untouched.
    """
    specs = OPERAND_SPECS.get(type(instr))
    if specs is None:
        return
    for spec in specs:
        if role is not None and spec.role is not role:
            continue
        if spec.container is _SCALAR:
            setattr(instr, spec.field, fn(getattr(instr, spec.field)))
        elif spec.container is OperandContainer.LIST:
            setattr(instr, spec.field, [fn(v) for v in getattr(instr, spec.field)])
        else:  # ARG_LIST
            for arg in getattr(instr, spec.field):
                arg.value = fn(arg.value)


# Block-id slots of the control-flow instructions: a scalar target or a list
# of them. The one table every pass that follows or rewrites an edge reads.
BLOCK_TARGET_FIELDS: Dict[type, Tuple[str, ...]] = {
    Jump: ('target',),
    CondBranch: ('true_target', 'false_target'),
    JumpTable: ('targets', 'default_target'),
    LookupTable: ('merge_target',),
}


# Instructions that end a basic block: the edge-carrying ones above plus the
# two returns.
TERMINATOR_TYPES: Tuple[type, ...] = (Return, ReturnFromInterrupt, *BLOCK_TARGET_FIELDS)


def iter_block_targets(instr):
    """Yield every block id *instr* can transfer control to."""
    for name in BLOCK_TARGET_FIELDS.get(type(instr), ()):
        value = getattr(instr, name)
        if isinstance(value, list):
            yield from value
        else:
            yield value


def map_block_targets(instr, fn):
    """In place, replace each block id of *instr* with fn(id). A list slot
    gets a new list, so a shallow copy of the instruction stays independent."""
    for name in BLOCK_TARGET_FIELDS.get(type(instr), ()):
        value = getattr(instr, name)
        if isinstance(value, list):
            setattr(instr, name, [fn(t) for t in value])
        else:
            setattr(instr, name, fn(value))


def iter_memory_operands(instr):
    """Yield every MemoryLocation operand of *instr*, from whichever field holds
    one. Read off the node itself, so no per-type list can fall out of date."""
    for f in dataclass_fields(instr):
        value = getattr(instr, f.name)
        if isinstance(value, MemoryLocation):
            yield value
        elif isinstance(value, list):
            yield from (v for v in value if isinstance(v, MemoryLocation))


def map_memory_operands(instr, fn):
    """In place, replace each MemoryLocation operand of *instr* with fn(loc)."""
    for f in dataclass_fields(instr):
        value = getattr(instr, f.name)
        if isinstance(value, MemoryLocation):
            setattr(instr, f.name, fn(value))
        elif isinstance(value, list) and any(isinstance(v, MemoryLocation) for v in value):
            setattr(instr, f.name, [fn(v) if isinstance(v, MemoryLocation) else v
                                    for v in value])


def count_vreg_reads(func) -> Dict[int, int]:
    """How many times each vreg is read anywhere in *func*, by vreg id.

    Across the whole function rather than per block, so a value also wanted in
    a later block counts as used there.
    """
    counts: Dict[int, int] = {}
    for block in func.blocks.values():
        for instr in block.instructions:
            for _spec, value in iter_operands(instr, role=OperandRole.READ):
                if isinstance(value, VirtualRegister):
                    counts[value.id] = counts.get(value.id, 0) + 1
    return counts


# Nodes whose codegen may overwrite every register: a call (the ABI's
# volatile registers) and inline asm (a black box). What a call actually
# keeps is in mir/call_effects.py; the MIR passes that test this stay
# conservative on purpose and ignore even `#[preserves(...)]`.
CLOBBERS_ALL = (Call, TraitDispatch, InlineAsm)


def fixed_clobbers(instr) -> FrozenSet[str]:
    """The registers *instr*'s fixed expansion overwrites, beyond any `dest`.

    Block moves and fills run MVN or a loop over A, X and Y; a clone up to
    AGGREGATE_COPY_UNROLL_MAX bytes is unrolled LDA/STA and touches only A;
    table dispatch indexes with `TAX`. A pointer access may load Y with its
    offset, which depends on where the pointer lands - each caller judges
    that itself. CLOBBERS_ALL nodes are the caller's to check.
    """
    if isinstance(instr, (BlockCopy, MemoryFill)):
        return _AXY
    if isinstance(instr, AggregateCopy):
        return _AXY if instr.count > AGGREGATE_COPY_UNROLL_MAX else _A
    if isinstance(instr, (JumpTable, LookupTable)):
        return _AX
    return frozenset()


def index_clobbers(instr, pin=None) -> FrozenSet[str]:
    """The index registers *instr*'s codegen overwrites, beyond the
    registers it names as written.

    Beyond `fixed_clobbers`, an indirect access with no index register loads
    Y with its own offset (`LDY #offset` before `(zp),Y` / `[zp],Y`), so a
    value kept in Y does not survive it - unless that value is the access's
    own pointer, which is read as `abs,Y` instead: a `HardwareRegister('Y')`,
    or a vreg *pin* places there. `index_register` marks the other shape,
    `PTR[i]`, where the index the access wants in Y *is* a value the
    allocator places there.
    """
    regs = fixed_clobbers(instr) & _XY
    if (isinstance(instr, (LoadIndirect, StoreIndirect, CompareIndirect))
            and instr.index_register is None
            and register_of(instr.pointer, pin) != 'Y'):
        regs = regs | _Y
    return regs


def index_writes(instr, pin=None, ignore: Optional[int] = None) -> FrozenSet[str]:
    """The index registers running *instr* may change.

    Both for a call or `asm!` (CLOBBERS_ALL; what a call keeps is
    mir/call_effects' to say, for the passes that trust it), its codegen's
    own writes (`index_clobbers`), and each register it writes a value into:
    one it names, or a vreg *pin* places in it (the allocator's placement, a
    loop-promotion hint). A Move from the register into itself writes
    nothing, nor does a def of the vreg *ignore* - the value whose register
    is being asked about.
    """
    if isinstance(instr, CLOBBERS_ALL):
        return _XY
    regs = set(index_clobbers(instr, pin))
    if isinstance(instr, Pull):
        written = [instr.register]
    else:
        written = [v for _, v in iter_operands(instr, role=OperandRole.WRITE)]
    kept = register_of(instr.source, pin) if isinstance(instr, Move) else None
    for v in written:
        if isinstance(v, VirtualRegister) and v.id == ignore:
            continue
        reg = register_of(v, pin)
        if reg in _XY and reg != kept:
            regs.add(reg)
    return frozenset(regs)


def register_of(value, pin=None) -> Optional[str]:
    """The register *value* is in: a `HardwareRegister`'s own, or where *pin*
    (vreg -> register name or None) places a vreg."""
    if isinstance(value, HardwareRegister):
        return value.name
    if isinstance(value, VirtualRegister) and pin is not None:
        return pin(value)
    return None


_A = frozenset('A')
_Y = frozenset('Y')
_XY = frozenset('XY')
_AX = frozenset('AX')
_AXY = frozenset('AXY')


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
    clobbers_attr: Optional[Any] = None     # ClobbersAttribute
    bank_attr: Optional[Any] = None         # BankAttribute
    interrupt_attr: Optional[Any] = None    # InterruptAttribute
    interrupt_modified_regs: Optional[set] = None  # Set of hw reg names modified in handler body
    # Codegen: the emitted register-save run and each RTI's register-restore
    # run, so they can be rebuilt once the body's real register writes are known.
    interrupt_save_run: List[Any] = field(default_factory=list)
    interrupt_restore_runs: List[List[Any]] = field(default_factory=list)
    inline_attr: Optional[Any] = None       # InlineAttribute
    is_entry: bool = False
    is_used: bool = False                   # #[used]: a dead-function-elimination root
    is_far: bool = False
    is_trait_method: bool = False  # True for trait impl methods (self passed in Y)
    is_recursive: bool = False  # True if function is in a recursive cycle

    # Inferred entry mode (M8 or M16, always X16)
    # Set based on A parameter type: u16 @ A -> M16, otherwise M8
    entry_m_mode: Optional[Any] = None      # ModeState (M8 or M16)

    # Inferred exit mode (M8 or M16, always X16)
    # Set based on return type: u16/i16 -> M16, otherwise M8
    # Determines mode at function exit (for return value)
    exit_m_mode: Optional[Any] = None       # ModeState (M8 or M16)

    # Far pointer stack parameter tracking
    # True if any stack parameters are far pointers (need D=S prologue or SET_DBR)
    has_far_ptr_stack_params: bool = False
    # Set of parameter indices that are far pointers on stack
    # (data pointers — used by SET_DBR strategy and the existing far_ptr cost model)
    far_ptr_param_indices: Set[int] = field(default_factory=set)
    # Set of parameter indices that are far function pointers on stack
    # (FunctionTypeInfo with is_far=True). Tracked separately from far_ptr_param_indices
    # so the cost model for D=S strategy can reason about indirect-call savings without
    # conflating fn pointers with data pointers under SET_DBR semantics.
    fn_ptr_param_indices: Set[int] = field(default_factory=set)
    # Strategy for far pointer access (D=S or SET_DBR), set by analysis pass
    far_ptr_strategy: Optional['FarPtrStrategy'] = None

    # Source location for debugging (from HIR)
    source_loc: Optional[Any] = None  # SourceLocation

    # ROM data tables for this function's local array/string/struct literal
    # initializers. Emitted adjacent to the function body (same ROM bank) so
    # the inlined MVN block-copy can reach them; NOT routed through the global
    # __init_start ROM data sections.
    local_rom_data: List['ROMDataRef'] = field(default_factory=list)

    # Virtual register allocator
    vreg_allocator: Optional[Any] = None  # VirtualRegisterAllocator

    # Register aliasing tracker
    alias_tracker: Optional[Any] = None  # RegisterAliasTracker

    # Stack parameter tracking for prologue generation
    # Maps parameter index to stack offset (from SP after return address)
    stack_param_offsets: Dict[int, int] = field(default_factory=dict)
    # Maps parameter index to allocated virtual register
    param_to_vreg: Dict[int, 'VirtualRegister'] = field(default_factory=dict)

    # Self pointer vreg for trait methods (passed in Y register)
    self_y_vreg: Optional['VirtualRegister'] = None

    # Far self D=S mode: True when trait method with far *self needs D=S prologue
    # (methods with calls, ROM/HW access, or other far ptr params).
    # Used under Default/Pascal ABI only; FixedStack uses self_far_uses_scratch instead.
    self_far_uses_d_equals_s: bool = False

    # Far self scratch mode: True when a FixedStack trait method with far *self
    # needs to save self across calls/ROM. Self is copied to a 3-byte zeropage
    # scratch slot in the prologue so DP stays at 0 and scratch params remain
    # reachable. Mutually exclusive with self_far_uses_d_equals_s.
    self_far_uses_scratch: bool = False
    # Zeropage address of the reserved 3-byte scratch slot for self when
    # self_far_uses_scratch is True. Holds (lo, hi, bank).
    self_scratch_addr: Optional[int] = None

    # Scratch parameter tracking (populated by analyze_scratch_params pre-pass)
    # Maps parameter index to scratch zero-page address for promoted stack params
    scratch_param_addrs: Dict[int, int] = field(default_factory=dict)

    # Scratch bytes the function's locals must avoid (set by the param
    # promotion pass): every function's scratch-param bytes, shared program
    # wide since a caller's scratch params persist while its callees run, and
    # the bytes of the trait methods this function dispatches (FixedStack).
    global_scratch_param_bytes: Set[int] = field(default_factory=set)
    trait_dispatch_scratch_bytes: Set[int] = field(default_factory=set)
    # The scratch bytes of the functions that may be on the call stack while
    # this one runs - their scratch params and far-self slots (FixedStack,
    # where those may be live across a call). Never borrowed, not even for
    # one instruction.
    caller_scratch_bytes: Set[int] = field(default_factory=set)

    # HW-promoted parameter registers (populated by fixedstack_params pre-pass)
    # Maps parameter index to hw register name for auto-promoted params
    hw_param_regs: Dict[int, str] = field(default_factory=dict)

    # Loop-promoted hardware register vregs (populated by loop_register_promotion)
    # Maps hw register name ('X', 'Y') to the vreg pre-allocated to that register
    loop_promoted_hw_vregs: Dict[str, 'VirtualRegister'] = field(default_factory=dict)

    # Pascal ABI: total parameter bytes for callee cleanup
    pascal_total_param_bytes: int = 0
    # Pascal ABI: result space bytes (size of return value pushed by caller)
    pascal_result_space_bytes: int = 0

    # Promoted aggregate local tracking
    # True if function has local struct/array variables promoted to static storage
    has_promoted_locals: bool = False

    # ABI info (populated at codegen time from ABIInfo.from_mir_function)
    abi_info: Optional[Any] = None        # ABIInfo

    # Codegen-populated stack usage (for stack depth analysis)
    codegen_frame_size: int = 0           # Local variable frame bytes (from slot allocator)
    codegen_prologue_bytes: int = 0       # Bytes pushed by prologue (preserves, DBR, etc.)
    codegen_max_live_frame_bytes_at_calls: int = 0  # Max live locals at any call site
    codegen_frame_dead_before_calls: bool = False    # True if frame was partially deallocated
    codegen_max_region_spill_bytes: int = 0  # Peak PHA/PHX/PHY region-spill depth across calls

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
    symbol_table: Optional[Any] = None  # SymbolTable from HIR
    stack_attr: Optional[Any] = None    # StackAttribute from #[stack(...)]
    snesrom_config: Optional[Any] = None  # ast.SnesRomDirective from #[snesrom(...)]
    rom_data_sections: List['ROMDataRef'] = field(default_factory=list)  # Array literal data
    trait_dispatch_info: Optional[dict] = None  # Trait dispatch tables from HIR
    asm_includes: List[Any] = field(default_factory=list)  # HIRIncludeAsm list — emitted as .INCLUDE per bank

    def __repr__(self):
        return f"MIRProgram({len(self.functions)} functions, {len(self.statics)} statics)"
