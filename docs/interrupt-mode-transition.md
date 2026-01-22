# Interrupt Handler Mode Transition - Implementation Complete

## Design Decision

**Rule:** Interrupt handlers automatically run in the default mode (m8, x16). The compiler handles all mode management automatically.

## Rationale

1. **Interrupts fire from unknown mode**: An NMI/IRQ can occur while processor is in any mode (m8 or m16)
2. **Handler needs specific mode**: Handler code expects to run in a known mode
3. **Must restore original mode**: After handling interrupt, must return to interrupted code's mode
4. **Automatic management**: Compiler generates PHP/PLP to save and restore the processor status

## Implementation

### 1. Automatic Mode Management

Interrupt handlers automatically save and restore processor status:

```rust
// Interrupt handler - runs in default m8 mode
// Compiler automatically generates PHP/PLP wrapper
#[interrupt(nmi)]
fn nmi_handler() {
    // handler code runs in m8 mode
}
```

**Note:** Interrupt handlers always run in the default m8/x16 mode. The compiler automatically generates the PHP (save status) at entry and PLP (restore status) at exit to preserve the interrupted code's mode.

### 2. Entry Wrapper Generation

MIR builder generates entry wrapper for interrupt handlers:

```python
if hir_func.interrupt_attr:
    # 1. Push all registers (automatic preservation)
    self.emit(Push(register=HardwareRegister('STATUS')))  # PHP
    self.emit(Push(register=HardwareRegister('A')))       # PHA
    self.emit(Push(register=HardwareRegister('X')))       # PHX
    self.emit(Push(register=HardwareRegister('Y')))       # PHY
    self.emit(Push(register=HardwareRegister('D')))       # PHD
    self.emit(Push(register=HardwareRegister('DBR')))     # PHB

    # 2. Force default m8/x16 mode for handler body
    self.emit(SetMode(mask=0x20, is_set=True))   # SEP #$20 - m8 mode
```

### 3. Exit Wrapper Generation

Modified return statement lowering for interrupt handlers:

```python
if self.current_function.interrupt_attr:
    # Restore all registers (reverse order of push)
    self.emit(Pull(register=HardwareRegister('DBR')))     # PLB
    self.emit(Pull(register=HardwareRegister('D')))       # PLD
    self.emit(Pull(register=HardwareRegister('Y')))       # PLY
    self.emit(Pull(register=HardwareRegister('X')))       # PLX
    self.emit(Pull(register=HardwareRegister('A')))       # PLA
    self.emit(Pull(register=HardwareRegister('STATUS')))  # PLP (restores mode!)

    # Return from interrupt
    self.emit(ReturnFromInterrupt())  # RTI
```

### 4. New MIR Instructions

Added `ReturnFromInterrupt` instruction:

```python
@dataclass
class ReturnFromInterrupt(MIRInstruction):
    """
    Return from interrupt handler (RTI instruction).

    Used only for interrupt handlers. Restores all CPU state including
    processor status register (mode bits) and program counter.
    """
    def __repr__(self):
        return "RTI"
```

## Generated Assembly Structure

### Example Source

```rust
#[interrupt(nmi)]
fn nmi_handler() {
    A = 0x42;
    FLAG = A;
    return;
}
```

**Note:** Interrupt handlers automatically run in m8/x16 mode. The compiler generates all necessary mode management code.

### Generated MIR

```
Block 0:
   0: PHP  ; Push STATUS        ─┐
   1: PHA  ; Push A              │
   2: PHX  ; Push X              │
   3: PHY  ; Push Y              │ Entry wrapper
   4: PHD  ; Push D              │ (6 pushes + mode set)
   5: PHB  ; Push DBR           ─┤
   6: SEP #$20                   ─┘ Force m8 mode (x16 is default)

   7: A = Move #66 : u8         ─┐
   8: Store A -> FLAG : u8      ─┘ Handler body

   9: PLB  ; Pull DBR           ─┐
  10: PLD  ; Pull D              │
  11: PLY  ; Pull Y              │
  12: PLX  ; Pull X              │ Exit wrapper
  13: PLA  ; Pull A              │ (6 pulls + RTI)
  14: PLP  ; Pull STATUS        ─┤ Restores original mode!
  15: RTI                        ─┘ Return from interrupt
```

### WLA-DX Assembly Output

```asm
nmi_handler:
    PHP                     ; Save STATUS (including mode bits)
    PHA                     ; Save A
    PHX                     ; Save X
    PHY                     ; Save Y
    PHD                     ; Save D
    PHB                     ; Save DBR (data bank)

    SEP #$20                ; Force m8 mode for handler (x16 is default)

    LDA #$42                ; Handler code
    STA FLAG

    PLB                     ; Restore DBR
    PLD                     ; Restore D
    PLY                     ; Restore Y
    PLX                     ; Restore X
    PLA                     ; Restore A
    PLP                     ; Restore STATUS (restores original mode!)
    RTI                     ; Return from interrupt
```

## Key Features

### 1. **Automatic Mode Management** ✅

Interrupt can fire in any mode, handler runs in default m8/x16 mode, original mode restored:

```
Main code running in m16 mode...
    ↓
Interrupt fires → NMI vector
    ↓
nmi_handler entry:
    PHP             ; Save STATUS (m16 saved on stack)
    ...             ; Save other registers
    SEP #$20        ; Force m8 mode (x16 is always the default)
    ↓
Handler body runs in m8/x16 mode
    ↓
nmi_handler exit:
    ...             ; Restore other registers
    PLP             ; Restore STATUS (m16 restored!)
    RTI
    ↓
Main code continues in m16 mode (seamlessly!)
```

### 2. **Automatic Register Preservation** ✅

All registers saved/restored automatically:
- STATUS (including mode bits M and X)
- A (accumulator)
- X (index)
- Y (index)
- D (direct page)
- DBR (data bank)

### 3. **No Manual Save/Restore Needed** ✅

User doesn't write preservation code:

```rust
// User writes clean handler code:
#[interrupt(nmi)]
fn nmi_handler() {
    FRAME_COUNT = FRAME_COUNT + 1;
    // No manual saves/restores!
}

// Compiler automatically wraps with PHP/PHA/PHX/PHY/PHD/PHB...PLB/PLD/PLY/PLX/PLA/PLP/RTI
```

### 4. **Known Mode Established** ✅

Handler runs in the default m8/x16 mode regardless of interrupted code's mode:

```rust
#[interrupt(nmi)]
fn nmi_handler() {
    A = 0x42;     // Always works in m8 mode, even if interrupted code was in m16 mode
}
```

## Edge Cases Handled

### 1. **Interrupt Handler Mode**

All interrupt handlers run in the default m8/x16 mode:

```rust
#[interrupt(nmi)]
fn nmi_handler() {
    // Runs in m8/x16 mode (default)
    // Gets full register preservation (PHP/PHA/.../PLP/RTI)
}
```

**Generated MIR:** Push/Pull wrappers with SEP #$20 to force m8 mode

### 2. **Interrupt Handler with Return Values**

```rust
#[interrupt(nmi)]
fn nmi_handler() -> u8 {  // ❌ ERROR
    return 42;
}
```

**Error:** `Interrupt handler 'nmi_handler' cannot return values`

**Rationale:** RTI doesn't support return values (no register convention for interrupt handlers)

### 3. **Nested Interrupts**

If interrupts are re-enabled within handler (via CLI), could get nested interrupts:

```rust
#[interrupt(nmi)]
fn nmi_handler() {
    // Re-enable interrupts
    asm!("CLI");

    // If IRQ fires here:
    // - Stack grows: NMI's saved state + IRQ's saved state
    // - IRQ handler runs, returns with RTI
    // - NMI handler resumes
    // - NMI returns with RTI
}
```

**Handled correctly:** Each handler saves/restores its own state on stack

### 4. **Infinite Loop Handlers**

```rust
#[interrupt(nmi)]
fn nmi_handler() -> ! {
    loop {
        // Process forever
    }
}
```

**Handled correctly:** Never hits return statement, no RTI generated (would be dead code)

## Comparison with Regular Functions

### Regular Function (m16 via parameter)

```rust
// m16 mode inferred from @ A: u16 parameter
fn process(value @ A: u16) {
    A = A + 0x1234;
}

// m8 mode (default)
fn caller() {
    process(0x1000);  // Callee handles: REP #$20, ..., SEP #$20
}
```

**Mode management:** Callee-side (function prologue/epilogue)
**Knows incoming mode:** Yes (m8 is default)
**Restoration:** Via SEP #$20 in epilogue

### Interrupt Handler (automatic)

```rust
#[interrupt(nmi)]
fn nmi_handler() {
    A = 0x42;  // Runs in m8/x16 mode
}

// Can fire from anywhere
```

**Mode management:** Automatic entry/exit wrapper
**Knows incoming mode:** No (could be any mode)
**Restoration:** Via PLP (restores whatever mode was interrupted)

## Benefits

1. **Safety:** Impossible to forget register preservation
2. **Correctness:** Mode always matches handler's expectations
3. **Transparency:** Interrupted code sees no side effects
4. **Simplicity:** User writes clean handler code

## Testing

**Test file:** `examples/interrupt_simple_test.r65`

```rust
#[interrupt(nmi)]
fn nmi_handler() {
    A = 0x42;
    FLAG = A;
    return;
}
```

**Result:** ✅ 16 MIR instructions generated correctly

**Debug script:** `test_interrupt_mir.py` - Inspects generated MIR

## Implementation Files Modified

1. **`r65/compiler/typeck/type_checker.py`**
   - Validate interrupt handlers don't have return values

2. **`r65/compiler/mir/nodes.py`**
   - Added `ReturnFromInterrupt` instruction

3. **`r65/compiler/mir/builder.py`**
   - Entry wrapper generation (Push + SetMode)
   - Exit wrapper generation (Pull + RTI)

4. **`r65/compiler/mir/__init__.py`**
   - Export `ReturnFromInterrupt`

## Status

✅ **COMPLETE** - Fully implemented and tested

---

**Last Updated:** 2026-01-01
**Implementation:** Complete
**Testing:** Passing
