# Interrupt Handler Mode Transition

## Design Decision

**Rule:** Interrupt handlers automatically run in the default mode (m8, x16). The compiler handles all mode management automatically.

## Rationale

1. **Interrupts fire from unknown mode**: an NMI/IRQ keeps the M and X flags of the code it interrupted, which may be m16 or (inside a `STATUS.XY16 = false` region) x8
2. **Handler needs specific mode**: the body is compiled for m8/x16
3. **Must restore original mode**: RTI pulls the P the CPU pushed, so the interrupted widths come back without a PHP/PLP pair

## Entry Wrapper

```
REP #$30            ; 16-bit A and X/Y (REP #$10 when A is not saved)
PHA                 ; full 16-bit A, hidden high byte included   } only the
PHX / PHY / PHD / PHB                                            } registers the
                                                                 } body writes
[LDA #0 / TCD]      ; D = 0 when the scratch pool is saved
SEP #$20            ; m8 for the body — always
[scratch pool saves]
[frame allocation]
```

The REP/SEP pair is unconditional. A body that writes no register still
depends on the widths: `FRAME = FRAME + 1` folds to `INC FRAME`, a 16-bit
increment when the NMI lands in m16 code, and a `BIT $4211` acknowledge
reads two registers in m16. An index register left x8 would truncate X and
Y in the body. Before this was fixed, the SEP was skipped whenever nothing
was saved, so an `INC`-only handler ran at whatever width it interrupted.

### Which registers are saved

The emitted body decides. The prologue's save run and every RTI's restore
run are first emitted for all of A, X, Y, D and DBR; once the body is
emitted, `_reconcile_interrupt_saves` reads what it writes
(`clobber_summary.code_writes`, the scanner the call-site summaries use)
and rebuilds the runs for exactly that set. A call takes its callee's
summary (at -O1; at -O0, or for a callee with no summary, it writes
everything), `asm!` text writes everything, and a scratch pool forces A,
since the pool is put back with PLA/STA.

## Exit Wrapper

```
[frame deallocation: REP #$20 / TSC / CLC / ADC #n / TCS]
[scratch pool restores]
PLB / PLD / PLY / PLX     ; the saved registers, reverse order
REP #$20 / PLA            ; when A was saved: full 16-bit restore
RTI                       ; restores P, so the interrupted M/X widths
```

## Example

```rust
#[interrupt(nmi)]
fn nmi_handler() {
    V = V + 1;      // folds to INC on memory: no register written
}
```

```asm
nmi_handler:
    REP #$10        ; 16-bit X/Y
    SEP #$20        ; 8-bit A — the INC is one byte whatever was interrupted
    INC V
    RTI
```

## Edge Cases Handled

### 1. **Interrupt Handler Mode**

All interrupt handlers run in the default m8/x16 mode:

```rust
#[interrupt(nmi)]
fn nmi_handler() {
    // Runs in m8/x16 mode (default)
    // Saves the registers the body writes; REP/SEP sets m8/x16
}
```

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
**Restoration:** Via RTI (restores whatever mode was interrupted)

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

**Last Updated:** 2026-10-10
**Implementation:** Complete
**Testing:** Passing
