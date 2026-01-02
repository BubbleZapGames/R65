# Mode Transition Logic Analysis

## Current Implementation Review

### 1. Unknown Mode Handling ✅

**Logic:** Wrappers only generated when both caller and callee modes are fully known.
```python
mode_mismatch = (caller_mode != callee_mode and
                caller_mode.is_fully_known() and
                callee_mode.is_fully_known())
```

**Implications:**
- Functions with no `#[mode]` attribute have unknown mode
- No wrappers generated for unknown modes
- **Perfect for disassembly**: Preserves original code without injection
- **Safe for new code**: Programmer should specify modes explicitly

**Edge Cases:**
- Partial modes (e.g., `#[mode(m8)]` without X): Treated as unknown, no wrapper
- Mixed unknown/known: No wrapper generated (conservative approach)

---

### 2. Partial Mode Transitions ✅

**Logic:** Correctly handles M-only or X-only transitions via mask combination.

**Examples:**
```rust
// M changes only: (m8, x8) → (m16, x8)
REP #$20        ; Only M bit changes

// X changes only: (m8, x8) → (m8, x16)
REP #$10        ; Only X bit changes

// Both change same direction: (m8, x8) → (m16, x16)
REP #$30        ; Combined mask (0x20 | 0x10)

// Opposite directions: (m8, x16) → (m16, x8)
REP #$20        ; M: 8→16
SEP #$10        ; X: 16→8
```

**Implementation:**
```python
# Build mask for mode changes
sep_mask = 0  # Bits to set (8-bit mode)
rep_mask = 0  # Bits to clear (16-bit mode)

# M flag (bit 5, 0x20)
if from_mode.m_mode != to_mode.m_mode:
    if to_mode.m_mode == ModeState.M8:
        sep_mask |= 0x20
    elif to_mode.m_mode == ModeState.M16:
        rep_mask |= 0x20

# X flag (bit 4, 0x10)
if from_mode.x_mode != to_mode.x_mode:
    if to_mode.x_mode == XModeState.X8:
        sep_mask |= 0x10
    elif to_mode.x_mode == XModeState.X16:
        rep_mask |= 0x10

# Emit combined instructions
if sep_mask:
    emit(SetMode(mask=sep_mask, is_set=True))
if rep_mask:
    emit(SetMode(mask=rep_mask, is_set=False))
```

**Optimization:** Masks combined when transitions are in same direction (both SEP or both REP).

---

### 3. STATUS Preservation Logic

**Current Implementation:**
```python
if mode_mismatch and transition == ModeTransition.CALLER:
    if callee has #[preserves(STATUS)]:
        # Path A: Explicit mode transitions
        SEP/REP to switch to callee mode
        JSR callee
        SEP/REP to restore caller mode
    else:
        # Path B: Stack-based preservation
        PHP                     ; Save STATUS
        SEP/REP to switch to callee mode
        JSR callee
        PLP                     ; Restore STATUS
```

**Analysis:**

#### Path A: Callee Preserves STATUS
```
Caller mode: m8, x8
REP #$30                ; Switch to m16, x16 (callee's mode)
JSR callee              ; Callee preserves STATUS (guaranteed)
                        ; STATUS still has M=0, X=0 (m16, x16)
SEP #$30                ; Restore to m8, x8 (caller's mode)
```

**Contract:** Callee with `#[preserves(STATUS)]` promises:
- Won't execute SEP or REP
- Won't modify any STATUS bits
- Expects to be called in its declared mode

**Byte count:** 4 bytes (REP=2, SEP=2)

#### Path B: Callee Doesn't Preserve STATUS
```
Caller mode: m8, x8
PHP                     ; Push STATUS (saves all bits including M=1, X=1)
REP #$30                ; Switch to m16, x16
JSR callee              ; Callee might modify STATUS in any way
PLP                     ; Restore original STATUS (M=1, X=1 restored)
```

**Safety:** Works regardless of what callee does to STATUS internally.

**Byte count:** 4 bytes (PHP=1, REP=2, PLP=1)
**Stack usage:** 1 byte

---

## Edge Cases Identified

### 1. Internal Mode Changes

**Scenario:** Callee switches modes internally but returns to declared mode:
```rust
#[mode(m16, x16)]
fn callee() {
    // Start in m16
    SEP(0x20);      // Switch to m8 temporarily
    do_work();
    REP(0x20);      // Back to m16
    // Exit in m16
}
```

**Analysis:**
- Callee doesn't have `#[preserves(STATUS)]` (it modifies STATUS)
- Wrapper uses PHP/PLP path
- After call, caller's mode is correctly restored ✅

---

### 2. Recursive Calls

**Scenario:** Function calls itself
```rust
#[mode(m8, x8)]
fn recursive(n: u8) {
    if n > 0 {
        recursive(n - 1);  // Same mode
    }
}
```

**Analysis:**
- `caller_mode == callee_mode` → no wrapper generated ✅

---

### 3. Call Chains with Multiple Transitions

**Scenario:**
```rust
#[mode(m8, x8)]
fn caller() {
    bridge();  // m8 → m16
}

#[mode(m16, x16, transition=caller)]
fn bridge() {
    worker();  // m16 → m8
}

#[mode(m8, x8, transition=caller)]
fn worker() {
    // work
}
```

**Generated code:**
```
caller:
    ; Call to bridge (m8 → m16, transition=caller)
    PHP
    REP #$30
    JSR bridge
    PLP

bridge:
    ; Call to worker (m16 → m8, transition=caller)
    PHP
    REP #$30           ; Actually SEP #$30
    JSR worker
    PLP
```

**Analysis:** Each call site independently handles transitions ✅

---

### 4. Interrupt Handlers

**Question:** How do mode transitions interact with interrupt handlers?

**Interrupt handler structure:**
```rust
#[interrupt(nmi)]
#[mode(m8, x8)]
fn nmi_handler() {
    // Interrupt handler code
}
```

**Analysis:**
- Interrupts already have automatic register preservation (PHP/PHA/PHX/PHY/...)
- PHP at entry saves STATUS including current mode
- If handler has `#[mode]` attribute, it declares what mode it expects
- **Issue:** Who ensures the mode matches?
  - Hardware doesn't guarantee mode on interrupt
  - Handler might need mode transition wrapper at entry
  - **This is not currently implemented!** ⚠️

---

### 5. Unknown Mode Function Calls

**Scenario:** Disassembled function with no mode annotation
```rust
// Disassembled from ROM - mode unknown
fn original_routine() {
    // ...
}

#[mode(m8, x8)]
fn new_code() {
    original_routine();  // What happens here?
}
```

**Current behavior:**
- `callee_mode = ProcessorMode.unknown()`
- `mode_mismatch` check fails (callee not fully known)
- No wrapper generated
- **Programmer must manually ensure correct mode** ✅

**For disassembly:** This is correct! Preserves original calling convention.

---

### 6. Function Pointers (Not Yet Implemented)

**Question:** How would indirect calls work?
```rust
type Handler = fn(x @ A: u8) -> u8;
static mut CALLBACK: Handler;

fn indirect_call() {
    CALLBACK(10);  // How to handle mode transition?
}
```

**Challenges:**
- Mode of target function unknown at compile time
- Function pointer type could encode mode and transition strategy
- Trampoline might need to handle transition dynamically
- **Needs design** 🔴

---

### 7. Bank Boundaries and Far Calls

**Question:** Do far calls (JSL/RTL) need special mode handling?

**Current implementation:**
- `is_far` flag passed to Call instruction
- Mode transitions independent of near/far distinction
- Generates same wrappers for JSR and JSL ✅

**DBR + Mode interaction:**
- `data_bank=auto`: Callee sets DBR to its program bank
- Could combine with mode transition for single wrapper
- **Optimization opportunity:** Combine PHB+PHP → PHB+PHP+SEP/REP+... ⚡

---

### 8. Batching Optimization (Not Implemented)

**Scenario:** Multiple calls to same-mode functions
```rust
#[mode(m8, x8)]
fn caller() {
    func1();  // m8 → m16
    func2();  // m8 → m16
    func3();  // m8 → m16
}

#[mode(m16, x16, transition=caller)]
fn func1() { }

#[mode(m16, x16, transition=caller)]
fn func2() { }

#[mode(m16, x16, transition=caller)]
fn func3() { }
```

**Current (inefficient):**
```
PHP
REP #$30
JSR func1
PLP

PHP
REP #$30
JSR func2
PLP

PHP
REP #$30
JSR func3
PLP
```

**Optimized (batched):**
```
REP #$30     ; Switch once
JSR func1
JSR func2
JSR func3
SEP #$30     ; Restore once
```

**Challenge:** Requires basic block analysis to identify consecutive same-mode calls
**Benefit:** Significant code size and performance improvement
**Status:** Not implemented, mentioned in original design docs ⚡

---

### 9. transition=auto Implementation Status

**Current status:** Validation only, not implemented

**What's validated:**
```python
if transition == ModeTransition.AUTO:
    if func_decl.preserves_attr and 'STATUS' in func_decl.preserves_attr.registers:
        raise TypeCheckError(
            "Function cannot use transition=auto with #[preserves(STATUS)]"
        )
```

**What's missing:** Callee-side wrapper generation
- Need to wrap function body with mode transition
- Structure:
  ```
  function_entry:
      PHP                          ; Save caller's STATUS
      SEP/REP to switch to declared mode
      ; ... function body ...
      PLP                          ; Restore caller's STATUS
      RTS/RTL
  ```

**Challenge:** Where to insert wrapper?
- Entry point: Before first instruction
- Exit points: Before ALL return statements (could be multiple!)
- Need to track all exit blocks

**Interaction with register preservation:**
- If function already has `#[preserves(A, X, Y)]`, stack order matters
- Could conflict with hand-written preservation code
- **Needs careful design** 🔴

---

## Disassembly Use Case Analysis

### Requirements

For reverse engineering SNES ROMs:
1. Disassemble existing assembly to R65
2. Preserve original behavior exactly
3. Don't inject code that wasn't there
4. Allow incremental annotation/improvement

### How Current Implementation Supports This

#### ✅ **Unknown Modes Respected**
```rust
// Disassembled function - original mode unknown
fn subroutine_80C234() {
    A = HWREG;
    X = 0;
}

// Can call without wrapper injection
fn main() {
    subroutine_80C234();  // No wrapper, preserves original behavior
}
```

#### ✅ **Explicit Opt-In for Wrappers**
```rust
// Annotated after analysis
#[mode(m8, x8)]
fn subroutine_80C234() {
    A = HWREG;
    X = 0;
}

// Can still call without wrapper (transition=none is default)
#[mode(m16, x16)]
fn new_function() {
    subroutine_80C234();  // No wrapper unless transition=caller
}
```

#### ✅ **Original SEP/REP Preserved**
```rust
// If original had explicit mode switching:
fn original_code() {
    SEP(0x30);  // Original instruction preserved
    process_8bit();
    REP(0x30);  // Original instruction preserved
}
```

#### ✅ **Incremental Annotation**
```rust
// Stage 1: Disassemble without modes
fn func1() { }
fn func2() { }

// Stage 2: Add modes after analysis
#[mode(m8, x8)]
fn func1() { }

#[mode(m16, x16)]
fn func2() { }

// Stage 3: Opt-in to wrappers for new code
#[mode(m8, x8, transition=caller)]  // Only when ready!
fn new_func() {
    func2();  // Now gets wrapper
}
```

### Potential Issues for Disassembly

#### ⚠️ **Parser Limitation**
- Named attribute arguments (`transition=caller`) currently don't parse
- Workaround: MIRBuilder defaults to `transition=none`
- **Impact:** Low - default behavior is correct for disassembly

#### ⚠️ **Mode Inference**
- Static analysis could infer modes from SEP/REP instructions
- Would help with incremental annotation
- **Not currently implemented**
- Example:
  ```rust
  fn analyze_me() {
      SEP(0x30);   // Could infer: starts unknown, now m8/x8
      work();
      REP(0x20);   // Could infer: now m16/x8
      more_work();
  }
  ```

#### ⚠️ **Interrupt Mode Mismatch**
- Interrupts can fire in any mode
- Handler might expect specific mode
- No automatic mode setup on interrupt entry
- **Needs design:** Auto-insert SEP/REP at interrupt entry?

---

## Recommendations

### High Priority

1. **✅ DONE - Review and validate current implementation**
   - Unknown mode handling: Correct
   - Partial transitions: Correct
   - STATUS preservation: Correct

2. **🔴 TODO - Implement interrupt handler mode entry**
   - Add automatic mode transition at interrupt entry if mode declared
   - Structure:
     ```
     nmi_handler:
         ; Auto-generated by #[interrupt]
         PHP
         PHA
         PHX
         PHY
         PHD
         PHB

         ; Auto-generated by #[mode(m8, x8)]
         SEP #$30     ; Ensure handler's expected mode

         ; User's handler body
         ...

         ; Auto-generated exit
         PLB
         PLD
         PLY
         PLX
         PLA
         PLP
         RTI
     ```

3. **🟡 TODO - Document parser limitation**
   - Add note about named attribute arguments
   - Explain workaround (defaults work correctly)

### Medium Priority

4. **⚡ TODO - Implement batching optimization**
   - Detect consecutive calls to same-mode functions
   - Hoist mode transitions outside call sequence
   - Significant performance win for tight loops

5. **🔴 TODO - Design and implement transition=auto**
   - Callee-side wrapper generation
   - Handle multiple return paths
   - Coordinate with register preservation

6. **🔴 TODO - Design function pointer mode handling**
   - Encode mode/transition in function pointer types
   - Generate appropriate trampolines
   - Consider dynamic vs static mode checking

### Low Priority

7. **⚡ TODO - Mode inference from SEP/REP**
   - Static analysis to infer modes
   - Help with disassembly annotation
   - Optional, not required for correctness

8. **📝 TODO - Add comprehensive tests**
   - Test all edge cases identified
   - Test disassembly workflow
   - Test mode transitions with far calls, DBR changes

---

## Summary

**Current implementation is sound and correct** ✅

**Key strengths:**
- Conservative approach: No wrappers for unknown modes
- Perfect for disassembly: Preserves original behavior
- Correct handling of partial transitions
- Proper STATUS preservation logic
- Explicit opt-in via transition attribute

**Key gaps:**
- Interrupt handler mode entry (safety issue)
- transition=auto not implemented (functional gap)
- No batching optimization (performance opportunity)
- Parser doesn't support named attributes (tooling issue)

**For disassembly use case:** Current implementation is **excellent** - defaults are conservative and preserve original code.
