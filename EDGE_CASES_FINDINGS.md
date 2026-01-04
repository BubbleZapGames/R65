# R65 Language Edge Cases and Deficiencies

This document tracks edge cases, potential bugs, and design deficiencies discovered during systematic testing.

---

## FIXED BUGS

### 1. **Shift by More Than Width** - FIXED
**Severity**: Medium

**Issue**: Shifting an 8-bit value by more than 8 bits generated incorrect code (only 7 shifts).

**Fix**: Optimizes to `LDA #$00` when shift count >= type bit width.

---

### 6. **Mode Mixing Function Calls** - FIXED
**Severity**: Critical

**Issue**: Compiler allowed m8 function to call m16 function without mode transition, generating incorrect code.

**Fix**: Added compile-time validation that rejects calls with mismatched modes unless explicit `transition` attribute is present.

---

### 8. **Signed vs Unsigned Comparisons** - FIXED
**Severity**: Critical

**Issue**: Compiler used unsigned branch instructions (BCC/BCS) for both signed and unsigned comparisons.

**Fix**: Implemented correct BVC/EOR/#$80/BMI pattern for signed comparisons.

---

### 10. **Division Operator Not Implemented** - FIXED
**Severity**: Medium

**Issue**: Type checker accepted division by powers of 2, but code generator did not implement it.

**Fix**: Implemented LSR-based code generation for power-of-2 divisors (1, 2, 4, 8).

---

### 11. **Multiplication Operator Not Implemented** - FIXED
**Severity**: Medium

**Issue**: Type checker accepted multiplication by powers of 2, but code generator did not implement it.

**Fix**: Implemented ASL-based code generation for power-of-2 multipliers (1, 2, 4, 8).

---

### 7. **Register Preservation Violations** - FIXED
**Severity**: Medium

**Issue**: Compiler did not validate or enforce `#[preserves(...)]` contracts.

**Fix**: Compiler now auto-generates save/restore code (PHA/PLA, PHX/PLX, PHY/PLY, PHP/PLP, PHD/PLD, PHB/PLB) for preserved registers. Functions can freely modify preserved registers - the compiler handles saving and restoring.

**Generated code example:**
```asm
; #[preserves(X, Y)]
preserves_xy:
    PHX          ; Auto-generated save
    PHY          ; Auto-generated save
    ; ... function body modifies X, Y ...
    PLY          ; Auto-generated restore
    PLX          ; Auto-generated restore
    RTS
```

---

### 9. **Const/Literal Overflow Detection** - FIXED
**Severity**: Low

**Issue**: Compiler detected literal overflow but reported "type mismatch" instead of explicit overflow error.

**Fix**: Added overflow detection in `_raise_type_mismatch_error` helper. Now reports:
```
Literal value 256 exceeds maximum for type u8 (255)
  Valid range for u8: 0 to 255
  Suggestion: Use a larger type (e.g., u16) or reduce the value
```

---

## USABILITY IMPROVEMENTS

### 2. **Register Aliasing Semantics** - Documentation Needed
**Severity**: Low

**Issue**: Register aliases are references to registers, not value captures. After a function call, the alias reflects the current register value, not the original.

```rust
fn test_alias_and_call() -> u8 {
    let value @ A = 42;
    let result: u8 = func1(value);  // func1 returns 43 in A
    return value;                    // Returns 43, not 42!
}
```

**Recommendation**: Add prominent documentation that register aliases are **live references**. Consider warning when alias is used after non-preserving call.

---

### 3. **Uninitialized Static Variables**
**Severity**: Low
**Status**: Working as designed

**Issue**: Static variables without initializers have undefined values (SNES RAM not zeroed).

**Recommendation**: Consider optional compiler warning for uninitialized statics.

---

### 4. **Type Inference Limitations**
**Severity**: Low
**Status**: By design

**Issue**: Let bindings for function returns require explicit type annotations.

```rust
let result = get_value();      // ERROR
let result: u8 = get_value();  // OK
```

**Recommendation**: Document this limitation clearly.

---

## EDGE CASES TO VERIFY

### 5. **Division by Zero**
**Severity**: High (but expected for systems language)
**Status**: To verify

**Issue**: Runtime division by zero cannot be caught at compile time.

**Expected**: Undefined behavior at runtime (likely crash or hang).

**Recommendation**: Document as undefined behavior.

---

## TESTING CHECKLIST

### Completed
- [x] Shift overflow edge case - **FIXED** (#1)
- [x] Mode mixing calls - **FIXED** (#6)
- [x] Signed comparison code generation - **FIXED** (#8)
- [x] Division operator - **FIXED** (#10)
- [x] Multiplication operator - **FIXED** (#11)
- [x] Register alias after function call - **DOCUMENTED** (#2)
- [x] Const expression overflow - **FIXED** (#9)
- [x] Preserves violation detection - **FIXED** (#7)

### Pending
- [ ] Division by zero behavior
- [ ] Array bounds (documented as no checking)
- [ ] Function pointer type safety
- [ ] Nested register parameters
- [ ] Memory initialization edge cases

---

## SUMMARY

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | Shift by more than width | Medium | **Fixed** |
| 6 | Mode mixing function calls | Critical | **Fixed** |
| 8 | Signed vs unsigned comparisons | Critical | **Fixed** |
| 10 | Division operator | Medium | **Fixed** |
| 11 | Multiplication operator | Medium | **Fixed** |
| 7 | Register preservation | Medium | **Fixed** |
| 9 | Literal overflow error messages | Low | **Fixed** |
| 2 | Register alias semantics | Low | Document |
| 3 | Uninitialized static variables | Low | By Design |
| 4 | Type inference limitations | Low | By Design |
| 5 | Division by zero | High | To Verify |

### Priority Order
1. **#2**: Document register alias semantics
2. **#5**: Verify/document division by zero behavior

---

*Last Updated: 2026-01-03*
