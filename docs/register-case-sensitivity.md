# Register Case Sensitivity Implementation

## Summary

Implemented strict case sensitivity for 65816 processor registers. All registers must be uppercase.

## Register Names (All Uppercase)

- **Single-character**: A, X, Y, D, S
- **Multi-character**: DBR, PBR, STATUS

## Implementation Details

### Grammar Changes

Updated `compiler/frontend/grammar.lark`:
- Changed `STATUS` to `STATUS` in REGISTER terminal definition
- Registers now: `/A\b/ | /X\b/ | /Y\b/ | /STATUS\b/ | /D\b/ | /DBR\b/ | /PBR\b/ | /S\b/`

### Parser Validation

Added validation in `compiler/frontend/parser.py`:
- `_validate_identifier_not_register()` method validates multi-character register names
- Only validates DBR, PBR, STATUS to avoid false positives
- Single-letter identifiers (a, x, y, d, s) allowed as variable names
- Provides helpful error messages: "Did you mean 'STATUS'?"

### Validation Strategy

**What triggers validation:**
- Using wrong-case multi-character register names (status, dbr, pbr, etc.)
- Suggests correct uppercase version in error message

**What doesn't trigger validation:**
- Single-letter variable names (a, x, y, d, s) - common in programming
- Properly uppercase register references (A, X, Y, D, S, DBR, PBR, STATUS)

## Examples

### Valid Code
```rust
fn test() {
    A = 10;           // OK - register A
    STATUS = 0x00;    // OK - register STATUS
    DBR = 0x7E;       // OK - register DBR
    
    let a = 5;        // OK - variable 'a'
    let x: u8 = 10;   // OK - parameter 'x'
}
```

### Invalid Code
```rust
fn test() {
    status = 10;      // ERROR: suggests STATUS
    STATUS = 10;      // ERROR: suggests STATUS
    dbr = 0x7E;       // ERROR: suggests DBR
    Dbr = 0x7E;       // ERROR: suggests DBR
}
```

## Files Modified

1. `compiler/frontend/grammar.lark` - Updated REGISTER terminal
2. `compiler/frontend/parser.py` - Added validation logic
3. `compiler/frontend/lexer.py` - Updated validation (kept for consistency)
4. `tests/test_lexer.py` - Updated test to use STATUS
5. `CLAUDE.md` - Updated all register references to STATUS

## Test Results

- ✅ All 15 lexer tests pass
- ✅ All 18 parser tests pass
- ✅ Case sensitivity validation working correctly
- ✅ Helpful error messages for wrong-case registers
- ✅ Single-letter variables work without false positives

## Design Rationale

### Why Only Validate Multi-Character Registers?

Single-letter identifiers like 'a', 'x', 'y', 'd', 's' are extremely common as:
- Loop counters
- Function parameters
- Mathematical variables
- Coordinate names

Validating these would cause too many false positives and make the language difficult to use.

Multi-character register names (DBR, PBR, STATUS) are:
- Less commonly used as variable names
- More likely to be typos of register names
- Worth validating to catch case errors

This provides a good balance between catching genuine errors and allowing natural code.

---

*Implemented: 2025-01-03*
*Test STATUS: 33/33 passing*
