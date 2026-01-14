# Array Bounds Checking

R65 provides **compile-time bounds checking** for array accesses with constant indices, while allowing unbounded access for dynamic indices to maintain hardware performance.

## Design Philosophy

**Hardware-Appropriate Safety**: Catch obvious programming mistakes at compile time without imposing runtime costs on the resource-constrained 6502/65816 target.

## Constant Index Bounds Checking

When the array index is a compile-time constant, the compiler verifies it's within bounds:

```rust
#[ram]
static mut BUFFER: [u8; 256] = [0; 256];

fn process() {
    BUFFER[0] = 42;      // OK - within bounds
    BUFFER[255] = 99;    // OK - within bounds
    
    BUFFER[256] = 1;     // COMPILE ERROR: index 256 >= array size 256
    BUFFER[300] = 5;     // COMPILE ERROR: index 300 >= array size 256
}
```

**Error message:**
```
error: array index 300 is out of bounds for array of size 256
   --> src/main.r65:10:5
    |
10  |     BUFFER[300] = 5;
    |     ^^^^^^^^^^^^^^^^
```

## Unbounded Dynamic Access

When the index is a variable or non-constant expression, no bounds checking is performed:

```rust
fn process() {
    let index = 10;
    BUFFER[index] = 5;   // No bounds check - programmer responsibility
    
    let offset = A + 1;
    BUFFER[offset] = 7;  // No bounds check - programmer responsibility
}
```

**Rules:**
- No compile-time bounds checking for non-constant indices
- No runtime bounds checking (too expensive on 6502/65816)
- Out-of-bounds access is undefined behavior
- Programmer is responsible for ensuring valid indices

## Constant Expression Detection

The compiler performs constant folding and evaluation to identify constant indices:

```rust
const ARRAY_SIZE: u16 = 256;
const LAST_INDEX: u16 = ARRAY_SIZE - 1;  // 255

fn process() {
    BUFFER[ARRAY_SIZE - 1] = 99;  // OK - 255
    BUFFER[LAST_INDEX] = 100;     // OK - 255
    
    BUFFER[ARRAY_SIZE] = 0;       // ERROR - 256 >= 256
}
```

**Supported constant expressions:**
- Integer literals
- Const variables
- Arithmetic operations (`+`, `-`, `*`, `/`, `%`)
- Bitwise operations (`&`, `|`, `^`, `<<`, `>>`)
- Type casts with `as`
- Array `len()` method (e.g., `BUFFER.len()` returns `256`)

## Implementation Details

**Compiler phases involved:**
1. **Const Evaluation**: Evaluates constant expressions and caches results
2. **Type Checking**: Validates array access for constant indices
3. **Error Reporting**: Provides detailed error messages with bounds information

**Performance impact:**
- Zero runtime overhead (compile-time only)
- No impact on dynamic array access
- Minimal compile-time cost for constant evaluation

## Best Practices

**1. Use constants for array sizes:**
```rust
const BUFFER_SIZE: u16 = 256;
static mut BUFFER: [u8; BUFFER_SIZE] = [0; BUFFER_SIZE];
```

**2. Validate dynamic indices when necessary:**
```rust
fn safe_write(index: u16, value: u8) {
    if index < BUFFER_SIZE {
        BUFFER[index] = value;
    }
}
```

**3. Consider sentinel values for error conditions:**
```rust
fn read_buffer(index: u16) -> u8 {
    if index >= BUFFER_SIZE {
        return 0xFF;  // Error indicator
    }
    return BUFFER[index];
}
```

## Relationship to Hardware

This design aligns with 6502/65816 programming practices:

- **Direct addressing**: No bounds checking on hardware
- **Performance-critical**: Runtime checks too expensive
- **Programmer responsibility**: Hardware-level control expected
- **Compile-time assistance**: Catch obvious mistakes early

## Error Messages

**Out of bounds error:**
```
error: array index 300 is out of bounds for array of size 256
   --> src/main.r65:15:5
    |
15  |     BUFFER[300] = 5;
    |     ^^^^^^^^^^^^^^^^
    |
    = note: array indices must be less than the array size
    = help: use a valid index in range 0..255
```

**Negative index error:**
```
error: array index -1 is out of bounds for array of size 256
   --> src/main.r65:20:5
    |
20  |     BUFFER[-1] = 5;
    |     ^^^^^^^^^^^^^^^
    |
    = note: array indices cannot be negative
```

## Future Considerations

**Potential enhancements (not currently planned):**
- Optional runtime bounds checking with compiler flag
- Static analysis for common dynamic access patterns
- Integration with sanitizers for debugging

**Current limitations:**
- No bounds checking for multi-dimensional arrays
- No checking for pointer-based array access
- Constant evaluation limited to expressions without function calls

---

*This feature provides safety for common programming mistakes while maintaining the performance and hardware-transparency principles of R65.*