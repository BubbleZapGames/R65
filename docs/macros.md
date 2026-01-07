# Macros Design

## Overview

R65 provides a simplified macro system inspired by Rust's `macro_rules!` but designed for the constraints and use cases of 6502/65816 development. The goal is to cover ~80% of macro use cases with ~30% of Rust's complexity.

**Design Philosophy**:
- **Simple pattern matching**: One pattern per macro (no multiple arms)
- **Basic repetition**: Comma-separated repetition only (`$(...),*`)
- **No hygiene**: Like C macros, generated names can collide (programmer's responsibility)
- **Token-based**: Operates on token streams, not AST nodes
- **Compile-time only**: All expansion happens before parsing

---

## Syntax Overview

### Macro Definition

```rust
macro_rules! name($param1:fragment, $param2:fragment) {
    // body with $param1 and $param2 substituted
}
```

### Macro Invocation

```rust
name!(arg1, arg2);
```

### Complete Example

```rust
// Define a macro to increment a register twice
macro_rules! inc_twice($reg:reg) {
    $reg++;
    $reg++;
}

// Use the macro
fn main() {
    inc_twice!(X);  // Expands to: X++; X++;
}
```

---

## Fragment Types

Fragment specifiers determine what kind of syntax a parameter can match.

| Fragment | Matches | Examples | Use Case |
|----------|---------|----------|----------|
| `$x:expr` | Any expression | `a + b`, `42`, `foo()`, `arr[i]` | Computations, values |
| `$x:ident` | Single identifier | `my_var`, `BUFFER`, `foo` | Names, labels |
| `$x:literal` | Literal value | `42`, `0xFF`, `true`, `"str"` | Constants |
| `$x:ty` | Type | `u8`, `u16`, `[u8; 16]`, `Player` | Generic over types |
| `$x:reg` | Hardware register | `A`, `X`, `Y` | Register operations |
| `$x:tt` | Token tree | Anything | Catch-all, code blocks |

### Fragment Details

#### `expr` - Expressions

Matches any valid R65 expression:

```rust
macro_rules! double($val:expr) {
    ($val) + ($val)
}

double!(5)           // Expands to: (5) + (5)
double!(x + 1)       // Expands to: (x + 1) + (x + 1)
double!(arr[i])      // Expands to: (arr[i]) + (arr[i])
```

**Note**: Expressions are wrapped in parentheses during expansion to preserve precedence.

**Warning**: Expression is evaluated multiple times if used multiple times in body. For side-effect-free expressions only, or use a local binding.

#### `ident` - Identifiers

Matches a single identifier (variable name, function name, etc.):

```rust
macro_rules! declare_counter($name:ident) {
    #[zeropage]
    static mut $name: u8 = 0;
}

declare_counter!(FRAME_COUNT);
// Expands to:
// #[zeropage]
// static mut FRAME_COUNT: u8 = 0;
```

#### `literal` - Literals

Matches numeric, boolean, or string literals:

```rust
macro_rules! repeat_byte($count:literal, $value:literal) {
    [$value; $count]
}

repeat_byte!(16, 0xFF)  // Expands to: [0xFF; 16]
```

#### `ty` - Types

Matches type expressions:

```rust
macro_rules! declare_buffer($name:ident, $element:ty, $size:literal) {
    #[ram]
    static mut $name: [$element; $size];
}

declare_buffer!(SPRITE_DATA, u16, 128);
// Expands to:
// #[ram]
// static mut SPRITE_DATA: [u16; 128];
```

#### `reg` - Registers

Matches hardware register names (A, X, Y only):

```rust
macro_rules! save_and_clear($reg:reg) {
    let saved = $reg;
    $reg = 0;
}

save_and_clear!(X);
// Expands to:
// let saved = X;
// X = 0;
```

#### `tt` - Token Tree

Matches any single token or balanced group `(...)`, `[...]`, or `{...}`. Use as catch-all or for code blocks:

```rust
macro_rules! forward($($tokens:tt),*) {
    other_macro_rules!($($tokens),*)
}

// Use tt with braces for code blocks
macro_rules! time_it($body:tt) {
    let start = TIMER;
    $body
    let elapsed = TIMER - start;
}

time_it!({ process_frame(); });
// Expands to:
// let start = TIMER;
// { process_frame(); }
// let elapsed = TIMER - start;
```

---

## Repetition

R65 macros support a single repetition form: `$(...),*` (comma-separated, zero or more).

### Basic Repetition

```rust
macro_rules! sum($($val:expr),*) {
    A = 0;
    $(A = A + $val;)*
}

sum!(1, 2, 3);
// Expands to:
// A = 0;
// A = A + 1;
// A = A + 2;
// A = A + 3;

sum!();
// Expands to:
// A = 0;
// (empty repetition)
```

### Repetition with Multiple Captures

Multiple parameters can be captured together:

```rust
macro_rules! init_vars($($name:ident = $value:expr),*) {
    $(let mut $name = $value;)*
}

init_vars!(x = 10, y = 20, z = 30);
// Expands to:
// let mut x = 10;
// let mut y = 20;
// let mut z = 30;
```

### Counting Repetitions

Use a counter pattern for indexed access:

```rust
macro_rules! indexed_store($base:expr, $($val:expr),*) {
    {
        let mut __idx: u8 = 0;
        $(
            $base[__idx] = $val;
            __idx++;
        )*
    }
}

indexed_store!(BUFFER, 10, 20, 30);
// Expands to:
// {
//     let mut __idx: u8 = 0;
//     BUFFER[__idx] = 10; __idx++;
//     BUFFER[__idx] = 20; __idx++;
//     BUFFER[__idx] = 30; __idx++;
// }
```

### Repetition Separators

The separator is always comma. Other separators are not supported:

```rust
// Supported
$($x:expr),*     // Comma-separated

// NOT Supported
$($x:expr);*     // Semicolon-separated (use different pattern)
$($x:expr)*      // No separator
```

---

## Expansion Rules

### Order of Expansion

1. **Macro definitions** are collected during initial lexing
2. **Macro invocations** are expanded before parsing
3. **Expansion is recursive**: Macros can invoke other macros
4. **Depth limit**: Maximum 64 levels of nesting (prevents infinite recursion)

### Token Substitution

Parameters are substituted as token sequences:

```rust
macro_rules! wrap($e:expr) {
    ($e)
}

wrap!(a + b)
// Tokens: ( a + b )
```

### Expression Parenthesization

Expression fragments are automatically parenthesized to preserve precedence:

```rust
macro_rules! double($e:expr) {
    $e * 2
}

double!(1 + 2)
// Without parens: 1 + 2 * 2 = 5 (wrong!)
// With parens: (1 + 2) * 2 = 6 (correct)
// Expands to: (1 + 2) * 2
```

### Identifier Concatenation

Identifiers cannot be concatenated (unlike C's `##`):

```rust
// NOT SUPPORTED
macro_rules! make_name($prefix:ident) {
    let $prefix_counter = 0;  // Does NOT create "foo_counter"
}

// Workaround: Pass full name
macro_rules! make_counter($name:ident) {
    let $name = 0;
}
make_counter!(foo_counter);
```

---

## Hygiene (None)

R65 macros have **no hygiene** - they operate like C preprocessor macros. Names generated by macros can collide with names in the calling scope.

### Name Collision Example

```rust
macro_rules! with_temp($body:tt) {
    let temp = 0;
    $body
}

fn example() {
    let temp = 42;
    with_temp!({ temp = temp + 1; });  // Collision! Which 'temp'?
}
```

### Mitigation Strategies

1. **Use unlikely names with prefixes**:
```rust
macro_rules! with_temp($body:tt) {
    let __macro_temp = 0;  // Unlikely to collide
    $body
}
```

2. **Use block scope**:
```rust
macro_rules! scoped_temp($body:tt) {
    {
        let temp = 0;  // Scoped to this block
        $body
    }
}
```

3. **Accept name as parameter**:
```rust
macro_rules! with_temp($temp_name:ident, $body:tt) {
    let $temp_name = 0;
    $body
}

with_temp!(my_temp, { my_temp = my_temp + 1; });
```

---

## Common Patterns

### Hardware Register Setup

```rust
macro_rules! setup_dma($channel:literal, $src:expr, $dst:expr, $size:expr) {
    DMASRC[$channel] = $src;
    DMADST[$channel] = $dst;
    DMASIZE[$channel] = $size;
    DMACTL[$channel] = 0x01;  // Start transfer
}

setup_dma!(0, SPRITE_DATA, 0x0000, 512);
```

### Loop Unrolling

```rust
macro_rules! unroll4($body:tt) {
    $body
    $body
    $body
    $body
}

// Unroll a tight loop
unroll4!({ A = *PTR; PTR++; *DST = A; DST++; });
```

### Conditional Compilation (Limited)

```rust
// Define feature flags as macros
macro_rules! DEBUG() { }  // Empty = disabled
// macro_rules! DEBUG() { log_state(); }  // Uncomment to enable

macro_rules! debug_only($body:tt) {
    DEBUG!()  // Expands to nothing or debug code
}
```

### Lookup Table Generation

```rust
macro_rules! sin_table($name:ident, $size:literal) {
    #[rom]
    static $name: [u8; $size] = [
        // Pre-computed at compile time
        // (Requires const evaluation support)
    ];
}
```

### Multi-Register Operations

```rust
macro_rules! push_all($($reg:reg),*) {
    $(asm!("PH" + stringify!($reg));)*
}

macro_rules! pop_all($($reg:reg),*) {
    $(asm!("PL" + stringify!($reg));)*
}

// Save all registers
push_all!(A, X, Y);
// ... do work ...
pop_all!(Y, X, A);  // Reverse order!
```

### Struct-like Initialization

```rust
macro_rules! sprite($x:expr, $y:expr, $tile:expr, $attr:expr) {
    {
        SPRITE_X = $x;
        SPRITE_Y = $y;
        SPRITE_TILE = $tile;
        SPRITE_ATTR = $attr;
    }
}

sprite!(100, 50, 0x10, 0x00);
```

### Assert (Debug Only)

```rust
macro_rules! assert($cond:expr) {
    if !($cond) {
        asm!("BRK");  // Trigger debugger
    }
}

assert!(health <= 100);
```

### Bitfield Access

```rust
macro_rules! get_bits($value:expr, $mask:literal, $shift:literal) {
    (($value) & $mask) >> $shift
}

macro_rules! set_bits($target:expr, $mask:literal, $shift:literal, $value:expr) {
    $target = ($target & ~$mask) | (($value << $shift) & $mask)
}

let priority = get_bits!(attr, 0xC0, 6);
set_bits!(attr, 0xC0, 6, 2);
```

---

## Recursive Macros

Macros can call themselves or other macros:

```rust
macro_rules! countdown($n:literal) {
    A = $n;
    // Note: Can't actually recurse with decremented value
    // without const evaluation. This is a limitation.
}

// Macros calling other macros works:
macro_rules! inner($x:expr) {
    $x + 1
}

macro_rules! outer($x:expr) {
    inner!($x) * 2
}

outer!(5)  // Expands to: ((5) + 1) * 2
```

### Recursion Limit

Maximum expansion depth is 64 levels. Exceeding this is a compile error:

```rust
macro_rules! infinite() {
    infinite!()  // ERROR: macro expansion depth exceeded (64 levels)
}
```

---

## Built-in Macros

### `stringify!` - Convert Arguments to String Literal

The `stringify!` macro is a built-in macro that converts its arguments into a string literal. This is useful for debugging, logging, or metaprogramming.

#### Syntax

```rust
stringify!(arg1, arg2, ...)
```

#### Behavior

- Joins all arguments with spaces
- Escapes special characters (quotes, backslashes, newlines, tabs)
- Returns a string literal

#### Examples

```rust
fn debug_print() {
    // Single argument
    stringify!(Hello);           // Expands to: "Hello"

    // Multiple arguments - joined with spaces
    stringify!(Hello World 123); // Expands to: "Hello World 123"

    // Empty arguments
    stringify!();                // Expands to: ""

    // Special characters are escaped
    stringify!(Say "Hi" \n);    // Expands to: "Say \"Hi\" \\n"
}

// Useful for debugging
fn log_debug(message: u8) {
    stringify!(Debug: message = message);
    // Expands to: "Debug: message = message"
}
```

#### Usage Notes

- **Statement context only**: Currently supported only as a statement (`stringify!(...);`)
- **No evaluation**: Arguments are treated as literal tokens, not evaluated
- **Token-based**: Operates on the token level, like all R65 macros
- **Escaping**: Automatically handles special characters for safe string literals

---

## Scope and Visibility

### Global Scope

All macros are globally visible after definition. No visibility modifiers:

```rust
// In header.r65
macro_rules! common_pattern($x:expr) {
    $x + 1
}

// In main.r65
include!("header.r65")
let y = common_pattern!(5);  // Works
```

### Definition Order

Macros must be defined before use:

```rust
foo!(5);  // ERROR: macro 'foo' not defined

macro_rules! foo($x:expr) { $x }

foo!(5);  // OK
```

### Shadowing

Later definitions shadow earlier ones:

```rust
macro_rules! greet() { "Hello" }
let a = greet!();  // "Hello"

macro_rules! greet() { "Hi" }
let b = greet!();  // "Hi"
```

---

## Error Handling

### Compile-Time Errors

| Error | Cause | Example |
|-------|-------|---------|
| Undefined macro | Using macro before definition | `foo!()` without defining `foo` |
| Argument count mismatch | Wrong number of arguments | `foo!(1, 2)` when `foo` takes 1 |
| Fragment mismatch | Argument doesn't match fragment type | `$x:literal` with `a + b` |
| Recursion limit | Too many nested expansions | Infinite recursion |
| Unclosed repetition | Missing `)` in `$(...)` | `$($x:expr,*` |

### Error Messages

Errors point to the invocation site with expansion context:

```
error: macro 'foo' expects 2 arguments, found 1
  --> main.r65:10:5
   |
10 |     foo!(42);
   |     ^^^^^^^^
   |
note: macro 'foo' defined here
  --> macros.r65:5:1
   |
5  | macro_rules! foo($a:expr, $b:expr) { ... }
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

### Debugging Macro Expansion

Use `--dump-macros` flag to see expanded code:

```bash
r65c game.r65 --dump-macros
```

Output shows before/after expansion:

```
=== Macro Expansion ===
Before: inc_twice!(X)
After:  X++; X++;
```

---

## Limitations

### What's NOT Supported

| Feature | Rust | R65 | Reason |
|---------|------|-----|--------|
| Multiple patterns | ✅ | ❌ | Complexity |
| `$x:stmt` fragment | ✅ | ❌ | Hard to parse; use `$x:tt` with `{...}` |
| Nested repetition | `$($($x)*)` | ❌ | Rarely needed |
| Repetition separators | `;`, `:`, etc. | `,` only | Simplicity |
| Hygiene | ✅ | ❌ | Implementation cost |
| Procedural macros | ✅ | ❌ | Way too complex |
| `$crate` | ✅ | ❌ | No module system |
| Identifier concatenation | `##` (C) | ❌ | Token-based expansion |
| Conditional compilation | `#[cfg]` | ❌ | Use const + dead code elimination |
| Variadic without separator | `$($x)*` | ❌ | Ambiguity |

### Workarounds

**Multiple patterns** → Define separate macros:
```rust
// Instead of multiple arms:
macro_rules! foo($x:expr) { ... }
macro_rules! foo_pair($x:expr, $y:expr) { ... }
```

**Statement fragment** → Use `tt` with braces:
```rust
// Instead of $body:stmt, use $body:tt and pass blocks
macro_rules! wrapper($body:tt) {
    setup();
    $body
    cleanup();
}

wrapper!({ do_thing(); });  // Pass code in braces
```

**Identifier concatenation** → Pass full name:
```rust
// Instead of: $prefix ## _counter
macro_rules! make_counter($full_name:ident) {
    static mut $full_name: u8 = 0;
}
make_counter!(sprite_counter);
```

**Conditional compilation** → Use const evaluation:
```rust
const DEBUG: bool = true;

fn maybe_log() {
    if DEBUG {  // Compiler eliminates dead branch
        log_state();
    }
}
```

---

## Implementation Notes

### Compiler Pipeline Integration

```
Source → Lexer → [Macro Collection] → [Macro Expansion] → Parser → AST → ...
                       ↓                      ↓
              MacroDefinition table    Token stream rewriting
```

### Data Structures

```python
@dataclass
class MacroDefinition:
    name: str
    params: List[MacroParam]      # [(name, fragment_type, is_repeated)]
    body_tokens: List[Token]      # Raw token stream
    source_loc: SourceLocation

@dataclass
class MacroParam:
    name: str                     # Without $ prefix
    fragment_type: str            # 'expr', 'ident', 'literal', etc.
    is_repeated: bool             # True if inside $()*

class MacroExpander:
    macros: Dict[str, MacroDefinition]
    expansion_depth: int = 0
    max_depth: int = 64
```

### Expansion Algorithm

1. **Collect**: First pass collects all `macro_rules!` definitions
2. **Match**: For each invocation, match arguments against parameters
3. **Capture**: Extract token sequences for each parameter
4. **Substitute**: Replace `$param` with captured tokens
5. **Recurse**: Re-scan result for nested macro invocations
6. **Limit**: Track depth, error if > 64

### Performance Considerations

- Macro expansion happens once at compile time
- Expanded code may be larger (no runtime cost for expansion itself)
- Repeated use of `$expr` may duplicate code (consider local bindings)

---

## Comparison to Rust Macros

| Aspect | Rust `macro_rules!` | R65 `macro_rules!` |
|--------|---------------------|--------------|
| Patterns per macro | Multiple (with `=>`) | One |
| Repetition forms | `*`, `+`, `?` | `*` only |
| Repetition separators | Any token | Comma only |
| Fragment types | 10+ | 6 |
| Hygiene | Yes (identifier scoping) | No |
| Recursion | Unlimited (with limits) | 64 levels |
| Visibility | `#[macro_export]`, scoped | Global |
| Procedural macros | Yes (`proc_macro`) | No |
| Complexity | High | Low |
| Learning curve | Steep | Gentle |

---

## Examples

### Complete DMA Transfer Macro

```rust
// Define hardware registers
#[hw(0x4300)]
static mut DMACTL: [u8; 8];
#[hw(0x4302)]
static mut DMASRC: [u16; 8];
#[hw(0x4304)]
static mut DMABANK: [u8; 8];
#[hw(0x4305)]
static mut DMASIZE: [u16; 8];

macro_rules! dma_transfer($channel:literal, $src:expr, $bank:expr, $size:expr) {
    DMASRC[$channel] = $src;
    DMABANK[$channel] = $bank;
    DMASIZE[$channel] = $size;
    DMACTL[$channel] = 0x01;
}

fn upload_tiles() {
    dma_transfer!(0, TILE_DATA as u16, 0x00, 4096);
}
```

### Variadic Print for Debugging

```rust
#[hw(0x21FC)]
static mut DEBUG_PORT: u8;

macro_rules! debug_bytes($($val:expr),*) {
    $(DEBUG_PORT = $val;)*
}

fn checkpoint() {
    debug_bytes!(0xDE, 0xAD, 0xBE, 0xEF);  // Send marker bytes
}
```

### State Machine Helper

```rust
macro_rules! state_handler($state:ident, $body:tt) {
    if current_state == State::$state {
        $body
    }
}

fn update() {
    state_handler!(Idle, {
        if input_pressed() {
            current_state = State::Running;
        }
    });

    state_handler!(Running, {
        player_x += velocity;
        if player_x > 240 {
            current_state = State::Idle;
        }
    });
}
```

### Memory Fill

```rust
macro_rules! memset($dst:expr, $val:expr, $count:expr) {
    {
        let mut __i: u16 = 0;
        while __i < $count {
            $dst[__i] = $val;
            __i++;
        }
    }
}

fn clear_screen() {
    memset!(VRAM, 0x00, 2048);
}
```

### Register Preservation Wrapper

```rust
macro_rules! preserve_a($body:tt) {
    {
        let __saved_a = A;
        $body
        A = __saved_a;
    }
}

fn utility_function() {
    preserve_a!({
        A = 0;
        call_external();
    });
    // A is restored here
}
```

---

## Future Enhancements

### Possible Additions

1. **`+` repetition** (one or more): `$($x:expr),+`
2. **`?` repetition** (zero or one): `$($x:expr)?`
3. **Additional separators**: `$($x:expr);*`
4. **`stringify!`**: Convert tokens to string literal
5. **`concat_idents!`**: Identifier concatenation
6. **Scoped macros**: Limit visibility to file

### Not Planned

- Procedural macros (too complex for target use case)
- Full hygiene (implementation cost too high)
- Arbitrary const evaluation in macros

---

**STATUS**: Design Complete
**Last Updated**: 2026-01-06
**Estimated Implementation**: 2-3 weeks
