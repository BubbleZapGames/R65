# Macros Design

## Overview

R65 provides a simplified macro system inspired by Rust's `macro_rules!` but designed for the constraints and use cases of 6502/65816 development. The goal is to cover ~80% of macro use cases with ~30% of Rust's complexity.

**Design Philosophy**:
- **Two definition forms**: a concise R65 shorthand (single arm) and Rust-style multiple arms
- **Basic repetition**: Comma-separated repetition only (`$(...),*`)
- **No hygiene**: Like C macros, generated names can collide (programmer's responsibility)
- **AST-based**: Operates on parsed AST nodes (expansion happens after parsing)
- **Compile-time only**: All expansion happens before HIR lowering

---

## Syntax Overview

### Macro Definition

R65 accepts two equivalent forms. The **shorthand** carries a single pattern:

```rust
macro_rules! name($param1:fragment, $param2:fragment) {
    // body with $param1 and $param2 substituted
}
```

The **multi-arm** form (Rust-style) carries several `(pattern) => { body }` arms,
separated by `;` (the final `;` is optional):

```rust
macro_rules! name {
    ($param:fragment)             => { /* body for one argument */ };
    ($a:fragment, $b:fragment)    => { /* body for two arguments */ };
}
```

The shorthand is exactly a single-arm macro; see [Multiple Arms](#multiple-arms) below.

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

## Multiple Arms

A macro may define several arms, each with its own pattern and body. At a call
site the compiler picks the **first arm that matches** the arguments (top to bottom).

```rust
macro_rules! use_var {
    ($name:ident)              => { $name = 0; };
    ($name:ident, $v:literal)  => { $name = $v; };
}

fn main() {
    use_var!(score);       // arm 1 -> score = 0;
    use_var!(score, 5);    // arm 2 -> score = 5;
}
```

Arms are separated by `;`; the semicolon after the last arm is optional. Each arm
body is wrapped in `{ }` (after the `=>`), and each arm may use its own fragment
types and `$(...),*` repetition.

### How an Arm Is Selected

An arm matches when **both** hold:

1. **Argument count** fits the arm's parameters (exact for fixed arms;
   "at least the leading count" for an arm ending in `$(...),*`).
2. **Fragment type** of each argument is compatible with the corresponding
   parameter. This lets arms of the *same* arity be distinguished:

   ```rust
   macro_rules! load {
       ($r:reg)     => { A = $r; };   // load!(X)   -> A = X;
       ($v:literal) => { A = $v; };   // load!(42)  -> A = 42;
       ($n:ident)   => { A = $n; };   // load!(foo) -> A = foo;
   }
   ```

   Argument classification follows the lexer: `reg` matches a hardware register
   (`A`, `X`, `Y`, `B`, `D`, `S`, `STATUS`, `DBR`, `PBR`); `literal` matches a
   single integer/string/char/boolean literal; `ident` matches a non-register
   identifier; `ty` matches a type; `expr` and `tt` match anything.

Because the first compatible arm wins, **order specific arms before catch-all
arms**. An `($x:expr)` or `($x:tt)` arm matches any single argument, so place it
last:

```rust
macro_rules! describe {
    ($r:reg)  => { /* register case */ };
    ($x:expr) => { /* everything else */ };   // catch-all goes last
}
```

If no arm matches, compilation fails with an error listing every arm's signature.

### Relationship to the Shorthand

The shorthand `macro_rules! name(params) { body }` is exactly a single-arm macro —
it is sugar for `macro_rules! name { (params) => { body } }`. Fragment types are
only consulted to choose **between** arms, so a single-arm macro matches on
argument count alone (its fragment types are documentation only). This keeps every
existing single-arm macro behaving exactly as before.

---

## Method Macros (in `impl` blocks)

A macro declared inside an `impl` block is invoked on a receiver, and `self`
inside the body is replaced by that receiver:

```rust
struct Sprite { x: u8, y: u8 }

impl Sprite {
    macro_rules! move_by($dx:expr, $dy:expr) {
        self.x = self.x + $dx;
        self.y = self.y + $dy;
    }
}

PLAYER.move_by!(1, 2);      // -> PLAYER.x = PLAYER.x + 1; ...
```

Multiple arms work exactly as they do for free macros.

### As a statement or as a value

A method macro expands in either position, and the body decides which one it
fits. A body made of statements is used for its effect:

```rust
impl Q10 {
    macro_rules! clamp($lo:expr, $hi:expr) {
        {
            if self < $lo { self = $lo; }
            if self > $hi { self = $hi; }
        }
    }
}

VELOCITY.clamp!(Q10(0 - 256), Q10(256));
```

A body that is a single expression — usually a
[block expression](control-flow.md) — produces a value:

```rust
impl Sprite {
    macro_rules! area() { { let w: u8 = self.x; w * 2 } }
}

let a: u8 = PLAYER.area!();
OUT = PLAYER.area!() + 1;
```

Using a statement-bodied macro for its value is an error naming the macro:

```rust
OUT = PLAYER.move_by!(1, 2);
// macro error: method macro 'move_by' does not produce a value, so it cannot
// be used here
```

**Rules**: the receiver is substituted textually, so a body naming `self` more
than once evaluates it that many times — pass a place, not a costly expression.
Arms are selected by the receiver's type when it can be resolved, otherwise by
macro name across all `impl` blocks; an ambiguous name is an error.

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
    static $name: [u8; $size] = [  // Immutable = ROM
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

- **Both statement and expression contexts**: Can be used as a standalone statement or within expressions
- **No evaluation**: Arguments are treated as literal tokens, not evaluated
- **Escaping**: Automatically handles special characters for safe string literals

---

### `compile_error!` - Emit Compile-Time Error

Causes compilation to fail with a custom error message:

```rust
compile_error!("This platform is not supported");
```

Useful in macros for guarding against invalid usage.

### `const_assert!` - Compile-Time Assertion

Evaluates a constant expression and emits a compile error if it is false:

```rust
const_assert!(BUFFER_SIZE <= 256);
const_assert!(TILE_WIDTH * TILE_HEIGHT == 64);
```

Both arguments must be compile-time constants.

---

## Inline Assembly (`asm!`)

The `asm!` statement embeds raw 65816 assembly instructions directly into R65 code.

### Basic Syntax

```rust
asm!("instruction");
asm!("instr1", "instr2", "instr3");  // Multiple instructions
```

### Format String Substitution

Named parameters allow dynamic construction of assembly instructions at compile time:

```rust
asm!("LD{REG} #{VAL}", REG="A", VAL=42);  // Generates: LDA #42
asm!("ST{REG} $2100", REG="X");           // Generates: STX $2100
```

- Placeholders use `{name}` syntax
- Values must be **string literals** or **integer literals** (not identifiers or expressions)
- Named arguments apply to all instructions in the statement

### Use with `stringify!`

Combine with `stringify!` in macros to generate register-specific instructions:

```rust
macro_rules! push($reg:reg) {
    asm!("PH{R}", R=stringify!($reg));
}

push!(A);  // Generates: PHA
push!(X);  // Generates: PHX
```

**Important**: Use `stringify!($param)` rather than `$param` directly, because format arguments only accept literals. Using `stringify!` converts any macro parameter (including identifiers) to a string literal.

### Notes

- **Compile-time only**: All substitution happens at compile time
- **Verbatim output**: Resulting strings are emitted as-is to assembly
- **Register clobbering**: Compiler assumes all registers may be modified

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

Use `--dump-ast` to see the AST after macro expansion, or `--dump-hir` to see the final lowered representation:

```bash
r65c game.r65 --dump-ast
r65c game.r65 --dump-hir
```

---

## Limitations

### What's NOT Supported

| Feature | Rust | R65 | Reason |
|---------|------|-----|--------|
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
Source → Lexer → Parser → AST → [Macro Collection] → [Macro Expansion] → HIR → ...
                                        ↓                      ↓
                               MacroDefinition table    AST node rewriting
```

**Note**: Macros are expanded **after** parsing at the AST level, not before parsing at the token level.

### Data Structures

```python
@dataclass
class MacroDefinition:
    name: str
    arms: List[MacroArmDef]       # One arm for the shorthand; many for multi-arm
    source_loc: SourceLocation

@dataclass
class MacroArmDef:
    params: List[MacroParam]      # [(name, fragment_type, is_repeated)]
    body_tokens: List[Token]      # Raw token stream for this arm

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

1. **Parse**: Source is fully parsed into AST (macros are parsed as AST nodes)
2. **Collect**: First pass collects all `macro_rules!` definitions (with their arms) from AST
3. **Select**: For each invocation, pick the first arm whose argument count and
   (for multi-arm macros) fragment types match
4. **Capture**: Extract AST subtrees for each parameter of the selected arm
5. **Substitute**: Replace `$param` with captured AST nodes in that arm's body
6. **Recurse**: Re-scan result for nested macro invocations
7. **Limit**: Track depth, error if > 64

### Performance Considerations

- Macro expansion happens once at compile time
- Expanded code may be larger (no runtime cost for expansion itself)
- Repeated use of `$expr` may duplicate code (consider local bindings)

---

## Comparison to Rust Macros

| Aspect | Rust `macro_rules!` | R65 `macro_rules!` |
|--------|---------------------|--------------|
| Patterns per macro | Multiple (with `=>`) | Multiple (with `=>`), or single via shorthand |
| Arm selection | Full pattern matching | Argument count + fragment type |
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
4. **`concat_idents!`**: Identifier concatenation
5. **Scoped macros**: Limit visibility to file

### Not Planned

- Procedural macros (too complex for target use case)
- Full hygiene (implementation cost too high)
- Arbitrary const evaluation in macros

---

**STATUS**: Implemented
**Last Updated**: 2026-02-10
