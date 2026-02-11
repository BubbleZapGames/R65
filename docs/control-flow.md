# Control Flow Structures Design

## Overview

R65 provides control flow structures that map efficiently to 65816 branch and jump instructions while maintaining readable, structured code.

**Design Principles**:
- Clear mapping to assembly (branches and jumps)
- Handle branch distance limitations transparently
- No runtime overhead for structured control flow
- Support for early exit patterns common in systems programming

---

## Branch Distance Limitations

The 65816 has two types of control flow instructions:

### Conditional Branches (±127 bytes)
- **Instructions**: BEQ, BNE, BCC, BCS, BMI, BPL, BVC, BVS
- **Range**: 8-bit signed offset (−128 to +127 bytes)
- **Performance**: 2-3 cycles (2 if not taken, 3 if taken)
- **Use**: Short jumps within small code blocks

### Unconditional Jumps (full address space)
- **Instructions**: JMP (absolute), JMP (indirect), BRA (relative), BRL (long)
- **Range**: Full 64KB (JMP), ±127 bytes (BRA), ±32KB (BRL)
- **Performance**: 3-4 cycles
- **Use**: Long jumps, function calls, far branches

### Compiler Strategy

The compiler automatically handles branch distance through a post-optimization fixup pass:

1. **Generate code normally**: Emit conditional branches to target labels
2. **Run peephole optimization**: Finalize instruction sequences
3. **Branch fixup pass**: Calculate actual distances and fix long branches

**Long branch fixup pattern:**
```asm
// Original (fails if target > 127 bytes):
    BEQ far_target
    JMP other_target

// Fixed by compiler:
    BNE __branch_skip_0      ; Inverted condition (nearby)
    JMP far_target           ; JMP can reach anywhere
__branch_skip_0:
    JMP other_target
```

**Branch inversion table:**
| Original | Inverted |
|----------|----------|
| BEQ | BNE |
| BNE | BEQ |
| BCC | BCS |
| BCS | BCC |
| BMI | BPL |
| BPL | BMI |
| BVC | BVS |
| BVS | BVC |

This is **transparent to the programmer** - write code naturally and the compiler handles it.

---

## If Statements

### Basic If

**Syntax**:
```rust
if condition {
    // body
}
```

**Semantics**:
- Condition must be type `bool` or comparable expression
- Body executes only if condition is true
- No `else` clause

**Assembly Mapping**:
```rust
if a == b {
    do_something();
}
// LDA a
// CMP b
// BNE skip
// JSR do_something
// skip:
```

**Examples**:
```rust
if x > 10 {
    x = 0;
}

if (flags & 0x80) != 0 {
    handle_error();
}

if ready {  // bool variable
    start_game();
}
```

---

### If-Else

**Syntax**:
```rust
if condition {
    // true branch
} else {
    // false branch
}
```

**Assembly Mapping**:
```rust
if a == 0 {
    process_zero();
} else {
    process_nonzero();
}

// LDA a
// BNE else_block
// JSR process_zero
// JMP end
// else_block:
// JSR process_nonzero
// end:
```

**Examples**:
```rust
if health == 0 {
    game_over();
} else {
    continue_game();
}

if button_pressed {
    jump();
} else {
    walk();
}
```

---

### If-Else If-Else Chain

**Syntax**:
```rust
if condition1 {
    // branch 1
} else if condition2 {
    // branch 2
} else if condition3 {
    // branch 3
} else {
    // default branch
}
```

**Assembly Mapping**:
```rust
if x < 10 {
    category = 0;
} else if x < 20 {
    category = 1;
} else if x < 30 {
    category = 2;
} else {
    category = 3;
}

// LDA x
// CMP #10
// BCS check2      // Branch if x >= 10
// LDA #0
// STA category
// JMP end
// check2:
// LDA x
// CMP #20
// BCS check3
// LDA #1
// STA category
// JMP end
// check3:
// LDA x
// CMP #30
// BCS else_block
// LDA #2
// STA category
// JMP end
// else_block:
// LDA #3
// STA category
// end:
```

**Optimization**: Compiler can reorder conditions or use jump tables for better performance.

---

## Loop Constructs

### Infinite Loop: `loop`

**Syntax**:
```rust
loop {
    // body (repeats forever)
}
```

**Semantics**:
- Repeats indefinitely
- Must use `break` to exit or `return` to exit function
- Common pattern for main game loops and event loops

**Assembly Mapping**:
```rust
loop {
    update();
}

// loop_start:
// JSR update
// JMP loop_start
```

**Examples**:
```rust
// Main game loop
#[entry]
fn main() -> ! {
    init();
    loop {
        wait_vblank();
        update_game();
        render();
    }
}

// Infinite wait
loop {
    if (STATUS & 0x01) != 0 {
        break;
    }
}
```

---

### While Loop

**Syntax**:
```rust
while condition {
    // body
}
```

**Semantics**:
- Condition checked **before** each iteration
- Body may never execute if condition is initially false
- Loop exits when condition becomes false

**Assembly Mapping**:
```rust
while count > 0 {
    process();
    count -= 1;
}

// loop_start:
// LDA count
// BEQ loop_end      // Exit if count == 0
// JSR process
// DEC count
// JMP loop_start
// loop_end:
```

**Examples**:
```rust
while !ready {
    wait();
}

while (JOYPAD & 0x80) != 0 {
    process_input();
}

let mut index = 0;
while index < 10 {
    buffer[index] = 0;
    index += 1;
}
```

---

### For Loop (Range-Based)

**Syntax**:
```rust
for variable in start..end {
    // body
}
```

**Semantics**:
- Iterates from `start` (inclusive) to `end` (exclusive)
- Loop variable is automatically declared as mutable
- Range `start..end` must be integer expressions
- Equivalent to `let mut i = start; while i < end { body; i += 1; }`

**Assembly Mapping**:
```rust
for i in 0..10 {
    process(i);
}

// Desugars to:
// let mut i = 0;
// while i < 10 {
//     process(i);
//     i = i + 1;
// }

// loop_start:
// LDA i
// CMP #10
// BCS loop_end
// JSR process
// INC i
// JMP loop_start
// loop_end:
```

**Examples**:
```rust
// Simple iteration
for i in 0..256 {
    buffer[i] = 0;
}

// Nested loops for 2D processing
for y in 0..8 {
    for x in 0..8 {
        process_tile(x, y);
    }
}

// Using constants
const WIDTH: u8 = 32;
const HEIGHT: u8 = 28;
for row in 0..HEIGHT {
    for col in 0..WIDTH {
        draw_cell(col, row);
    }
}

// Labeled for loop (see Labeled Break/Continue)
'rows: for y in 0..8 {
    'cols: for x in 0..8 {
        if tile[y][x] == target {
            break 'rows;  // Exit both loops
        }
    }
}
```

**Note**: Iterator-based for loops (`for item in collection`) are not supported. Only range syntax `start..end` works.

---

## Break Statement

**Syntax**:
```rust
break;           // Break innermost loop
break 'label;    // Break labeled loop
```

**Semantics**:
- `break;` immediately exits the innermost loop (`loop`, `while`, `for`, or `loop-while`)
- `break 'label;` exits the loop with the specified label
- Execution continues after the target loop
- **Not allowed** outside of loops (compile error)
- Label must refer to an enclosing loop (compile error otherwise)

**Assembly Mapping**:
```rust
loop {
    if ready {
        break;
    }
    wait();
}

// loop_start:
// LDA ready
// BEQ not_ready
// JMP loop_end      // break
// not_ready:
// JSR wait
// JMP loop_start
// loop_end:
```

**Examples**:
```rust
// Search for value in array
let mut index = 0;
let mut found = false;
while index < 256 {
    if buffer[index] == target {
        found = true;
        break;
    }
    index += 1;
}

// Infinite loop with exit condition
loop {
    let input = read_controller();
    if input == 0 {
        break;
    }
    process(input);
}

// Labeled break - exit outer loop from inner loop
'outer: for y in 0..8 {
    for x in 0..8 {
        if tile[y][x] == target {
            break 'outer;  // Exit both loops immediately
        }
    }
}

// Labeled break with loop
'search: loop {
    'inner: while A < 10 {
        if found {
            break 'search;  // Exit outer loop
        }
        A = A + 1;
    }
}
```

---

## Continue Statement

**Syntax**:
```rust
continue;           // Continue innermost loop
continue 'label;    // Continue labeled loop
```

**Semantics**:
- `continue;` skips rest of current iteration of innermost loop
- `continue 'label;` skips to next iteration of the labeled loop
- Jumps to loop condition check (for `while`/`for`) or loop start (for `loop`)
- **Not allowed** outside of loops (compile error)
- Label must refer to an enclosing loop (compile error otherwise)

**Assembly Mapping**:
```rust
while index < 10 {
    if skip_table[index] {
        index += 1;
        continue;
    }
    process(index);
    index += 1;
}

// loop_start:
// LDA index
// CMP #10
// BCS loop_end
//
// LDX index
// LDA skip_table,X
// BEQ not_skipped
// INC index
// JMP loop_start     // continue
//
// not_skipped:
// LDA index
// JSR process
// INC index
// JMP loop_start
// loop_end:
```

**Examples**:
```rust
let mut i = 0;
while i < 100 {
    i += 1;

    if (i & 0x01) != 0 {  // Skip odd numbers
        continue;
    }

    process_even(i);
}

loop {
    let status = read_status();

    if status == 0xFF {
        continue;  // Ignore invalid status
    }

    handle(status);

    if done {
        break;
    }
}

// Labeled continue - skip to next iteration of outer loop
'rows: for y in 0..HEIGHT {
    for x in 0..WIDTH {
        if skip_row[y] {
            continue 'rows;  // Skip rest of this row, go to next y
        }
        process_cell(x, y);
    }
}

// Labeled continue with nested processing
'outer: loop {
    'inner: while A < 10 {
        if should_restart {
            continue 'outer;  // Restart outer loop
        }
        A = A + 1;
    }
    break;
}
```

---

## Return Statement

**Syntax**:
```rust
return;              // Return from void function (or implicit A)
return value;        // Return single value
return a, b;         // Return multiple values
return a, b, c;      // Return three values
```

**Semantics**:
- Immediately exits current function
- Returns control to caller
- Optional return value(s)
- Can appear anywhere in function body

### Return Value Conventions

**No explicit return**:
```rust
fn get_status() -> u8 {
    A = STATUS_REG;
    // Implicitly returns A
}
```

**Explicit register return**:
```rust
fn get_xy() -> (u8, u8) {
    X = PLAYER_X;
    Y = PLAYER_Y;
    return X, Y;
}
```

**Early return**:
```rust
fn validate(input @ A: u8) -> u8 {
    if input == 0 {
        return 0;  // Early exit
    }

    if input > 100 {
        return 100;  // Early exit
    }

    return input;  // Normal return
}
```

**Assembly Mapping**:
```rust
fn process(value @ A: u8) -> u8 {
    if value == 0 {
        return 0xFF;
    }
    return value;
}

// LDA value (already in A)
// BNE not_zero
// LDA #$FF
// RTS              // Early return
// not_zero:
// ; value already in A
// RTS              // Normal return
```

---

## Nested Loops and Break/Continue

### Nested Loops

```rust
let mut y = 0;
while y < 8 {
    let mut x = 0;
    while x < 8 {
        process_tile(x, y);
        x += 1;
    }
    y += 1;
}
```

### Break from Inner Loop

```rust
let mut found = false;
let mut y = 0;
while y < 8 {
    let mut x = 0;
    while x < 8 {
        if tile[y][x] == target {
            found = true;
            break;  // Only breaks inner loop
        }
        x += 1;
    }

    if found {
        break;  // Break outer loop
    }
    y += 1;
}
```

### Labeled Break/Continue

Labels allow breaking or continuing outer loops from nested code:

**Syntax**:
```rust
'label: loop { }      // Label a loop
'label: while { }     // Label a while
'label: for { }       // Label a for
break 'label;         // Break to labeled loop
continue 'label;      // Continue labeled loop
```

**Example - Search with early exit**:
```rust
'outer: for y in 0..8 {
    for x in 0..8 {
        if tile[y][x] == target {
            break 'outer;  // Exit both loops immediately
        }
    }
}
// Execution continues here after break 'outer
```

**Example - Skip to next outer iteration**:
```rust
'rows: for y in 0..HEIGHT {
    'cols: for x in 0..WIDTH {
        if should_skip_row(y) {
            continue 'rows;  // Skip to next row
        }
        if should_skip_col(x) {
            continue 'cols;  // Skip to next column
        }
        process(x, y);
    }
}
```

**Label Rules**:
- Labels start with `'` followed by an identifier and `:`
- Labels are only valid on `loop`, `while`, and `for` statements
- `break 'label` and `continue 'label` must reference an enclosing labeled loop
- Referencing a non-existent label is a compile error

---

## Never Type: `!`

Functions that never return use the `!` type:

```rust
#[entry]
fn main() -> ! {
    init();
    loop {
        update();
    }
    // No return needed - ! means "never returns"
}

fn fatal_error() -> ! {
    SCREEN = 0x00;  // Black screen
    loop {
        stp();  // Stop processor
    }
}
```

**Semantics**:
- Function **never** returns to caller
- Common for entry points and error handlers
- Compiler error if function can return

**Code Generation**:
- If control flow reaches a return point, `WAI` is emitted instead of `RTS`/`RTL`
- This is a safety measure - properly written `-> !` functions should have infinite loops
- The `WAI` instruction halts the CPU until an interrupt, providing a safe fallback

**Assembly Mapping**:
```rust
fn infinite() -> ! {
    loop { }
}

// infinite:
// loop_start:
// JMP loop_start
// ; No RTS - loop never exits

fn broken_never() -> ! {
    // Oops - forgot the infinite loop!
}

// broken_never:
// WAI             ; Safety fallback - halts CPU
```

---

## Conditional Expressions (Ternary)

### Not Included (Use If-Else Instead)

C-style ternary operator (`? :`) is **not included** in initial version:

```rust
// NOT supported:
let x = condition ? true_val : false_val;

// Use if-else instead:
let x = if condition {
    true_val
} else {
    false_val
};
```

### If-Else as Expression

If-else can be used as an expression when both branches are present:
```rust
let category: u8 = if x < 10 {
    0
} else if x < 20 {
    1
} else {
    2
};
```

Both branches must return the same type. The `else` branch is required when using if as an expression.

### Block Expressions

A block `{ ... }` can be used as an expression. The last expression in the block (without a semicolon) is the block's value:

```rust
let result: u8 = {
    let temp: u8 = compute();
    temp + 1
};
```

Block expressions are useful for complex initializations that need intermediate variables.

---

## Optimization Opportunities

### Condition Inversion (Default Strategy)

The compiler uses inverted conditions as its default code generation strategy for all conditional branches:

```rust
// Source:
if x != 0 {
    process();
}
next();

// Naive assembly:
// LDA x
// BNE true_block
// JMP skip
// true_block:
// JSR process
// skip:
// JSR next

// Optimized (no JMP needed):
// LDA x
// BEQ skip
// JSR process
// skip:
// JSR next
```

### Dead Code Elimination

```rust
if false {
    unreachable();  // Removed by optimizer
}

if true {
    always_runs();  // Condition removed
}
```

### Constant Folding

```rust
const DEBUG: bool = false;

if DEBUG {
    log_message();  // Entire block removed if DEBUG = false
}
```

---

## Assembly Label Generation

The compiler generates labels systematically:

### Label Naming Convention

```
function_name__L{block_id}
```

Examples:
- `main__L0`, `main__L1`, `main__L2`
- `update__L0`, `update__L1`

Block IDs are sequential integers assigned during MIR CFG construction. There are no descriptive suffixes like `_if` or `_loop` — all blocks use the same `__L{id}` format.

Additionally, comparison helpers use `__SCMP{N}` labels (globally unique counter).

### Label Scoping

Labels are function-scoped to avoid conflicts:

```rust
fn foo() {
    loop { break; }  // foo__L0, foo__L1, foo__L2
}

fn bar() {
    loop { break; }  // bar__L0, bar__L1, bar__L2
}
```

---

## Short-Circuit Evaluation

Logical operators `&&` and `||` use short-circuit evaluation:

### Logical AND (`&&`)

```rust
if expensive_check1() && expensive_check2() {
    execute();
}

// expensive_check2() only called if expensive_check1() returns true

// Assembly:
// JSR expensive_check1
// ; result in A
// BEQ skip           // If false, skip second check
// JSR expensive_check2
// BEQ skip
// JSR execute
// skip:
```

### Logical OR (`||`)

```rust
if quick_check() || slow_check() {
    execute();
}

// slow_check() only called if quick_check() returns false

// Assembly:
// JSR quick_check
// BNE execute        // If true, skip second check
// JSR slow_check
// BEQ skip
// execute:
// JSR execute
// skip:
```

### Chained Conditions

```rust
if a && b && c {
    execute();
}

// Assembly:
// LDA a
// BEQ skip
// LDA b
// BEQ skip
// LDA c
// BEQ skip
// JSR execute
// skip:
```

---

## Control Flow in Register Context

### Preserving Registers Across Branches

```rust
fn process(value @ A: u8) -> u8 {
    if value > 100 {
        return 100;
    }
    return value;
}

// Compiler ensures A contains correct value on all paths:
// LDA value (already in A)
// CMP #100
// BCC not_too_high
// LDA #100           // Overwrite A with 100
// RTS
// not_too_high:
// ; A still contains original value
// RTS
```

### Register Aliasing in Loops

```rust
fn sum_array(arr: *u8) -> u8 {
    let total @ A = 0;
    let index @ X = 0;

    while index < 10 {
        total = total + arr[index];
        index += 1;
    }

    return total;
}

// Compiler keeps A and X allocated throughout loop:
// LDA #0             // total = 0
// LDX #0             // index = 0
// loop_start:
// CPX #10
// BCS loop_end
// CLC
// ADC arr,X         // total += arr[index]
// INX               // index++
// JMP loop_start
// loop_end:
// RTS               // total in A
```

---

## Error Conditions

### Break/Continue Outside Loop

```rust
fn invalid() {
    break;  // ERROR: break outside of loop
}

fn also_invalid() {
    if true {
        continue;  // ERROR: continue outside of loop
    }
}
```

### Unreachable Code

Unreachable code after `return` or infinite loops is silently removed by dead code elimination. No warning is currently emitted.

### Missing Return

The compiler does not currently validate that all code paths return a value. Functions with missing returns on some paths will compile but may produce incorrect results.

---

## Examples

### Game State Machine

```rust
enum GameState { Menu, Playing, Paused, GameOver }

#[zeropage]
static mut STATE: GameState = GameState::Menu;

fn main() -> ! {
    loop {
        if STATE == GameState::Menu {
            update_menu();
        } else if STATE == GameState::Playing {
            update_game();
        } else if STATE == GameState::Paused {
            update_pause();
        } else if STATE == GameState::GameOver {
            update_game_over();
        }

        render();
        wait_vblank();
    }
}
```

### Polling Loop with Timeout

```rust
fn wait_ready(timeout @ X: u8) -> bool {
    loop {
        if (STATUS & READY_BIT) != 0 {
            return true;
        }

        if timeout == 0 {
            return false;
        }

        timeout -= 1;
        wait_frame();
    }
}
```

### Memory Copy with Count

```rust
fn copy_memory(src: *u8, dst: *u8, count @ X: u8) {
    if count == 0 {
        return;
    }

    let mut index @ Y = 0;
    loop {
        dst[index] = src[index];
        index += 1;
        count -= 1;

        if count == 0 {
            break;
        }
    }
}
```

### Binary Search

```rust
fn binary_search(target @ A: u8) -> u8 {
    let mut low: u8 = 0;
    let mut high: u8 = 255;

    while low <= high {
        let mid = low + ((high - low) >> 1);
        let value = table[mid];

        if value == target {
            return mid;
        } else if value < target {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    return 0xFF;  // Not found
}
```

---

## Match Expressions

### Basic Match

**Syntax**:
```rust
match scrutinee {
    pattern1 => expression1,
    pattern2 => expression2,
    _ => default_expression,
}
```

**Semantics**:
- Evaluates `scrutinee` once, then tests each arm's pattern in order
- Executes the first matching arm's expression
- All arms must return the same type
- Match must be **exhaustive**: every possible value must be covered (use `_` wildcard for remaining values)
- Trailing comma after the last arm is optional

**Supported scrutinee types**: `u8`, `i8`, `u16`, `i16`, `bool`, enums

### Pattern Types

#### Literal Patterns

Match against integer or boolean constants:

```rust
let result: u8 = match tile_id {
    0 => 10,
    1 => 20,
    2 => 30,
    _ => 0,
};
```

#### Enum Patterns

Match against enum variants:

```rust
enum GameState { Menu = 0, Playing, Paused, GameOver }

let cost: u8 = match state {
    GameState::Menu => 0,
    GameState::Playing => 1,
    GameState::Paused => 0,
    GameState::GameOver => 0,
};
```

When all enum variants are covered, no wildcard `_` arm is needed.

#### Range Patterns

Match against a contiguous range of integer values:

```rust
let category: u8 = match tile_id {
    0..=15 => 1,      // inclusive: matches 0, 1, ..., 15
    16..32 => 2,       // exclusive: matches 16, 17, ..., 31
    32..=47 => 3,
    _ => 0,
};
```

- `start..=end` — **inclusive** range (matches start through end)
- `start..end` — **exclusive** range (matches start through end−1)
- Both endpoints must be integer literals
- Empty ranges are a compile error (`5..5`, `5..=3`)
- Range patterns only match integer scrutinee types (`u8`, `i8`, `u16`, `i16`)

#### Or Patterns

Combine multiple patterns with `|`:

```rust
let result: u8 = match input {
    0 | 1 | 2 => 10,
    3 | 4 | 5 => 20,
    _ => 0,
};
```

Range patterns can appear inside OR patterns:

```rust
let zone: u8 = match tile_id {
    0..=3 | 10..=13 => 1,    // two ranges in one arm
    4..=9 => 2,
    _ => 0,
};
```

#### Wildcard Pattern

`_` matches any value (catch-all):

```rust
let result: u8 = match val {
    0 => 100,
    _ => 0,       // matches everything else
};
```

#### Identifier Pattern

Binds the matched value to a variable:

```rust
let result: u8 = match val {
    0 => 100,
    other => other + 1,   // 'other' holds the matched value
};
```

### Exhaustiveness

Match expressions must cover all possible values. The compiler enforces this:

- **bool**: must cover both `true` and `false` (or use `_`)
- **enum**: must cover all variants (or use `_`)
- **integer types**: must include a `_` or identifier pattern (too many values to enumerate)

```rust
// OK: all bool values covered
let x: u8 = match flag {
    true => 1,
    false => 0,
};

// ERROR: non-exhaustive - missing 'false'
let x: u8 = match flag {
    true => 1,
};

// OK: wildcard covers remaining integer values
let x: u8 = match val {
    0 => 10,
    _ => 0,
};
```

### Match as Expression

Match is an expression — it produces a value that can be used in `let` bindings, return statements, or anywhere an expression is expected:

```rust
let category: u8 = match tile_id {
    0..=15 => 0,
    16..=31 => 1,
    _ => 2,
};

return match state {
    GameState::Playing => 1,
    _ => 0,
};
```

### Optimization

The compiler automatically selects the best code generation strategy:

- **Lookup table**: When patterns form a dense range and all arm bodies are compile-time constants, the compiler emits an inline ROM table for O(1) lookup
- **Jump table**: When patterns are dense but bodies aren't constant, emits an indexed jump table (JMP (addr,X)) for O(1) dispatch
- **Branch chain**: For sparse patterns or few arms, emits a sequential comparison chain

Range patterns expand to individual values for density analysis and can trigger table optimizations. However, or-patterns (`a | b`) currently cause a fallback to branch chains — they skip table optimization.

**Assembly Mapping** (branch chain):
```rust
match val {
    0 => handle_zero(),
    1 => handle_one(),
    _ => handle_other(),
}

// CMP #0
// BNE _check_1
// JSR handle_zero
// JMP _merge
// _check_1:
// CMP #1
// BNE _default
// JSR handle_one
// JMP _merge
// _default:
// JSR handle_other
// _merge:
```

**Assembly Mapping** (range pattern branch chain):
```rust
match val {
    0..=15 => handle_low(),
    _ => handle_high(),
}

// CMP #0
// BCC _default        ; val < 0? (unsigned: can't happen for u8, but generated)
// CMP #16
// BCS _default        ; val >= 16? → default
// JSR handle_low
// JMP _merge
// _default:
// JSR handle_high
// _merge:
```

---

## Future Enhancements

### Iterator-Based For Loops

```rust
// Possible future syntax for iterating over collections:
for item in array {
    process(item);
}

// Note: Range-based for loops (for i in 0..10) are already implemented
```

---

## Implementation Notes

### Control Flow Graph (CFG)

The MIR phase builds a CFG for:
- Dead code elimination
- Reachability analysis
- Register allocation across basic blocks
- Optimization opportunities

### Branch Prediction Hints (Future)

Consider annotations for branch prediction:
```rust
#[likely]
if common_case {
    // ...
}

#[unlikely]
if rare_error {
    // ...
}
```

Could influence code layout for better cache behavior (though less relevant for 65816).

---

**STATUS**: Implementation Complete
**Last Updated**: 2026-02-10
**Not Yet Implemented**: Unreachable code warnings, missing return path validation
