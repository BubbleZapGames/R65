# ABI Models and Calling Convention

## Overview

R65's calling convention is **explicit and flexible**, designed to match hand-written assembly patterns from existing SNES code while providing type safety and mode tracking. The programmer specifies the parameter-passing mechanism for each parameter, and the compiler enforces consistency.

R65 supports three compile-wide ABI models selected via the `--abi` flag. Each model controls how stack parameters are passed, who cleans up the stack, and how return values are delivered. Per-function details (register bindings, preserves, far/near) apply within any model.

```bash
r65c game.r65 -o game.asm              # Default ABI (implicit)
r65c game.r65 -o game.asm --abi FixedStack
r65c game.r65 -o game.asm --abi Pascal
```

---

## Parameter Passing

### Register Parameters (@ A, @ X, @ Y)

**Syntax**: `param @ Register: Type`

**Mechanism**: Caller places value in specified register. Zero overhead when values are already in the right registers.

```rust
fn add(a @ A: u8, b @ X: u16) -> u8 {
    // a already in A, b already in X
    return A;
}
```

**Generated assembly (caller)**:
```asm
LDA #10        ; a @ A
LDX #20        ; b @ X
JSR add
; Result in A
```

**Characteristics**:
- Fastest (no memory access)
- Limited to 3 registers (A, X, Y), plus B in m8 mode
- **X/Y must be u16** (always x16 mode)
- Zero overhead if values already in registers

### B Register Parameters (@ B, m8 Only)

In m8 mode (default, or when A parameter is u8), the B register (high byte of the 16-bit accumulator) is available as a fourth parameter slot.

**Syntax**: `param @ B: Type`

```rust
fn pack_word(low @ A: u8, high @ B: u8) -> u16 {
    return A as u16 | ((B as u16) << 8);
}
```

**Key rules**:
- **m8 only**: Compiler error if B used with `@ A: u16` (m16 mode)
- **Caller setup**: Caller sets B via XBA instruction before the call
- **Mixed parameters**: B can be combined with A, X, Y
- **Not preserved**: B cannot appear in `#[preserves(...)]` (B shares hardware with A)

**Performance**: B access via XBA costs 3 cycles. Common in hand-written 65816 assembly for byte packing.

### Variable-Bound Parameters (@ VARIABLE)

**Syntax**: `param @ VARIABLE: Type`

**Mechanism**: Caller writes to a specific memory location (typically zero-page).

```rust
#[zeropage(0x10)]
static mut INPUT_A: u8;

#[zeropage(0x11)]
static mut INPUT_B: u8;

fn add(a @ INPUT_A: u8, b @ INPUT_B: u8) -> u8 {
    return a + b;
}
```

**Generated assembly**:
```asm
; Caller:
LDA #10
STA $10        ; INPUT_A
LDA #20
STA $11        ; INPUT_B
JSR add

; Callee:
add:
    LDA $10        ; Load from INPUT_A
    CLC
    ADC $11        ; Add INPUT_B
    RTS
```

**Characteristics**:
- Fast (zero-page is 3-4 cycles)
- Zero overhead when caller already has values in those locations
- Very common in hand-written SNES assembly

### Stack Parameters

**Syntax**: `param: Type`

**Mechanism**: Pushed by caller, accessed via stack-relative addressing in the callee.

```rust
fn add(a: u8, b: u8) -> u8 {
    return a + b;
}

let result = add(10, 20);
```

**Generated assembly (Default ABI)**:

**Caller**:
```asm
LDA #20        ; Second parameter (pushed first, right-to-left)
PHA
LDA #10        ; First parameter
PHA
JSR add
PLX            ; Caller cleans up 1st param byte (preserves A return)
PLX            ; Caller cleans up 2nd param byte
```

**Callee**:
```asm
add:
    LDA $03,S      ; Load first parameter (stack-relative)
    CLC
    ADC $04,S      ; Add second parameter
    RTS            ; Just return — caller handles cleanup
```

**Characteristics**:
- Slower than register/variable (stack access is expensive on 65816)
- Reentrant (supports recursion)
- Unlimited parameters
- Push order: right to left (first parameter closest to return address)

A pre-codegen analysis pass promotes eligible stack parameters to direct-page scratch registers when available (`--disable-scratch-parameters` turns this off). Promoted parameters are passed via `STA $dp` instead of PHA, avoiding stack-relative addressing entirely.

### Parameter Ordering Rules

Stack parameters must appear before register/variable-bound parameters in the signature:

```rust
// ERROR: Stack parameter after aliased parameter
fn bad(reg @ A: u8, stack: u8) { }

// OK: Stack parameters first
fn good(stack: u8, reg @ A: u8) { }

// OK: All same type
fn all_stack(a: u8, b: u8, c: u8) { }
fn all_register(a @ A: u8, b @ X: u16) { }
```

**Reason**: Stack layout must be determined before register operations.

### Argument Setup Preserves an A-Resident Value

Setting up one argument can destroy another. A `@ B` argument reaches B through
A (`LDA; XBA`), and a `@ X`/`@ Y` argument from memory or the stack does too
(`LDA; TAX` — the 65816 has no stack-relative `LDX`/`LDY`). If the value being
passed to the `@ A` parameter already lives in A, that value is gone.

Ordering cannot fix it: the `@ A` argument emits *nothing* when the value is
already in A, so the value has to survive every other argument's setup, and the
`SELF_Y` load, all the way to the call. The compiler parks it and puts it back:

```rust
// stdlib: far fn mul8(multA @ A: u8, multB @ B: u8) -> u16
fn scaled(v @ A: u8, k: u8) -> u16 { return mul8(v, k); }
```
```asm
scaled:
    XBA         ; park v in B
    LDA $03,S   ; k
    XBA         ; A = v, B = k -- restores A *and* delivers the B argument
    JSR mul8
```

The parking place is chosen in this order, and costs nothing when no argument's
setup would clobber A:

| where | when | cost |
|---|---|---|
| **B** (`XBA` … `XBA`) | an 8-bit value, and a `@ B` argument is the only clobberer | free — the `@ B` argument's own `XBA` is the restore |
| **Y**, then **X** | the register is neither an argument target nor an argument source | `TAY`/`TYA`, 4 cycles |
| **stack** (`PHA`/`PLA`) | both index registers are taken | 7 cycles; stack-relative sources shift with the push automatically |

`TAY`/`TAX` move index-width bits whatever the accumulator mode is, so only the
restore needs its mode pinned — and it is pinned in **both** directions. A `TYA`
executed in m16 writes 16 bits into A, and the high half *is* B, which by then
holds a `@ B` argument.

### Storing From an Index Register

X and Y are unconditionally 16-bit in R65, so `STX`/`STY` write **two** bytes. A
1-byte destination therefore cannot use them at all and is routed through the
accumulator:

```asm
STY $10        ; WRONG for a u8 destination -- writes $10 and $11
SEP #$20       ; the compiler emits this instead
TYA            ; M-sized: copies only Y's low byte, leaving B alone
STA $10
```

The accumulator is narrowed *before* the transfer, not after. `TYA` run in m16
copies 16 bits, which both widens the following store and overwrites `B` with the
index register's high byte.

The same routing covers the addressing modes `STX`/`STY` lack — stack-relative,
self-indexed (`STX addr,X`), and long — so a `#[ram]` destination works rather
than failing to assemble. A genuine 2-byte value still stores directly, at no
cost.

### Zero-Cost Calls

When arguments already match parameter aliases, the compiler emits no setup code:

```rust
fn process(value @ A: u8, index @ X: u16) { }

fn caller() {
    let v @ A = compute();
    let i @ X = 0;
    process(v, i);  // Zero overhead — A and X already set
}
```

**Generated**:
```asm
JSR compute     ; Result in A
LDX #0
JSR process     ; Direct call, no setup
```

---

## Return Values

### Implicit A Return

Functions without an explicit return statement return A:

```rust
fn get_status() -> u8 {
    A = HWREG;
    // Implicitly returns A
}
```

### Explicit Register Return

```rust
fn get_index() -> u16 {
    X = 100;
    return X;  // Transferred to A for single-value return
}
```

**Convention**: A single return value is delivered in A (transferred if needed).

### Multiple Return Values

```rust
fn divide(dividend @ A: u8, divisor @ X: u16) -> u8, u16 {
    // quotient in A, remainder in X
    return A, X;
}

let q, r = divide(100, 7);
```

**Convention**: The return type lists the value types (`-> u8, u16`); the compiler assigns registers from those types and the mode. First return in A, second in X (or B in m8), third in Y. No parentheses in the `return` statement.

### B Register Returns (m8 Only)

In m8 mode, B is the second return register when a function returns two 8-bit values. Declare it with a `-> u8, u8` return type — the first value lands in A, the second in B:

```rust
fn unpack_word(value: u16) -> u8, u8 {
    A = value as u8;
    B = (value >> 8) as u8;
    return A, B;
}
```

The `return` statement supplies the values in declared order; the registers they happen to be read from are independent of the registers they are returned in. For example, `return B, A;` against `-> u8, u8` returns B's value first (in register A) and A's value second (in register B). The first value always lands in A — there is no return type that places a value in B without also using A.

**Caller reads B** via XBA:
```asm
JSR get_high_byte
XBA                ; Exchange B into A
STA result         ; Store the second return value
```

### Zero-Page Returns

```rust
#[zeropage(0x10)]
static mut RESULT: u16;

fn calculate(p @ PARAM: u16) {
    RESULT = p + 100;
    // Returns via RESULT (no explicit return)
}

calculate();
let output = RESULT;  // Read result from zero-page
```

Very common in hand-written SNES code — zero-page is used for both parameters and returns.

### Mixed Returns (Register + Zero-Page)

```rust
#[zeropage(0x10)]
static mut RESULT: u16;

fn mixed_return() {
    RESULT = 1000;
    return X, RESULT;  // X register + RESULT variable
}
```

### Return Signature Consistency

**All return paths in a function must return the same registers/variables in the same order.**

```rust
// GOOD: All paths return A
fn good(val: u16) -> u8 {
    if val > 100 {
        return A;
    } else {
        return A;
    }
}

// BAD: Different registers returned
fn bad(val: u16) {
    if val == 1 {
        return A;     // Signature: (A)
    } else {
        return X;     // Signature: (X) — MISMATCH!
    }
}
```

The compiler validates consistency and produces an error on mismatch:
```
error: inconsistent return signature
  --> file.r65:5:16
   |
3  |         return RESULT;
   |                ------ first return has signature: (RESULT)
5  |         return RESULT2;
   |                ^^^^^^^ returns (RESULT2), expected (RESULT)
```

---

## Structs and Arrays (Pass by Reference)

Structs and arrays cannot be passed by value, returned by value, or directly assigned. This is a deliberate restriction — copying large data structures is expensive on 6502/65816 and the cost should be explicit.

```rust
// ERROR: Cannot pass struct by value
fn bad(player: Player) { }

// CORRECT: Pass by pointer
fn process_player(player: *Player) {
    player.health = player.health - 1;
}

// CORRECT: Return pointer to static data
fn get_player() -> *Player {
    return &PLAYER;
}

// CORRECT: Copy fields individually
PLAYER1.x = PLAYER2.x;
PLAYER1.y = PLAYER2.y;
PLAYER1.health = PLAYER2.health;
```

### Newtypes Are the Exception

A [newtype](type-system.md#newtypes) is not an aggregate. Its payload is a
scalar of at most two bytes, so there is nothing to copy that would not already
fit in a register — it passes, returns, and assigns by value like the `u8` or
`i16` it wraps, and rides in `A`/`X`/`Y` under every parameter mechanism.

```rust
struct TileId(u8);

fn bump(t @ A: TileId) -> TileId { return t + 1; }   // register, 0 cycles
fn combine(a: TileId, b: TileId) -> TileId { ... }   // stack, like two u8s
```

### `self` by Value

Because a newtype fits in a register, its methods take **`self` by value**, and
`self` is bound to the accumulator. Pointer self (`*self`) on a newtype is a
compile error, and bare `self` on a struct or union is too — one self form per
type.

```rust
struct TileId(u8);

impl TileId {
    fn raw(self) -> u8        { return self.0; }
    fn bumped(self) -> TileId { return TileId(self.0 + 1); }
}
```

`bumped` is the whole method — self arrives in `A` and the result leaves in `A`:

```asm
TileId__bumped:
    INC A
    RTS
```

`raw` is a retype of a value already in `A`, so it inlines away to nothing at
the call site. Mutation is expressed by returning a new value; there is no
in-place form, which is why the compound-assignment operator traits
(`AddAssign` and friends) do not apply to newtypes — see
[operator-overloading.md](operator-overloading.md).

Two consequences of binding `self` to `A`:

- **Neither `A` nor `B` is available to a parameter.** `B` is the accumulator's
  high byte, not a register of its own, so `self` in `A` claims both. Both
  `fn m(self, x @ A: u8)` and `fn m(self, x @ B: u8)` are errors naming the
  conflict; bind `x` to `X`/`Y` or pass it on the stack. The restriction is
  specific to a by-value `self` — a `*self` method is stack-passed and claims
  neither register, and free functions may still bind `@ B` freely.
- A 2-byte payload puts the method in **m16** on entry, exactly as `@ A: u16`
  does for a free function.

Unlike trait methods — which receive `*self` in `Y` and are never inlined —
newtype methods are ordinary static-dispatch functions and remain inlinable.

---

## Register Preservation

### Caller-Save (Default)

All registers are **caller-save** by default. No automatic preservation:

```rust
fn caller() {
    let value @ A = 10;
    SAVED = value;     // Must save A manually
    callee();
    A = SAVED;         // Restore after call
}
```

### Preserves Attribute

Functions declare which registers they preserve. The compiler automatically generates save/restore code.

`A` cannot be preserved by a function that returns a value. The save/restore is a
bracket around the whole body, so the restore runs *after* the result is in A:

```rust
#[preserves(A)]
fn bump(v @ A: u8) -> u8 { return A + 1; }   // ERROR: A holds the return value
```
```asm
bump:
    PHA        ; preserve A
    INC A      ; the result
    PLA        ; restore A - overwrites the result
    RTS
```

A void function may preserve A (it has no result to lose), as may a `-> !`
function (it never reaches its epilogue).

```rust
#[preserves(X, Y)]
fn careful(input @ A: u8) -> u8 {
    X = 20;  // Freely modify — saved at entry, restored at exit
    Y = 30;
    return A;
}
```

**Generated**:
```asm
careful:
    PHX                ; Auto-save X
    PHY                ; Auto-save Y
    ; ... function body ...
    PLY                ; Auto-restore Y
    PLX                ; Auto-restore X
    RTS
```

**Valid registers**: `A`, `X`, `Y`, `STATUS`, `D`, `DBR`. **Invalid**: `B`, `PBR`, `S`.

---

## Near vs Far Calls

### Near (JSR/RTS)

**Syntax**: `fn name() { }`

Same-bank call using 16-bit address.

```rust
fn local_function() { }

fn caller() {
    local_function();  // JSR local_function / RTS
}
```

- 6 cycles (JSR) + 6 cycles (RTS) = 12 cycles overhead
- Same bank only
- Return address: 2 bytes on stack

### Far (JSL/RTL)

**Syntax**: `far fn name() { }`

Cross-bank call using 24-bit address. `#[bank(n)]` sets the bank context:

```rust
#[bank(1)]
far fn remote_function() { }

fn caller() {
    remote_function();  // JSL remote_function / RTL
}
```

- 8 cycles (JSL) + 6 cycles (RTL) = 14 cycles overhead
- Cross-bank capable
- Return address: 3 bytes on stack

**Auto-Bank Mode**: `#[bank(auto)]` for automatic placement. Requires `far fn` and `far static` for ROM statics:
```rust
#[bank(auto)]
far fn auto_placed() { }
far static DATA: [u8; 256] = [0; 256];
```

### Data Bank Register (DBR) Management

Far functions can specify DBR handling via `#[mode(databank=...)]`:

**`databank=none`** (default): No DBR management. Programmer handles it manually.

**`databank=inline`**: Callee saves/restores DBR:
```rust
#[bank(1)]
#[mode(databank=inline)]
far fn auto_dbr() { }
```
```asm
auto_dbr:
    PHB            ; Save DBR
    LDA #$01
    PHA
    PLB            ; Set DBR = 1
    ; ... function body ...
    PLB            ; Restore DBR
    RTL
```

**`databank=caller`**: Caller manages DBR (useful for batching multiple far calls):
```rust
#[bank(1)]
#[mode(databank=caller)]
far fn caller_dbr() { }
```
```asm
caller:
    PHB
    LDA #$01
    PHA
    PLB            ; Set DBR = 1
    JSL caller_dbr
    PLB            ; Restore DBR
```

**Far pointer stack params**: When a function takes `far *T` as a stack parameter, the compiler picks per-function between **D_EQUALS_S** (PHD/TSC/TCD; enables `[dp],Y`) and **SET_DBR** (PHB/PLB to the param's bank; enables `(d,S),Y`) using a cost model. See [Register/Memory Configuration](register_memory_config.md) for the full decision tree, prologue/epilogue shapes, and per-strategy access costs.

### Cross-Bank Call Validation

Near functions can only call near functions in the **same bank**. JSR uses a 16-bit address and cannot cross bank boundaries.

| Caller Bank | Callee Bank | Callee Type | Allowed? |
|-------------|-------------|-------------|----------|
| 0 | 0 | `fn` | Yes |
| 0 | 1 | `fn` | **No** (compile error) |
| 0 | 1 | `far fn` | Yes |
| 1 | 0 | `fn` | **No** (compile error) |
| Any | Any | `far fn` | Yes |

---

## Function Pointers

```rust
type NearFunc = fn(u8) -> u8;        // JSR/RTS
type FarFunc = far fn(u8) -> u8;     // JSL/RTL
```

The type system enforces near vs far. Indirect calls use a trampoline:

```rust
type Callback = fn(input @ A: u8) -> u8;

#[ram]
static mut HANDLER: Callback;

fn caller() {
    let result @ A = HANDLER(10);
}
```

**Generated**:
```asm
caller:
    LDA #10
    JSR call_trampoline

call_trampoline:
    JMP (HANDLER)  ; Indirect jump
```

**Far indirect fast path:** for `far fn(...)` indirect calls where the
function pointer lives in a zeropage scratch slot (DP-addressable),
the compiler skips the generic trampoline and emits a 4-instruction
sequence using the 65816's `JML [d]` opcode (long indirect via DP).
PHK / PEA / JML [d] saves ~62 cycles and ~24 bytes per call. See
[register_memory_config.md §2.5](register_memory_config.md#25-indirect-call-lowering-jml-d-fast-path)
for details and soundness.

---

## Mode Transitions

Mode transitions are handled automatically by the compiler:

- **Default mode**: m8 (8-bit A), x16 (16-bit X/Y)
- **m16 mode**: Inferred when function has `@ A: u16` parameter
- **X/Y always u16**: No x8 mode in R65

```rust
fn process_byte(value @ A: u8) -> u8 { return value + 1; }    // m8
fn process_word(value @ A: u16) -> u16 { return value + 1; }  // m16

fn caller() {
    let byte = process_byte(10);     // m8 → m8, no transition
    let word = process_word(1000);   // callee switches to m16, restores m8
}
```

**Generated (m16 callee)**:
```asm
process_word:
    REP #$20       ; Switch to m16
    ; ... function body ...
    SEP #$20       ; Restore m8
    RTS
```

---

## ABI Models

### Default

The Default ABI uses PHA-based argument passing with caller PLX cleanup. This eliminates a permanent outgoing-arg area from caller stack frames, producing smaller frames and smaller code.

**Parameter passing**: Register and variable-bound parameters work as declared. Stack parameters are pushed via PHA before each call (right-to-left) and cleaned up by the caller via PLX after the call returns.

**Stack frame**: Space for locals and preserved registers, but no outgoing argument area:
```
[locals]
[preserved regs]   <- PHX, PHY, etc.
[return address]   <- 2 bytes (JSR) or 3 bytes (JSL)
[caller's frame]
```

Frame allocation uses `TSC / SEC / SBC #size / TCS` for frames larger than 4 bytes, or one `PHB` per byte for small frames. Deallocation mirrors this — `PLA` per byte for small frames, `TSC / CLC / ADC / TCS` for large ones.

**Return values**: Hardware registers: A (first), B (second, m8 only), X, Y. Callee loads values into registers before returning.

**Cleanup**: Callee simply executes `RTS` (or `RTL`). Caller cleans up pushed arguments via PLX (2 bytes per PLX, preserves A return value).

**Characteristics**:
- Supports recursion
- Unlimited stack parameters
- Scratch promotion reduces stack traffic for leaf-like functions
- No permanent outgoing area — smaller frames
- PHA is 1 byte vs STA d,S at 2 bytes — smaller code
- Region spill analysis saves/restores hardware registers around calls when needed

### FixedStack

FixedStack eliminates stack-passed parameters entirely. All parameters must fit in hardware registers or direct-page scratch locations. This produces smaller, more predictable stack frames at the cost of limiting parameter count.

**Parameter passing**: The parameter promotion pass converts all would-be stack parameters into hardware register or scratch DP assignments. Register bindings (`@ A`, `@ X`, `@ Y`) are honored. Remaining parameters are assigned to available scratch DP addresses. If there are more parameters than available locations, compilation fails.

**Stack frame**: Without stack parameters or an outgoing area, the frame is minimal:
```
[locals]
[preserved regs]
[return address]
```

Frame allocation always uses `PHB` per byte (never `TSC/SBC/TCS`). Deallocation always uses `PLA` per byte. This keeps the stack pointer movement predictable and bounded.

**Return values**: Identical to Default — hardware registers A, B, X, Y.

**Cleanup**: No parameters to clean up. Just `RTS` / `RTL`.

**Restrictions**:
- **No recursion.** Recursive functions are rejected at compile time. Since all parameters go through fixed locations, a recursive call would overwrite the caller's parameters.
- **Limited parameter count.** Bounded by available hardware registers (A, X, Y, B) plus scratch DP slots.

**Use cases**: Interrupt handlers and NMI routines where stack depth must be predictable. Performance-critical inner loops. Programs that need static stack depth analysis. Environments with very small stacks (e.g., 256 bytes).

### Pascal

The Pascal ABI implements an Apple IIGS / classic Pascal calling convention. All parameters go on the stack regardless of register binding annotations. The callee cleans up parameters before returning, and return values are passed through caller-allocated stack space rather than registers.

**Parameter passing**: All parameters are pushed onto the stack via PHA, left-to-right — the first parameter is pushed first (deepest), and the last parameter sits closest to the return address. Register binding annotations (`@ A`, `@ X`, `@ Y`) are ignored. No scratch promotion occurs.

Before pushing parameters, the caller pushes **result space** — enough bytes for the return type.

**Caller sequence**:
```asm
; 1. Push result space (if non-void return)
PHA                  ; result space byte(s)

; 2. Push parameters left-to-right
LDA <param0>
PHA                  ; first param (deepest)
LDA <param1>
PHA                  ; second param
; ...

; 3. Call
JSR callee

; 4. Pull result (callee already cleaned params)
PLA                  ; result value
```

**Stack layout (callee's view)**:
```
[result space]     <- caller will PLA this after return
[param0]           <- deepest parameter
[param1]
...
[paramN]           <- closest to return address
[return address]   <- 2 bytes (JSR) or 3 bytes (JSL)
[preserved regs]
[locals]
SP ->
```

**Return values**: Callee writes its return value into the result space on the stack via `STA offset,S` before the epilogue.

**Cleanup**: Callee removes parameter bytes (but not result space) using a save-adjust-restore sequence:
```asm
PLX                  ; save return address in X
TSC / CLC / ADC #param_bytes / TCS   ; slide SP past params
PHX                  ; push return address back
RTS                  ; return to caller
```

After return, SP points at the result space. The caller pulls it with `PLA`.

**Characteristics**:
- All parameters on stack — simple, uniform convention
- Callee cleanup — caller doesn't need to know parameter byte count
- Stack result space — return values don't occupy registers
- Register bindings are ignored
- No scratch promotion
- Supports recursion

**Use cases**: Interoperability with Apple IIGS toolbox routines or Orca/Pascal code. Research and exploration of alternative calling conventions.

### Comparison Table

| Feature | Default | FixedStack | Pascal |
|---|---|---|---|
| Stack parameters | Yes (PHA push) | No | Yes (all params) |
| Register parameters | Yes | Yes | No (ignored) |
| Scratch promotion | Yes | All params | No |
| Outgoing arg area | No (PHA push) | No | No (PHA push) |
| Parameter cleanup | Caller (PLX) | None | Callee |
| Return mechanism | Registers (A,B,X,Y) | Registers (A,B,X,Y) | Stack result space |
| Recursion | Yes | No | Yes |
| Frame alloc (small) | PHB per byte | PHB per byte | PHB per byte |
| Frame alloc (large) | TSC/SBC/TCS | PHB per byte | TSC/SBC/TCS |
| Frame dealloc (small) | PLA per byte | PLA per byte | PLA per byte |
| Frame dealloc (large) | TSC/ADC/TCS | PLA per byte | TSC/ADC/TCS |

---

## Stack Frame Layout

### Frame Organization

```
                  (high address)
    +---------------------------+
    | Parameter N               |  <- pushed by caller (Default/Pascal)
    +---------------------------+
    | ...                       |
    +---------------------------+
    | Parameter 1               |
    +---------------------------+
    | Return Address            |  <- 2 bytes (JSR) or 3 bytes (JSL)
    +---------------------------+
    | Preserved Registers       |  <- PHX, PHY, etc. from #[preserves]
    +---------------------------+
    | Local 1                   |
    +---------------------------+
    | Local 2                   |  <- SP
    +---------------------------+
                  (low address, growing down)
```

**Stack grows downward** (toward lower addresses).

**Prologue** (frame allocation):
```asm
example:
    TSC
    SEC
    SBC #<locals_size>   ; Allocate space for locals
    TCS
```

**Epilogue** (frame deallocation — Default ABI):
```asm
    TSC
    CLC
    ADC #<locals_size>   ; Deallocate locals only
    TCS
    RTS                  ; Caller cleans up parameters
```

### Stack Depth Considerations

The 65816 `LDA d,S` instruction uses an unsigned 8-bit offset (0-255), so a single function can only address 255 bytes from its SP. In practice this is not a limiting factor:

- **Default**: Per-function frame is locals + preserves. No outgoing area inflation, so frames are typically small.
- **FixedStack**: No outgoing area, no stack params. Frames are just locals + preserves — often just a few bytes.
- **Pascal**: Parameters are pushed per-call (not part of the frame), so the frame itself stays small. The callee needs to reach the result space through frame + prologue + return address + params, which could approach the 255-byte limit for functions with many parameters and large frames.

---

## Performance Characteristics

```
Register parameters:      0-3 cycles (setup)
Variable-bound (zp):      3-6 cycles (memory writes)
Stack parameters:         5-10 cycles (per parameter push)

Near call (JSR/RTS):      12 cycles
Far call (JSL/RTL):       14 cycles
Indirect call:            18-24 cycles (trampoline)

Mode transition (auto):   +6 cycles (REP/SEP)

Caller stack cleanup:     ~4 cycles per parameter byte (PLX)
Pascal callee cleanup:    ~24 cycles (PLX + TSC/CLC/ADC/TCS + PHX, constant)
```

**Fastest**: Register parameters + near call + no mode transition.

---

## Summary

### Decision Tree

**1. Choose parameter passing**:
- Few parameters, performance critical → Register aliases
- Shared communication area (hand-written style) → Zero-page variables
- Many parameters, need reentrancy → Stack

**2. Choose return mechanism**:
- Register return → `return A` or `return A, X`
- Zero-page return → Write to zero-page variable
- Mixed → Combine both

**3. Enforce consistency**:
- All return paths must have identical signatures

**4. Choose call type**:
- Same bank → near `fn()`
- Cross-bank → `far fn()`

**5. Mode is automatic**:
- Inferred from `@ A: u16` parameter
- X/Y always u16 (x16 mode)
- For data bank management, use `#[mode(databank=...)]`

**6. Declare preservation**:
- `#[preserves(X, Y)]` for callee-save registers

**7. Choose ABI model**:
- Default → General purpose, supports recursion, caller cleanup
- FixedStack → Predictable stack depth, no recursion
- Pascal → Apple IIGS interop, callee cleanup

---

*Last Updated: 2026-02-24*
