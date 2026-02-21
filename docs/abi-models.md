# ABI Models

R65 supports three compile-wide ABI models selected via the `--abi` flag. Each model controls how parameters are passed, how stack frames are allocated, how return values are delivered, and who cleans up the stack after a call.

```bash
r65c game.r65 -o game.asm              # Default ABI (implicit default)
r65c game.r65 -o game.asm --abi FixedStack
r65c game.r65 -o game.asm --abi Pascal
```

The ABI model is a compile-wide policy — every function in the program uses the same convention. Per-function details (register bindings, preserves, far/near) still apply within the chosen model.

---

## Default

The Default ABI is the default convention. It uses PHA-based argument passing with caller PLX cleanup. This eliminates the permanent outgoing-arg area from caller stack frames, producing smaller frames and smaller code.

### Parameter Passing

Register and variable-bound parameters work as declared — the caller places the value in the specified register or memory location before the call. Stack parameters are always pushed via PHA before each call (right-to-left, so the first parameter ends up closest to the return address) and cleaned up by the caller via PLX after the call returns.

A pre-codegen analysis pass promotes eligible stack parameters to direct-page scratch registers when available (`--disable-scratch-parameters` turns this off). Promoted parameters are passed via `STA $dp` instead of PHA, avoiding stack-relative addressing entirely.

### Stack Frame

The caller's frame includes space for locals and preserved registers, but no outgoing argument area:

```
[locals]
[preserved regs]   ← PHX, PHY, etc.
[return address]   ← 2 bytes (JSR) or 3 bytes (JSL)
[caller's frame]
```

Frame allocation uses `TSC / SEC / SBC #size / TCS` for frames larger than 4 bytes, or one `PHB` per byte for small frames. Deallocation mirrors this — `PLA` per byte for small frames, `TSC / CLC / ADC / TCS` for large ones.

### Return Values

Return values are passed in hardware registers: A (first), B (second, m8 only), X, Y. The callee loads values into the appropriate registers before returning.

### Cleanup

The callee does **not** clean up parameters. It simply executes `RTS` (or `RTL` for far functions). The caller cleans up pushed arguments via PLX after the call returns (2 bytes per PLX, preserving the A return value).

### Characteristics

- Supports recursion
- Unlimited stack parameters
- Scratch promotion reduces stack traffic for leaf-like functions
- No permanent outgoing area — smaller frames
- PHA is 1 byte vs STA d,S at 2 bytes — smaller code
- Region spill analysis saves/restores hardware registers around calls when needed

---

## FixedStack

FixedStack is a restricted ABI that eliminates stack-passed parameters entirely. All parameters must fit in hardware registers or direct-page scratch locations. This produces smaller, more predictable stack frames at the cost of limiting parameter count.

### Parameter Passing

The parameter promotion pass converts all would-be stack parameters into hardware register or scratch DP assignments. Register bindings (`@ A`, `@ X`, `@ Y`) are honored. Remaining parameters are assigned to available scratch DP addresses. If there are more parameters than available locations, compilation fails.

There is no outgoing argument area — the caller sets up registers and/or scratch locations directly, then calls.

### Stack Frame

Without stack parameters or an outgoing area, the frame is minimal:

```
[locals]
[preserved regs]
[return address]
```

Frame allocation always uses `PHB` per byte (never `TSC/SBC/TCS`). Deallocation always uses `PLA` per byte. This keeps the stack pointer movement predictable and bounded — the stack pointer only changes by known constant amounts.

### Return Values

Identical to Default — hardware registers A, B, X, Y.

### Cleanup

The callee has no parameters to clean up. Just `RTS` / `RTL`.

### Restrictions

- **No recursion.** Recursive functions are rejected at compile time. Since all parameters go through fixed locations (registers and scratch DP), a recursive call would overwrite the caller's parameters.
- **Limited parameter count.** Bounded by available hardware registers (A, X, Y, B) plus scratch DP slots.

### Use Cases

- Interrupt handlers and NMI routines where stack depth must be predictable
- Performance-critical inner loops where stack-relative addressing overhead matters
- Programs that need static stack depth analysis (no recursion, no unbounded SP growth)
- Environments with very small stacks (e.g., 256 bytes)

---

## Pascal

The Pascal ABI implements an Apple IIGS / classic Pascal calling convention. All parameters go on the stack regardless of register binding annotations. The callee cleans up parameters before returning, and return values are passed through caller-allocated stack space rather than registers.

### Parameter Passing

All parameters are pushed onto the stack via `PHA`, left-to-right — the first parameter is pushed first (ending up deepest), and the last parameter sits closest to the return address. Register binding annotations (`@ A`, `@ X`, `@ Y`) are ignored. No scratch promotion occurs.

Before pushing parameters, the caller pushes **result space** — enough bytes for the return type. For a `u8` return, 1 byte; for `u16`, 2 bytes; for void, nothing.

### Caller Sequence

```
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

### Stack Layout (callee's view)

After the call, with the callee's frame set up:

```
[result space]     ← caller will PLA this after return
[param0]           ← deepest parameter
[param1]
...
[paramN]           ← closest to return address
[return address]   ← 2 bytes (JSR) or 3 bytes (JSL)
[preserved regs]
[locals]
SP →
```

### Return Values

The callee writes its return value into the result space on the stack via `STA offset,S` before beginning the epilogue. The offset is computed from the current SP through the frame, prologue, return address, and parameter bytes to reach the result space above all of that.

### Cleanup

The callee removes parameter bytes (but **not** result space) in the epilogue. Since the return address sits between the parameters and SP, the cleanup uses a save-adjust-restore sequence:

```asm
PLX                  ; save return address in X
TSC / CLC / ADC #param_bytes / TCS   ; slide SP past params
PHX                  ; push return address back
RTS                  ; return to caller
```

After return, SP points at the result space. The caller pulls it with `PLA`.

### Frame Allocation

Uses `PHB` per byte for small frames (4 bytes or less), `TSC / SEC / SBC / TCS` for larger frames — same as Default.

### Characteristics

- All parameters on stack — simple, uniform calling convention
- Callee cleanup — caller doesn't need to know parameter byte count
- Stack result space — return values don't occupy registers
- Register bindings are ignored — `@ A` has no effect
- No scratch promotion
- Supports recursion (all state is on the stack)

### Use Cases

- Interoperability with Apple IIGS toolbox routines or Orca/Pascal code
- Research and exploration of alternative calling conventions
- Situations where a uniform stack-only ABI simplifies code generation

---

## Comparison

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

## Stack Depth Considerations

The 65816 `LDA d,S` instruction uses an unsigned 8-bit offset (0-255), so a single function can only address 255 bytes from its SP. In practice this is not a limiting factor:

- **Default**: Per-function frame is locals + preserves. No outgoing area inflation, so frames are typically small. The hard stack limit (total RAM allocated for stack) is reached by call nesting long before any single frame approaches 255 bytes.
- **FixedStack**: No outgoing area, no stack params. Frames are just locals + preserves — often just a few bytes. The hard stack limit is reached by call nesting long before any single frame approaches 255 bytes.
- **Pascal**: Parameters are pushed per-call (not part of the frame), so the frame itself stays small. The callee needs to reach the result space through frame + prologue + return address + params, which could approach the 255-byte limit for functions with many parameters and large frames.

---

*Last Updated: 2026-02-20*
