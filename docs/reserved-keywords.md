# Reserved Keywords in R65

R65 reserves all Rust keywords to maintain compatibility and prevent future conflicts. This document lists all reserved keywords organized by category.

## Currently Implemented Keywords (22)

These keywords are actively used in R65:

| Keyword | Purpose |
|---------|---------|
| `fn` | Function definition |
| `let` | Variable binding |
| `mut` | Mutable modifier |
| `const` | Compile-time constant |
| `static` | Static variable |
| `if` | Conditional statement |
| `else` | Conditional alternative |
| `loop` | Infinite loop |
| `while` | Conditional loop |
| `for` | For loop (`for i in 0..n`) |
| `in` | For loop range |
| `match` | Pattern matching |
| `break` | Exit loop |
| `continue` | Skip to next iteration |
| `return` | Return from function |
| `struct` | Structure definition |
| `enum` | Enumeration definition |
| `type` | Type alias |
| `include` | File inclusion |
| `asm` | Inline assembly |
| `as` | Type casting |
| `macro_rules` | Macro definition |
| `impl` | Trait/method implementation | Struct methods |
| `trait` | Trait definition | Abstract traits |
| `self` | Current module/instance | Impl method parameter |
| `dyn` | Trait object | Pointer to trait object |

## Built-in Functions (4)

These are treated as keywords because they map to special hardware instructions:

| Keyword | Purpose |
|---------|---------|
| `mvn` | Block move forward (65816) |
| `mvp` | Block move backward (65816) |
| `wai` | Wait for interrupt |
| `stp` | Stop processor |

## Reserved Rust Keywords (17)

Currently unused but reserved for future implementation:

| Keyword | Rust Purpose | R65 STATUS |
|---------|-------------|----------------|
| `where` | Generic constraints | Reserved for future generics |
| `use` | Import items | Reserved (we use `include!` instead) |
| `pub` | Public visibility | Reserved (no module system currently) |
| `mod` | Module definition | Reserved (no module system currently) |
| `crate` | Crate root | Reserved (no module system currently) |
| `Self` | Current type | Reserved for future methods |
| `super` | Parent module | Reserved (no module system currently) |
| `async` | Async function | Reserved (not planned for 65816) |
| `await` | Await async value | Reserved (not planned for 65816) |
| `move` | Move closure | Reserved (closures not planned) |
| `ref` | Reference binding | Reserved (no lifetimes) |
| `extern` | External linkage | **In use** — `extern fn` / `extern static` for asm interop (see [Assembly Interop](#assembly-interop)) |
| `unsafe` | Unsafe code block | Reserved (but **not used** - all R65 code has direct hardware access) |

## Strict Reserved Keywords (13)

Reserved by Rust for future use - we reserve them for compatibility:

| Keyword | STATUS |
|---------|--------|
| `abstract` | Reserved for future use |
| `become` | Reserved for future use |
| `box` | Reserved for future use |
| `do` | Reserved for future use |
| `final` | Reserved for future use |
| `macro` | Reserved for future use |
| `override` | Reserved for future use |
| `priv` | Reserved for future use |
| `typeof` | Reserved for future use |
| `unsized` | Reserved for future use |
| `virtual` | Reserved for future use |
| `yield` | Reserved for future use |
| `try` | Reserved for future use |

## Special Modifier Keyword (1)

| Keyword | Purpose |
|---------|---------|
| `far` | Far function call (JSL/RTL) or far pointer type |
| `near` | Near function call (JSR/RTS) or near pointer type |

## Assembly Interop

`extern` declares a symbol implemented in an included `.s` file. Pair it with
`include_asm!("file.s")` to bring the asm into the build.

```rust
// Brings the .s file's contents into the current bank/section
include_asm!("vendor/sound.s");

// Body-less declarations — symbols resolved at link time
extern fn sound_tick(a @ A: u8) -> u8;
extern far fn sound_play_song(id @ A: u8);
extern static SONG_TABLE: [u8; 64];
extern static mut SOUND_RAM: [u8; 256];
```

Rules:
- `extern fn` is body-less and ends in `;`. A body block (`{ }`) is a parse error.
- Defaults to all-clobbered. Add `#[preserves(X, Y)]` to assert what the asm
  callee preserves — the compiler trusts the annotation.
- `extern fn` lowers to `JSR symbol` (near, current bank); `extern far fn`
  lowers to `JSL symbol` (24-bit).
- `extern static` cannot carry storage attributes (`#[ram]`, `#[zeropage]`,
  etc.) — the asm file owns placement. Reads/writes resolve to the bare label.
- `include_asm!` paths resolve relative to the including `.r65` file (same
  search rules as `include_bytes!`). The expansion lands inside the surrounding
  `.BANK`/`.SECTION` window, so the included file should not carry its own
  `.BANK` directives.

## Total Count

**57 reserved keywords** in total:
- 22 currently implemented
- 4 built-in functions
- 17 reserved Rust keywords
- 13 strict reserved keywords
- 2 special modifier (`far`, `near`)

## Important Notes

1. **Case Sensitivity**: All keywords are **case-sensitive** and must be lowercase (except `Self`).
   - ✅ `fn` is a keyword
   - ❌ `Fn` and `FN` are valid identifiers

2. **Word Boundaries**: Keywords use word boundaries, so they don't match partial words.
   - ✅ `impl` is a keyword
   - ✅ `implementation` is a valid identifier

3. **No Keyword as Identifiers**: You **cannot** use any reserved keyword as:
   - Variable names
   - Function names
   - Type names
   - Field names
   - Any other identifier

4. **Future Compatibility**: By reserving all Rust keywords, we ensure that:
   - R65 code won't break if we add new features
   - The language remains familiar to Rust programmers
   - Migration from Rust is easier

## Rationale

### Why Reserve Unused Keywords?

1. **Future-Proofing**: Allows adding features later without breaking existing code
2. **Rust Compatibility**: Makes the language familiar to Rust programmers
3. **Consistency**: Prevents confusion about which Rust keywords work in R65
4. **Best Practice**: Following Rust's example of reserving keywords early

### Keywords We'll Never Use

Some reserved keywords will likely **never** be implemented in R65:

- `async`/`await` - No async runtime on 65816
- `move` - No closures planned
- `unsafe` - All R65 code has direct hardware access by design

However, we still reserve them to maintain maximum compatibility with Rust syntax highlighters and tools.

---

## Hardware Register Names

65816 processor registers are **not keywords** but are recognized as special identifiers. All register names must be **uppercase**:

| Register | Type | Description |
|----------|------|-------------|
| `A` | u8/u16 | Accumulator (u8 in m8, u16 in m16) |
| `B` | u8 | High byte of accumulator (m8 mode only) |
| `X` | u16 | X index register (always 16-bit) |
| `Y` | u16 | Y index register (always 16-bit) |
| `D` | u16 | Direct Page register |
| `S` | u16 | Stack Pointer |
| `DBR` | u8 | Data Bank Register |
| `PBR` | u8 | Program Bank Register (read-only) |
| `STATUS` | u8 | Processor status flags |

### Case Sensitivity Rules

**Registers must be uppercase:**
```rust
A = 10;           // OK - register A
STATUS = 0x00;    // OK - register STATUS
DBR = 0x7E;       // OK - register DBR
```

**Lowercase/mixed-case are valid variable names:**
```rust
let a = 5;        // OK - variable 'a'
let x: u8 = 10;   // OK - variable 'x'
let status = 0;   // OK - variable 'status'
```

### Validation Strategy

- **Single-letter** lowercase (a, x, y, d, s) are allowed as variable names (common in programming)
- **Multi-character** wrong-case (status, dbr, pbr) trigger helpful errors suggesting the uppercase version
- Example: `status = 10;` → Error: "Did you mean 'STATUS'?"
