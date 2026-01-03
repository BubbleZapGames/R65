# Source Interleaving Format Options

This document explores various formats for interleaving R65 source code with generated assembly, evaluating readability, tooling compatibility, and use cases.

---

## Format 1: Vertical Pipe (Current Design)

**Description**: R65 source prefixed with `; |`, assembly instructions follow.

**Example:**
```asm
; | fn calculate(a @ A: u8, b @ X: u8) -> u8 {
; |     let sum @ A = a + b;  // Add values
STX $16             ; Add values | Save b
LDA $16             ; Add values | Load b
CLC                 ; Add values | Clear carry
ADC A               ; Add values | Add a
; |     return sum;
RTS
; | }
```

**Pros:**
- Clear visual separation with `|` character
- Assembly-comment compatible (WLA-DX ignores `;` lines)
- Preserves R65 structure and indentation
- R65 comments extracted to assembly comments

**Cons:**
- R65 code and assembly somewhat separated visually
- Requires scrolling to see assembly for distant source lines

**Best for:** Standard assembly output, WLA-DX compatibility, general use

---

## Format 2: Side-by-Side Columns

**Description**: R65 on left (40 chars), assembly on right, separated by `|`

**Example:**
```asm
fn calculate(a @ A: u8, b @ X: u8)  |  calculate:
    -> u8 {                          |
    let sum @ A = a + b; // Add      |      STX $16        ; Save b
                                     |      LDA $16        ; Load b
                                     |      CLC            ; Clear carry
                                     |      ADC A          ; Add a
    return sum;                      |      RTS
}                                    |
```

**Pros:**
- Direct correspondence between source and assembly
- Easy to scan both at once
- Familiar to developers who've used side-by-side diffs
- Compact representation

**Cons:**
- Fixed column width may truncate long lines
- Harder to generate (requires column alignment)
- May not assemble directly (depends on assembler)
- Difficult to read for wide source lines

**Best for:** Documentation, learning, compiler output visualization

**Implementation Note:** Could be an alternate output mode (`--format=sidebyside`)

---

## Format 3: Block Sections

**Description**: Complete R65 statement in comment block, followed by all its assembly.

**Example:**
```asm
; ┌─────────────────────────────────────────────────────────
; │ fn calculate(a @ A: u8, b @ X: u8) -> u8 {
; │     let sum @ A = a + b;  // Add values
; └─────────────────────────────────────────────────────────
calculate:
    STX $16             ; Add values | Save b
    LDA $16             ; Add values | Load b
    CLC                 ; Add values | Clear carry
    ADC A               ; Add values | Add a

; ┌─────────────────────────────────────────────────────────
; │     return sum;
; │ }
; └─────────────────────────────────────────────────────────
    RTS
```

**Pros:**
- Very clear statement boundaries
- Grouped by logical units
- Easy to identify which assembly corresponds to which source

**Cons:**
- Verbose (takes more vertical space)
- Box drawing characters may not work in all editors
- Harder to generate
- More visual clutter

**Best for:** Teaching, detailed debugging, complex statements

---

## Format 4: Nested Indentation

**Description**: Assembly is indented under the R65 source it implements.

**Example:**
```asm
; fn calculate(a @ A: u8, b @ X: u8) -> u8 {
calculate:

;     let sum @ A = a + b;  // Add values
        STX $16             ; Save b
        LDA $16             ; Load b
        CLC                 ; Clear carry
        ADC A               ; Add a

;     return sum;
        RTS

; }
```

**Pros:**
- Hierarchical structure shows compilation relationship
- Assembly clearly "belongs to" the source above it
- Natural reading flow (source → implementation)
- Clean, minimal syntax

**Cons:**
- Unusual assembly style (most assembly isn't indented like this)
- Labels at different indentation than instructions
- May confuse assemblers that are whitespace-sensitive

**Best for:** Understanding compiler transformations, teaching

---

## Format 5: Inline Expansion

**Description**: R65 source on its own line, assembly immediately below (indented).

**Example:**
```asm
calculate:
// fn calculate(a @ A: u8, b @ X: u8) -> u8 {

// let sum @ A = a + b;  // Add values
    STX $16             ; Save b
    LDA $16             ; Load b
    CLC                 ; Clear carry
    ADC A               ; Add a

// return sum;
    RTS

// }
```

**Pros:**
- Uses `//` for R65 source (feels natural for Rust-like syntax)
- Assembly at proper indentation level
- Compact and readable
- Easy to scan (alternate between `//` and instructions)

**Cons:**
- `//` may be confused with actual comments in some assemblers
- Less visual separation than pipe format
- Could be ambiguous which assembly belongs to which source

**Best for:** Quick scanning, minimal visual overhead

---

## Format 6: Tagged Sections

**Description**: Explicit tags mark source vs assembly sections.

**Example:**
```asm
; [R65]
; fn calculate(a @ A: u8, b @ X: u8) -> u8 {
;     let sum @ A = a + b;  // Add values

; [ASM]
calculate:
    STX $16             ; Save b
    LDA $16             ; Load b
    CLC                 ; Clear carry
    ADC A               ; Add a

; [R65]
;     return sum;
; }

; [ASM]
    RTS
```

**Pros:**
- Explicit section markers
- Easy to parse programmatically
- No ambiguity about what's source vs assembly
- Could support multiple source languages

**Cons:**
- Verbose
- Tags add visual noise
- Requires more vertical space
- Repetitive for many small statements

**Best for:** Multi-language output, tool processing, formal documentation

---

## Format 7: Diff-Style

**Description**: Inspired by unified diff format, using `+ ` prefix for assembly.

**Example:**
```asm
  fn calculate(a @ A: u8, b @ X: u8) -> u8 {
+ calculate:
      let sum @ A = a + b;  // Add values
+     STX $16             ; Save b
+     LDA $16             ; Load b
+     CLC                 ; Clear carry
+     ADC A               ; Add a
      return sum;
+     RTS
  }
```

**Pros:**
- Familiar to developers who use diff/patch tools
- Clear visual distinction (prefix vs no prefix)
- Shows what was "added" by compiler
- Compact

**Cons:**
- Not valid assembly (needs preprocessing)
- `+` prefix unusual for assembly files
- Can't be directly assembled
- Indentation of R65 code unclear (no comment marker)

**Best for:** Compiler development, seeing what was generated, diffs

---

## Format 8: Alternating Blocks (Markdown-Style)

**Description**: R65 in code fence, assembly in code fence, repeated.

**Example:**
```asm
; ```r65
; fn calculate(a @ A: u8, b @ X: u8) -> u8 {
;     let sum @ A = a + b;  // Add values
; ```

calculate:
    STX $16             ; Save b
    LDA $16             ; Load b
    CLC                 ; Clear carry
    ADC A               ; Add a

; ```r65
;     return sum;
; }
; ```

    RTS
```

**Pros:**
- Markdown-compatible (could be rendered in docs)
- Clear boundaries between R65 and assembly
- Familiar syntax to many developers
- Could support syntax highlighting in editors

**Cons:**
- Very verbose
- Code fences add significant visual noise
- Takes much more vertical space
- Harder to scan quickly

**Best for:** Documentation generation, Markdown rendering, GitHub/GitLab display

---

## Format 9: Compact Reference

**Description**: Minimal format - just line numbers and statement type.

**Example:**
```asm
; L5: fn calculate(a @ A: u8, b @ X: u8) -> u8
calculate:

; L6: let sum @ A = a + b
    STX $16
    LDA $16
    CLC
    ADC A

; L7: return sum
    RTS
```

**Pros:**
- Very compact
- Easy to cross-reference with source file
- Minimal visual overhead
- Fast to generate

**Cons:**
- Loses actual source text
- Must have source file open to understand
- No inline comments preserved
- Less self-documenting

**Best for:** Space-constrained output, when source is always available

---

## Format 10: Hybrid Columns (Wide Format)

**Description**: R65 in left column (60 chars), assembly in right (80+ chars total).

**Example:**
```asm
; R65 Source                                      | Assembly
;─────────────────────────────────────────────────┼──────────────────────────
; fn calculate(a @ A: u8, b @ X: u8) -> u8 {     | calculate:
;     let sum @ A = a + b;  // Add values         |     STX $16        ; Save b
;                                                  |     LDA $16        ; Load b
;                                                  |     CLC            ; Clear carry
;                                                  |     ADC A          ; Add a
;     return sum;                                  |     RTS
; }                                                |
```

**Pros:**
- Very clear separation
- Professional appearance
- Good for wide terminals (120+ columns)
- Both sides are complete and readable

**Cons:**
- Requires wide display (minimum 100 columns)
- Complex to generate (alignment, wrapping)
- Wastes space on narrow displays
- Header/separator lines add overhead

**Best for:** Wide-screen displays, presentations, documentation PDFs

---

## Format 11: Interleaved with Statement IDs

**Description**: Each R65 statement gets an ID, assembly references it.

**Example:**
```asm
; [S1] fn calculate(a @ A: u8, b @ X: u8) -> u8 {
calculate:

; [S2] let sum @ A = a + b;  // Add values
    STX $16             ; [S2] Save b
    LDA $16             ; [S2] Load b
    CLC                 ; [S2] Clear carry
    ADC A               ; [S2] Add a

; [S3] return sum;
    RTS                 ; [S3]

; [S4] }
```

**Pros:**
- Explicit mapping from assembly back to source
- Easy to trace which statement generated what
- Supports out-of-order generation
- Could enable interactive tools (click S2 → highlight assembly)

**Cons:**
- Statement IDs add visual noise
- Requires generating and tracking IDs
- More verbose than simple pipe format
- IDs may not be intuitive

**Best for:** Debuggers, IDE integration, traceability, optimization remarks

---

## Format 12: Minimal Inline

**Description**: R65 source only when it changes (no repeat on each line).

**Example:**
```asm
calculate:
; fn calculate(a @ A: u8, b @ X: u8) -> u8 {
;     let sum @ A = a + b;  // Add values
    STX $16             ; Save b
    LDA $16             ; Load b
    CLC                 ; Clear carry
    ADC A               ; Add a
;     return sum;
    RTS
; }
```

**Pros:**
- Most compact format with source
- Minimal duplication
- Easy to read (source appears once)
- Natural assembly style

**Cons:**
- Harder to see correspondence for multi-instruction statements
- Can't tell at a glance which assembly goes with which source
- Sequential reading required

**Best for:** Experienced developers, compact output

---

## Comparison Matrix

| Format | Readability | Compactness | Assembly Compatible | Tool Friendly | Learning Curve |
|--------|-------------|-------------|---------------------|---------------|----------------|
| 1. Vertical Pipe | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Yes | ⭐⭐⭐⭐ | Low |
| 2. Side-by-Side | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ No | ⭐⭐⭐ | Medium |
| 3. Block Sections | ⭐⭐⭐⭐⭐ | ⭐⭐ | ✅ Yes | ⭐⭐⭐ | Low |
| 4. Nested Indent | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⚠️ Maybe | ⭐⭐⭐ | Medium |
| 5. Inline Expansion | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⚠️ Maybe | ⭐⭐⭐⭐ | Low |
| 6. Tagged Sections | ⭐⭐⭐ | ⭐⭐ | ✅ Yes | ⭐⭐⭐⭐⭐ | Low |
| 7. Diff-Style | ⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ No | ⭐⭐⭐⭐ | Medium |
| 8. Markdown Blocks | ⭐⭐⭐ | ⭐ | ✅ Yes | ⭐⭐⭐⭐⭐ | Low |
| 9. Compact Ref | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ Yes | ⭐⭐ | Medium |
| 10. Wide Columns | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⚠️ Maybe | ⭐⭐⭐ | Low |
| 11. Statement IDs | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ Yes | ⭐⭐⭐⭐⭐ | Medium |
| 12. Minimal Inline | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ Yes | ⭐⭐⭐ | Low |

---

## Recommendations

### Default Format: **Vertical Pipe (Format 1)**
- Best balance of readability, compatibility, and tooling support
- Works with all assemblers
- Clear visual separation
- Industry-standard comment format

### Alternative Format: **Side-by-Side Columns (Format 2)**
- For documentation, PDFs, learning materials
- Controlled via `--format=sidebyside` or `--format=columns`
- Not directly assemblable, but excellent for understanding

### Debug Format: **Statement IDs (Format 11)**
- For compiler development and debugging
- Controlled via `--format=debug` or `--trace-statements`
- Enables precise statement tracking

### Minimal Format: **Compact Reference (Format 9)**
- For space-constrained environments
- Controlled via `--format=minimal` or `--source-refs-only`
- Cross-reference to source file

---

## Implementation Strategy

### Core Support
Implement Format 1 (Vertical Pipe) as the default with full support.

### Additional Formats
Add command-line flag for format selection:
```bash
r65c compile program.r65 --interleave-source --format=<format>
```

**Format options:**
- `pipe` - Vertical pipe (default)
- `sidebyside` - Side-by-side columns
- `nested` - Nested indentation
- `inline` - Inline expansion
- `debug` - Statement IDs
- `minimal` - Compact reference

### Format-Specific Options
```bash
# Side-by-side column width
r65c compile program.r65 --format=sidebyside --source-width=50

# Statement ID prefix
r65c compile program.r65 --format=debug --id-prefix="S"

# Disable box drawing
r65c compile program.r65 --format=blocks --no-box-chars
```

---

## Use Case Recommendations

| Use Case | Recommended Format | Why |
|----------|-------------------|-----|
| General development | Vertical Pipe | Assembles directly, clear |
| Learning R65 | Side-by-Side or Block | See direct correspondence |
| Documentation | Side-by-Side or Markdown | Renders well in docs |
| Debugging compiler | Statement IDs or Tagged | Precise traceability |
| Space-constrained | Minimal or Compact Ref | Smallest output |
| Code review | Vertical Pipe or Nested | Clear, familiar |
| Presentation slides | Wide Columns or Block | Professional appearance |
| CI/CD logs | Minimal or Pipe | Compact, scannable |

---

**Recommendation**: Start with **Format 1 (Vertical Pipe)** as the default, add **Format 2 (Side-by-Side)** as an alternative for documentation and learning, and consider **Format 11 (Statement IDs)** for advanced debugging features.

All formats can coexist with a simple `--format` flag to switch between them.
