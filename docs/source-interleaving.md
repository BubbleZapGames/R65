# Source Code Interleaving in Assembly Output

## Overview

R65 supports interleaving original source code as comments in the generated WLA-DX assembly output. This feature enhances readability, debugging, and understanding of the compilation process by showing the original R65 code alongside the generated assembly instructions.

**Design Goals:**
- Make assembly output self-documenting
- Aid in debugging and optimization
- Help users learn how R65 constructs map to assembly
- Support reverse engineering workflows
- Maintain clean, readable output

---

## Feature Specification

### Compiler Flag

```bash
# Enable source interleaving
r65c compile program.r65 --interleave-source

# Short form
r65c compile program.r65 -s

# Disable (default)
r65c compile program.r65
```

### Output Format

**Format**: Source lines appear as comments above the corresponding assembly instructions.

**Syntax**:
- R65 source code: `; <indent>| <source_line>`
- R65 comments: `; <indent>// <comment_text>` or `; <indent>/* <comment_text> */`
- Assembly comments: `<instruction>    ; <comment>` (inline, right-aligned to column 24)

The `|` character visually separates the interleaved R65 source from R65 comments and assembly comments.

---

## Indentation Mirroring

Assembly instructions mirror the **control flow depth** of the corresponding R65 source code.

### Indentation Rules

1. **Base Level**: Function body starts at 0 indentation
2. **Control Flow Blocks**: Each nesting level adds 4 spaces
3. **Assembly Instructions**: Match the indentation of the R65 statement they implement
4. **Multi-Instruction Sequences**: All instructions for one R65 statement share same indentation

### Indentation Levels

```
Level 0: Function level
Level 1: First-level block (if, loop, etc.)
Level 2: Nested block
Level 3: Doubly-nested block
...
```

---

## Comment Handling

### Three Types of Comments

1. **R65 Source Code** - Original program statements
2. **R65 Comments** - Developer documentation in source
3. **Assembly Comments** - Compiler-generated explanations

### R65 Comment Preservation

**Line Comments** (`//`):
```rust
fn example() {
    // This is important
    let x @ A = 10;
}
```

Generated:
```asm
example:
; | fn example() {
; |     // This is important
; |     let x @ A = 10;
LDA #10
; | }
```

**Block Comments** (`/* */`):
```rust
fn example() {
    /* Multi-line
       explanation */
    let x @ A = 10;
}
```

Generated:
```asm
example:
; | fn example() {
; |     /* Multi-line
; |        explanation */
; |     let x @ A = 10;
LDA #10
; | }
```

**Inline Comments**:
```rust
fn example() {
    let x @ A = 10;  // Initialize value
}
```

Generated:
```asm
example:
; | fn example() {
; |     let x @ A = 10;  // Initialize value
LDA #10
; | }
```

### R65 Comments as Assembly Comments

**R65 inline comments are extracted and attached to the generated assembly instructions.**

**Example with Inline R65 Comment:**

R65 Source:
```rust
fn calculate(a @ A: u8, b @ X: u8) -> u8 {
    let sum @ A = a + b;  // Add the values
    return sum;
}
```

Generated Assembly:
```asm
calculate:
; | fn calculate(a @ A: u8, b @ X: u8) -> u8 {
; |     let sum @ A = a + b;  // Add the values
STX $16             ; Add the values
LDA $16             ; Add the values
CLC                 ; Add the values
ADC A               ; Add the values
; |     return sum;
RTS
; | }
```

**Multiple Instructions**: When one R65 statement generates multiple assembly instructions, the R65 comment is attached to **all** generated instructions.

### Combined Assembly Comments

**R65 Comment + Compiler-Generated Comment:**

When both exist, the format is: `; <r65_comment> | <compiler_comment>`

R65 Source:
```rust
fn example() {
    let x @ A = value + 1;  // Increment value
}
```

Generated Assembly:
```asm
example:
; | fn example() {
; |     let x @ A = value + 1;  // Increment value
LDA value           ; Increment value | Load operand
CLC                 ; Increment value | Clear carry
ADC #1              ; Increment value | Add immediate
STA x               ; Increment value | Store result
; | }
```

**Block Comments Before Statement:**

R65 Source:
```rust
fn example() {
    // This is a critical section
    // Do not modify without review
    process_data();
}
```

Generated Assembly:
```asm
example:
; | fn example() {
; |     // This is a critical section
; |     // Do not modify without review
; |     process_data();
JSR process_data    ; This is a critical section; Do not modify without review
; | }
```

**Block comments** immediately preceding a statement are concatenated and attached to the first instruction.

### Comment Indentation Rules

**R65 Comments**: Mirror the indentation of the R65 source code
```asm
; | fn example() {
; |     // Function-level comment (depth 0)
; |     if condition {
; |         // Block-level comment (depth 1)
; |         statement;
; |     }
; | }
```

**Assembly Comments**: Right-aligned to column 24 (after instruction)
```asm
LDA #10             ; Load immediate value
STA $20             ; Store to zero-page
```

---

## Examples

### Example 1: Simple Function

**R65 Source:**
```rust
// m8 mode (default)
fn calculate(a @ A: u8, b @ X: u8) -> u8 {
    let sum @ A = a + b;
    return sum;
}
```

**Generated Assembly (without interleaving):**
```asm
; ============================================================================
; Function: calculate
; Mode: m8 (default)
; ============================================================================
calculate:
    STX $16
    LDA $16
    CLC
    ADC A
    RTS
```

**Generated Assembly (with interleaving):**
```asm
; ============================================================================
; Function: calculate
; Mode: m8 (default)
; ============================================================================
calculate:
    ; | fn calculate(a @ A: u8, b @ X: u8) -> u8 {
    ; |     let sum @ A = a + b;
    STX $16         ; Save X
    LDA $16         ; Load b
    CLC             ; Clear carry
    ADC A           ; Add a
    ; |     return sum;
    RTS
    ; | }
```

---

### Example 2: If Statement

**R65 Source:**
```rust
fn check_value(x @ A: u8) {
    if x > 100 {
        set_flag();
    } else {
        clear_flag();
    }
}
```

**Generated Assembly (with interleaving):**
```asm
check_value:
    ; | fn check_value(x @ A: u8) {
    ; |     if x > 100 {
    CMP #100
    BCC else_block
        ; |         set_flag();
        JSR set_flag
        JMP end_if
    ; |     } else {
else_block:
        ; |         clear_flag();
        JSR clear_flag
    ; |     }
end_if:
    RTS
    ; | }
```

---

### Example 3: Nested Control Flow

**R65 Source:**
```rust
fn process_data() {
    let mut i @ X = 0;
    while i < 10 {
        if buffer[i] != 0 {
            process_item(i);
        }
        i = i + 1;
    }
}
```

**Generated Assembly (with interleaving):**
```asm
process_data:
    ; | fn process_data() {
    ; |     let mut i @ X = 0;
    LDX #0
    ; |     while i < 10 {
loop_start:
    CPX #10
    BCS loop_end
        ; |         if buffer[i] != 0 {
        LDA buffer,X
        BEQ skip_process
            ; |             process_item(i);
            TXA
            JSR process_item
        ; |         }
skip_process:
        ; |         i = i + 1;
        INX
        JMP loop_start
    ; |     }
loop_end:
    RTS
    ; | }
```

---

### Example 4: Complex Expression

**R65 Source:**
```rust
fn calculate_damage(attack: u8, defense: u8) -> u8 {
    let damage @ A = attack - defense;
    if damage < 5 {
        damage = 5;
    }
    return damage;
}
```

**Generated Assembly (with interleaving):**
```asm
calculate_damage:
    ; | fn calculate_damage(attack: u8, defense: u8) -> u8 {
    ; |     let damage @ A = attack - defense;
    LDA attack
    SEC
    SBC defense
    ; |     if damage < 5 {
    CMP #5
    BCS skip_min
        ; |         damage = 5;
        LDA #5
    ; |     }
skip_min:
    ; |     return damage;
    RTS
    ; | }
```

---

## Implementation Design

### Source Tracking

**AST/MIR Node Metadata:**
- Every AST and MIR node carries source location information:
  - `line_number: int` - Original source line
  - `column: int` - Column position
  - `control_flow_depth: int` - Nesting level (0 = function level)
  - `attached_comment: str` - R65 comment associated with this statement (if any)
  - `preceding_comments: List[str]` - Block comments immediately before this statement

**Source Line Cache:**
```python
class SourceLineCache:
    """Cache of original source file lines."""

    def __init__(self, source_file: str):
        with open(source_file, 'r') as f:
            self.lines = f.readlines()

    def get_line(self, line_num: int) -> str:
        """Get original source line (1-indexed)."""
        if 1 <= line_num <= len(self.lines):
            return self.lines[line_num - 1].rstrip()
        return ""
```

### Comment Extraction During Parsing

**Parser Enhancement:**

The parser extracts comments during tokenization and attaches them to AST nodes.

```python
class CommentExtractor:
    """Extract and associate comments with statements."""

    def extract_inline_comment(self, source_line: str) -> Optional[str]:
        """Extract inline comment from source line."""
        # Match // comment
        match = re.search(r'//\s*(.+)$', source_line)
        if match:
            return match.group(1).strip()
        return None

    def extract_block_comment(self, token) -> Optional[str]:
        """Extract block comment token."""
        if token.type == 'BLOCK_COMMENT':
            # Remove /* */ delimiters
            comment = token.value[2:-2].strip()
            return comment
        return None

    def attach_comments_to_node(self, node: ASTNode, source_line: str,
                                preceding_tokens: List[Token]):
        """Attach comments to AST node."""
        # Extract inline comment
        inline = self.extract_inline_comment(source_line)
        if inline:
            node.attached_comment = inline

        # Extract preceding block comments
        preceding = []
        for token in preceding_tokens:
            if token.type == 'LINE_COMMENT':
                # Strip // prefix
                preceding.append(token.value[2:].strip())
            elif token.type == 'BLOCK_COMMENT':
                preceding.append(self.extract_block_comment(token))

        if preceding:
            node.preceding_comments = preceding
```

**MIR Propagation:**

Comments are propagated from AST to MIR nodes during lowering:

```python
class MIRBuilder:
    def lower_statement(self, ast_stmt: ASTStatement) -> List[MIRInstruction]:
        """Lower AST statement to MIR, preserving comments."""
        mir_instrs = self._generate_mir(ast_stmt)

        # Attach comments to first instruction
        if mir_instrs and ast_stmt.attached_comment:
            mir_instrs[0].attached_comment = ast_stmt.attached_comment

        if mir_instrs and ast_stmt.preceding_comments:
            mir_instrs[0].preceding_comments = ast_stmt.preceding_comments

        return mir_instrs
```


### Assembly Emitter Enhancement

**Indentation Tracking:**
```python
class AssemblyEmitter:
    def __init__(self, interleave_source=False, source_file=None):
        self.interleave_source = interleave_source
        self.source_cache = SourceLineCache(source_file) if source_file else None
        self.current_indent_level = 0
        self.last_emitted_line = -1  # Prevent duplicate source lines

    def set_indent_level(self, level: int):
        """Set current indentation level (0-based)."""
        self.current_indent_level = level

    def emit_source_line(self, line_num: int, indent_level: int):
        """Emit source line as comment."""
        if not self.interleave_source or line_num == self.last_emitted_line:
            return

        source_line = self.source_cache.get_line(line_num)
        if source_line:
            indent = "    " * indent_level
            self.output.append(f"{indent}; | {source_line}")
            self.last_emitted_line = line_num

    def emit_instruction(self, mnemonic: str, operands: str = "",
                        comment: str = "", r65_comment: str = "",
                        indent_level: int = None):
        """Emit assembly instruction with indentation and comments."""
        if indent_level is None:
            indent_level = self.current_indent_level

        indent = "    " * indent_level

        if operands:
            instr = f"{indent}{mnemonic} {operands}"
        else:
            instr = f"{indent}{mnemonic}"

        # Build comment string
        final_comment = ""
        if r65_comment and comment:
            # Both R65 and compiler comment
            final_comment = f"{r65_comment} | {comment}"
        elif r65_comment:
            # Only R65 comment
            final_comment = r65_comment
        elif comment:
            # Only compiler comment
            final_comment = comment

        if final_comment:
            # Align comments to column 24
            instr = instr.ljust(24) + f"; {final_comment}"

        self.output.append(instr)
```

### Instruction Selection Integration

**Modified InstructionSelector:**
```python
class InstructionSelector:
    def select_instruction(self, instr: MIRInstruction):
        """Generate assembly for MIR instruction."""

        # Emit source line if enabled
        if self.emitter.interleave_source:
            # Emit preceding block comments
            if instr.preceding_comments:
                for comment in instr.preceding_comments:
                    indent = "    " * instr.control_flow_depth
                    self.emitter.output.append(f"{indent}; | // {comment}")

            # Emit source line
            self.emitter.emit_source_line(
                instr.source_line,
                instr.control_flow_depth
            )

        # Set indentation level for assembly instructions
        self.emitter.set_indent_level(instr.control_flow_depth)

        # Extract R65 comment from instruction
        r65_comment = instr.attached_comment or ""

        # Generate assembly based on instruction type
        # Pass r65_comment to all emit calls
        if isinstance(instr, Move):
            self._emit_move(instr, r65_comment)
        elif isinstance(instr, BinaryOp):
            self._emit_binary_op(instr, r65_comment)
        # ... etc

    def _emit_move(self, instr: Move, r65_comment: str):
        """Emit move instruction with R65 comment."""
        # Example: LDA #$42
        self.emitter.emit_instruction(
            "LDA",
            f"#{instr.value}",
            comment="Load immediate",
            r65_comment=r65_comment
        )

    def _emit_binary_op(self, instr: BinaryOp, r65_comment: str):
        """Emit binary operation with R65 comment."""
        if instr.op == "+":
            self.emitter.emit_instruction("CLC", comment="Clear carry",
                                         r65_comment=r65_comment)
            self.emitter.emit_instruction("ADC", instr.operand,
                                         comment="Add with carry",
                                         r65_comment=r65_comment)
        # ... other operations
```

---

## Source Line Emission Strategy

### When to Emit Source Lines

**Emit source line when:**
1. Entering a new statement (not already emitted)
2. Control flow depth changes
3. First instruction of a basic block

**Don't emit when:**
1. Same line already emitted
2. Internal compiler-generated instructions (no source mapping)
3. Optimization passes that rearrange code (optional: can be disabled)

### Grouping Multi-Line Constructs

**Single Line for Multi-Instruction Sequences:**
```rust
let sum @ A = a + b;  // Multiple assembly instructions
```

Generated:
```asm
    ; | let sum @ A = a + b;
    LDA a           ; Load a
    CLC             ; Clear carry
    ADC b           ; Add b
    STA sum         ; Store result
```

**Statement Boundaries:**
- Emit source comment before the **first** instruction of the statement
- Don't repeat for subsequent instructions of same statement

---

## Control Flow Depth Calculation

### Depth Assignment Rules

**Function Body**: depth = 0
```rust
fn example() {          // depth 0
    statement;          // depth 0
}
```

**If Statement**: depth += 1 for each branch
```rust
fn example() {
    if condition {      // condition: depth 0
        statement;      // depth 1
    } else {
        statement;      // depth 1
    }
}
```

**Nested Blocks**: accumulate depth
```rust
fn example() {
    while condition {   // condition: depth 0
        if x > 0 {      // condition: depth 1
            statement;  // depth 2
        }
    }
}
```

**Loop Constructs**: depth += 1 inside loop body
```rust
fn example() {
    loop {
        statement;      // depth 1
        if x {
            statement;  // depth 2
        }
    }
}
```

---

## Edge Cases

### Case 1: Empty Blocks

**R65 Source:**
```rust
if condition {
    // Empty block
}
```

**Generated:**
```asm
    ; | if condition {
    LDA condition
    BEQ skip
    ; | }
skip:
```

### Case 2: Inline Assembly

**R65 Source:**
```rust
fn wait() {
    asm!("WAI");
}
```

**Generated:**
```asm
wait:
    ; | fn wait() {
    ; |     asm!("WAI");
    WAI             ; Inline assembly
    RTS
    ; | }
```

### Case 3: Compiler-Generated Code

**Mode Transitions** (compiler-generated for m16 functions):
```asm
    ; | fn process(value @ A: u16) {  // m16 mode inferred from @ A: u16
    REP #$20        ; Compiler-generated (switch to 16-bit A)
    ; |     statement;
    LDA #$1234
    ; | }
    SEP #$20        ; Compiler-generated (restore to 8-bit A)
    RTS
```

**Strategy**: Emit source line for user-written code only. Compiler-generated mode transitions get standard comments.

---

## Configuration Options

### Command-Line Flags

```bash
# Enable source interleaving
--interleave-source, -s

# Control indentation width (default: 4 spaces)
--indent-width=2

# Show only function headers as source
--source-headers-only

# Disable all source interleaving
--no-source
```

### Configuration File

**r65.toml** (future):
```toml
[codegen]
interleave_source = true
indent_width = 4
source_headers_only = false
```

---

## Benefits

### For Developers

1. **Debugging**: See exactly what source generated what assembly
2. **Learning**: Understand compiler output by comparing source and assembly
3. **Optimization**: Identify inefficiencies by examining generated code
4. **Verification**: Confirm compiler correctness

### For Reverse Engineering

1. **Documentation**: Self-documenting assembly for disassembled ROMs
2. **Annotation**: Original intent visible alongside generated code
3. **Modification**: Easier to understand what to change when patching

### For Education

1. **Teaching**: Show students how high-level constructs map to assembly
2. **Learning 65816**: See idiomatic assembly patterns
3. **Compiler Understanding**: Demystify compilation process

---

## Performance Considerations

**Compile Time:**
- Minimal impact (< 5% slowdown)
- Source line cache is O(1) lookup
- Indentation calculation is O(1) per instruction

**Output Size:**
- Source interleaving increases file size by ~30-50%
- Each source line adds ~40-60 bytes (comment overhead)
- Trade-off: readability vs file size

**Recommendation**: Enable for development builds, disable for production/release builds.

---

## Future Enhancements

### Enhanced Source Mapping

**Source Maps** (JSON format):
```json
{
  "version": 1,
  "file": "program.r65",
  "mappings": [
    {"source_line": 42, "asm_lines": [100, 101, 102]},
    {"source_line": 43, "asm_lines": [103, 104]}
  ]
}
```

### Interactive Debugging

**Debugger Integration:**
- Map assembly breakpoints back to R65 source
- Step through R65 source while viewing assembly
- Variable inspection at source level

### Optimization Remarks

**Compiler Annotations:**
```asm
    ; | let sum = a + b;
    ; OPTIMIZATION: Loop unrolled 4x
    LDA a
    CLC
    ADC b
```

---

## Implementation Checklist

- [ ] Add `--interleave-source` flag to CLI
- [ ] Implement `SourceLineCache` class
- [ ] Implement `CommentExtractor` class
- [ ] Add source location tracking to AST nodes
- [ ] Add comment tracking to AST nodes (`attached_comment`, `preceding_comments`)
- [ ] Update parser to extract and attach comments
- [ ] Add control flow depth calculation to MIR builder
- [ ] Propagate comments from AST to MIR nodes
- [ ] Enhance `AssemblyEmitter` with indentation tracking
- [ ] Enhance `AssemblyEmitter.emit_instruction()` to accept `r65_comment` parameter
- [ ] Modify `InstructionSelector` to emit source lines and comments
- [ ] Update all `_emit_*` methods to accept and use `r65_comment`
- [ ] Add indentation mirroring for control flow
- [ ] Handle edge cases (empty blocks, inline asm, compiler-generated)
- [ ] Write tests for source interleaving
- [ ] Write tests for comment extraction and preservation
- [ ] Document in user guide
- [ ] Add examples to documentation

---

## Example Full Output

**R65 Source (complete.r65):**
```rust
#[zeropage(0x20)]
static mut COUNTER: u8 = 0;

// m8 mode (default)
fn increment() {
    // Load current counter value
    let value @ A = COUNTER;

    // Don't overflow past 255
    if value < 255 {
        value = value + 1;  // Increment
        COUNTER = value;    // Save back
    }
}
```

**Generated Assembly (with --interleave-source):**
```asm
; ============================================================================
; Generated by R65 Compiler
; Source: complete.r65
; ============================================================================

.65816

; ============================================================================
; Zero-Page Allocations
; ============================================================================
.DEFINE COUNTER $20

; ============================================================================
; Bank 0 - Main Code
; ============================================================================
.BANK 0 SLOT 0
.ORG 0

; ============================================================================
; Function: increment
; Source: complete.r65:5
; Mode: m8 (default)
; ============================================================================
increment:
; | // m8 mode (default)
; | fn increment() {
; |     // Load current counter value
; |     let value @ A = COUNTER;
LDA COUNTER         ; Load current counter value
; |
; |     // Don't overflow past 255
; |     if value < 255 {
CMP #255            ; Don't overflow past 255
BCS end_if
    ; |         value = value + 1;  // Increment
    INC A               ; Increment
    ; |         COUNTER = value;    // Save back
    STA COUNTER         ; Save back
; |     }
end_if:
RTS
; | }
```

---

## Alternative Format Options

The default **Vertical Pipe** format (`; |`) can be changed via command-line flags. Available formats:

| Format | Flag | Best For |
|--------|------|----------|
| **Vertical Pipe** (default) | `--format=pipe` | General development, assembles directly |
| **Side-by-Side** | `--format=sidebyside` | Documentation, learning (40-char columns) |
| **Block Sections** | `--format=blocks` | Teaching, detailed debugging |
| **Nested Indent** | `--format=nested` | Understanding compiler transformations |
| **Statement IDs** | `--format=debug` | Debuggers, precise traceability |
| **Compact Reference** | `--format=minimal` | Space-constrained output |

### Side-by-Side Example

```asm
fn calculate(a @ A: u8, b @ X: u8)  |  calculate:
    -> u8 {                          |
    let sum @ A = a + b; // Add      |      STX $16        ; Save b
                                     |      LDA $16        ; Load b
                                     |      CLC            ; Clear carry
    return sum;                      |      RTS
}                                    |
```

### Statement ID Example

```asm
; [S1] fn calculate(a @ A: u8, b @ X: u8) -> u8 {
calculate:
; [S2] let sum @ A = a + b;
    STX $16             ; [S2] Save b
    LDA $16             ; [S2] Load b
    CLC                 ; [S2] Clear carry
; [S3] return sum;
    RTS                 ; [S3]
```

**Note**: Only the Vertical Pipe format produces directly-assemblable output. Other formats are for documentation and debugging.

---

**Status**: Design Complete - Ready for Implementation
**Last Updated**: 2026-01-02
