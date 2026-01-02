# Parser Fix: Named Attribute Arguments

## Summary

**Fixed:** Parser now supports named arguments in attributes (`name=value` syntax)

## Problem

The parser only supported positional arguments in attributes:

```rust
// ✅ Worked before
#[mode(m8, x8)]

// ❌ Failed before - parsed 'transition=auto' as assignment expression
#[mode(m8, x8, transition=auto)]
```

**Error:** `HIR error: Expected identifier, got Assignment`

## Root Cause

### Grammar Issue

```lark
# OLD (wrong)
attribute_arg: expr
```

This treated `transition=auto` as an assignment expression, creating:
```python
AttributeArg(name=None, value=Assignment(target='transition', value='auto'))
```

### Transformer Issue

The transformer created all `AttributeArg` objects with `name=None`, even though the HIR processor expected an optional name field.

## Solution

### 1. Updated Grammar

```lark
# NEW (correct)
attribute_arg: (IDENT "=")? expr  -> attribute_arg
```

This makes the `IDENT "="` part optional:
- With name: `transition=auto` → captures IDENT + expr
- Without name: `m8` → captures just expr

### 2. Updated Transformer

```python
def attribute_arg(self, items):
    """
    Attribute argument - can be named or positional.

    Grammar: (IDENT "=")? expr

    Returns AttributeArg with optional name.
    """
    items = self._filter_tokens(items, keep_types={'IDENT'})

    # Check if we have a name (IDENT token followed by expression)
    if len(items) == 2 and isinstance(items[0], LarkToken):
        # Named argument: name=value
        name = items[0].value
        value = items[1]
        return ast.AttributeArg(name=name, value=value)
    else:
        # Positional argument: just value
        value = items[0]
        return ast.AttributeArg(name=None, value=value)
```

Now creates correct `AttributeArg` structure:
- Named: `AttributeArg(name='transition', value=Identifier('auto'))`
- Positional: `AttributeArg(name=None, value=Identifier('m8'))`

## Files Modified

### Grammar
- **`r65/compiler/frontend/grammar.lark`**
  - Line 58: Changed `attribute_arg: expr` to `attribute_arg: (IDENT "=")? expr -> attribute_arg`

### Parser/Transformer
- **`r65/compiler/frontend/parser.py`**
  - Updated `attribute_arg()` method to handle named arguments
  - Updated `attribute_args()` to pass through `AttributeArg` objects

## Testing

### Test 1: Named Arguments ✅
```rust
#[mode(m8, x8, transition=auto)]
```

**Parsed:**
- `[0]` name=None, value=Identifier('m8')
- `[1]` name=None, value=Identifier('x8')
- `[2]` name='transition', value=Identifier('auto')

### Test 2: Backward Compatibility ✅
```rust
#[mode(m8, x8)]
```

**Parsed:**
- `[0]` name=None, value=Identifier('m8')
- `[1]` name=None, value=Identifier('x8')

### Test 3: Interrupt Handler ✅
```rust
#[interrupt(nmi)]
#[mode(m8, x8, transition=auto)]
fn nmi_handler() {
    return;
}
```

**Result:** Builds MIR successfully with 16 instructions (entry/exit wrapper)

### Test 4: transition=caller ✅
```rust
#[mode(m16, x16, transition=caller)]
fn process() { }

#[mode(m8, x8)]
fn caller() {
    process();  // Generates PHP/REP/Call/PLP wrapper
}
```

**Result:** Generates correct caller-side wrapper

### Test 5: transition=caller + preserves(STATUS) ✅
```rust
#[mode(m16, x16, transition=caller)]
#[preserves(STATUS)]
fn process() { }

#[mode(m8, x8)]
fn caller() {
    process();  // Generates REP/Call/SEP wrapper (no PHP/PLP)
}
```

**Result:** Generates correct SEP/REP wrapper without stack operations

## Examples Working

### Interrupt Handlers
```rust
#[interrupt(nmi)]
#[mode(m8, x8, transition=auto)]
fn nmi_handler() {
    A = 0x42;
    FLAG = A;
    return;
}
```

**Generated MIR:**
```
 0: PHP          ; ─┐
 1: PHA          ;  │
 2: PHX          ;  │ Entry wrapper
 3: PHY          ;  │
 4: PHD          ;  │
 5: PHB          ; ─┤
 6: SEP #$30     ; ─┘ Force mode
 7-8: [body]     ;    Handler code
 9: PLB          ; ─┐
10: PLD          ;  │
11: PLY          ;  │ Exit wrapper
12: PLX          ;  │
13: PLA          ;  │
14: PLP          ; ─┤ Restore mode
15: RTI          ; ─┘ Return
```

### Mode Transitions
```rust
#[mode(m16, x16, transition=caller)]
fn needs_16bit() {
    A = 0x1234;
}

#[mode(m8, x8)]
fn caller() {
    needs_16bit();  // Auto-wrapped with PHP/REP/Call/PLP
}
```

**Generated MIR for caller:**
```
0: PHP              ; Save STATUS
1: REP #$30         ; Switch to m16/x16
2: Call needs_16bit
3: PLP              ; Restore STATUS
```

## Benefits

1. **Self-Documenting Code**: `transition=auto` is visible in source
2. **Type Safety**: Named arguments prevent errors
3. **Flexibility**: Mix positional and named arguments
4. **Backward Compatible**: Old code still works

## Validation Now Working

### Interrupt + Mode Requires transition=auto
```rust
// ❌ ERROR
#[interrupt(nmi)]
#[mode(m8, x8)]
fn handler() { }

// ✅ CORRECT
#[interrupt(nmi)]
#[mode(m8, x8, transition=auto)]
fn handler() { }
```

### transition=auto + preserves(STATUS) Conflicts
```rust
// ❌ ERROR
#[mode(m8, x8, transition=auto)]
#[preserves(STATUS)]
fn helper() { }
```

**Error:** `transition=auto requires modifying STATUS, conflicts with preservation`

## Full Pipeline Working

1. **Parser** ✅ - Parses `transition=auto` correctly
2. **HIR Builder** ✅ - Processes named arguments correctly
3. **Type Checker** ✅ - Validates interrupt handler rules
4. **MIR Builder** ✅ - Generates correct wrappers

## Test Files

### Working Examples
- `examples/interrupt_simple_test.r65` - Interrupt with transition=auto
- `examples/transition_caller_working.r65` - transition=caller tests
- `examples/mixed_mode_test.r65` - transition=none (backward compat)

### Test Scripts
- `test_named_attributes.py` - Parser unit tests
- `test_interrupt_mir.py` - Interrupt handler MIR verification
- `test_transition_caller.py` - Caller wrapper verification

## Status

✅ **COMPLETE** - Parser fully supports named attribute arguments

All mode transition features now functional:
- `transition=none` (default)
- `transition=auto` (interrupt handlers)
- `transition=caller` (explicit mode management)

---

**Last Updated:** 2026-01-01
**Status:** Complete and tested
