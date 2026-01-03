# Type Checker Refactoring Analysis

**Date**: 2026-01-02
**File**: `r65/compiler/typeck/type_checker.py`
**Size**: 689 lines
**Status**: Analysis Complete

## Executive Summary

The type_checker.py file is **well-architected** with good delegation to specialized classes. However, there are **3 minor improvement opportunities** that could reduce duplication and improve maintainability:

1. **Function lookup pattern** - duplicated 2 times (~10 lines)
2. **Mode retrieval pattern** - duplicated 2 times (~8 lines)
3. **Type validation helpers** - repeated validation patterns (~15 lines)

**Total potential savings**: ~30-35 lines
**Priority**: 🟡 LOW-MEDIUM (nice-to-have improvements)

---

## Issue 1: Function Declaration Lookup Duplication

### Problem

The pattern for looking up a function declaration from the program appears twice:

**Location 1**: `check_function_call()` lines 443-448
```python
# Look up HIR function declaration from program
func_decl = None
for decl in self.program.declarations:
    if isinstance(decl, HIRFunctionDecl) and decl.name == func_symbol.name:
        func_decl = decl
        break

if not func_decl:
    raise TypeCheckError(...)
```

**Location 2**: `check_function_address()` lines 523-527
```python
# Find function declaration
func_decl = None
for decl in self.program.declarations:
    if isinstance(decl, HIRFunctionDecl) and decl.name == func_symbol.name:
        func_decl = decl
        break

if not func_decl:
    raise TypeCheckError(...)
```

### Proposed Solution

Extract to helper method:

```python
def _lookup_function_decl(self, func_name: str, source_loc=None) -> HIRFunctionDecl:
    """
    Look up function declaration by name.

    Args:
        func_name: Name of function to find
        source_loc: Source location for error reporting

    Returns:
        HIRFunctionDecl

    Raises:
        TypeCheckError: If function not found
    """
    for decl in self.program.declarations:
        if isinstance(decl, HIRFunctionDecl) and decl.name == func_name:
            return decl

    raise TypeCheckError(
        f"Function '{func_name}' not found",
        source_loc=source_loc
    )
```

**Usage**:
```python
# In check_function_call
func_decl = self._lookup_function_decl(func_symbol.name, expr.source_loc)

# In check_function_address
func_decl = self._lookup_function_decl(func_symbol.name, expr.source_loc)
```

**Impact**:
- Lines saved: ~10 lines
- Maintainability: ⭐⭐⭐⭐ - Single source of truth for function lookup
- Consistency: Ensures same error messages everywhere

---

## Issue 2: Mode Retrieval Pattern Duplication

### Problem

The pattern for getting the current mode (from mode_tracker or fallback to current_mode) appears twice:

**Location 1**: `check_let_statement()` lines 177-180
```python
# Get mode at this statement
if self.mode_tracker:
    mode = self.mode_tracker.get_mode_at_statement(stmt)
else:
    mode = self.current_mode
```

**Location 2**: `check_expression()` for HIRRegister, lines 264-267
```python
# Get register type from current mode
if self.mode_tracker:
    mode = self.mode_tracker.get_mode_at_statement(expr)
else:
    mode = self.current_mode
```

### Proposed Solution

Extract to helper method:

```python
def _get_mode_at(self, stmt_or_expr) -> ProcessorMode:
    """
    Get processor mode at a statement or expression.

    Args:
        stmt_or_expr: Statement or expression node

    Returns:
        ProcessorMode (from mode tracker if available, else current mode)
    """
    if self.mode_tracker:
        return self.mode_tracker.get_mode_at_statement(stmt_or_expr)
    return self.current_mode
```

**Usage**:
```python
# In check_let_statement
mode = self._get_mode_at(stmt)

# In check_expression (HIRRegister case)
mode = self._get_mode_at(expr)
```

**Impact**:
- Lines saved: ~6 lines
- Clarity: ⭐⭐⭐⭐ - Clear intent with descriptive name
- Consistency: ⭐⭐⭐⭐ - Uniform mode retrieval

---

## Issue 3: Type Validation Helper Opportunities

### Problem

Several places validate types and raise similar errors:

**Pattern A**: Boolean type validation (appears 5 times)
- Lines 150-154: If condition must be boolean
- Lines 167-171: While condition must be boolean
- Lines 348-352: Left operand of && must be bool
- Lines 353-357: Right operand of || must be bool
- Lines 374-378: Operand of ! must be bool

**Pattern B**: Integer type validation (appears 2 times)
- Lines 383-387: Operand of ~ must be integer
- Lines 392-396: Operand of - must be integer
- Lines 608-612: Array index must be integer

### Proposed Solution

Extract validation helpers:

```python
def _require_boolean_type(self, expr_type: TypeInfo, context: str, source_loc=None):
    """
    Validate that a type is boolean, raise error if not.

    Args:
        expr_type: Type to check
        context: Context string for error message (e.g., "if condition")
        source_loc: Source location for error

    Raises:
        TypeCheckError: If type is not boolean
    """
    if not TypeUtils.is_boolean_type(expr_type):
        raise TypeCheckError(
            f"{context} must be boolean, found {expr_type}",
            source_loc=source_loc
        )

def _require_integer_type(self, expr_type: TypeInfo, context: str, source_loc=None):
    """
    Validate that a type is integer, raise error if not.

    Args:
        expr_type: Type to check
        context: Context string for error message (e.g., "array index")
        source_loc: Source location for error

    Raises:
        TypeCheckError: If type is not integer
    """
    if not TypeUtils.is_integer_type(expr_type):
        raise TypeCheckError(
            f"{context} must be integer, found {expr_type}",
            source_loc=source_loc
        )
```

**Usage**:
```python
# In check_statement (if statement)
cond_type = self.check_expression(stmt.condition)
self._require_boolean_type(cond_type, "If condition", stmt.condition.source_loc)

# In check_statement (while statement)
cond_type = self.check_expression(stmt.condition)
self._require_boolean_type(cond_type, "While condition", stmt.condition.source_loc)

# In check_binary_op (logical operators)
left_type = self.check_expression(expr.left)
self._require_boolean_type(left_type, f"Left operand of '{expr.op}'", expr.left.source_loc)

# In check_unary_op (! operator)
operand_type = self.check_expression(expr.operand)
self._require_boolean_type(operand_type, "Operand of '!'", expr.operand.source_loc)

# In check_unary_op (~ and - operators)
operand_type = self.check_expression(expr.operand)
self._require_integer_type(operand_type, f"Operand of '{expr.op}'", expr.operand.source_loc)

# In check_array_index
index_type = self.check_expression(expr.index)
self._require_integer_type(index_type, "Array index", expr.index.source_loc)
```

**Impact**:
- Lines saved: ~15 lines
- Consistency: ⭐⭐⭐⭐⭐ - Uniform error messages
- Maintainability: ⭐⭐⭐⭐ - Easy to update error format

---

## Non-Issues (Already Well-Designed)

### Delegation to Specialized Classes ✅

The type checker properly delegates to:
- `ModeTracker` - Mode analysis
- `CFGBuilder` - Control flow graphs
- `TypeUtils` - Type utilities
- `OperatorValidator` - Operator validation
- `PreservationChecker` - Register preservation
- `TypeInference` - Type inference

This is **excellent architecture** - no changes needed.

### Expression Type Checking Dispatch ✅

The `check_expression()` method uses a clear if-elif dispatch pattern that's appropriate for this use case. While it could theoretically use a dictionary dispatch, the current approach is:
- Clear and readable
- Easy to debug
- Handles type checking context properly
- Not worth refactoring

---

## Summary of Proposed Improvements

| Improvement | Lines Saved | Priority | Complexity |
|-------------|-------------|----------|------------|
| Function lookup helper | ~10 lines | 🟡 MEDIUM | Easy |
| Mode retrieval helper | ~6 lines | 🟡 MEDIUM | Easy |
| Type validation helpers | ~15 lines | 🟡 MEDIUM | Easy |
| **TOTAL** | **~30 lines** | - | - |

**Total Potential Reduction**: ~30 lines (~4% of file)

---

## Implementation Priority

### Recommended Order

1. **Function lookup helper** (highest value)
   - Used in two critical paths
   - Improves error consistency
   - Simple extraction

2. **Mode retrieval helper** (medium value)
   - Clarifies mode tracking logic
   - Simple extraction

3. **Type validation helpers** (nice-to-have)
   - Improves error message consistency
   - Multiple call sites make it worthwhile
   - Easy to implement

---

## Testing Strategy

- Run all existing tests after each refactoring
- Verify error messages remain clear and helpful
- No behavioral changes expected

---

## Conclusion

The type_checker.py file is **well-architected** with proper separation of concerns. The proposed refactorings are **minor quality improvements** that would:

1. Reduce duplication (~30 lines saved)
2. Improve consistency (especially error messages)
3. Make future maintenance easier

**Recommendation**: These are **optional improvements** - the code is already high quality. Implement if doing general code cleanup, but not urgent.

**Priority Level**: 🟡 LOW-MEDIUM (after HIGH and MEDIUM priorities in other files)
