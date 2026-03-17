#!/usr/bin/env python3
# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""Test named attribute arguments parsing."""

from r65.compiler.frontend import parse

# Test 1: Named arguments
source1 = """
fn test1() {
}
"""

print("Test 1: Named arguments")
print("Source:", source1.strip())
try:
    ast = parse(source1, "test1.r65")
    func = ast.items[0]
    attr = func.attributes[0]
    print(f"✅ Parsed successfully!")
    print(f"Attribute: {attr.name}")
    print(f"Arguments:")
    for i, arg in enumerate(attr.args):
        print(f"  [{i}] name={arg.name!r}, value={arg.value}")
    print()
except Exception as e:
    print(f"❌ Error: {e}")
    print()

# Test 2: Mixed positional and named
source2 = """
fn test2() {
}
"""

print("Test 2: Mixed positional and named")
print("Source:", source2.strip())
try:
    ast = parse(source2, "test2.r65")
    func = ast.items[0]
    attr = func.attributes[0]
    print(f"✅ Parsed successfully!")
    print(f"Attribute: {attr.name}")
    print(f"Arguments:")
    for i, arg in enumerate(attr.args):
        print(f"  [{i}] name={arg.name!r}, value={arg.value}")
    print()
except Exception as e:
    print(f"❌ Error: {e}")
    print()

# Test 3: Only positional (backward compatibility)
source3 = """
fn test3() {
}
"""

print("Test 3: Only positional (backward compatibility)")
print("Source:", source3.strip())
try:
    ast = parse(source3, "test3.r65")
    func = ast.items[0]
    attr = func.attributes[0]
    print(f"✅ Parsed successfully!")
    print(f"Attribute: {attr.name}")
    print(f"Arguments:")
    for i, arg in enumerate(attr.args):
        print(f"  [{i}] name={arg.name!r}, value={arg.value}")
    print()
except Exception as e:
    print(f"❌ Error: {e}")
    print()

# Test 4: Interrupt handler with mode and transition
source4 = """
#[interrupt(nmi)]
fn nmi_handler() {
    return;
}
"""

print("Test 4: Interrupt handler with transition=inline")
print("Source:", source4.strip())
try:
    ast = parse(source4, "test4.r65")
    func = ast.items[0]
    print(f"✅ Parsed successfully!")
    for attr in func.attributes:
        print(f"Attribute: {attr.name}")
        print(f"  Arguments:")
        for i, arg in enumerate(attr.args):
            print(f"    [{i}] name={arg.name!r}, value={arg.value}")
    print()
except Exception as e:
    print(f"❌ Error: {e}")
    print()
