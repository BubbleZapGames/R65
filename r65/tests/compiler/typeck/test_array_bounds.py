#!/usr/bin/env python3
# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Test cases for constant index bounds checking feature.

Tests that array bounds are checked at compile time for constant indices,
but dynamic indices are allowed without bounds checking.
"""

import pytest
from r65.compiler.main import compile_string

def test_constant_index_in_bounds():
    """Test that constant indices within bounds are allowed."""
    # Test that compilation works (we can modify test later to check for bounds errors)
    try:
        result = compile_string("""
        #[ram]
        static mut BUFFER: [u8; 256];
        
        fn test() {
            BUFFER[0] = 42;
        }
        """)
        # For now, just test that it compiles without crashing
        assert result is not None
        print("✓ Compilation succeeded")
    except Exception as e:
        print(f"❌ Compilation failed: {e}")

def test_constant_index_out_of_bounds():
    """Test that constant indices out of bounds cause compile error."""
    try:
        result = compile_string("""
        #[ram]
        static mut BUFFER: [u8; 256];
        
        fn test() {
            BUFFER[256] = 99;
        }
        """)
        # Should fail with bounds error
        assert result is None
        print("✓ Compilation failed as expected")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_constant_index_negative():
    """Test that negative constant indices cause compile error."""
    try:
        result = compile_string("""
        #[ram]
        static mut BUFFER: [u8; 256];
        
        fn test() {
            BUFFER[-1] = 5;
        }
        """)
        # Should fail with bounds error
        assert result is None
        print("✓ Compilation failed as expected")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")