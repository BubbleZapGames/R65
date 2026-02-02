"""
Virtual register allocation for MIR.

Virtual registers are unlimited during MIR construction.
They are mapped to scratch registers or stack during code generation.
"""

from r65.compiler.mir.nodes import *
from typing import Optional, Any


class VirtualRegisterAllocator:
    """
    Allocates virtual registers (unlimited).

    Virtual registers are placeholders that will be mapped to:
    - Scratch registers (designated zero-page locations)
    - Stack locations
    during code generation.

    The allocator simply assigns unique IDs - actual hardware allocation
    happens later in the code generation phase.
    """

    def __init__(self):
        self.next_id = 0

    def alloc(self, type_info: Any, hint: Optional[str] = None,
              register_hint: Optional[str] = None) -> VirtualRegister:
        """
        Allocate a new virtual register.

        Args:
            type_info: TypeInfo for size tracking
            hint: Optional name hint for debugging
            register_hint: Optional hardware register hint ('X', 'Y') for loop variables

        Returns:
            New VirtualRegister with unique ID
        """
        vreg = VirtualRegister(
            id=self.next_id,
            type_info=type_info,
            hint=hint,
            register_hint=register_hint
        )
        self.next_id += 1
        return vreg

    def reset(self):
        """Reset allocator (for testing or new function)."""
        self.next_id = 0
