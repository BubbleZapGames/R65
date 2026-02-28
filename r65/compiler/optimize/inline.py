"""
Function Inlining Optimization Pass.

Replaces call sites with inlined function bodies to eliminate JSR/RTS overhead
(12 cycles) and JSL/RTL overhead (14 cycles).

The inlining pass operates on MIR after HIR lowering, before code generation.
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from copy import deepcopy

from r65.compiler.mir.nodes import (
    MIRProgram, MIRFunction, BasicBlock,
    MIRInstruction, VirtualRegister, HardwareRegister, Immediate,
    Move, Jump, Return, Call, CondBranch, JumpTable, LookupTable,
    Load, Store, LoadIndirect, StoreIndirect,
    BinaryOp, UnaryOp, Compare, TypeConvert, ToBool,
    InlineAsm, ReturnFromInterrupt, TraitDispatch,
    Argument, ArgumentMechanism,
)
from r65.compiler.hir import RegisterBinding, VariableBinding
from r65.compiler.hir.attributes import InlineMode


# Size thresholds for inlining decisions
INLINE_THRESHOLD_WITH_ATTR = 30   # Max instructions for #[inline] functions
INLINE_THRESHOLD_NO_ATTR = 3      # Max instructions for unmarked functions (very conservative)


class InlinabilityChecker:
    """
    Determines whether a function can and should be inlined.

    Hard requirements (must all be true):
    1. Not recursive (direct or mutual)
    2. Not a far fn (cross-bank calls can't be inlined)
    3. Not an interrupt handler
    4. Not the entry point
    5. No inline assembly (asm!())

    Heuristics for inlining decisions:
    - Explicit inlining (always checked):
      - Marked #[inline] or #[inline(always)] → inline if < 30 instructions
    - Implicit inlining (only when implicit_inline=True, i.e., -O2):
      - Called exactly once → always inline (no code size increase)
      - No attribute → inline only if very small (< 3 instructions)

    Note: Trivial getters/setters are auto-marked with #[inline] at HIR level.
    """

    def __init__(self, mir_program: MIRProgram, implicit_inline: bool = True):
        """
        Initialize the inlinability checker.

        Args:
            mir_program: The MIR program to analyze
            implicit_inline: If True, allow implicit inlining (called-once and small
                           functions without #[inline] attribute). If False, only
                           inline functions with explicit #[inline] or #[inline(always)].
        """
        self.mir_program = mir_program
        self.implicit_inline = implicit_inline
        self.func_map: Dict[str, MIRFunction] = {f.name: f for f in mir_program.functions}
        self.call_graph: Dict[str, Set[str]] = {}
        self.call_counts: Dict[str, int] = {}
        self._build_call_graph()
        self._count_calls()
        self._find_recursive_functions()

    def _build_call_graph(self):
        """Build a call graph mapping function names to called functions."""
        for func in self.mir_program.functions:
            called = set()
            for block in func.blocks.values():
                for instr in block.instructions:
                    if isinstance(instr, Call) and isinstance(instr.function, str):
                        called.add(instr.function)
            self.call_graph[func.name] = called

    def _count_calls(self):
        """Count how many times each function is called."""
        self.call_counts = {f.name: 0 for f in self.mir_program.functions}
        for func in self.mir_program.functions:
            for block in func.blocks.values():
                for instr in block.instructions:
                    if isinstance(instr, Call) and isinstance(instr.function, str):
                        if instr.function in self.call_counts:
                            self.call_counts[instr.function] += 1

    def _find_recursive_functions(self):
        """Find all functions involved in recursion (direct or mutual)."""
        self.recursive_functions: Set[str] = set()

        # Check for direct recursion
        for func_name, called in self.call_graph.items():
            if func_name in called:
                self.recursive_functions.add(func_name)

        # Check for mutual recursion using transitive closure
        for start_func in self.call_graph:
            visited = set()
            stack = list(self.call_graph.get(start_func, set()))

            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)

                if current == start_func:
                    self.recursive_functions.add(start_func)
                    break

                stack.extend(self.call_graph.get(current, set()))

    def _count_instructions(self, func: MIRFunction) -> int:
        """Count the total number of instructions in a function."""
        count = 0
        for block in func.blocks.values():
            count += len(block.instructions)
        return count

    def _has_inline_asm(self, func: MIRFunction) -> bool:
        """Check if function contains inline assembly."""
        for block in func.blocks.values():
            for instr in block.instructions:
                if isinstance(instr, InlineAsm):
                    return True
        return False

    def can_inline(self, func_name: str) -> bool:
        """
        Check if a function can be inlined (hard requirements).

        Returns:
            True if the function passes all hard requirements for inlining.
        """
        if func_name not in self.func_map:
            return False

        func = self.func_map[func_name]

        # Not recursive
        if func_name in self.recursive_functions:
            return False

        # Not a far function
        if func.is_far:
            return False

        # Not an interrupt handler
        if func.interrupt_attr is not None:
            return False

        # Not an entry point
        if func.is_entry:
            return False

        # Not __init_start
        if func_name == "__init_start":
            return False

        # No inline assembly
        if self._has_inline_asm(func):
            return False

        return True

    def should_inline(self, func_name: str) -> bool:
        """
        Check if a function should be inlined (heuristics).

        Returns:
            True if the function should be inlined based on heuristics.
        """
        if not self.can_inline(func_name):
            return False

        func = self.func_map[func_name]
        instr_count = self._count_instructions(func)

        # Check for #[inline(never)] - never inline these functions
        if func.inline_attr is not None and func.inline_attr.mode == InlineMode.NEVER:
            return False

        # Marked #[inline] or #[inline(always)] → inline if under threshold (explicit inlining)
        if func.inline_attr is not None:
            return instr_count < INLINE_THRESHOLD_WITH_ATTR

        # Below this point is implicit inlining - only allowed when implicit_inline=True
        if not self.implicit_inline:
            return False

        # Called exactly once → always inline (unless #[inline(never)] which is handled above)
        if self.call_counts.get(func_name, 0) == 1:
            return True

        # No attribute → inline only if very small
        return instr_count < INLINE_THRESHOLD_NO_ATTR


class BlockCloner:
    """
    Clones basic blocks with fresh virtual register and block IDs.

    Handles remapping of:
    - Virtual register IDs
    - Block IDs (for jump targets)
    - Preserves hardware register references
    """

    def __init__(self, caller_func: MIRFunction, callee_func: MIRFunction):
        """
        Initialize the block cloner.

        Args:
            caller_func: The function receiving the inlined code
            callee_func: The function being inlined
        """
        self.caller_func = caller_func
        self.callee_func = callee_func
        self.vreg_map: Dict[int, VirtualRegister] = {}
        self.block_map: Dict[int, int] = {}

    def _get_next_block_id(self) -> int:
        """Get the next available block ID in the caller function."""
        if not self.caller_func.blocks:
            return 0
        return max(self.caller_func.blocks.keys()) + 1

    def _remap_vreg(self, vreg: VirtualRegister) -> VirtualRegister:
        """
        Get or create a remapped virtual register.

        Args:
            vreg: Original virtual register from callee

        Returns:
            New virtual register in caller's address space
        """
        if vreg.id not in self.vreg_map:
            new_vreg = self.caller_func.vreg_allocator.alloc(
                vreg.type_info,
                f"inlined_{vreg.hint}" if vreg.hint else None
            )
            self.vreg_map[vreg.id] = new_vreg
        return self.vreg_map[vreg.id]

    def _remap_operand(self, operand):
        """
        Remap an operand, handling virtual registers.

        Args:
            operand: Any MIR operand

        Returns:
            Remapped operand
        """
        if isinstance(operand, VirtualRegister):
            return self._remap_vreg(operand)
        elif isinstance(operand, (HardwareRegister, Immediate)):
            return operand
        else:
            # Memory locations and other operands pass through
            return operand

    def _remap_block_id(self, block_id: int) -> int:
        """
        Get or create a remapped block ID.

        Args:
            block_id: Original block ID from callee

        Returns:
            New block ID in caller's block space
        """
        if block_id not in self.block_map:
            # Find a new block ID that doesn't conflict
            new_id = self._get_next_block_id()
            # Get existing mapped values (excluding the one we're about to add)
            existing_mapped = set(self.block_map.values())
            # Find an ID that's not in caller's blocks and not already mapped
            while new_id in self.caller_func.blocks or new_id in existing_mapped:
                new_id += 1
            self.block_map[block_id] = new_id
        return self.block_map[block_id]

    def _clone_instruction(self, instr: MIRInstruction) -> MIRInstruction:
        """
        Clone an instruction with remapped operands.

        Args:
            instr: Original instruction from callee

        Returns:
            Cloned instruction with remapped operands
        """
        # Deep copy first to handle any nested structures
        cloned = deepcopy(instr)

        # Remap based on instruction type
        if isinstance(cloned, Move):
            cloned.dest = self._remap_operand(cloned.dest)
            cloned.source = self._remap_operand(cloned.source)

        elif isinstance(cloned, Load):
            cloned.dest = self._remap_operand(cloned.dest)
            # MemoryLocation passes through

        elif isinstance(cloned, Store):
            cloned.source = self._remap_operand(cloned.source)
            # MemoryLocation passes through

        elif isinstance(cloned, LoadIndirect):
            cloned.dest = self._remap_operand(cloned.dest)
            cloned.pointer = self._remap_operand(cloned.pointer)

        elif isinstance(cloned, StoreIndirect):
            cloned.source = self._remap_operand(cloned.source)
            cloned.pointer = self._remap_operand(cloned.pointer)

        elif isinstance(cloned, BinaryOp):
            cloned.dest = self._remap_operand(cloned.dest)
            cloned.left = self._remap_operand(cloned.left)
            cloned.right = self._remap_operand(cloned.right)

        elif isinstance(cloned, UnaryOp):
            cloned.dest = self._remap_operand(cloned.dest)
            cloned.operand = self._remap_operand(cloned.operand)

        elif isinstance(cloned, Compare):
            cloned.left = self._remap_operand(cloned.left)
            cloned.right = self._remap_operand(cloned.right)

        elif isinstance(cloned, TypeConvert):
            cloned.dest = self._remap_operand(cloned.dest)
            cloned.source = self._remap_operand(cloned.source)

        elif isinstance(cloned, ToBool):
            cloned.dest = self._remap_operand(cloned.dest)
            cloned.source = self._remap_operand(cloned.source)

        elif isinstance(cloned, Jump):
            cloned.target = self._remap_block_id(cloned.target)

        elif isinstance(cloned, CondBranch):
            cloned.condition = self._remap_operand(cloned.condition)
            cloned.true_target = self._remap_block_id(cloned.true_target)
            cloned.false_target = self._remap_block_id(cloned.false_target)

        elif isinstance(cloned, JumpTable):
            cloned.scrutinee = self._remap_operand(cloned.scrutinee)
            cloned.targets = [self._remap_block_id(t) for t in cloned.targets]
            cloned.default_target = self._remap_block_id(cloned.default_target)

        elif isinstance(cloned, LookupTable):
            cloned.dest = self._remap_operand(cloned.dest)
            cloned.scrutinee = self._remap_operand(cloned.scrutinee)
            cloned.merge_target = self._remap_block_id(cloned.merge_target)

        elif isinstance(cloned, Call):
            # Remap arguments
            for arg in cloned.args:
                arg.value = self._remap_operand(arg.value)
            # Remap return registers
            cloned.returns = [self._remap_operand(r) for r in cloned.returns]

        elif isinstance(cloned, Return):
            cloned.values = [self._remap_operand(v) for v in cloned.values]

        # Other instructions (SetMode, Push, Pull, etc.) pass through mostly unchanged

        return cloned

    def clone_blocks(self) -> Dict[int, BasicBlock]:
        """
        Clone all blocks from the callee with remapped IDs.

        Returns:
            Dictionary of new block ID to cloned BasicBlock
        """
        cloned_blocks = {}

        # First pass: assign new block IDs
        for old_id in self.callee_func.blocks:
            self._remap_block_id(old_id)

        # Second pass: clone blocks with remapped instructions
        for old_id, block in self.callee_func.blocks.items():
            new_id = self.block_map[old_id]

            new_block = BasicBlock(
                block_id=new_id,
                instructions=[self._clone_instruction(i) for i in block.instructions],
                predecessors=[],  # Will be rebuilt
                successors=[],    # Will be rebuilt
            )

            cloned_blocks[new_id] = new_block

        # Third pass: update predecessor/successor relationships
        for old_id, block in self.callee_func.blocks.items():
            new_id = self.block_map[old_id]
            new_block = cloned_blocks[new_id]

            new_block.predecessors = [self.block_map[p] for p in block.predecessors
                                      if p in self.block_map]
            new_block.successors = [self.block_map[s] for s in block.successors
                                    if s in self.block_map]

        return cloned_blocks

    def get_entry_block_id(self) -> int:
        """Get the remapped entry block ID."""
        return self.block_map[self.callee_func.entry_block_id]


class FunctionInliner:
    """
    Main function inlining pass.

    Algorithm:
    1. Count calls to each function
    2. Find inline candidates (call sites where should_inline() returns true)
    3. For each candidate (bottom-up order):
       a. Clone callee's basic blocks with fresh vreg/block IDs
       b. Set up parameter bindings (map callee params to caller args)
       c. Replace Return instructions with Move + Jump to merge block
       d. Split call block: pre-call instructions → jump to inlined entry
       e. Create merge block with post-call instructions
       f. Add cloned blocks to caller's CFG
    """

    def __init__(self, verbose: bool = False, implicit_inline: bool = True):
        """
        Initialize the function inliner.

        Args:
            verbose: If True, print information about inlined functions
            implicit_inline: If True, allow implicit inlining (called-once and small
                           functions). If False, only inline explicitly marked functions.
        """
        self.verbose = verbose
        self.implicit_inline = implicit_inline

    def run(self, mir_program: MIRProgram) -> int:
        """
        Run the inlining pass on the MIR program.

        Args:
            mir_program: The MIR program to optimize

        Returns:
            Number of call sites inlined
        """
        checker = InlinabilityChecker(mir_program, implicit_inline=self.implicit_inline)
        inlined_count = 0

        # Process functions - we may need multiple passes as inlining
        # can expose new inlining opportunities
        # Limit iterations to prevent infinite loops
        max_iterations = 100
        iteration = 0

        changed = True
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            for caller_func in mir_program.functions:
                # Find all call sites that should be inlined
                call_sites = self._find_inline_candidates(caller_func, checker)

                for block_id, instr_idx, call_instr in call_sites:
                    callee_name = call_instr.function
                    if not isinstance(callee_name, str):
                        continue

                    if not checker.should_inline(callee_name):
                        continue

                    callee_func = checker.func_map.get(callee_name)
                    if callee_func is None:
                        continue

                    # Perform inlining
                    if self._inline_call(caller_func, block_id, instr_idx,
                                         call_instr, callee_func):
                        inlined_count += 1
                        changed = True

                        if self.verbose:
                            print(f"  Inlined {callee_name} into {caller_func.name}")

                        # Rebuild checker since call counts changed
                        checker = InlinabilityChecker(mir_program, implicit_inline=self.implicit_inline)
                        break  # Restart the loop with updated function

                if changed:
                    break

        if self.verbose and inlined_count > 0:
            print(f"Function inlining: {inlined_count} call site(s) inlined")

        return inlined_count

    def _find_inline_candidates(
        self,
        func: MIRFunction,
        checker: InlinabilityChecker
    ) -> List[Tuple[int, int, Call]]:
        """
        Find all call sites that are candidates for inlining.

        Args:
            func: Function to search
            checker: Inlinability checker

        Returns:
            List of (block_id, instruction_index, Call) tuples
        """
        candidates = []

        for block_id, block in func.blocks.items():
            for idx, instr in enumerate(block.instructions):
                if isinstance(instr, Call) and isinstance(instr.function, str):
                    if checker.should_inline(instr.function):
                        candidates.append((block_id, idx, instr))

        return candidates

    def _inline_call(
        self,
        caller: MIRFunction,
        block_id: int,
        instr_idx: int,
        call: Call,
        callee: MIRFunction
    ) -> bool:
        """
        Inline a single call site.

        Args:
            caller: Calling function
            block_id: Block containing the call
            instr_idx: Index of call instruction in block
            call: The Call instruction
            callee: Function being called

        Returns:
            True if inlining was successful
        """
        call_block = caller.blocks[block_id]

        # Clone callee's blocks
        cloner = BlockCloner(caller, callee)
        cloned_blocks = cloner.clone_blocks()
        inlined_entry_id = cloner.get_entry_block_id()

        # Override source_loc on all inlined instructions to use call site location
        # This ensures debug info points to where the function was called, not defined
        call_site_loc = call.source_loc
        if call_site_loc:
            for block in cloned_blocks.values():
                for instr in block.instructions:
                    instr.source_loc = call_site_loc

        # Create merge block for code after the call
        merge_block_id = max(caller.blocks.keys()) + 1
        while merge_block_id in cloned_blocks:
            merge_block_id += 1

        # Split instructions: before call, call itself, after call
        pre_call_instrs = call_block.instructions[:instr_idx]
        post_call_instrs = call_block.instructions[instr_idx + 1:]

        # Create merge block with post-call instructions
        merge_block = BasicBlock(
            block_id=merge_block_id,
            instructions=list(post_call_instrs),
            predecessors=[],  # Will be filled in
            successors=list(call_block.successors),
        )

        # Update call block: keep pre-call instructions, add jump to inlined entry
        call_block.instructions = list(pre_call_instrs)
        call_block.instructions.append(Jump(target=inlined_entry_id))
        call_block.successors = [inlined_entry_id]

        # Set up parameter bindings for inlined code.
        # Returns two lists: register_instrs (set up hw regs) and stack_instrs
        # (bind stack params to vregs).
        register_instrs, stack_instrs = self._create_param_bindings(call, callee, cloner)

        # Insert instructions in the correct order at the inlined entry block:
        #   1. register_instrs: Load argument values into hardware registers
        #      (e.g., LDA #33 for val @ A)
        #   2. reg_save instructions from callee: Move(dest=vreg, source=HardwareRegister)
        #      (e.g., save A to saved_val_vreg before anything clobbers A)
        #   3. stack_instrs: Bind stack param arguments to callee vregs
        #      (e.g., Move(dest=ptr_vreg, source=Immediate(addr)) — may clobber A in codegen)
        #   4. Rest of callee body
        inlined_entry = cloned_blocks[inlined_entry_id]
        reg_save_count = 0
        for instr in inlined_entry.instructions:
            if (isinstance(instr, Move) and
                isinstance(instr.source, HardwareRegister) and
                isinstance(instr.dest, VirtualRegister)):
                reg_save_count += 1
            else:
                break
        inlined_entry.instructions = (
            register_instrs +
            inlined_entry.instructions[:reg_save_count] +
            stack_instrs +
            inlined_entry.instructions[reg_save_count:]
        )
        inlined_entry.predecessors.append(block_id)

        # Create result vreg if call has return values
        result_vreg = call.returns[0] if call.returns else None

        # Replace Return instructions with Move + Jump to merge block
        return_blocks = []

        # Determine if the function returns via A register (default for u8/i8/u16/i16)
        # In this case, the return value vreg's value is actually in A, not in memory
        # Default calling convention: integer types return in A unless explicitly bound elsewhere
        returns_via_a = (
            callee.return_type is not None and
            callee.return_type.name in ('u8', 'i8', 'u16', 'i16')
        )

        # Collect REMAPPED callee parameter vregs — these may be hw-promoted to
        # X/Y after inlining (FixedStack ABI), so we can't assume they'll be in A
        # at return. Must use remapped vregs since cloned blocks have remapped values.
        callee_param_vregs = set()
        for vreg in callee.param_to_vreg.values():
            callee_param_vregs.add(cloner._remap_vreg(vreg))

        for cloned_block in cloned_blocks.values():
            new_instructions = []
            for instr in cloned_block.instructions:
                if isinstance(instr, Return):
                    # Move return value to result vreg if needed
                    if result_vreg and instr.values:
                        return_value = instr.values[0]

                        # The return_value vreg is already remapped by clone_blocks().
                        # Do NOT call _remap_operand again — that would create a
                        # disconnected new vreg.
                        if returns_via_a and isinstance(return_value, VirtualRegister):
                            if return_value not in callee_param_vregs:
                                # Computed result: ALU ops leave result in A
                                return_value = HardwareRegister('A')
                            # else: parameter passthrough — use remapped vreg directly.
                            # With FixedStack ABI the param may be hw-promoted to X/Y
                            # after inlining, so we can't assume it's in A.

                        new_instructions.append(Move(
                            dest=result_vreg,
                            source=return_value,
                            type_info=callee.return_type
                        ))
                    # Jump to merge block
                    new_instructions.append(Jump(target=merge_block_id))
                    return_blocks.append(cloned_block.block_id)
                else:
                    new_instructions.append(instr)
            cloned_block.instructions = new_instructions
            # Update successors if this was a return block
            if cloned_block.block_id in return_blocks:
                cloned_block.successors = [merge_block_id]

        # Set merge block predecessors
        merge_block.predecessors = return_blocks

        # Add cloned blocks to caller
        for new_id, new_block in cloned_blocks.items():
            caller.blocks[new_id] = new_block

        # Add merge block
        caller.blocks[merge_block_id] = merge_block

        # Update original successors' predecessors
        for succ_id in merge_block.successors:
            if succ_id in caller.blocks:
                succ_block = caller.blocks[succ_id]
                if block_id in succ_block.predecessors:
                    succ_block.predecessors.remove(block_id)
                if merge_block_id not in succ_block.predecessors:
                    succ_block.predecessors.append(merge_block_id)

        return True

    def _create_param_bindings(
        self,
        call: Call,
        callee: MIRFunction,
        cloner: BlockCloner
    ) -> List[MIRInstruction]:
        """
        Create instructions to bind call arguments to callee parameters.

        Returns two lists:
        - register_instrs: Move arg value into hardware register (must run first)
        - stack_instrs: Move arg value into callee vreg (must run after reg saves)

        Args:
            call: The Call instruction
            callee: Function being inlined
            cloner: Block cloner for vreg remapping

        Returns:
            Tuple of (register_instrs, stack_instrs)
        """
        register_instrs = []
        stack_instrs = []

        for i, (arg, param) in enumerate(zip(call.args, callee.parameters)):
            if isinstance(param.binding, RegisterBinding):
                # Register parameter: the call lowering would have set up the
                # hardware register. After inlining, we must emit this setup
                # explicitly since the Call instruction is removed.
                hw_reg = HardwareRegister(param.binding.register_name)
                register_instrs.append(Move(
                    dest=hw_reg,
                    source=arg.value,
                    type_info=param.param_type
                ))

            elif isinstance(param.binding, VariableBinding):
                # Variable-bound: caller already stored to the variable
                pass

            else:
                # Stack parameter: map caller's arg to callee's param vreg
                # Get the callee's vreg for this parameter
                param_idx = i
                if param_idx in callee.param_to_vreg:
                    callee_vreg = callee.param_to_vreg[param_idx]
                    # Get the remapped vreg
                    inlined_vreg = cloner._remap_vreg(callee_vreg)
                    # Move argument value to inlined vreg
                    stack_instrs.append(Move(
                        dest=inlined_vreg,
                        source=arg.value,
                        type_info=param.param_type
                    ))

        return register_instrs, stack_instrs
