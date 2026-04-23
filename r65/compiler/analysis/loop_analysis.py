# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Shared loop analysis utilities.

Provides back-edge-based loop detection and nesting-depth computation
reused by the inliner, loop unroller, and register promotion passes.
"""

from typing import Dict, List, Set, Tuple

from r65.compiler.mir.nodes import MIRFunction, Jump, CondBranch


def find_loops(func: MIRFunction) -> List[Tuple[int, Set[int]]]:
    """
    Find natural loops via back-edge detection.

    A back-edge is a jump whose target block_id <= source block_id.

    Returns:
        List of (header_block_id, body_block_ids) tuples.
    """
    loops = []

    for block_id, block in func.blocks.items():
        if not block.instructions:
            continue

        for instr in block.instructions:
            target = None
            if isinstance(instr, Jump):
                target = instr.target
            elif isinstance(instr, CondBranch):
                if instr.true_target <= block_id:
                    target = instr.true_target
                if instr.false_target <= block_id:
                    body = _find_loop_body(func, instr.false_target, block_id)
                    if body:
                        loops.append((instr.false_target, body))
                    if target is None:
                        continue

            if target is not None and target <= block_id:
                body = _find_loop_body(func, target, block_id)
                if body:
                    loops.append((target, body))

    return loops


def _find_loop_body(func: MIRFunction, header: int, back_edge_source: int) -> Set[int]:
    """Reverse-walk from back_edge_source to header to collect loop body blocks."""
    body = {header}
    if header == back_edge_source:
        return body

    worklist = [back_edge_source]
    body.add(back_edge_source)

    while worklist:
        block_id = worklist.pop()
        block = func.blocks.get(block_id)
        if block is None:
            continue
        for pred_id in block.predecessors:
            if pred_id not in body:
                body.add(pred_id)
                worklist.append(pred_id)

    return body


def compute_block_nesting(func: MIRFunction) -> Dict[int, int]:
    """
    Compute the loop nesting depth for each basic block.

    Returns:
        Dict mapping block_id → nesting depth (0 = not inside any loop).
    """
    loops = find_loops(func)
    depth: Dict[int, int] = {}
    for _header, body in loops:
        for bid in body:
            depth[bid] = depth.get(bid, 0) + 1
    return depth
