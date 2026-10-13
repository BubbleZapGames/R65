# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Shared loop analysis utilities.

Provides back-edge-based loop detection and nesting-depth computation
reused by the inliner, loop unroller, and register promotion passes.
"""

from typing import Dict, List, Set, Tuple

from r65.compiler.mir.nodes import MIRFunction, iter_block_targets


def find_loops(func: MIRFunction) -> List[Tuple[int, Set[int]]]:
    """
    Find natural loops via back-edge detection.

    A back-edge is a jump to a block that is still on the depth-first search
    stack from the entry - an ancestor of the source. Block IDs are not a
    substitute: they are handed out as the lowering asks for them, so an
    outer `if`'s merge block is numbered before an inner `if`'s blocks, and
    the inner one's jump to it goes to a lower ID without being a loop.

    Returns:
        List of (header_block_id, body_block_ids) tuples, in block order.
    """
    back_edges = find_back_edges(func)
    loops = []

    for block_id, block in func.blocks.items():
        for instr in block.instructions:
            for target in iter_block_targets(instr):
                if (block_id, target) in back_edges:
                    loops.append((target, _find_loop_body(func, target, block_id)))

    return loops


def find_back_edges(func: MIRFunction) -> Set[Tuple[int, int]]:
    """`{(source, target)}` for every CFG edge whose target is a DFS ancestor.

    The search starts at the entry block, then from any block it never
    reached, so unreachable code is classified the same way.
    """
    state: Dict[int, int] = {}          # absent: unvisited, 1: on stack, 2: done
    back_edges: Set[Tuple[int, int]] = set()

    starts = [func.entry_block_id] + [bid for bid in func.blocks
                                      if bid != func.entry_block_id]
    for start in starts:
        if start in state or start not in func.blocks:
            continue
        state[start] = 1
        stack = [(start, iter(_successors(func.blocks[start])))]
        while stack:
            block_id, succs = stack[-1]
            for succ in succs:
                if succ not in func.blocks:
                    continue
                if state.get(succ) == 1:
                    back_edges.add((block_id, succ))
                elif succ not in state:
                    state[succ] = 1
                    stack.append((succ, iter(_successors(func.blocks[succ]))))
                    break
            else:
                state[block_id] = 2
                stack.pop()

    return back_edges


def _successors(block) -> List[int]:
    """Branch targets of *block*: its recorded successors plus its jumps'."""
    succs = list(block.successors)
    for instr in block.instructions:
        succs.extend(iter_block_targets(instr))
    return list(dict.fromkeys(succs))


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
