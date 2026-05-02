# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Tests for trait-impl-resolved CallGraph.

CallGraphAnalyzer takes an optional trait_dispatch_info dict (from
MIRProgram) and uses it to (a) populate CallGraph.trait_impls and (b)
add caller -> impl edges for every TraitDispatch encountered.
"""

import pytest

from r65.compiler.mir.nodes import (
    MIRFunction, MIRProgram, BasicBlock, Return, TraitDispatch,
    VirtualRegister,
)
from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator
from r65.compiler.hir.types import BasicTypeInfo
from r65.compiler.analysis.call_graph import CallGraphAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_func(name, *, instructions=None):
    """Build a minimal MIRFunction with optional instructions."""
    if instructions is None:
        instructions = [Return()]
    block = BasicBlock(
        block_id=0,
        instructions=instructions,
        predecessors=[],
        successors=[],
    )
    return MIRFunction(
        name=name,
        parameters=[],
        return_type=BasicTypeInfo('u8'),
        blocks={0: block},
        entry_block_id=0,
        exit_block_ids=[0],
        vreg_allocator=VirtualRegisterAllocator(),
    )


def _make_trait_info(trait_name, methods, implementors):
    """Build a trait_dispatch_info-style dict for one trait.

    implementors is a list of (struct_name, type_id, list_of_mangled_names).
    """
    return {
        trait_name: {
            'is_far': False,
            'methods': methods,
            'implementors': [
                {'struct': s, 'type_id': tid, 'mangled': mangled}
                for s, tid, mangled in implementors
            ],
        }
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTraitImpls:
    """Verify trait_impls is populated and impl edges are added."""

    def test_basic_two_impls(self):
        """One trait with one method, two impls; both edges present."""
        td = TraitDispatch(
            trait_name='Drawable',
            method_name='draw',
            method_index=0,
            self_ptr=None,
        )
        caller = _make_func('caller', instructions=[td, Return()])
        player_draw = _make_func('Player__draw')
        enemy_draw = _make_func('Enemy__draw')

        prog = MIRProgram(
            functions=[caller, player_draw, enemy_draw],
            trait_dispatch_info=_make_trait_info(
                'Drawable',
                methods=['draw'],
                implementors=[
                    ('Player', 1, ['Player__draw']),
                    ('Enemy', 2, ['Enemy__draw']),
                ],
            ),
        )

        graph = CallGraphAnalyzer(prog).analyze()

        # trait_impls populated
        assert graph.trait_impls[('Drawable', 'draw')] == {
            'Player__draw', 'Enemy__draw'
        }
        # caller is tagged as indirect_caller (existing behavior preserved)
        assert 'caller' in graph.indirect_callers
        # Edges added
        callees = graph.get_callees('caller')
        assert 'Player__draw' in callees
        assert 'Enemy__draw' in callees

    def test_resolve_trait_method_api(self):
        """resolve_trait_method returns the impl set."""
        prog = MIRProgram(
            functions=[],
            trait_dispatch_info=_make_trait_info(
                'Drawable',
                methods=['draw', 'get_width'],
                implementors=[
                    ('Player', 1, ['Player__draw', 'Player__get_width']),
                    ('Enemy', 2, ['Enemy__draw', 'Enemy__get_width']),
                ],
            ),
        )
        graph = CallGraphAnalyzer(prog).analyze()

        assert graph.resolve_trait_method('Drawable', 'draw') == {
            'Player__draw', 'Enemy__draw'
        }
        assert graph.resolve_trait_method('Drawable', 'get_width') == {
            'Player__get_width', 'Enemy__get_width'
        }
        # Unknown method -> empty set, never raises
        assert graph.resolve_trait_method('Drawable', 'unknown') == set()
        assert graph.resolve_trait_method('Unknown', 'x') == set()

    def test_no_trait_dispatches_unchanged(self):
        """A program without TraitDispatch behaves exactly like before."""
        f = _make_func('main')
        prog = MIRProgram(functions=[f])  # trait_dispatch_info is None

        graph = CallGraphAnalyzer(prog).analyze()

        assert graph.trait_impls == {}
        assert graph.indirect_callers == set()
        assert graph.get_callees('main') == set()

    def test_program_with_traits_but_no_dispatch_calls(self):
        """trait_impls is populated even if no function dispatches the trait."""
        f = _make_func('main')
        prog = MIRProgram(
            functions=[f],
            trait_dispatch_info=_make_trait_info(
                'Drawable',
                methods=['draw'],
                implementors=[('Player', 1, ['Player__draw'])],
            ),
        )

        graph = CallGraphAnalyzer(prog).analyze()

        # trait_impls is populated from the registry
        assert graph.trait_impls[('Drawable', 'draw')] == {'Player__draw'}
        # No edge added because no caller dispatches it
        assert graph.get_callees('main') == set()

    def test_explicit_trait_dispatch_info_override(self):
        """Passing trait_dispatch_info explicitly overrides the program's."""
        td = TraitDispatch(
            trait_name='Drawable',
            method_name='draw',
            method_index=0,
            self_ptr=None,
        )
        caller = _make_func('caller', instructions=[td, Return()])
        prog = MIRProgram(
            functions=[caller],
            trait_dispatch_info=None,
        )

        explicit = _make_trait_info(
            'Drawable',
            methods=['draw'],
            implementors=[('Foo', 1, ['Foo__draw'])],
        )
        graph = CallGraphAnalyzer(prog, trait_dispatch_info=explicit).analyze()

        assert 'Foo__draw' in graph.get_callees('caller')

    def test_unresolved_trait_dispatch_falls_back_to_indirect(self):
        """If trait_dispatch_info is missing, no impl edges; still indirect."""
        td = TraitDispatch(
            trait_name='Mystery',
            method_name='do_thing',
            method_index=0,
            self_ptr=None,
        )
        caller = _make_func('caller', instructions=[td, Return()])
        prog = MIRProgram(functions=[caller])

        graph = CallGraphAnalyzer(prog).analyze()

        assert 'caller' in graph.indirect_callers
        assert graph.get_callees('caller') == set()
