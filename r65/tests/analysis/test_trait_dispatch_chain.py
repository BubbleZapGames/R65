# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Unit tests for analyze_trait_dispatch_chains.

Verifies that the chain pass correctly assigns ChainRole values to
runs of TraitDispatch on the same far-self vreg, and rejects runs
that would be unsound (intervening RAM access, near-pointer deref,
hostile call, redefined self vreg, etc).
"""

import pytest

from r65.compiler.mir.nodes import (
    MIRFunction, MIRProgram, BasicBlock, Return, TraitDispatch, ChainRole,
    Argument, ArgumentMechanism, VirtualRegister, Call, Load, Store,
    LoadIndirect, StoreIndirect, MemoryLocation, Move, Immediate,
)
from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator
from r65.compiler.hir.types import BasicTypeInfo
from r65.compiler.analysis.call_graph import CallGraphAnalyzer
from r65.compiler.analysis.far_ptr_strategy import (
    analyze_trait_dispatch_chains,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_func(name, blocks, entry=0, exits=None):
    if exits is None:
        exits = list(blocks.keys())
    return MIRFunction(
        name=name,
        parameters=[],
        return_type=BasicTypeInfo('u8'),
        blocks=blocks,
        entry_block_id=entry,
        exit_block_ids=exits,
        vreg_allocator=VirtualRegisterAllocator(),
    )


def _make_block(block_id, instructions, preds=(), succs=()):
    return BasicBlock(
        block_id=block_id,
        instructions=list(instructions),
        predecessors=list(preds),
        successors=list(succs),
    )


def _make_self_vreg(vid=1):
    return VirtualRegister(id=vid, type_info=BasicTypeInfo('u16'))


def _make_dispatch(self_vreg, trait='T', method='m'):
    arg = Argument(
        value=self_vreg,
        mechanism=ArgumentMechanism.SELF_Y,
    )
    return TraitDispatch(
        trait_name=trait,
        method_name=method,
        method_index=0,
        self_ptr=self_vreg,
        args=[arg],
        returns=[],
        is_far=True,
        self_is_far=True,
    )


def _trait_info(trait='T', method='m', impls=None):
    """Build a trait_dispatch_info dict where every impl is a function
    name; caller is responsible for adding empty MIRFunctions for those
    names so they pass the DBR-independence check."""
    if impls is None:
        impls = ['Impl1__m']
    return {
        trait: {
            'is_far': True,
            'methods': [method],
            'implementors': [
                {'struct': name.split('__')[0], 'type_id': i + 1,
                 'mangled': [name]}
                for i, name in enumerate(impls)
            ],
        }
    }


def _multi_method_trait_info(trait='T', methods=None, impls_per_method=None):
    """Build a trait_dispatch_info dict for a trait with multiple methods.

    Args:
        trait: trait name
        methods: list of method names, e.g. ['m1', 'm2']
        impls_per_method: dict {method_name: [impl_func_name, ...]}.
            All methods must have the same set of implementor structs.
            We index by struct: implementor i gets mangled[k] from
            impls_per_method[methods[k]][i].
    """
    if methods is None:
        methods = ['m1', 'm2']
    if impls_per_method is None:
        impls_per_method = {m: [f'Impl1__{m}'] for m in methods}

    n_impls = len(impls_per_method[methods[0]])
    implementors = []
    for i in range(n_impls):
        mangled = [impls_per_method[m][i] for m in methods]
        struct = mangled[0].split('__')[0]
        implementors.append({
            'struct': struct,
            'type_id': i + 1,
            'mangled': mangled,
        })
    return {
        trait: {
            'is_far': True,
            'methods': list(methods),
            'implementors': implementors,
        }
    }


def _empty_impl_func(name):
    """A trivial leaf impl — passes _impl_is_dbr_independent."""
    return _make_func(name, {0: _make_block(0, [Return()])})


def _build_program(caller_blocks, trait='T', method='m',
                   impl_names=('Impl1__m',), extra_funcs=()):
    caller = _make_func('caller', caller_blocks)
    impls = [_empty_impl_func(n) for n in impl_names]
    funcs = [caller] + impls + list(extra_funcs)
    return MIRProgram(
        functions=funcs,
        trait_dispatch_info=_trait_info(trait, method, list(impl_names)),
    )


def _run_analysis(program):
    cg = CallGraphAnalyzer(program).analyze()
    analyze_trait_dispatch_chains(program, cg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChainAssignment:
    """Test that ChainRole values are assigned correctly."""

    def test_empty_block(self):
        """Block with no dispatches: no roles assigned."""
        prog = _build_program({0: _make_block(0, [Return()])})
        _run_analysis(prog)
        # Nothing to assert beyond 'no crash'.

    def test_single_dispatch_remains_solo(self):
        """One dispatch stays SOLO (the default)."""
        s = _make_self_vreg()
        td = _make_dispatch(s)
        prog = _build_program({0: _make_block(0, [td, Return()])})
        _run_analysis(prog)
        assert td.self_chain_role == ChainRole.SOLO

    def test_two_dispatch_chain_start_end(self):
        """Two same-self dispatches: roles START + END."""
        s = _make_self_vreg()
        a = _make_dispatch(s)
        b = _make_dispatch(s)
        prog = _build_program({0: _make_block(0, [a, b, Return()])})
        _run_analysis(prog)
        assert a.self_chain_role == ChainRole.START
        assert b.self_chain_role == ChainRole.END

    def test_three_dispatch_chain_start_middle_end(self):
        """Three same-self dispatches: START, MIDDLE, END."""
        s = _make_self_vreg()
        a = _make_dispatch(s)
        b = _make_dispatch(s)
        c = _make_dispatch(s)
        prog = _build_program({0: _make_block(0, [a, b, c, Return()])})
        _run_analysis(prog)
        assert a.self_chain_role == ChainRole.START
        assert b.self_chain_role == ChainRole.MIDDLE
        assert c.self_chain_role == ChainRole.END


class TestChainBreaks:
    """Test that intervening operations break the chain."""

    def test_break_by_ram_store(self):
        """A RAM store between dispatches forces SOLO."""
        s = _make_self_vreg()
        a = _make_dispatch(s)
        b = _make_dispatch(s)
        # Synthetic Store to RAM
        ram_loc = MemoryLocation(
            storage_type='ram', address=0x7E0200, symbol=None
        )
        ram_store = Store(
            source=Immediate(value=1),
            dest=ram_loc,
            type_info=BasicTypeInfo('u8'),
        )
        prog = _build_program({0: _make_block(0, [a, ram_store, b, Return()])})
        _run_analysis(prog)
        assert a.self_chain_role == ChainRole.SOLO
        assert b.self_chain_role == ChainRole.SOLO

    def test_break_by_near_pointer_deref(self):
        """A near pointer deref between dispatches forces SOLO."""
        s = _make_self_vreg()
        ptr_vreg = VirtualRegister(id=99, type_info=BasicTypeInfo('u16'))
        dest_vreg = VirtualRegister(id=100, type_info=BasicTypeInfo('u8'))
        a = _make_dispatch(s)
        b = _make_dispatch(s)
        near_load = LoadIndirect(
            dest=dest_vreg, pointer=ptr_vreg,
            is_far=False, type_info=BasicTypeInfo('u8'),
        )
        prog = _build_program({0: _make_block(0, [a, near_load, b, Return()])})
        _run_analysis(prog)
        assert a.self_chain_role == ChainRole.SOLO
        assert b.self_chain_role == ChainRole.SOLO

    def test_break_by_call_to_unknown_function(self):
        """A direct call to an unknown function forces SOLO."""
        s = _make_self_vreg()
        a = _make_dispatch(s)
        b = _make_dispatch(s)
        # Call to a function not in the program (external/unknown).
        ext_call = Call(function='external_fn', args=[], returns=[])
        prog = _build_program({0: _make_block(0, [a, ext_call, b, Return()])})
        _run_analysis(prog)
        assert a.self_chain_role == ChainRole.SOLO
        assert b.self_chain_role == ChainRole.SOLO

    def test_break_by_self_redef(self):
        """Redefining the self vreg between dispatches forces SOLO."""
        s = _make_self_vreg()
        a = _make_dispatch(s)
        b = _make_dispatch(s)
        # A Move that writes to s
        mv = Move(
            dest=s,
            source=Immediate(value=0x1234),
            type_info=BasicTypeInfo('u16'),
        )
        prog = _build_program({0: _make_block(0, [a, mv, b, Return()])})
        _run_analysis(prog)
        assert a.self_chain_role == ChainRole.SOLO
        assert b.self_chain_role == ChainRole.SOLO

    def test_break_by_different_self(self):
        """Two dispatches on different selves do not chain."""
        s1 = _make_self_vreg(vid=1)
        s2 = _make_self_vreg(vid=2)
        a = _make_dispatch(s1)
        b = _make_dispatch(s2)
        prog = _build_program({0: _make_block(0, [a, b, Return()])})
        _run_analysis(prog)
        assert a.self_chain_role == ChainRole.SOLO
        assert b.self_chain_role == ChainRole.SOLO

    def test_break_by_unknown_impl_set(self):
        """If the trait has no impls in trait_dispatch_info, no chain."""
        s = _make_self_vreg()
        a = _make_dispatch(s)
        b = _make_dispatch(s)
        prog = MIRProgram(
            functions=[_make_func('caller', {0: _make_block(0, [a, b, Return()])})],
            trait_dispatch_info=None,
        )
        _run_analysis(prog)
        assert a.self_chain_role == ChainRole.SOLO
        assert b.self_chain_role == ChainRole.SOLO


class TestImplDbrIndependence:
    """Test that impl bodies are checked for DBR independence."""

    def test_impl_with_ram_access_breaks_chain(self):
        """If an impl accesses RAM, the chain is rejected."""
        s = _make_self_vreg()
        a = _make_dispatch(s)
        b = _make_dispatch(s)

        # Impl that does a RAM access — DBR-dependent, so chain rejected.
        ram_loc = MemoryLocation(
            storage_type='ram', address=0x7E0300, symbol=None
        )
        bad_impl_block = _make_block(0, [
            Store(
                source=Immediate(value=0),
                dest=ram_loc,
                type_info=BasicTypeInfo('u8'),
            ),
            Return(),
        ])
        bad_impl = _make_func('Bad__m', {0: bad_impl_block})

        prog = MIRProgram(
            functions=[
                _make_func('caller', {0: _make_block(0, [a, b, Return()])}),
                bad_impl,
            ],
            trait_dispatch_info=_trait_info(impls=['Bad__m']),
        )
        _run_analysis(prog)
        assert a.self_chain_role == ChainRole.SOLO
        assert b.self_chain_role == ChainRole.SOLO

    def test_impl_with_near_deref_breaks_chain(self):
        """An impl that does a near pointer deref disqualifies the chain."""
        s = _make_self_vreg()
        a = _make_dispatch(s)
        b = _make_dispatch(s)

        ptr_v = VirtualRegister(id=50, type_info=BasicTypeInfo('u16'))
        dest_v = VirtualRegister(id=51, type_info=BasicTypeInfo('u8'))
        bad_block = _make_block(0, [
            LoadIndirect(
                dest=dest_v, pointer=ptr_v,
                is_far=False, type_info=BasicTypeInfo('u8'),
            ),
            Return(),
        ])
        bad_impl = _make_func('Bad__m', {0: bad_block})
        prog = MIRProgram(
            functions=[
                _make_func('caller', {0: _make_block(0, [a, b, Return()])}),
                bad_impl,
            ],
            trait_dispatch_info=_trait_info(impls=['Bad__m']),
        )
        _run_analysis(prog)
        assert a.self_chain_role == ChainRole.SOLO
        assert b.self_chain_role == ChainRole.SOLO


class TestCrossBlockChains:
    """Test chains across straight-line CFG paths."""

    def test_cross_block_straight_line(self):
        """Chain extends across a straight-line block boundary."""
        s = _make_self_vreg()
        a = _make_dispatch(s)
        b = _make_dispatch(s)
        # Block 0 -> Block 1, single edge in each direction.
        blocks = {
            0: _make_block(0, [a], succs=[1]),
            1: _make_block(1, [b, Return()], preds=[0]),
        }
        prog = _build_program(blocks)
        _run_analysis(prog)
        assert a.self_chain_role == ChainRole.START
        assert b.self_chain_role == ChainRole.END

    def test_cross_block_join_rejected(self):
        """A successor block with multiple predecessors breaks the chain."""
        s = _make_self_vreg()
        a = _make_dispatch(s)
        b = _make_dispatch(s)
        # Block 1 has two predecessors (0 and 2) — not straight-line.
        blocks = {
            0: _make_block(0, [a], succs=[1]),
            1: _make_block(1, [b, Return()], preds=[0, 2]),
            2: _make_block(2, [Return()], succs=[1]),
        }
        prog = _build_program(blocks)
        _run_analysis(prog)
        assert a.self_chain_role == ChainRole.SOLO
        assert b.self_chain_role == ChainRole.SOLO

    def test_cross_block_branch_rejected(self):
        """A predecessor with multiple successors breaks the chain."""
        s = _make_self_vreg()
        a = _make_dispatch(s)
        b = _make_dispatch(s)
        # Block 0 has two successors (1 and 2) — not straight-line.
        blocks = {
            0: _make_block(0, [a], succs=[1, 2]),
            1: _make_block(1, [b, Return()], preds=[0]),
            2: _make_block(2, [Return()], preds=[0]),
        }
        prog = _build_program(blocks)
        _run_analysis(prog)
        assert a.self_chain_role == ChainRole.SOLO
        assert b.self_chain_role == ChainRole.SOLO


class TestChainAcrossMethods:
    """v1.5: chain extends across different methods of the same trait."""

    def _build_two_method_program(self, caller_blocks, m1_impl_blocks=None,
                                  m2_impl_blocks=None):
        """Helper: program with trait T having methods m1 and m2, each
        with one implementor (Impl1__m1 / Impl1__m2)."""
        if m1_impl_blocks is None:
            m1_impl_blocks = {0: _make_block(0, [Return()])}
        if m2_impl_blocks is None:
            m2_impl_blocks = {0: _make_block(0, [Return()])}
        caller = _make_func('caller', caller_blocks)
        impl_m1 = _make_func('Impl1__m1', m1_impl_blocks)
        impl_m2 = _make_func('Impl1__m2', m2_impl_blocks)
        return MIRProgram(
            functions=[caller, impl_m1, impl_m2],
            trait_dispatch_info=_multi_method_trait_info(
                trait='T',
                methods=['m1', 'm2'],
                impls_per_method={
                    'm1': ['Impl1__m1'],
                    'm2': ['Impl1__m2'],
                },
            ),
        )

    def test_chain_different_methods_same_trait_coalesces(self):
        """v1.5: two same-self dispatches to different methods of the
        same trait coalesce into START + END."""
        s = _make_self_vreg()
        a = _make_dispatch(s, trait='T', method='m1')
        b = _make_dispatch(s, trait='T', method='m2')
        prog = self._build_two_method_program(
            {0: _make_block(0, [a, b, Return()])}
        )
        _run_analysis(prog)
        assert a.self_chain_role == ChainRole.START
        assert b.self_chain_role == ChainRole.END

    def test_chain_three_methods_mixed(self):
        """v1.5: three dispatches with method pattern [m1, m2, m1]
        produce START + MIDDLE + END."""
        s = _make_self_vreg()
        a = _make_dispatch(s, trait='T', method='m1')
        b = _make_dispatch(s, trait='T', method='m2')
        c = _make_dispatch(s, trait='T', method='m1')
        prog = self._build_two_method_program(
            {0: _make_block(0, [a, b, c, Return()])}
        )
        _run_analysis(prog)
        assert a.self_chain_role == ChainRole.START
        assert b.self_chain_role == ChainRole.MIDDLE
        assert c.self_chain_role == ChainRole.END

    def test_chain_breaks_when_one_method_has_dirty_impl(self):
        """If one method's impl set fails DBR-independence, chain
        rejects when that method appears in the run.

        Verifies the union-of-impls accounting: when extending from m1
        to m2, the new method's impls must also pass — and m2's impl
        does a RAM access here, so the chain should fall back to SOLO.
        """
        s = _make_self_vreg()
        a = _make_dispatch(s, trait='T', method='m1')
        b = _make_dispatch(s, trait='T', method='m2')

        # Bad m2 impl: writes to RAM.
        ram_loc = MemoryLocation(
            storage_type='ram', address=0x7E0400, symbol=None
        )
        bad_m2_blocks = {0: _make_block(0, [
            Store(
                source=Immediate(value=1),
                dest=ram_loc,
                type_info=BasicTypeInfo('u8'),
            ),
            Return(),
        ])}
        prog = self._build_two_method_program(
            {0: _make_block(0, [a, b, Return()])},
            m2_impl_blocks=bad_m2_blocks,
        )
        _run_analysis(prog)
        # m1 alone is fine but the chain extension to m2 fails — so
        # both end up SOLO.
        assert a.self_chain_role == ChainRole.SOLO
        assert b.self_chain_role == ChainRole.SOLO
