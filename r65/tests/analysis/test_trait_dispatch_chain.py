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
    TypeConvert, BinaryOp,
)
from r65.compiler.mir.virtual_registers import VirtualRegisterAllocator
from r65.compiler.hir.types import BasicTypeInfo, PointerTypeInfo, TraitTypeInfo
from r65.compiler.analysis.call_graph import CallGraphAnalyzer
from r65.compiler.analysis.far_ptr_strategy import (
    analyze_trait_dispatch_chains,
    _chain_self_root,
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


# ---------------------------------------------------------------------------
# v1.6: cast-transparency
# ---------------------------------------------------------------------------

def _ptr_to_trait(trait_name='Drawable', is_far=True):
    """Build a PointerTypeInfo to a TraitTypeInfo."""
    return PointerTypeInfo(
        is_far=is_far,
        pointee_type=TraitTypeInfo(name=trait_name),
    )


def _ptr_to_struct(name='Player', is_far=True):
    """Build a PointerTypeInfo to a struct (modeled as BasicTypeInfo)."""
    return PointerTypeInfo(
        is_far=is_far,
        pointee_type=BasicTypeInfo(name=name),
    )


class TestChainSelfRoot:
    """v1.6: tests for the _chain_self_root walker."""

    def _build_func_with_blocks(self, blocks):
        return _make_func('caller', blocks)

    def test_chain_self_root_no_def(self):
        """A vreg with no def in the function (e.g. a parameter) is
        its own root."""
        v = VirtualRegister(id=10, type_info=_ptr_to_trait())
        func = self._build_func_with_blocks({
            0: _make_block(0, [Return()])
        })
        root = _chain_self_root(v, func)
        assert root.id == v.id

    def test_chain_self_root_through_move(self):
        """A vreg defined as Move from another vreg resolves to the
        source vreg."""
        src = VirtualRegister(id=20, type_info=_ptr_to_trait('A'))
        alias = VirtualRegister(id=21, type_info=_ptr_to_trait('B'))
        mv = Move(dest=alias, source=src, type_info=_ptr_to_trait('B'))
        func = self._build_func_with_blocks({
            0: _make_block(0, [mv, Return()])
        })
        root = _chain_self_root(alias, func)
        assert root.id == src.id

    def test_chain_self_root_through_typeconvert_same_far(self):
        """A TypeConvert between two same-is_far pointer types is
        followed."""
        src = VirtualRegister(id=30, type_info=_ptr_to_trait('A'))
        alias = VirtualRegister(id=31, type_info=_ptr_to_trait('B'))
        tc = TypeConvert(
            dest=alias, source=src,
            source_type=_ptr_to_trait('A'),
            target_type=_ptr_to_trait('B'),
        )
        func = self._build_func_with_blocks({
            0: _make_block(0, [tc, Return()])
        })
        root = _chain_self_root(alias, func)
        assert root.id == src.id

    def test_chain_self_root_stops_at_far_to_near(self):
        """A TypeConvert that changes is_far is NOT followed."""
        src = VirtualRegister(id=40, type_info=_ptr_to_trait('A', is_far=True))
        alias = VirtualRegister(id=41, type_info=_ptr_to_trait('A', is_far=False))
        tc = TypeConvert(
            dest=alias, source=src,
            source_type=_ptr_to_trait('A', is_far=True),
            target_type=_ptr_to_trait('A', is_far=False),
        )
        func = self._build_func_with_blocks({
            0: _make_block(0, [tc, Return()])
        })
        root = _chain_self_root(alias, func)
        # Walk does not cross the far→near boundary.
        assert root.id == alias.id

    def test_chain_self_root_stops_at_load(self):
        """A vreg defined by a Load (memory read) is its own root —
        the address is fresh, not aliased to anything."""
        v = VirtualRegister(id=50, type_info=_ptr_to_trait())
        loc = MemoryLocation(
            storage_type='ram', address=0x7E0500, symbol=None
        )
        ld = Load(dest=v, source=loc, type_info=_ptr_to_trait())
        func = self._build_func_with_blocks({
            0: _make_block(0, [ld, Return()])
        })
        root = _chain_self_root(v, func)
        assert root.id == v.id

    def test_chain_self_root_stops_at_binaryop(self):
        """A vreg defined by BinaryOp (e.g. pointer arithmetic /
        field-offset cast) is its own root — the address has shifted."""
        base = VirtualRegister(id=60, type_info=_ptr_to_struct())
        offset_ptr = VirtualRegister(id=61, type_info=_ptr_to_trait())
        bop = BinaryOp(
            dest=offset_ptr, left=base, right=Immediate(value=4),
            op='+', type_info=_ptr_to_trait(),
        )
        func = self._build_func_with_blocks({
            0: _make_block(0, [bop, Return()])
        })
        root = _chain_self_root(offset_ptr, func)
        # offset_ptr's def is a BinaryOp — not a transparent cast.
        assert root.id == offset_ptr.id

    def test_chain_self_root_chained_moves(self):
        """Three-step alias chain a → b → c collapses to root a."""
        a = VirtualRegister(id=70, type_info=_ptr_to_trait('A'))
        b = VirtualRegister(id=71, type_info=_ptr_to_trait('B'))
        c = VirtualRegister(id=72, type_info=_ptr_to_trait('C'))
        mv1 = Move(dest=b, source=a, type_info=_ptr_to_trait('B'))
        mv2 = Move(dest=c, source=b, type_info=_ptr_to_trait('C'))
        func = self._build_func_with_blocks({
            0: _make_block(0, [mv1, mv2, Return()])
        })
        root = _chain_self_root(c, func)
        assert root.id == a.id


class TestChainAcrossTraitCast:
    """v1.6: chain extends across trait pointer aliases of the same
    underlying object."""

    def test_chain_across_trait_cast_coalesces(self):
        """Two dispatches via different *Trait aliases of the same
        underlying root chain together."""
        # root vreg is the original pointer (e.g. &PLAYER).
        root_v = VirtualRegister(id=80, type_info=_ptr_to_struct('Player'))
        # Two trait-pointer aliases of the same root.
        as_a = VirtualRegister(id=81, type_info=_ptr_to_trait('TraitA'))
        as_b = VirtualRegister(id=82, type_info=_ptr_to_trait('TraitB'))
        cast_a = TypeConvert(
            dest=as_a, source=root_v,
            source_type=_ptr_to_struct('Player'),
            target_type=_ptr_to_trait('TraitA'),
        )
        cast_b = TypeConvert(
            dest=as_b, source=root_v,
            source_type=_ptr_to_struct('Player'),
            target_type=_ptr_to_trait('TraitB'),
        )
        td_a = _make_dispatch(as_a, trait='TraitA', method='ma')
        td_b = _make_dispatch(as_b, trait='TraitB', method='mb')

        impl_a = _empty_impl_func('Impl1__ma')
        impl_b = _empty_impl_func('Impl1__mb')

        # Two single-method traits.
        info = {
            'TraitA': {
                'is_far': True,
                'methods': ['ma'],
                'implementors': [{
                    'struct': 'Player', 'type_id': 1,
                    'mangled': ['Impl1__ma'],
                }],
            },
            'TraitB': {
                'is_far': True,
                'methods': ['mb'],
                'implementors': [{
                    'struct': 'Player', 'type_id': 1,
                    'mangled': ['Impl1__mb'],
                }],
            },
        }
        prog = MIRProgram(
            functions=[
                _make_func('caller', {0: _make_block(0, [
                    cast_a, cast_b, td_a, td_b, Return()
                ])}),
                impl_a, impl_b,
            ],
            trait_dispatch_info=info,
        )
        _run_analysis(prog)
        # The two dispatches' self vregs have the same root; cross-trait
        # chain extends.
        assert td_a.self_chain_role == ChainRole.START
        assert td_b.self_chain_role == ChainRole.END

    def test_chain_rejects_unrelated_pointer(self):
        """Two dispatches whose self vregs have UNRELATED defs (e.g.
        each loaded separately from memory) do not chain."""
        v1 = VirtualRegister(id=90, type_info=_ptr_to_trait('TraitA'))
        v2 = VirtualRegister(id=91, type_info=_ptr_to_trait('TraitB'))
        # Each defined via Load — distinct roots, no aliasing.
        loc1 = MemoryLocation(
            storage_type='ram', address=0x7E0600, symbol=None
        )
        loc2 = MemoryLocation(
            storage_type='ram', address=0x7E0700, symbol=None
        )
        ld1 = Load(dest=v1, source=loc1, type_info=_ptr_to_trait('TraitA'))
        ld2 = Load(dest=v2, source=loc2, type_info=_ptr_to_trait('TraitB'))
        td1 = _make_dispatch(v1, trait='TraitA', method='ma')
        td2 = _make_dispatch(v2, trait='TraitB', method='mb')

        # The Loads are RAM stores — these would also break DBR-
        # independence, so to isolate the root-mismatch we put the
        # Loads BEFORE the chain-region. But Load is RAM access between
        # dispatches would break it; here we have the chain be
        # [td1, td2], and the Loads are at the top of the block.
        # Actually — we need to put loads BEFORE both dispatches so
        # they don't appear "between". RAM-relative Loads would still
        # break the inter-dispatch run, but in our check there's
        # nothing between td1 and td2.
        impl_a = _empty_impl_func('Impl1__ma')
        impl_b = _empty_impl_func('Impl1__mb')
        info = {
            'TraitA': {
                'is_far': True, 'methods': ['ma'],
                'implementors': [{'struct': 'A', 'type_id': 1,
                                  'mangled': ['Impl1__ma']}],
            },
            'TraitB': {
                'is_far': True, 'methods': ['mb'],
                'implementors': [{'struct': 'B', 'type_id': 2,
                                  'mangled': ['Impl1__mb']}],
            },
        }
        prog = MIRProgram(
            functions=[
                _make_func('caller', {0: _make_block(0, [
                    ld1, ld2, td1, td2, Return()
                ])}),
                impl_a, impl_b,
            ],
            trait_dispatch_info=info,
        )
        _run_analysis(prog)
        # Different roots — no chain.
        assert td1.self_chain_role == ChainRole.SOLO
        assert td2.self_chain_role == ChainRole.SOLO

    def test_chain_self_root_stops_at_field_offset(self):
        """A cast that includes pointer arithmetic / field offset
        (modeled here as BinaryOp) is NOT considered same-root; its
        own pointer is its root."""
        root_v = VirtualRegister(id=100, type_info=_ptr_to_struct('Obj'))
        offset_v = VirtualRegister(id=101, type_info=_ptr_to_trait('SomeTrait'))
        bop = BinaryOp(
            dest=offset_v, left=root_v, right=Immediate(value=8),
            op='+', type_info=_ptr_to_trait('SomeTrait'),
        )
        # Two dispatches — one on root_v, one on offset_v. They should
        # NOT chain.
        td_root = _make_dispatch(root_v, trait='RootTrait', method='r')
        td_off = _make_dispatch(offset_v, trait='SomeTrait', method='s')
        impl_r = _empty_impl_func('Impl1__r')
        impl_s = _empty_impl_func('Impl1__s')
        info = {
            'RootTrait': {
                'is_far': True, 'methods': ['r'],
                'implementors': [{'struct': 'Obj', 'type_id': 1,
                                  'mangled': ['Impl1__r']}],
            },
            'SomeTrait': {
                'is_far': True, 'methods': ['s'],
                'implementors': [{'struct': 'Sub', 'type_id': 2,
                                  'mangled': ['Impl1__s']}],
            },
        }
        prog = MIRProgram(
            functions=[
                _make_func('caller', {0: _make_block(0, [
                    bop, td_root, td_off, Return()
                ])}),
                impl_r, impl_s,
            ],
            trait_dispatch_info=info,
        )
        _run_analysis(prog)
        # Distinct roots — no chain.
        assert td_root.self_chain_role == ChainRole.SOLO
        assert td_off.self_chain_role == ChainRole.SOLO


# ---------------------------------------------------------------------------
# v2 commit A: cross-block extension via CFG diamonds
# ---------------------------------------------------------------------------

class TestDiamondChainExtension:
    """v2(A): chain extends through if/else diamonds when both arms are
    DBR-independent and don't redefine the chain root self vreg."""

    def test_chain_extends_through_diamond(self):
        """Dispatch in entry block, both arms are DBR-independent (each
        does only a ROM-LONG read), dispatch in join block. The chain
        should extend across the diamond."""
        s = _make_self_vreg()
        a = _make_dispatch(s)
        b = _make_dispatch(s)
        # Both arms read from ROM (DBR-independent).
        rom_loc = MemoryLocation(
            storage_type='rom', address=0x008000, symbol=None,
        )
        rom_dest_v1 = VirtualRegister(id=200, type_info=BasicTypeInfo('u8'))
        rom_dest_v2 = VirtualRegister(id=201, type_info=BasicTypeInfo('u8'))
        # Block layout:
        #   0: a; branch to (1, 2)
        #   1: load ROM; jump to 3
        #   2: load ROM; jump to 3
        #   3: b; return
        blocks = {
            0: _make_block(0, [a], succs=[1, 2]),
            1: _make_block(1, [
                Load(dest=rom_dest_v1, source=rom_loc,
                     type_info=BasicTypeInfo('u8')),
            ], preds=[0], succs=[3]),
            2: _make_block(2, [
                Load(dest=rom_dest_v2, source=rom_loc,
                     type_info=BasicTypeInfo('u8')),
            ], preds=[0], succs=[3]),
            3: _make_block(3, [b, Return()], preds=[1, 2]),
        }
        prog = _build_program(blocks)
        _run_analysis(prog)
        assert a.self_chain_role == ChainRole.START
        assert b.self_chain_role == ChainRole.END

    def test_chain_breaks_when_one_arm_does_ram_access(self):
        """Same diamond shape but one arm does a non-LONG RAM access.
        The bridge must be rejected and both dispatches remain SOLO."""
        s = _make_self_vreg()
        a = _make_dispatch(s)
        b = _make_dispatch(s)
        ram_loc = MemoryLocation(
            storage_type='ram', address=0x7E0800, symbol=None,
        )
        rom_loc = MemoryLocation(
            storage_type='rom', address=0x008000, symbol=None,
        )
        d1 = VirtualRegister(id=210, type_info=BasicTypeInfo('u8'))
        d2 = VirtualRegister(id=211, type_info=BasicTypeInfo('u8'))
        blocks = {
            0: _make_block(0, [a], succs=[1, 2]),
            1: _make_block(1, [
                Load(dest=d1, source=ram_loc,
                     type_info=BasicTypeInfo('u8')),
            ], preds=[0], succs=[3]),
            2: _make_block(2, [
                Load(dest=d2, source=rom_loc,
                     type_info=BasicTypeInfo('u8')),
            ], preds=[0], succs=[3]),
            3: _make_block(3, [b, Return()], preds=[1, 2]),
        }
        prog = _build_program(blocks)
        _run_analysis(prog)
        assert a.self_chain_role == ChainRole.SOLO
        assert b.self_chain_role == ChainRole.SOLO

    def test_chain_breaks_when_arm_redefines_root(self):
        """One arm reassigns the self vreg. The chain must break."""
        s = _make_self_vreg()
        a = _make_dispatch(s)
        b = _make_dispatch(s)
        # Arm 1 redefines s via a Move from Immediate (DBR-independent
        # in itself, but invalidates the chain root).
        redef = Move(
            dest=s,
            source=Immediate(value=0x1234),
            type_info=BasicTypeInfo('u16'),
        )
        blocks = {
            0: _make_block(0, [a], succs=[1, 2]),
            1: _make_block(1, [redef], preds=[0], succs=[3]),
            2: _make_block(2, [], preds=[0], succs=[3]),
            3: _make_block(3, [b, Return()], preds=[1, 2]),
        }
        prog = _build_program(blocks)
        _run_analysis(prog)
        assert a.self_chain_role == ChainRole.SOLO
        assert b.self_chain_role == ChainRole.SOLO

    def test_chain_breaks_on_loop_back_edge(self):
        """A back-edge from a candidate site rejects the diamond bridge."""
        s = _make_self_vreg()
        a = _make_dispatch(s)
        b = _make_dispatch(s)
        # Block 0 has 2 successors but block 2 is a back-edge into 0
        # (so 0 has predecessor 2 — visited will not contain 2 when
        # we first reach 0, but the arm shape is wrong). Actually the
        # cleanest way: make 0 → 1 → 0 (a loop). The diamond walker
        # rejects when arm targets are already in visited / when arm
        # successors don't form a strict diamond.
        blocks = {
            0: _make_block(0, [a], preds=[1], succs=[1, 2]),
            1: _make_block(1, [b], preds=[0], succs=[0]),  # back-edge
            2: _make_block(2, [Return()], preds=[0]),
        }
        prog = _build_program(blocks)
        _run_analysis(prog)
        # Loops are rejected entirely; dispatches stay SOLO.
        assert a.self_chain_role == ChainRole.SOLO
        assert b.self_chain_role == ChainRole.SOLO

    def test_chain_breaks_on_three_way_branch(self):
        """A switch-like CFG with 3 successors is rejected outright."""
        s = _make_self_vreg()
        a = _make_dispatch(s)
        b = _make_dispatch(s)
        blocks = {
            0: _make_block(0, [a], succs=[1, 2, 3]),
            1: _make_block(1, [], preds=[0], succs=[4]),
            2: _make_block(2, [], preds=[0], succs=[4]),
            3: _make_block(3, [], preds=[0], succs=[4]),
            4: _make_block(4, [b, Return()], preds=[1, 2, 3]),
        }
        prog = _build_program(blocks)
        _run_analysis(prog)
        assert a.self_chain_role == ChainRole.SOLO
        assert b.self_chain_role == ChainRole.SOLO

    def test_chain_breaks_when_arm_has_extra_successors(self):
        """An arm whose own successor count != 1 is rejected (nested
        branching is not a strict diamond)."""
        s = _make_self_vreg()
        a = _make_dispatch(s)
        b = _make_dispatch(s)
        blocks = {
            0: _make_block(0, [a], succs=[1, 2]),
            # Arm 1 has 2 successors of its own — nested diamond.
            1: _make_block(1, [], preds=[0], succs=[3, 5]),
            2: _make_block(2, [], preds=[0], succs=[3]),
            3: _make_block(3, [b, Return()], preds=[1, 2, 5]),
            5: _make_block(5, [], preds=[1], succs=[3]),
        }
        prog = _build_program(blocks)
        _run_analysis(prog)
        assert a.self_chain_role == ChainRole.SOLO
        assert b.self_chain_role == ChainRole.SOLO
