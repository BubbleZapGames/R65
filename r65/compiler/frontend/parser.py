# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Parser for R65 using Lark.

Transforms Lark parse trees into our custom AST.
"""
from pathlib import Path
from lark import Lark, Transformer, Token as LarkToken, v_args
from lark.exceptions import UnexpectedToken, UnexpectedCharacters, UnexpectedEOF, VisitError
from r65.compiler.frontend import ast
from r65.compiler.frontend.derive_compat import derive_hint, derive_trait_names
from r65.compiler.frontend.tokens import (
    split_integer_suffix, parse_integer, decode_literal, ESCAPE_HINT,
)
from r65.compiler.errors import ParseError, SourceLocation, get_source_line
from dataclasses import dataclass
from typing import List, Optional


# Load the grammar
GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"
with open(GRAMMAR_PATH) as f:
    GRAMMAR = f.read()

# Options the LALR tables are built from. Named once so the cache and the
# fallback cannot drift apart.
_LARK_OPTIONS = dict(
    parser='lalr',              # LALR is much faster than Earley
    lexer='contextual',         # Contextual lexer for LALR
    start='start',
    keep_all_tokens=True,
    propagate_positions=True,   # Enable source location tracking
)

# Shared by the lexer (--dump-tokens), which lexes with this same instance.
# Matches `*.lark_cache` in .gitignore.
_CACHE_PATH = GRAMMAR_PATH.with_suffix(".parser.lark_cache")


def _build_lark() -> Lark:
    """Build the Lark parser, reusing serialized LALR tables when possible.

    Building the LALR tables costs ~0.6s per process (the e2e suite starts
    one per test, so the in-process `Parser._lark_cache` does not help it);
    loading them back is ~0.05s. Lark invalidates the cache itself when the
    grammar or the options change, so there is no staleness to manage here.
    """
    try:
        return Lark(GRAMMAR, cache=str(_CACHE_PATH), **_LARK_OPTIONS)
    except Exception:
        # An installed package may sit on a read-only path, and a cache torn
        # by a killed or concurrent writer will fail to load. Neither is a
        # reason to refuse to compile. Drop a bad cache so the next run can
        # rebuild it rather than silently paying the full cost forever.
        try:
            _CACHE_PATH.unlink(missing_ok=True)
        except OSError:
            pass
        return Lark(GRAMMAR, **_LARK_OPTIONS)


def render_ast_type(node) -> str:
    """Render an AST type node back to source form, for use in error hints."""
    if isinstance(node, ast.BasicType):
        return node.name
    if isinstance(node, ast.PointerType):
        return ('far *' if node.is_far else '*') + render_ast_type(node.pointee_type)
    if isinstance(node, ast.ArrayType):
        size = getattr(node.size, 'value', None)
        return f"[{render_ast_type(node.element_type)}; {size if size is not None else 'N'}]"
    if isinstance(node, ast.Identifier):
        return node.name
    return '<type>'



def _tok(item):
    """A Lark token's text, or *item* itself when it is already a value."""
    return item.value if isinstance(item, LarkToken) else item


@dataclass
class _TailBlock:
    """An if or else body as parsed, with its trailing expression apart.

    ASTBuilder._if_forms turns it into a Block or a BlockExpression once the
    whole `if` is known; it never leaves the builder.
    """
    statements: list
    tail: Optional[ast.Expression]
    source_loc: Optional[SourceLocation]


class ASTBuilder(Transformer):
    """
    Lark Transformer that builds our AST from the parse tree.

    Each method corresponds to a grammar rule and transforms
    the matched nodes into AST nodes.
    """

    def __init__(self, filename: str = "<input>", included_from=None, origin_package=None):
        """
        Initialize the AST builder.

        Args:
            filename: Current source file name
            included_from: SourceLocation of include! statement if this is included
            origin_package: The package the file belongs to ("stdlib 0.4.0")
        """
        super().__init__()
        self.filename = filename
        self.included_from = included_from
        self.origin_package = origin_package

    def _make_source_loc(self, meta):
        """Create a SourceLocation from Lark meta info."""
        if meta is None:
            return None
        return SourceLocation(
            file_path=self.filename,
            line=getattr(meta, 'line', 0),
            column=getattr(meta, 'column', 0),
            included_from=self.included_from,
            origin_package=self.origin_package,
        )

    def _source_loc_from_token(self, token):
        """Create a SourceLocation from a LarkToken's line/column."""
        if not isinstance(token, LarkToken):
            return None
        return SourceLocation(
            file_path=self.filename,
            line=getattr(token, 'line', 0) or 0,
            column=getattr(token, 'column', 0) or 0,
            included_from=self.included_from,
            origin_package=self.origin_package,
        )

    def _take(self, items, idx, tok_type):
        """Consume items[idx] if it is a LarkToken of tok_type.

        Returns (present, new_idx). Collapses the recurring optional-modifier
        scan (FAR/NEAR/MUT/CONST/STAR/DYN) into one call.
        """
        if idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == tok_type:
            return True, idx + 1
        return False, idx

    def _take_call_kind(self, items, idx):
        """Consume an optional FAR or NEAR function modifier.

        A bare `fn` is far but demotable; `far fn` pins it far.
        Returns (call_kind, new_idx).
        """
        is_far, is_near, idx = self._take_far_near(items, idx)
        if is_far:
            return ast.CallKind.FORCE_FAR, idx
        return (ast.CallKind.NEAR if is_near else ast.CallKind.FAR), idx

    def _take_far_near(self, items, idx):
        """Consume an optional FAR or NEAR pointer modifier (mutually exclusive).

        Returns (is_far, is_near, new_idx).
        """
        is_far, idx = self._take(items, idx, 'FAR')
        if is_far:
            return True, False, idx
        is_near, idx = self._take(items, idx, 'NEAR')
        return False, is_near, idx

    def _filter_tokens(self, items, keep_types=None):
        """
        Filter out punctuation tokens, keeping only semantic content.

        Args:
            items: List of items from Lark
            keep_types: Optional set of token types to keep

        Returns:
            List with only AST nodes and relevant tokens
        """
        if keep_types is None:
            keep_types = {'IDENT', 'MUT', 'FAR', 'INTEGER', 'STRING', 'BOOLEAN', 'REGISTER'}

        result = []
        for item in items:
            if isinstance(item, LarkToken):
                if item.type in keep_types:
                    result.append(item)
            else:
                # Keep all AST nodes
                result.append(item)
        return result

    def _collect_attributes(self, items: list, start_idx: int):
        """
        Collect attributes from items list starting at index.

        Args:
            items: Filtered items list
            start_idx: Starting index

        Returns:
            Tuple of (attributes_list, next_index)
        """
        attrs = []
        idx = start_idx
        while idx < len(items) and isinstance(items[idx], ast.Attribute):
            attrs.append(items[idx])
            idx += 1
        return attrs, idx

    @staticmethod
    def _strip_doc_comment(text: str) -> str:
        """Strip doc comment prefix from a single comment token."""
        if text.startswith('///'):
            return text[3:]  # Strip ///
        elif text.startswith('//!'):
            return text[3:]  # Strip //!
        elif text.startswith(('/**', '/*!')):
            # Block doc comment (outer /** or inner /*!): strip delimiters and
            # the common leading "* " from each line.
            inner = text[3:-2]
            stripped = []
            for line in inner.split('\n'):
                s = line.lstrip()
                if s.startswith('* '):
                    stripped.append(s[2:])
                elif s.startswith('*') and (len(s) == 1 or not s[1:].strip()):
                    stripped.append('')
                else:
                    stripped.append(line)
            return '\n'.join(stripped)
        return text

    def _collect_doc_comments(self, items: list, start_idx: int):
        """
        Collect doc comment strings from items list starting at index.

        Returns:
            Tuple of (doc_string_or_None, next_index)
        """
        doc_parts = []
        idx = start_idx
        while idx < len(items) and isinstance(items[idx], str) and items[idx].startswith('__doc__:'):
            doc_parts.append(items[idx][8:])  # Strip __doc__: prefix
            idx += 1
        doc = '\n'.join(doc_parts).strip() if doc_parts else None
        return doc, idx

    def doc_comment(self, items):
        """Transform doc comment rule into a tagged string."""
        token = items[0]
        return '__doc__:' + self._strip_doc_comment(str(token))

    # ========================================================================
    # Program
    # ========================================================================

    # Messages for reserved Rust keywords that appear as dangling Token items
    # via the grammar's _reserved_keyword rule. Maps keyword → (error_msg, hint).
    _KEYWORD_MESSAGES = {
        'pub':      ("'pub' visibility is not supported in R65", "all declarations are globally visible; remove 'pub'"),
        'async':    ("'async' functions are not supported in R65", "use interrupt handlers (#[interrupt]) for async events"),
        'await':    ("'await' is not supported in R65", "R65 has no async runtime"),
        'unsafe':   ("'unsafe' is not supported in R65", "all R65 code has direct hardware access; 'unsafe' is unnecessary"),
        'use':      ("'use' imports are not supported in R65", "use include!(\"file.r65\") for file inclusion"),
        'crate':    ("'crate' is not supported in R65", "use include!(\"file.r65\") for file inclusion"),
        'move':     ("'move' closures are not supported in R65", "use function pointers (fn() or far fn())"),
        'ref':      ("'ref' patterns are not supported in R65", "use raw pointers (*u8)"),
        'where':    ("'where' clauses are not supported in R65", "R65 has no generics or trait bounds"),
        'yield':    ("'yield' is not supported in R65", "R65 has no generators"),
        'super':    ("'super' is not supported in R65", "R65 has no module system; use include!()"),
        'abstract': ("'abstract' is not supported in R65", "use traits for polymorphism"),
        'try':      ("'try' is not supported in R65", "use return codes or error flags"),
        'box':      ("'box' is not supported in R65", "use raw pointers (*u8)"),
        'do':       ("'do' is not supported in R65", "use loop or while"),
        'priv':     ("'priv' is not supported in R65", "all declarations are globally visible"),
    }

    def start(self, items):
        """Start rule - returns a Program node."""
        # Collect inner doc comments (//! and /*! */) from the beginning
        inner_doc_parts = []
        decl_items = []
        for item in items:
            if isinstance(item, LarkToken) and item.type in ('DOC_INNER', 'DOC_BLOCK_INNER'):
                inner_doc_parts.append(self._strip_doc_comment(str(item)))
            elif isinstance(item, LarkToken) and item.type == 'KEYWORD':
                kw = item.value
                msg, hint = self._KEYWORD_MESSAGES.get(kw, (
                    f"'{kw}' is a reserved keyword not supported in R65", None
                ))
                source_loc = SourceLocation(
                    file_path=self.filename,
                    line=getattr(item, 'line', 0),
                    column=getattr(item, 'column', 0),
                )
                raise ParseError(msg, source_loc, hint=hint)
            else:
                decl_items.append(item)
        doc = '\n'.join(inner_doc_parts).strip() if inner_doc_parts else None
        return ast.Program(items=decl_items, doc=doc)

    # ========================================================================
    # Declarations
    # ========================================================================

    @v_args(tree=True)
    def function_decl(self, tree):
        """Function declaration."""
        items = self._filter_tokens(tree.children, keep_types={'IDENT', 'MUT', 'FAR', 'NEAR', 'CONST', 'INTEGER', 'STRING', 'BOOLEAN', 'REGISTER'})

        # Collect doc comments
        doc, idx = self._collect_doc_comments(items, 0)

        # Collect attributes
        attrs, idx = self._collect_attributes(items, idx)

        # Check for const
        is_const, idx = self._take(items, idx, 'CONST')

        # Functions are far by default; `near fn` opts out, and `far fn`
        # pins the function far so far_to_near never demotes it
        call_kind, idx = self._take_call_kind(items, idx)

        name = items[idx]
        idx += 1

        # Params (optional)
        params = []
        if idx < len(items) and isinstance(items[idx], list):
            params = items[idx]
            idx += 1

        # Return type (optional)
        return_type = None
        if idx < len(items) and not isinstance(items[idx], ast.Block):
            return_type = items[idx]
            idx += 1

        # Body
        body = items[idx]

        return ast.FunctionDecl(
            attributes=attrs,
            call_kind=call_kind,
            name=_tok(name),
            params=params,
            return_type=return_type,
            body=body,
            is_const=is_const,
            doc=doc,
            source_loc=self._make_source_loc(tree.meta)
        )

    def param_list(self, items):
        """Parameter list."""
        # Filter out commas, keep only Parameter nodes
        return [item for item in items if isinstance(item, ast.Parameter)]

    def param(self, items):
        """Function parameter."""
        items = self._filter_tokens(items, keep_types={'IDENT', 'FAR', 'NEAR', 'STAR', 'REGISTER'})

        idx = 0

        # Check for far/near modifier (for pointer parameters)
        is_far, is_near, idx = self._take_far_near(items, idx)

        # Check for pointer (*) prefix
        is_pointer, idx = self._take(items, idx, 'STAR')

        # Name
        name = _tok(items[idx])
        idx += 1

        # Check for binding (@ register or variable)
        binding = None
        if idx < len(items) - 1:  # There's something between name and type
            binding_node = items[idx]
            binding = binding_node
            idx += 1

        # Type is always last (this is the pointee type if is_pointer is True)
        param_type = items[-1]

        # If this is a pointer parameter, wrap the type in PointerType
        if is_pointer:
            param_type = ast.PointerType(is_far=is_far, pointee_type=param_type)

        return ast.Parameter(name=name, binding=binding, param_type=param_type)

    @v_args(tree=True)
    def _safe_ptr_error_parts(self, tree, *, with_mut=False):
        """Extract (name_str, mut_str, source_loc) for a safe-pointer error hint.

        Shared by the param/static/struct-field/let safe-pointer error rules.
        with_mut also recognizes a leading MUT token (static declarations).
        """
        keep = {'IDENT', 'AMPER', 'MUT'} if with_mut else {'IDENT', 'AMPER'}
        items = self._filter_tokens(tree.children, keep_types=keep)
        name = None
        is_mut = False
        for item in items:
            if isinstance(item, LarkToken):
                if item.type == 'MUT':
                    is_mut = True
                elif item.type == 'IDENT':
                    name = item.value
                    break
            elif isinstance(item, tuple) and len(item) >= 2:
                # safe_ptr_pattern returns a tuple (let statements)
                name = item[1]
                break
        name_str = name if name else "name"
        mut_str = "mut " if is_mut else ""
        source_loc = self._make_source_loc(tree.meta) if hasattr(tree, 'meta') else None
        return name_str, mut_str, source_loc

    def param_safe_ptr_error(self, tree):
        """Error handler for safe pointer syntax in parameters."""
        name, _, loc = self._safe_ptr_error_parts(tree)
        raise ParseError(
            "safe pointers are not supported in R65", source_loc=loc,
            hint=f"use '{name}: *type' instead of '&{name}: type' for pointer parameters")

    def binding(self, items):
        """Binding for @ operator."""
        item = items[0]
        if item.type == 'REGISTER':
            return ast.Register(name=item.value)
        else:
            return ast.Identifier(name=item.value)

    def return_type(self, items):
        """Return type."""
        items = self._filter_tokens(items, keep_types={'EXCLAMATION'})
        if items:
            item = items[0]
            if isinstance(item, LarkToken) and item.value == '!':
                return ast.NeverType()
            return item
        return None

    @v_args(tree=True)
    def static_decl(self, tree):
        """Static variable declaration."""
        items = self._filter_tokens(tree.children, keep_types={'IDENT', 'MUT', 'FAR', 'NEAR', 'STAR', 'INTEGER', 'STRING', 'BOOLEAN', 'REGISTER'})

        # Collect doc comments
        doc, idx = self._collect_doc_comments(items, 0)

        # Collect attributes
        attrs, idx = self._collect_attributes(items, idx)

        # Check for far/near modifier (for pointer declarations)
        is_far, is_near, idx = self._take_far_near(items, idx)

        # Check for mut token
        is_mut, idx = self._take(items, idx, 'MUT')

        # Check for pointer (*) prefix
        is_pointer, idx = self._take(items, idx, 'STAR')

        # Name
        name = _tok(items[idx])
        idx += 1

        # Type (optional - can be inferred for include_bytes!)
        var_type = None
        if idx < len(items) and isinstance(items[idx], ast.Type):
            var_type = items[idx]
            idx += 1

        # If this is a pointer declaration, wrap the type in PointerType
        # Note: The is_far on StaticDecl is for auto-bank (far static),
        # not for the pointer type. For far pointers, use type: far *T
        if is_pointer:
            if var_type is None:
                raise self._make_error("pointer static declarations require a type annotation",
                                       tree.meta)
            var_type = ast.PointerType(is_far=False, pointee_type=var_type)

        # Initializer (optional)
        initializer = items[idx] if idx < len(items) else None

        return ast.StaticDecl(
            attributes=attrs,
            is_far=is_far,
            is_mut=is_mut,
            name=name,
            var_type=var_type,
            initializer=initializer,
            doc=doc,
            source_loc=self._make_source_loc(tree.meta)
        )

    def var_name(self, items):
        """Variable name - can be IDENT or REGISTER token."""
        items = self._filter_tokens(items, keep_types={'IDENT', 'REGISTER'})
        token = items[0]
        return _tok(token)

    @v_args(tree=True)
    def static_decl_safe_ptr_error(self, tree):
        """Error handler for safe pointer syntax in static declarations."""
        name_str, mut_str, source_loc = self._safe_ptr_error_parts(tree, with_mut=True)
        raise ParseError(
            f"safe pointers are not supported in R65",
            source_loc=source_loc,
            hint=f"use 'static {mut_str}{name_str}: *type' instead of 'static {mut_str}&{name_str}: type'"
        )

    @v_args(tree=True)
    def const_decl(self, tree):
        """Const declaration."""
        items = self._filter_tokens(tree.children)
        doc, idx = self._collect_doc_comments(items, 0)
        name = _tok(items[idx])
        const_type = items[idx + 1]
        value = items[idx + 2]
        return ast.ConstDecl(name=name, const_type=const_type, value=value,
                             doc=doc, source_loc=self._make_source_loc(tree.meta))

    @v_args(tree=True)
    def struct_decl(self, tree):
        """Struct declaration."""
        return self._aggregate_decl(
            tree.children, is_union=False,
            source_loc=self._make_source_loc(tree.meta))

    @v_args(tree=True)
    def union_decl(self, tree):
        """Union declaration - same field syntax as a struct, overlaid layout."""
        return self._aggregate_decl(
            tree.children, is_union=True,
            source_loc=self._make_source_loc(tree.meta))

    @v_args(tree=True)
    def newtype_decl(self, tree):
        """Newtype declaration: struct TileId(u8);"""
        items = self._filter_tokens(tree.children, keep_types={'IDENT'})
        doc, idx = self._collect_doc_comments(items, 0)
        loc = self._make_source_loc(tree.meta)
        attrs, idx = self._collect_attributes(items, idx)
        self._reject_derive_on_type(attrs, items, idx, loc)
        name = _tok(items[idx])
        inner_type = items[idx + 1]
        return ast.NewtypeDecl(
            name=name,
            inner_type=inner_type,
            doc=doc,
            attributes=attrs,
            source_loc=loc
        )

    @v_args(tree=True)
    def newtype_arity_error(self, tree):
        """Error handler for a newtype with anything but one field: struct Pair(u8, u8);

        The single-field form is the newtype and parses (see `newtype_decl`);
        only two-or-more fields — and the empty form — land here.
        """
        items = self._filter_tokens(tree.children, keep_types={'IDENT'})
        _, idx = self._collect_doc_comments(items, 0)
        _, idx = self._collect_attributes(items, idx)
        name = _tok(items[idx])
        types = [item for item in items[idx + 1:] if not isinstance(item, LarkToken)]

        # Give each positional field a name the author can rename.
        if types:
            fields = ', '.join(
                f"field{i}: {render_ast_type(t)}" for i, t in enumerate(types))
        else:
            fields = "value: <type>"

        raise ParseError(
            "a newtype wraps exactly one field",
            source_loc=self._make_source_loc(tree.meta),
            hint=f"give each field a name: 'struct {name} {{ {fields} }}'"
        )

    @v_args(tree=True)
    def newtype_field_access(self, tree):
        """Newtype payload access: t.0"""
        base = None
        index = 0
        for item in tree.children:
            if isinstance(item, LarkToken) and item.type == 'INTEGER':
                index = int(item.value, 0)
            elif not isinstance(item, LarkToken):
                base = item
        return ast.NewtypeFieldAccess(
            base=base,
            index=index,
            source_loc=self._make_source_loc(tree.meta)
        )

    def _aggregate_decl(self, items, is_union: bool, source_loc=None):
        """Build a StructDecl from struct_decl/union_decl children."""
        items = self._filter_tokens(items)
        doc, idx = self._collect_doc_comments(items, 0)
        attrs, idx = self._collect_attributes(items, idx)
        self._reject_derive_on_type(attrs, items, idx, source_loc)
        name = _tok(items[idx])
        fields = [item for item in items[idx + 1:] if isinstance(item, ast.StructField)]
        return ast.StructDecl(name=name, fields=fields, doc=doc, is_union=is_union,
                              attributes=attrs, source_loc=source_loc)

    def _reject_derive_on_type(self, attrs, items, idx, source_loc):
        """Reject `#[derive(...)]` on a type declaration, naming the type.

        Type declarations accept attributes (`#[soa]` on a struct), so
        `#[derive(Clone)] struct Point` parses; reject it here, where the type
        name is known for the hint (the HIR attribute processor has none). The
        source scan in `Parser._check_rust_feature_hints` covers
        `enum`/`trait`/`type`, which take no attributes.
        """
        for attr in attrs:
            if attr.name not in ('derive', 'proc_macro', 'proc_macro_derive',
                                 'proc_macro_attribute'):
                continue
            name = None
            if idx < len(items):
                name = (items[idx].value if isinstance(items[idx], LarkToken)
                        else getattr(items[idx], 'name', None))
            if attr.name == 'derive':
                raise ParseError(
                    "#[derive(...)] is not supported in R65",
                    source_loc=attr.source_loc or source_loc,
                    hint=derive_hint(derive_trait_names(attr), name))
            raise ParseError(
                f"#[{attr.name}] is not supported in R65",
                source_loc=attr.source_loc or source_loc,
                hint="procedural macros are not supported in R65 — use "
                     "`macro_rules!` for declarative macros (see docs/macros.md)")

    def struct_field(self, items):
        """Struct field."""
        items = self._filter_tokens(items, keep_types={'IDENT', 'FAR', 'NEAR', 'STAR'})

        # Collect doc comments
        doc, idx = self._collect_doc_comments(items, 0)

        # Check for far/near modifier (for pointer fields)
        is_far, is_near, idx = self._take_far_near(items, idx)

        # Check for pointer (*) prefix
        is_pointer, idx = self._take(items, idx, 'STAR')

        # Name
        name = _tok(items[idx])
        idx += 1

        # Type (this is the pointee type if is_pointer is True)
        field_type = items[idx]

        # If this is a pointer field, wrap the type in PointerType
        if is_pointer:
            field_type = ast.PointerType(is_far=is_far, pointee_type=field_type)

        return ast.StructField(name=name, field_type=field_type, doc=doc)

    @v_args(tree=True)
    def struct_field_safe_ptr_error(self, tree):
        """Error handler for safe pointer syntax in struct fields."""
        name, _, loc = self._safe_ptr_error_parts(tree)
        raise ParseError(
            "safe pointers are not supported in R65", source_loc=loc,
            hint=f"use '{name}: *type' instead of '&{name}: type' for struct fields")

    def enum_decl(self, items):
        """Enum declaration."""
        items = self._filter_tokens(items)
        doc, idx = self._collect_doc_comments(items, 0)
        name = _tok(items[idx])
        variants = [item for item in items[idx + 1:] if isinstance(item, ast.EnumVariant)]
        return ast.EnumDecl(name=name, variants=variants, doc=doc)

    def enum_variant(self, items):
        """Enum variant."""
        items = self._filter_tokens(items)
        name = _tok(items[0])
        value = items[1] if len(items) > 1 else None
        return ast.EnumVariant(name=name, value=value)

    def enum_variant_payload(self, items):
        """Enum variant carrying a payload: `C(Circle)`.

        The payload type is validated in the HIR builder, not here — it has to
        name a declared struct, and nothing at parse time knows what is declared.
        """
        items = self._filter_tokens(items)
        name = _tok(items[0])
        return ast.EnumVariant(name=name, value=None, payload_type=items[1])

    @v_args(tree=True)
    def impl_decl(self, tree):
        """Impl block declaration: impl StructName { methods and constants }

        Near/far is per method: far (JSL/RTL) by default, `near fn` for
        JSR/RTS. The self-pointer spelling (`*self` / `far *self`) is a
        separate axis.
        """
        items = self._filter_tokens(tree.children, keep_types={'IDENT'})

        # Collect doc comments
        doc, idx = self._collect_doc_comments(items, 0)

        # Struct name
        struct_name = _tok(items[idx])
        idx += 1

        # Collect methods, constants, and macros from remaining items
        methods = []
        constants = []
        macros = []
        for item in items[idx:]:
            if isinstance(item, ast.ImplMethod):
                methods.append(item)
            elif isinstance(item, ast.ImplConst):
                constants.append(item)
            elif isinstance(item, ast.MacroDecl):
                macros.append(item)

        return ast.ImplDecl(
            struct_name=struct_name,
            methods=methods,
            constants=constants,
            macros=macros,
            doc=doc,
            source_loc=self._make_source_loc(tree.meta)
        )

    @v_args(tree=True)
    def impl_method(self, tree):
        """Method declaration in impl block."""
        items = self._filter_tokens(tree.children, keep_types={'IDENT', 'FAR', 'NEAR', 'CONST'})

        # Collect doc comments, then attributes
        doc, idx = self._collect_doc_comments(items, 0)
        attrs, idx = self._collect_attributes(items, idx)

        # Check for const
        is_const, idx = self._take(items, idx, 'CONST')

        # Methods are far by default; `near fn` opts out, `far fn` pins far
        call_kind, idx = self._take_call_kind(items, idx)

        # Method name
        name = _tok(items[idx])
        idx += 1

        # Parameters (from impl_param_list)
        self_is_far = False
        self_by_value = False
        has_self = False
        params = []
        if idx < len(items):
            param_result = items[idx]
            if isinstance(param_result, tuple) and param_result[0] == 'impl_params':
                self_is_far = param_result[1]
                params = param_result[2]
                self_by_value = param_result[3]
                has_self = True
                idx += 1
            elif isinstance(param_result, list):
                # Regular param_list (no self)
                params = param_result
                idx += 1

        # Return type (optional)
        return_type = None
        if idx < len(items) and not isinstance(items[idx], ast.Block):
            return_type = items[idx]
            idx += 1

        # Body
        body = items[idx] if idx < len(items) else ast.Block(statements=[])

        return ast.ImplMethod(
            attributes=attrs,
            call_kind=call_kind,
            name=name,
            self_is_far=self_is_far,
            params=params,
            return_type=return_type,
            body=body,
            is_const=is_const,
            doc=doc,
            self_by_value=self_by_value,
            has_self=has_self,
            source_loc=self._make_source_loc(tree.meta)
        )

    def impl_param_list(self, items):
        """Parameter list for impl methods - may include self parameter."""
        # Check if first item is a self_param tuple
        if items and isinstance(items[0], tuple) and items[0][0] == 'self_param':
            self_is_far = items[0][1]
            self_by_value = items[0][2]
            # Rest are regular parameters
            params = [item for item in items[1:] if isinstance(item, ast.Parameter)]
            return ('impl_params', self_is_far, params, self_by_value)
        else:
            # No self param — an associated function. This alternative reduces to
            # `param_list`, which is itself a list, so `items` is [[Parameter, ...]]
            # and needs flattening; filtering it directly silently drops every
            # parameter.
            params = []
            for item in items:
                if isinstance(item, ast.Parameter):
                    params.append(item)
                elif isinstance(item, list):
                    params.extend(p for p in item if isinstance(p, ast.Parameter))
            return params

    def self_param(self, items):
        """Self parameter: *self, far *self, or near *self"""
        items = self._filter_tokens(items, keep_types={'FAR', 'NEAR', 'STAR', 'SELF'})

        is_far = False
        for item in items:
            if isinstance(item, LarkToken):
                if item.type == 'FAR':
                    is_far = True
                elif item.type == 'NEAR':
                    is_far = False

        return ('self_param', is_far, False)

    def self_by_value(self, items):
        """Bare `self`: the value in A (newtype, fieldless enum) or the element
        index in X (#[soa] struct, tagged enum). HIR checks it matches the type."""
        return ('self_param', False, True)

    @v_args(tree=True)
    def self_safe_ptr_error(self, tree):
        """Error handler for &self syntax (Rust safe reference)."""
        source_loc = self._make_source_loc(tree.meta) if hasattr(tree, 'meta') else None
        raise ParseError(
            "safe references are not supported in R65",
            source_loc=source_loc,
            hint="use '*self' instead of '&self' — R65 uses raw pointers"
        )

    def impl_const(self, items):
        """Associated constant in impl block: const NAME: type = value;"""
        items = self._filter_tokens(items)
        name = _tok(items[0])
        const_type = items[1]
        value = items[2]
        return ast.ImplConst(name=name, const_type=const_type, value=value)

    @v_args(tree=True)
    def impl_macro_short(self, tree):
        """Shorthand impl macro: macro_rules! name($param:type, ...) { body } (single arm)"""
        items = tree.children
        name = None
        params = []
        body_tokens = []
        doc, _ = self._collect_doc_comments(items, 0)

        for item in items:
            if isinstance(item, LarkToken) and item.type == 'IDENT':
                name = item.value
            elif isinstance(item, list):  # macro_params result
                params = item
            elif isinstance(item, ast.MacroParam):
                params.append(item)
            elif isinstance(item, tuple) and item[0] == 'macro_body':
                body_tokens = item[1]

        return ast.MacroDecl(
            name=name,
            arms=[ast.MacroArm(params=params, body_tokens=body_tokens)],
            doc=doc,
            source_loc=self._make_source_loc(tree.meta)
        )

    @v_args(tree=True)
    def impl_macro_multi(self, tree):
        """Multi-arm impl macro: macro_rules! name { (pat) => { body }; ... }"""
        items = tree.children
        name = None
        arms = []
        doc, _ = self._collect_doc_comments(items, 0)

        for item in items:
            if isinstance(item, LarkToken) and item.type == 'IDENT':
                name = item.value
            elif isinstance(item, ast.MacroArm):
                arms.append(item)

        return ast.MacroDecl(
            name=name,
            arms=arms,
            doc=doc,
            source_loc=self._make_source_loc(tree.meta)
        )

    @v_args(tree=True)
    def impl_trait_decl(self, tree):
        """Trait impl declaration: impl TraitName for StructName { methods and constants }"""
        items = self._filter_tokens(tree.children, keep_types={'IDENT'})

        # Collect doc comments
        doc, idx = self._collect_doc_comments(items, 0)

        # Trait name (first IDENT)
        trait_name = _tok(items[idx])
        idx += 1

        # Struct name (second IDENT, after FOR which is filtered out)
        struct_name = _tok(items[idx])
        idx += 1

        # Collect methods and constants from remaining items
        methods = []
        constants = []
        for item in items[idx:]:
            if isinstance(item, ast.ImplMethod):
                methods.append(item)
            elif isinstance(item, ast.ImplConst):
                constants.append(item)

        return ast.ImplDecl(
            struct_name=struct_name,
            methods=methods,
            constants=constants,
            trait_name=trait_name,
            doc=doc,
            source_loc=self._make_source_loc(tree.meta)
        )

    @v_args(tree=True)
    def trait_decl(self, tree):
        """Trait declaration: trait TraitName { methods and constants }"""
        items = self._filter_tokens(tree.children, keep_types={'IDENT', 'FAR'})

        # Collect doc comments
        doc, idx = self._collect_doc_comments(items, 0)

        # Trait name
        name = _tok(items[idx])
        idx += 1

        # Collect supertraits, methods, and constants
        supertraits = []
        methods = []
        constants = []
        for item in items[idx:]:
            if isinstance(item, tuple) and item and item[0] == 'trait_bounds':
                supertraits = item[1]
            elif isinstance(item, ast.ImplMethod):
                methods.append(item)
            elif isinstance(item, ast.TraitConst):
                constants.append(item)

        return ast.TraitDecl(
            name=name,
            methods=methods,
            constants=constants,
            supertraits=supertraits,
            doc=doc,
            source_loc=self._make_source_loc(tree.meta)
        )

    def trait_bounds(self, items):
        """Supertrait bounds: : Super1 + Super2"""
        names = [item.value for item in items
                 if isinstance(item, LarkToken) and item.type == 'IDENT']
        return ('trait_bounds', names)

    @v_args(tree=True)
    def trait_method(self, tree):
        """Method signature in trait declaration (no body)."""
        items = self._filter_tokens(tree.children, keep_types={'IDENT', 'FAR', 'NEAR'})

        doc, idx = self._collect_doc_comments(items, 0)

        # Trait methods are far by default; `near fn` opts out, `far fn`
        # pins every implementation far
        call_kind, idx = self._take_call_kind(items, idx)

        # Method name
        name = _tok(items[idx])
        idx += 1

        # Parameters (from trait_param_list)
        self_is_far = False
        self_by_value = False
        has_self = False
        params = []
        if idx < len(items):
            param_result = items[idx]
            if isinstance(param_result, tuple) and param_result[0] == 'impl_params':
                self_is_far = param_result[1]
                params = param_result[2]
                self_by_value = param_result[3]
                has_self = True
                idx += 1
            elif isinstance(param_result, list):
                params = param_result
                idx += 1

        # Return type (optional)
        return_type = None
        if idx < len(items) and not isinstance(items[idx], ast.Block):
            return_type = items[idx]
            idx += 1

        # Default body (optional) - present when the trait method has `{ ... }`
        default_body = None
        if idx < len(items) and isinstance(items[idx], ast.Block):
            default_body = items[idx]
            idx += 1

        return ast.ImplMethod(
            attributes=[],
            call_kind=call_kind,
            name=name,
            self_is_far=self_is_far,
            params=params,
            return_type=return_type,
            body=default_body,
            doc=doc,
            self_by_value=self_by_value,
            has_self=has_self,
            source_loc=self._make_source_loc(tree.meta)
        )

    def trait_param_list(self, items):
        """Parameter list for trait methods - self parameter required."""
        # Reuse impl_param_list logic
        if items and isinstance(items[0], tuple) and items[0][0] == 'self_param':
            self_is_far = items[0][1]
            self_by_value = items[0][2]
            params = [item for item in items[1:] if isinstance(item, ast.Parameter)]
            return ('impl_params', self_is_far, params, self_by_value)
        else:
            params = [item for item in items if isinstance(item, ast.Parameter)]
            return params

    def trait_const(self, items):
        """Associated constant declaration in trait: const NAME: type;"""
        items = self._filter_tokens(items)
        _, idx = self._collect_doc_comments(items, 0)
        name = _tok(items[idx])
        const_type = items[idx + 1]
        return ast.TraitConst(name=name, const_type=const_type)

    def type_alias(self, items):
        """Type alias."""
        items = self._filter_tokens(items)
        doc, idx = self._collect_doc_comments(items, 0)
        name = _tok(items[idx])
        aliased_type = items[idx + 1]
        return ast.TypeAlias(name=name, aliased_type=aliased_type, doc=doc)

    @v_args(tree=True)
    def include_stmt(self, tree):
        """Include statement."""
        items = self._filter_tokens(tree.children, keep_types={'STRING'})
        path = items[0].value.strip('"')  # Remove quotes
        return ast.IncludeStmt(path=path, source_loc=self._make_source_loc(tree.meta))

    @v_args(tree=True)
    def include_asm_stmt(self, tree):
        """include_asm! statement: `include_asm!("file.s");`"""
        items = self._filter_tokens(tree.children, keep_types={'STRING'})
        path = items[0].value.strip('"')
        return ast.IncludeAsmStmt(path=path, source_loc=self._make_source_loc(tree.meta))

    @v_args(tree=True)
    def extern_fn_decl(self, tree):
        """Extern function declaration: `extern [far|near] fn name(...) -> ret;`"""
        items = self._filter_tokens(tree.children, keep_types={'IDENT', 'FAR', 'NEAR', 'INTEGER', 'STRING', 'BOOLEAN', 'REGISTER'})

        doc, idx = self._collect_doc_comments(items, 0)
        attrs, idx = self._collect_attributes(items, idx)

        # Far by default, like every function; `extern near fn` opts out
        call_kind, idx = self._take_call_kind(items, idx)

        name = items[idx]
        idx += 1

        params = []
        if idx < len(items) and isinstance(items[idx], list):
            params = items[idx]
            idx += 1

        return_type = None
        if idx < len(items):
            return_type = items[idx]

        return ast.FunctionDecl(
            attributes=attrs,
            call_kind=call_kind,
            name=_tok(name),
            params=params,
            return_type=return_type,
            body=None,
            is_const=False,
            is_extern=True,
            doc=doc,
            source_loc=self._make_source_loc(tree.meta)
        )

    @v_args(tree=True)
    def extern_static_decl(self, tree):
        """Extern static declaration: `extern static [mut] NAME: Type;`"""
        items = self._filter_tokens(tree.children, keep_types={'IDENT', 'MUT', 'INTEGER', 'STRING', 'BOOLEAN', 'REGISTER'})

        doc, idx = self._collect_doc_comments(items, 0)
        attrs, idx = self._collect_attributes(items, idx)

        is_mut, idx = self._take(items, idx, 'MUT')

        name = items[idx]
        idx += 1

        var_type = items[idx]

        return ast.StaticDecl(
            attributes=attrs,
            is_far=False,
            is_mut=is_mut,
            name=_tok(name),
            var_type=var_type,
            initializer=None,
            is_extern=True,
            doc=doc,
            source_loc=self._make_source_loc(tree.meta)
        )

    def stack_directive(self, items):
        """Stack directive: #[stack(lower, upper)]"""
        lower, upper = self._filter_tokens(items, keep_types={'INTEGER'})
        return ast.StackDirective(lower=parse_integer(lower.value),
                                  upper=parse_integer(upper.value))

    def bank_directive(self, items):
        """Bank directive: #[bank(n)] or #[bank(auto)] - sets current ROM bank for following declarations."""
        # The grammar admits exactly one INTEGER or AUTO_KEYWORD.
        (token,) = self._filter_tokens(items, keep_types={'INTEGER', 'AUTO_KEYWORD'})
        if token.type == 'AUTO_KEYWORD':
            return ast.BankDirective(bank_number=None)
        return ast.BankDirective(bank_number=parse_integer(token.value))

    def snesrom_directive(self, items):
        """
        SNES ROM header directive: #[snesrom(name="...", ...)]

        Parses named arguments and flags for configuring the .SNESHEADER output.
        """
        # Collect all arguments (named args and flags)
        named_args = {}
        flags = set()

        for item in items:
            if isinstance(item, tuple):
                # Named argument: (name, value)
                name, value = item
                named_args[name] = value
            elif isinstance(item, str):
                # Flag argument
                flags.add(item)

        # Validate required argument
        if 'name' not in named_args:
            raise ValueError("#[snesrom] requires 'name' parameter")

        # Build directive with defaults
        name = named_args['name']
        rom_id = named_args.get('id', 'SNES')
        cartridge_type = named_args.get('cartridge_type', 0x00)
        sram_size = named_args.get('sram_size', 0x00)
        country = named_args.get('country', 0x01)
        version = named_args.get('version', 0x00)

        # ROM type flags (with mutual exclusivity for memory mapping)
        lorom = 'lorom' in flags
        hirom = 'hirom' in flags
        exhirom = 'exhirom' in flags
        slowrom = 'slowrom' in flags
        fastrom = 'fastrom' in flags

        # Default to lorom if no memory mapping specified
        if not lorom and not hirom and not exhirom:
            lorom = True

        # Default to slowrom if no speed specified
        if not slowrom and not fastrom:
            slowrom = True

        return ast.SnesRomDirective(
            name=name,
            id=rom_id,
            cartridge_type=cartridge_type,
            sram_size=sram_size,
            country=country,
            version=version,
            lorom=lorom,
            hirom=hirom,
            exhirom=exhirom,
            slowrom=slowrom,
            fastrom=fastrom
        )

    def snesrom_named_arg(self, items):
        """Named argument in snesrom directive: name=value"""
        items = self._filter_tokens(items, keep_types={'IDENT', 'STRING', 'INTEGER'})
        name = items[0].value
        value_token = items[1]

        if value_token.type == 'STRING':
            # Remove quotes from string
            value = value_token.value[1:-1]
        else:
            # Parse integer with base detection
            value = int(value_token.value, 0)

        return (name, value)

    def snesrom_flag_arg(self, items):
        """Flag argument in snesrom directive: flagname"""
        items = self._filter_tokens(items, keep_types={'IDENT'})
        return items[0].value

    # ========================================================================
    # Macros
    # ========================================================================

    @v_args(tree=True)
    def macro_decl_short(self, tree):
        """Shorthand macro: macro_rules! name($param:type, ...) { body } (single arm)"""
        items = tree.children
        doc, _ = self._collect_doc_comments(items, 0)
        name = None
        params = []
        body_tokens = []

        for item in items:
            if isinstance(item, LarkToken) and item.type == 'IDENT':
                name = item.value
            elif isinstance(item, list):  # macro_params result
                params = item
            elif isinstance(item, ast.MacroParam):
                params.append(item)
            elif isinstance(item, tuple) and item[0] == 'macro_body':
                body_tokens = item[1]

        return ast.MacroDecl(
            name=name,
            arms=[ast.MacroArm(params=params, body_tokens=body_tokens)],
            doc=doc,
            source_loc=self._make_source_loc(tree.meta)
        )

    @v_args(tree=True)
    def macro_decl_multi(self, tree):
        """Multi-arm macro: macro_rules! name { (pat) => { body }; ... }"""
        items = tree.children
        doc, _ = self._collect_doc_comments(items, 0)
        name = None
        arms = []

        for item in items:
            if isinstance(item, LarkToken) and item.type == 'IDENT':
                name = item.value
            elif isinstance(item, ast.MacroArm):
                arms.append(item)

        return ast.MacroDecl(
            name=name,
            arms=arms,
            doc=doc,
            source_loc=self._make_source_loc(tree.meta)
        )

    def macro_arm(self, items):
        """One arm of a multi-arm macro: (params) => { body }"""
        params = []
        body_tokens = []

        for item in items:
            if isinstance(item, list):  # macro_params result
                params = item
            elif isinstance(item, ast.MacroParam):
                params.append(item)
            elif isinstance(item, tuple) and item[0] == 'macro_body':
                body_tokens = item[1]

        return ast.MacroArm(params=params, body_tokens=body_tokens)

    def macro_params(self, items):
        """Macro parameter list."""
        result = []
        for item in items:
            if isinstance(item, ast.MacroParam):
                result.append(item)
        return result

    def macro_param_simple(self, items):
        """Simple macro parameter: $name:fragment"""
        items = self._filter_tokens(items, keep_types={'MACRO_VAR', 'FRAGMENT_TYPE'})
        name = items[0].value[1:]  # Remove $ prefix
        fragment_type = items[1].value
        return ast.MacroParam(name=name, fragment_type=fragment_type, is_repeated=False)

    def macro_param_repeated(self, items):
        """Repeated macro parameter: $($name:fragment),*"""
        items = self._filter_tokens(items, keep_types={'MACRO_VAR', 'FRAGMENT_TYPE'})
        name = items[0].value[1:]  # Remove $ prefix
        fragment_type = items[1].value
        return ast.MacroParam(name=name, fragment_type=fragment_type, is_repeated=True)

    def macro_body(self, items):
        """Macro body - collect all tokens as strings."""
        tokens = self._collect_macro_tokens(items)
        return ('macro_body', tokens)

    def _collect_macro_tokens(self, items) -> List[str]:
        """Recursively collect all tokens from macro body content."""
        result = []
        for item in items:
            if isinstance(item, LarkToken):
                result.append(item.value)
            elif isinstance(item, tuple):
                if item[0] == 'macro_body':
                    # Nested braces - content already includes braces due to keep_all_tokens=True
                    result.extend(item[1])
                elif item[0] == 'macro_rep':
                    # Repetition: $( ... ),* or $( ... )*
                    result.append('$(')
                    result.extend(item[1])
                    result.append(')')
                    result.append(item[2])  # ',*' or '*'
            elif isinstance(item, list):
                result.extend(self._collect_macro_tokens(item))
            elif isinstance(item, str):
                result.append(item)
        return result

    def macro_rep_comma(self, items):
        """Repetition with comma separator: $(...),*"""
        # Filter out structural tokens (DOLLAR, LPAR, RPAR, COMMA, STAR)
        # that Lark passes as rule children
        content = self._filter_rep_items(items)
        tokens = self._collect_macro_tokens(content)
        return ('macro_rep', tokens, ',*')

    def macro_rep_no_sep(self, items):
        """Repetition without separator: $(...)*"""
        # Filter out structural tokens (DOLLAR, LPAR, RPAR, STAR)
        # that Lark passes as rule children
        content = self._filter_rep_items(items)
        tokens = self._collect_macro_tokens(content)
        return ('macro_rep', tokens, '*')

    def _filter_rep_items(self, items):
        """Filter out structural tokens from macro repetition items."""
        result = []
        for item in items:
            if isinstance(item, LarkToken):
                # Skip structural tokens
                if item.type in ('DOLLAR', 'LPAR', 'RPAR', 'STAR', 'COMMA'):
                    continue
            result.append(item)
        return result

    def macro_token(self, items):
        """Single token in macro body."""
        if items:
            item = items[0]
            if isinstance(item, LarkToken):
                return item.value
            return str(item)
        return ''

    def _macro_invocation(self, tree, node_cls):
        """Build a macro-invocation node (name + args) from a Lark tree.

        Shared by the statement/inner/expression invocation rules, which differ
        only in the node class produced.
        """
        name = None
        args = []
        for item in tree.children:
            if isinstance(item, LarkToken) and item.type == 'IDENT':
                name = item.value
            elif isinstance(item, list):  # macro_args result
                args = item
        return node_cls(name=name, args=args, source_loc=self._make_source_loc(tree.meta))

    @v_args(tree=True)
    def macro_invocation_stmt(self, tree):
        """Top-level macro invocation: name!(args);"""
        return self._macro_invocation(tree, ast.MacroInvocationStmt)

    @v_args(tree=True)
    def macro_invocation_stmt_inner(self, tree):
        """Statement-level macro invocation: name!(args);"""
        return self._macro_invocation(tree, ast.MacroInvocationStmtInner)

    @v_args(tree=True)
    def macro_invocation_expr(self, tree):
        """Expression-level macro invocation: name!(args)"""
        return self._macro_invocation(tree, ast.MacroInvocation)

    def macro_args(self, items):
        """Macro arguments - list of token strings."""
        result = []
        for item in items:
            # Skip comma tokens between arguments
            if isinstance(item, LarkToken) and item.type == 'COMMA':
                continue
            elif isinstance(item, str):
                result.append(item)
            elif isinstance(item, list):
                # Flatten nested lists
                result.append(' '.join(item))
        return result

    def macro_arg(self, items):
        """Single macro argument - collect tokens."""
        tokens = []
        for item in items:
            if isinstance(item, LarkToken):
                tokens.append(item.value)
            elif isinstance(item, str):
                tokens.append(item)
            elif isinstance(item, list):
                tokens.extend(item)
        return ' '.join(tokens)

    def macro_arg_token(self, items):
        """Token in macro argument.

        Handles both single tokens and nested groups (parens, brackets, braces).
        For nested groups like { A = 0x0FFF; }, items will be:
        ['{', 'A', '=', '0x0FFF', ';', '}']
        """
        if not items:
            return ''
        # If multiple items, this is a nested group - join all tokens
        if len(items) > 1:
            tokens = []
            for item in items:
                if isinstance(item, LarkToken):
                    tokens.append(item.value)
                else:
                    tokens.append(str(item))
            return ' '.join(tokens)
        # Single item
        item = items[0]
        if isinstance(item, LarkToken):
            return item.value
        elif isinstance(item, list):
            return ' '.join(str(x) for x in item)
        return str(item)

    # ========================================================================
    # Attributes
    # ========================================================================

    def attribute(self, items):
        """Attribute."""
        items = self._filter_tokens(items)
        return items[0]  # Just return the attribute_inner result

    @v_args(tree=True)
    def attribute_inner(self, tree):
        """Attribute inner."""
        items = self._filter_tokens(tree.children)
        name = items[0]  # Will be string from attr_name
        # args will be a list from attribute_args if present
        args = items[1] if len(items) > 1 and isinstance(items[1], list) else []
        return ast.Attribute(name=name, args=args,
                             source_loc=self._make_source_loc(tree.meta))

    def attr_name(self, items):
        """Attribute name - can be IDENT or literal keyword."""
        if not items:
            return None
        item = items[0]
        if isinstance(item, LarkToken):
            return item.value
        return str(item)

    @v_args(tree=True)
    def cfg_attribute(self, tree):
        """cfg attribute: #[cfg(condition)]."""
        items = self._filter_tokens(tree.children)
        # items[0] should be the cfg_condition
        condition = items[0]
        return ast.Attribute(name='cfg', args=[ast.AttributeArg(name=None, value=condition)],
                             source_loc=self._make_source_loc(tree.meta))

    def cfg_condition(self, items):
        """Top-level cfg condition (cfg_any rule)."""
        return items[0]

    def cfg_any(self, items):
        """Any condition: condition1 || condition2 || ..."""
        items = self._filter_tokens(items)
        if len(items) == 1:
            return items[0]  # Single condition, no need for CfgAny wrapper
        
        # Extract actual conditions from any cfg_primary Trees
        actual_conditions = []
        for item in items:
            if hasattr(item, 'children') and len(item.children) >= 1:
                # Extract actual condition from Tree (skip tokens like parentheses)
                for child in item.children:
                    if not isinstance(child, LarkToken):  # Find AST node, not tokens
                        actual_conditions.append(child)
                        break
            else:
                actual_conditions.append(item)
        
        if len(actual_conditions) == 1:
            return actual_conditions[0]  # Single condition, no need for CfgAny wrapper
        return ast.CfgAny(conditions=actual_conditions)

    def cfg_all(self, items):
        """All condition: condition1 && condition2 && ..."""
        items = self._filter_tokens(items)
        if len(items) == 1:
            return items[0]  # Single condition, no need for CfgAll wrapper
        
        # Extract actual conditions from any cfg_primary Trees
        actual_conditions = []
        for item in items:
            if hasattr(item, 'children') and len(item.children) >= 1:
                # Extract actual condition from Tree (skip tokens like parentheses)
                for child in item.children:
                    if not isinstance(child, LarkToken):  # Find AST node, not tokens
                        actual_conditions.append(child)
                        break
            else:
                actual_conditions.append(item)
        
        if len(actual_conditions) == 1:
            return actual_conditions[0]  # Single condition, no need for CfgAll wrapper
        return ast.CfgAll(conditions=actual_conditions)

    def cfg_not(self, items):
        """Not condition: !condition."""
        items = self._filter_tokens(items)
        condition = items[0]
        
        # Handle case where condition is a Tree from cfg_primary
        if hasattr(condition, 'children') and len(condition.children) >= 1:
            # Extract the actual condition from Tree (skip tokens like parentheses)
            for child in condition.children:
                if not isinstance(child, LarkToken):  # Find the AST node, not tokens
                    condition = child
                    break
        
        return ast.CfgNot(condition=condition)

    def cfg_identifier(self, items):
        """Simple identifier condition."""
        items = self._filter_tokens(items, keep_types={'IDENT'})
        name = items[0].value if isinstance(items[0], LarkToken) else str(items[0])
        return ast.CfgIdentifier(name=name)

    def cfg_eq(self, items):
        """Equal comparison: key = "value"."""
        items = self._filter_tokens(items, keep_types={'IDENT', 'STRING'})
        key = items[0].value if isinstance(items[0], LarkToken) else str(items[0])
        value = items[1].value if isinstance(items[1], LarkToken) else str(items[1])
        value = value.strip('"')  # Remove quotes from string
        return ast.CfgComparison(key=key, operator='=', value=value)

    def cfg_ne(self, items):
        """Not equal comparison: key != "value"."""
        items = self._filter_tokens(items, keep_types={'IDENT', 'STRING'})
        key = items[0].value if isinstance(items[0], LarkToken) else str(items[0])
        value = items[1].value if isinstance(items[1], LarkToken) else str(items[1])
        value = value.strip('"')  # Remove quotes from string
        return ast.CfgComparison(key=key, operator='!=', value=value)

    def attribute_args(self, items):
        """Attribute arguments."""
        # Filter out comma tokens, keep only AttributeArg objects
        result = []
        for item in items:
            if not isinstance(item, LarkToken):
                result.append(item)
        return result

    def named_arg(self, items):
        """
        Named attribute argument: name=value

        Grammar: IDENT "=" expr
        """
        items = self._filter_tokens(items, keep_types={'IDENT'})
        name = items[0].value
        value = items[1]
        return ast.AttributeArg(name=name, value=value)

    def positional_arg(self, items):
        """
        Positional attribute argument: value (expr)

        Grammar: expr

        Can be an integer literal, identifier (for flags), or other expression.
        """
        value = items[0]
        return ast.AttributeArg(name=None, value=value)

    # ========================================================================
    # Statements
    # ========================================================================

    def function_body(self, items):
        """Function body: same as block."""
        statements = [item for item in items if not isinstance(item, LarkToken)]
        return ast.Block(statements=self._desugar_let_else(statements))

    @v_args(tree=True)
    def function_body_trailing(self, tree):
        """Function body with trailing expression (Rust-style implicit return).

        Wraps the trailing expression in an ExprStmt so the HIR builder's
        _add_implicit_return converts it to a return statement.
        """
        items = self._filter_tokens(tree.children)
        if not items:
            return ast.Block(statements=[])
        # Last item is the trailing expression, rest are statements
        trailing_expr = items[-1]
        statements = list(items[:-1])
        # Wrap trailing expression as an ExprStmt
        statements.append(ast.ExprStmt(
            expr=trailing_expr,
            source_loc=trailing_expr.source_loc
        ))
        return ast.Block(statements=self._desugar_let_else(statements))

    def block(self, items):
        """Block statement."""
        # Filter out brace tokens, keep only statement nodes
        statements = [item for item in items if not isinstance(item, LarkToken)]
        return ast.Block(statements=self._desugar_let_else(statements))

    @v_args(tree=True)
    def match_stmt(self, tree):
        """A match in statement position: the ExprStmt `match ..;` gives."""
        match = self._filter_tokens(tree.children)[0]
        return ast.ExprStmt(expr=match, source_loc=self._make_source_loc(tree.meta))

    def loop_body(self, items):
        """A loop body, which may end in a match with no `;`."""
        statements = self._filter_tokens(items)
        if statements and isinstance(statements[-1], ast.MatchExpression):
            tail = statements[-1]
            statements[-1] = ast.ExprStmt(expr=tail, source_loc=tail.source_loc)
        return ast.Block(statements=self._desugar_let_else(statements))

    @v_args(tree=True)
    def if_body(self, tree):
        """An if or else body; `_if_forms` decides whether it has a value."""
        statements = self._filter_tokens(tree.children)
        tail = None
        if statements and isinstance(statements[-1], ast.Expression):
            tail = statements.pop()
        return _TailBlock(statements, tail, self._make_source_loc(tree.meta))

    else_body = if_body

    def _if_forms(self, then_body, else_part):
        """The arms of an `if` statement: `(True, then, else)` as values, or
        `(False, then, else)` as statement blocks.

        The arms are values when the then arm ends in a match and the else
        has a value, as in `if c { match .. } else { 3 }` in a function's
        tail; before match_stmt that parsed as an if expression, and it still
        builds one.
        """
        else_value = None
        if then_body.tail is not None:
            if isinstance(else_part, _TailBlock) and else_part.tail is not None:
                else_value = self._tail_block_expr(else_part)
            elif (isinstance(else_part, ast.ExprStmt)
                    and self._is_if_value(else_part.expr)):
                else_value = self._else_value(else_part.expr)
        if else_value is not None:
            return True, self._tail_block_expr(then_body), else_value

        then_block = self._tail_block_stmt(then_body)
        if isinstance(else_part, _TailBlock):
            else_part = self._tail_block_stmt(else_part)
        elif else_part is not None and not isinstance(else_part, (ast.Block, ast.IfStmt)):
            # `else if let` is an ExprStmt, which HIR's _build_if would drop
            else_part = ast.Block(statements=[else_part], source_loc=else_part.source_loc)
        return False, then_block, else_part

    @staticmethod
    def _is_if_value(expr):
        """Whether an else-if statement is one `_if_forms` built as a value."""
        return (isinstance(expr, ast.IfExpression)
                or (isinstance(expr, ast.MatchExpression) and expr.origin is None))

    @staticmethod
    def _else_value(expr):
        """An if expression's else: `else if let` is a match, which HIR builds
        only inside a block expression."""
        if isinstance(expr, ast.MatchExpression):
            return ast.BlockExpression(statements=[], final_expr=expr,
                                       source_loc=expr.source_loc)
        return expr

    def _tail_block_expr(self, body):
        return self._let_else_block_expr(list(body.statements), body.tail, body.source_loc)

    def _tail_block_stmt(self, body):
        statements = list(body.statements)
        if body.tail is not None:
            statements.append(ast.ExprStmt(expr=body.tail, source_loc=body.tail.source_loc))
        return ast.Block(statements=self._desugar_let_else(statements))

    @staticmethod
    def _desugared_match(scrutinee, pattern, then_body, else_body, origin, loc):
        """`match scrutinee { pattern => then_body, _ => else_body }`.

        `if let`, `while let` and `let ... else` are all this match, so a
        payload binding in *pattern* is the same view of the pool a hand-written
        arm gets, with the same snapshot and rewrite checks.
        """
        return ast.MatchExpression(
            scrutinee=scrutinee,
            arms=[ast.MatchArm(pattern=pattern, body=then_body, source_loc=loc),
                  ast.MatchArm(pattern=ast.WildcardPattern(source_loc=loc),
                               body=else_body, source_loc=loc)],
            origin=origin, source_loc=loc)

    def _desugar_let_else(self, statements):
        """Fold each `let P = e else { .. };` and what follows it into a match.

        The bindings of P are in scope for the rest of the block, so the rest
        of the block is P's arm: `match e { P => { rest }, _ => { else } }`.
        """
        for k, stmt in enumerate(statements):
            if isinstance(stmt, ast.LetElseStmt):
                loc = stmt.source_loc
                rest = ast.Block(statements=self._desugar_let_else(statements[k + 1:]),
                                 source_loc=loc)
                match = self._desugared_match(stmt.scrutinee, stmt.pattern, rest,
                                              stmt.else_block, 'let_else', loc)
                return statements[:k] + [ast.ExprStmt(expr=match, source_loc=loc)]
        return statements

    @v_args(tree=True)
    def let_else_stmt(self, tree):
        """`let P = e else { .. };` — folded into a match by the enclosing block."""
        pattern, scrutinee, else_block = self._filter_tokens(tree.children)
        return ast.LetElseStmt(pattern=pattern, scrutinee=scrutinee,
                               else_block=else_block,
                               source_loc=self._make_source_loc(tree.meta))

    def single_pattern(self, items):
        """Single binding pattern: IDENT or IDENT @ binding"""
        items = self._filter_tokens(items)
        name = items[0].value
        binding = items[1] if len(items) > 1 else None
        return ('single', name, binding)

    def pointer_pattern(self, items):
        """Pointer pattern: *IDENT or *IDENT @ binding"""
        items = self._filter_tokens(items)
        name = _tok(items[0])
        binding = items[1] if len(items) > 1 else None
        return ('pointer', name, binding)

    @v_args(tree=True)
    def let_multi_stmt(self, tree):
        """Multi-binding let statement: let [mut] a, b = expr;"""
        items = self._filter_tokens(tree.children, keep_types={'IDENT', 'MUT'})
        idx = 0
        is_mut, idx = self._take(items, idx, 'MUT')
        # Collect IDENT names until we hit the initializer (non-LarkToken AST node)
        names = []
        while idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'IDENT':
            names.append(items[idx].value)
            idx += 1
        # Remaining item is the initializer expression
        initializer = items[idx] if idx < len(items) else None
        return ast.MultiLetStmt(
            names=names,
            is_mut=is_mut,
            initializer=initializer,
            source_loc=self._make_source_loc(tree.meta)
        )

    @v_args(tree=True)
    def let_stmt(self, tree):
        """Let statement."""
        items = self._filter_tokens(tree.children, keep_types={'IDENT', 'MUT', 'FAR', 'NEAR', 'INTEGER', 'STRING', 'BOOLEAN', 'REGISTER'})

        idx = 0

        # Check for far/near modifier (for pointer declarations)
        is_far, is_near, idx = self._take_far_near(items, idx)

        # Check for mut
        is_mut, idx = self._take(items, idx, 'MUT')

        # Get pattern (single or pointer)
        pattern_item = items[idx]
        idx += 1

        name = None
        binding = None
        is_pointer = False

        if isinstance(pattern_item, tuple):
            if pattern_item[0] == 'single':
                name = pattern_item[1]
                binding = pattern_item[2]
            elif pattern_item[0] == 'pointer':
                name = pattern_item[1]
                binding = pattern_item[2]
                is_pointer = True
        else:
            # Bare token; the let-pattern rules reduce to a tuple, so not expected
            name = pattern_item.value

        # Type annotation (optional) - comes after pattern
        var_type = None
        if idx < len(items) and isinstance(items[idx], ast.Type):
            var_type = items[idx]
            idx += 1

        # If this is a pointer declaration, wrap the type in PointerType
        if is_pointer and var_type is not None:
            var_type = ast.PointerType(is_far=is_far, pointee_type=var_type)

        # Initializer (always last)
        initializer = items[idx] if idx < len(items) else None

        # Require either type annotation or initializer (or both)
        if var_type is None and initializer is None:
            var_name = name if name else "variable"
            source_loc = self._make_source_loc(tree.meta) if hasattr(tree, 'meta') else None
            raise ParseError(
                f"variable '{var_name}' requires either a type annotation or an initializer",
                source_loc=source_loc,
                hint="add a type annotation (let x: u8;) or an initializer (let x = value;)"
            )

        return ast.LetStmt(
            is_mut=is_mut,
            name=name,
            binding=binding,
            var_type=var_type,
            initializer=initializer,
            source_loc=self._make_source_loc(tree.meta)
        )

    @v_args(tree=True)
    def let_stmt_safe_ptr_error(self, tree):
        """Error handler for safe pointer syntax in let statements."""
        name, _, loc = self._safe_ptr_error_parts(tree)
        raise ParseError(
            "safe pointers are not supported in R65", source_loc=loc,
            hint=f"use 'let {name}: *type' instead of 'let &{name}: type' for pointer variables")

    def safe_ptr_pattern(self, items):
        """Safe pointer pattern: &name or &name @ binding (for error handling)."""
        items = self._filter_tokens(items, keep_types={'IDENT', 'AMPER', 'REGISTER'})
        name = None
        binding = None
        for item in items:
            if isinstance(item, LarkToken) and item.type == 'IDENT':
                if name is None:
                    name = item.value
            elif isinstance(item, LarkToken) and item.type == 'REGISTER':
                binding = item.value
        return ('safe_ptr', name, binding)

    @v_args(tree=True)
    def expr_stmt(self, tree):
        """Expression statement."""
        items = self._filter_tokens(tree.children)
        return ast.ExprStmt(expr=items[0], source_loc=self._make_source_loc(tree.meta))

    @v_args(tree=True)
    def return_stmt(self, tree):
        """Return statement."""
        # Filter out 'return' keyword and semicolon, keep only expressions
        values = [item for item in tree.children if not isinstance(item, LarkToken)]
        # Flatten if we have a return_tuple (list of expressions)
        if values and isinstance(values[0], list):
            values = values[0]
        return ast.ReturnStmt(values=values, source_loc=self._make_source_loc(tree.meta))

    def return_tuple(self, items):
        """Multi-value return: `return a, b;` (no parentheses)."""
        # Filter out punctuation tokens
        return [item for item in items if not isinstance(item, LarkToken)]

    @v_args(tree=True)
    def break_stmt(self, tree):
        """Break statement with optional label and optional value."""
        label = None
        value = None
        for item in tree.children:
            if isinstance(item, LarkToken) and item.type == 'LABEL_REF':
                # Strip leading quote: 'outer -> outer
                label = item.value[1:]
            elif isinstance(item, ast.Expression):
                value = item
        return ast.BreakStmt(label=label, value=value, source_loc=self._make_source_loc(tree.meta))

    @v_args(tree=True)
    def continue_stmt(self, tree):
        """Continue statement with optional label."""
        label = None
        for item in tree.children:
            if isinstance(item, LarkToken) and item.type == 'LABEL_REF':
                # Strip leading quote: 'outer -> outer
                label = item.value[1:]
        return ast.ContinueStmt(label=label, source_loc=self._make_source_loc(tree.meta))

    @v_args(tree=True)
    def increment_stmt(self, tree):
        """Increment statement (x++;) - desugars to x += 1;"""
        items = self._filter_tokens(tree.children)
        lvalue = self._place(items[0], tree.meta)
        source_loc = self._make_source_loc(tree.meta)
        # Desugar to compound assignment: x++ becomes x += 1
        compound_assign = ast.CompoundAssignment(
            target=lvalue,
            operator='+',
            value=ast.IntegerLiteral(value=1, source_loc=source_loc),
            source_loc=source_loc
        )
        return ast.ExprStmt(expr=compound_assign, source_loc=source_loc)

    @v_args(tree=True)
    def decrement_stmt(self, tree):
        """Decrement statement (x--;) - desugars to x -= 1;"""
        items = self._filter_tokens(tree.children)
        lvalue = self._place(items[0], tree.meta)
        source_loc = self._make_source_loc(tree.meta)
        # Desugar to compound assignment: x-- becomes x -= 1
        compound_assign = ast.CompoundAssignment(
            target=lvalue,
            operator='-',
            value=ast.IntegerLiteral(value=1, source_loc=source_loc),
            source_loc=source_loc
        )
        return ast.ExprStmt(expr=compound_assign, source_loc=source_loc)

    @v_args(tree=True)
    def if_stmt(self, tree):
        """If statement."""
        items = self._filter_tokens(tree.children)
        condition = items[0]
        else_part = items[2] if len(items) > 2 else None
        loc = self._make_source_loc(tree.meta)
        is_value, then_block, else_block = self._if_forms(items[1], else_part)
        if is_value:
            return ast.ExprStmt(
                expr=ast.IfExpression(condition=condition, then_block=then_block,
                                      else_block=else_block, source_loc=loc),
                source_loc=loc)
        return ast.IfStmt(condition=condition, then_block=then_block, else_block=else_block,
                         source_loc=loc)

    @v_args(tree=True)
    def if_let_stmt(self, tree):
        """`if let P = e { .. } else ..` — `match e { P => { .. }, _ => else }`."""
        items = self._filter_tokens(tree.children)
        pattern, scrutinee = items[0], items[1]
        else_part = items[3] if len(items) > 3 else None
        loc = self._make_source_loc(tree.meta)
        is_value, then_block, else_block = self._if_forms(items[2], else_part)
        if is_value:                                     # as if_let_expr builds it
            match = self._desugared_match(scrutinee, pattern, then_block, else_block,
                                          None, loc)
            return ast.ExprStmt(expr=match, source_loc=loc)
        if else_block is None:
            else_block = ast.Block(statements=[], source_loc=loc)
        elif not isinstance(else_block, ast.Block):      # else if ...
            else_block = ast.Block(statements=[else_block], source_loc=loc)
        match = self._desugared_match(scrutinee, pattern, then_block, else_block,
                                      'if_let', loc)
        return ast.ExprStmt(expr=match, source_loc=loc)

    @v_args(tree=True)
    def while_let_stmt(self, tree):
        """`'l: while let P = e { .. }` — `'l: loop { match e { P => { .. }, _ => break 'l } }`.

        `continue` reaches the loop's top, so it re-tests e, as in Rust.
        """
        label = self._extract_label(tree.children)
        pattern, scrutinee, body = self._filter_tokens(tree.children)
        loc = self._make_source_loc(tree.meta)
        match = self._desugared_match(scrutinee, pattern, body,
                                      ast.BreakStmt(label=label, source_loc=loc),
                                      'if_let', loc)
        return ast.LoopStmt(
            body=ast.Block(statements=[ast.ExprStmt(expr=match, source_loc=loc)],
                           source_loc=loc),
            label=label, source_loc=loc)

    def else_clause(self, items):
        """Else clause."""
        items = self._filter_tokens(items)
        return items[0]

    def _extract_label(self, items):
        """Extract label from LABEL_DEF token if present."""
        for item in items:
            if isinstance(item, LarkToken) and item.type == 'LABEL_DEF':
                # Strip leading quote and trailing colon: 'outer: -> outer
                return item.value[1:-1]
        return None

    @v_args(tree=True)
    def loop_stmt(self, tree):
        """Loop statement with optional label."""
        label = self._extract_label(tree.children)
        items = self._filter_tokens(tree.children)
        return ast.LoopStmt(body=items[0], label=label,
                           source_loc=self._make_source_loc(tree.meta))

    @v_args(tree=True)
    def while_stmt(self, tree):
        """While statement with optional label."""
        label = self._extract_label(tree.children)
        items = self._filter_tokens(tree.children)
        condition = items[0]
        body = items[1]
        return ast.WhileStmt(condition=condition, body=body, label=label,
                            source_loc=self._make_source_loc(tree.meta))

    @v_args(tree=True)
    def for_stmt(self, tree):
        """For loop statement: for i in start..end { body } or start..=end"""
        label = self._extract_label(tree.children)
        variable = None
        exprs = []
        body = None
        inclusive = False

        for item in tree.children:
            if isinstance(item, LarkToken) and item.type == 'IDENT':
                variable = item.value
            elif isinstance(item, LarkToken) and item.type == 'DOTDOTEQ':
                inclusive = True
            elif isinstance(item, ast.Block):
                body = item
            elif isinstance(item, ast.Expression):
                exprs.append(item)

        if len(exprs) != 2:
            raise ParseError(f"For loop requires start and end expressions, got {len(exprs)}")

        return ast.ForStmt(
            variable=variable,
            start=exprs[0],
            end=exprs[1],
            body=body,
            label=label,
            inclusive=inclusive,
            source_loc=self._make_source_loc(tree.meta)
        )

    @v_args(tree=True)
    def asm_stmt(self, tree):
        """Inline assembly statement with optional format string support."""
        instructions = []
        format_args = {}

        for child in tree.children:
            if hasattr(child, 'type') and child.type == 'STRING':
                # Plain string instruction
                instructions.append(child.value.strip('"'))
            elif hasattr(child, 'data'):
                if child.data == 'asm_arg':
                    # asm_arg can be STRING or asm_named_arg
                    arg_child = child.children[0]
                    if hasattr(arg_child, 'type') and arg_child.type == 'STRING':
                        instructions.append(arg_child.value.strip('"'))
                    elif hasattr(arg_child, 'data') and arg_child.data == 'asm_named_arg':
                        # Named argument: IDENT = value
                        name, value = self._parse_asm_named_arg(arg_child)
                        format_args[name] = value
                elif child.data == 'asm_named_arg':
                    name, value = self._parse_asm_named_arg(child)
                    format_args[name] = value

        return ast.AsmStmt(
            instructions=instructions,
            format_args=format_args if format_args else None,
            source_loc=self._make_source_loc(tree.meta)
        )

    def _parse_asm_named_arg(self, tree):
        """Parse asm_named_arg: IDENT = asm_value"""
        name = None
        value = None

        for child in tree.children:
            if hasattr(child, 'type'):
                if child.type == 'IDENT':
                    name = child.value
            elif hasattr(child, 'data') and child.data == 'asm_value':
                # asm_value: expr (expressions are already transformed)
                val_child = child.children[0]
                if isinstance(val_child, ast.StringLiteral):
                    # String literal - extract the string value
                    value = val_child.value
                elif isinstance(val_child, ast.IntegerLiteral):
                    # Integer literal - extract the integer value
                    value = val_child.value
                elif isinstance(val_child, ast.Expression):
                    # Other expression - store for const evaluation in HIR
                    value = val_child

        return name, value

    # ========================================================================
    # Expressions
    # ========================================================================

    @v_args(tree=True)
    def integer(self, tree):
        """Integer literal."""
        items = tree.children
        raw = items[0].value
        digits, suffix = split_integer_suffix(raw)
        value = parse_integer(digits)
        return ast.IntegerLiteral(value=value, suffix=suffix, source_loc=self._make_source_loc(tree.meta))

    @v_args(tree=True)
    def char_literal(self, tree):
        """Character literal ('a', '\\n', '\\x7F', b'a'). Produces u8 integer."""
        raw = tree.children[0].value
        # Strip optional byte-literal 'b' prefix
        if raw.startswith('b'):
            raw = raw[1:]
        # Strip surrounding quotes
        inner = raw[1:-1]
        src_loc = self._make_source_loc(tree.meta)
        value = self._parse_char_content(inner, src_loc)
        return ast.IntegerLiteral(value=value, suffix='u8', source_loc=src_loc)

    def _parse_char_content(self, inner: str, src_loc) -> int:
        """Parse character literal content, returning the byte value (0-127, or 0-255 via \\xNN)."""
        if not inner.startswith('\\'):
            # A raw character must be 7-bit ASCII; \xNN spells 128-255.
            if len(inner) != 1:
                raise ParseError(
                    f"character literal must be a single ASCII character",
                    source_loc=src_loc,
                    hint="UTF-8/Unicode not supported; use \\xNN escapes for bytes 0x80-0xFF"
                )
            code = ord(inner)
            if code > 127:
                raise ParseError(
                    f"character literal '{inner}' (U+{code:04X}) is not 7-bit ASCII",
                    source_loc=src_loc,
                    hint=f"use \\x{code:02X} escape for byte 0x{code:02X}, or \\xNN for any byte 0x80-0xFF"
                )
        try:
            (value,) = decode_literal(inner)
        except ValueError as e:
            raise ParseError(f"invalid character literal '{inner}': {e}",
                             source_loc=src_loc, hint=ESCAPE_HINT)
        return value

    @v_args(tree=True)
    def boolean(self, tree):
        """Boolean literal."""
        items = tree.children
        value = items[0].value == 'true'
        return ast.BooleanLiteral(value=value, source_loc=self._make_source_loc(tree.meta))

    @v_args(tree=True)
    def identifier(self, tree):
        """Identifier."""
        items = tree.children
        token = items[0]
        identifier = token.value
        return ast.Identifier(name=identifier, source_loc=self._make_source_loc(tree.meta))

    @v_args(tree=True)
    def self_identifier(self, tree):
        """Self keyword used as identifier in method bodies."""
        # self refers to the self parameter in method bodies
        return ast.Identifier(name='self', source_loc=self._make_source_loc(tree.meta))

    @v_args(tree=True)
    def enum_variant_expr(self, tree):
        """Enum variant expression (e.g., Direction::North)."""
        # items: [IDENT, "::", IDENT]
        items = tree.children
        enum_name = items[0].value
        variant_name = items[2].value
        return ast.EnumVariantExpr(enum_name=enum_name, variant_name=variant_name, source_loc=self._make_source_loc(tree.meta))

    @v_args(tree=True)
    def register_ref(self, tree):
        """Register reference."""
        items = tree.children
        return ast.Register(name=items[0].value, source_loc=self._make_source_loc(tree.meta))

    @v_args(tree=True)
    def string_literal(self, tree):
        """String literal for byte array initialization."""
        items = tree.children
        token = items[0]
        # Remove surrounding quotes
        raw_value = token.value[1:-1]
        return ast.StringLiteral(value=raw_value, source_loc=self._make_source_loc(tree.meta))

    @v_args(tree=True)
    def include_bytes_expr(self, tree):
        """Include bytes expression (e.g., include_bytes!("data.bin"))."""
        # items: [INCLUDE_BYTES, "!", "(", STRING, ")"]
        # STRING token is at index 3, value includes quotes
        items = tree.children
        path = items[3].value.strip('"')
        return ast.IncludeBytesExpr(path=path, source_loc=self._make_source_loc(tree.meta))

    def array_fill_expr(self, items):
        """Array fill expression (e.g., [0; 256])."""
        # items: ["[", value_expr, ";", count_expr, "]"]
        items = self._filter_tokens(items)
        value = items[0]
        count = items[1]
        return ast.ArrayFillExpr(value=value, count=count)

    def array_literal_expr(self, items):
        """Array literal expression (e.g., [1, 2, 3])."""
        # items: ["[", expr, ",", expr, ..., "]"]
        items = self._filter_tokens(items)
        # All items are now the expressions
        return ast.ArrayLiteralExpr(elements=items)

    def initializer_array_literal(self, items):
        """Array literal in initializer context - allows struct literals as elements."""
        # items: ["[", initializer_expr, ",", initializer_expr, ..., "]"]
        items = self._filter_tokens(items)
        return ast.ArrayLiteralExpr(elements=items)

    def struct_literal_expr(self, items):
        """Struct literal expression (e.g., Player { x: 10, y: 20 })."""
        # items: [struct_name, field_init, field_init, ...]
        items = self._filter_tokens(items)
        struct_name = _tok(items[0])
        fields = [f for f in items[1:] if isinstance(f, ast.StructFieldInit)]
        return ast.StructLiteralExpr(struct_name=struct_name, fields=fields)

    def struct_field_init(self, items):
        """Struct field initializer (e.g., x: 10)."""
        items = self._filter_tokens(items)
        name = _tok(items[0])
        value = items[1]
        return ast.StructFieldInit(name=name, value=value)

    def paren(self, items):
        """Parenthesized expression."""
        # items = [LPAR, expr, RPAR] - return the middle element (the expression)
        return items[1]

    # ========================================================================
    # Pattern Matching
    # ========================================================================

    def match_expr(self, items):
        """Match expression."""
        items = self._filter_tokens(items)
        # items[0] is scrutinee, rest are match arms
        scrutinee = items[0]
        arms = items[1:]
        return ast.MatchExpression(scrutinee=scrutinee, arms=arms)

    @v_args(tree=True)
    def block_expr(self, tree):
        """Block expression: { statements; final_expr }

        The grammar rule is: block_expr: "{" statement* expr "}"
        After filtering, the last item is the final expression, everything before
        is statements.
        """
        items = self._filter_tokens(tree.children)
        if not items:
            raise ParseError("Empty block expression", self._make_source_loc(tree.meta))
        return self._let_else_block_expr(list(items[:-1]), items[-1],
                                         self._make_source_loc(tree.meta))

    def _let_else_block_expr(self, statements, final_expr, loc):
        """A block expression, with any `let ... else` folded as in a block.

        The rest of the block — its value included — becomes P's arm, so the
        match is the block's value and the else arm, which diverges, has none.
        """
        for k, stmt in enumerate(statements):
            if isinstance(stmt, ast.LetElseStmt):
                rest = self._let_else_block_expr(statements[k + 1:], final_expr, loc)
                match = self._desugared_match(stmt.scrutinee, stmt.pattern, rest,
                                              stmt.else_block, 'let_else',
                                              stmt.source_loc)
                return ast.BlockExpression(statements=statements[:k],
                                           final_expr=match, source_loc=loc)
        return ast.BlockExpression(statements=statements, final_expr=final_expr,
                                   source_loc=loc)

    @v_args(tree=True)
    def if_expr(self, tree):
        """If expression: if cond { expr } else { expr }

        The grammar rule is: if_expr: IF expr block_expr (ELSE (if_expr | block_expr))
        After filtering tokens: [condition, then_block_expr, else_block_or_if_expr]
        """
        items = self._filter_tokens(tree.children)
        condition = items[0]
        then_block = items[1]
        else_block = items[2] if len(items) > 2 else None
        if else_block is None:
            raise ParseError(
                "if expression requires an else branch",
                self._make_source_loc(tree.meta)
            )
        return ast.IfExpression(
            condition=condition,
            then_block=then_block,
            else_block=self._else_value(else_block),
            source_loc=self._make_source_loc(tree.meta)
        )

    @v_args(tree=True)
    def if_let_expr(self, tree):
        """`if let P = e { a } else { b }` as a value — `match e { P => a, _ => b }`."""
        pattern, scrutinee, then_block, else_block = self._filter_tokens(tree.children)
        return self._desugared_match(scrutinee, pattern, then_block, else_block,
                                     None, self._make_source_loc(tree.meta))

    @v_args(tree=True)
    def loop_expr(self, tree):
        """Loop expression: loop { ... break value; ... }"""
        label = self._extract_label(tree.children)
        items = self._filter_tokens(tree.children)
        body = items[0]
        return ast.LoopExpression(
            body=body,
            label=label,
            source_loc=self._make_source_loc(tree.meta)
        )

    def match_arm(self, items):
        """Match arm."""
        items = self._filter_tokens(items)
        # items[0] is pattern, items[1] is body expression or statement
        return ast.MatchArm(pattern=items[0], body=items[1])

    @v_args(tree=True)
    def match_arm_return(self, tree):
        """Return statement in match arm (no semicolon)."""
        values = [item for item in tree.children if not isinstance(item, LarkToken)]
        if values and isinstance(values[0], list):
            values = values[0]
        return ast.ReturnStmt(values=values, source_loc=self._make_source_loc(tree.meta))

    @v_args(tree=True)
    def match_arm_break(self, tree):
        """Break statement in match arm (no semicolon)."""
        label = None
        value = None
        for item in tree.children:
            if isinstance(item, LarkToken) and item.type == 'LABEL_REF':
                label = item.value[1:]
            elif isinstance(item, ast.Expression):
                value = item
        return ast.BreakStmt(label=label, value=value, source_loc=self._make_source_loc(tree.meta))

    @v_args(tree=True)
    def match_arm_continue(self, tree):
        """Continue statement in match arm (no semicolon)."""
        label = None
        for item in tree.children:
            if isinstance(item, LarkToken) and item.type == 'LABEL_REF':
                label = item.value[1:]
        return ast.ContinueStmt(label=label, source_loc=self._make_source_loc(tree.meta))

    @v_args(tree=True)
    def match_arm_block(self, tree):
        """Statement-only block in match arm (no trailing expression)."""
        items = self._filter_tokens(tree.children)
        return ast.Block(statements=self._desugar_let_else(list(items)),
                         source_loc=self._make_source_loc(tree.meta))

    def pattern_range(self, items):
        """Range pattern (e.g., 0..5 or 0..=5, supports negative bounds)."""
        items = self._filter_tokens(items, keep_types={'INTEGER', 'DOTDOT', 'DOTDOTEQ'})
        # items are: [pattern_int_value, DOTDOT/DOTDOTEQ, pattern_int_value]
        # pattern_int sub-rules are already resolved to ints by pattern_int()
        filtered = []
        range_op = None
        for item in items:
            if isinstance(item, LarkToken) and item.type in ('DOTDOT', 'DOTDOTEQ'):
                range_op = item
            elif isinstance(item, int):
                filtered.append(item)
            elif isinstance(item, LarkToken) and item.type == 'INTEGER':
                filtered.append(parse_integer(item.value))
        inclusive = range_op is not None and range_op.type == 'DOTDOTEQ'
        return ast.RangePattern(start=filtered[0], end=filtered[1], inclusive=inclusive)

    @v_args(tree=True)
    def pattern_int(self, tree):
        """Integer in pattern context (possibly negative)."""
        has_neg = any(
            isinstance(t, LarkToken) and t == '-' for t in tree.children
        )
        int_token = next(
            t for t in tree.children
            if isinstance(t, LarkToken) and t.type == 'INTEGER'
        )
        value = parse_integer(int_token.value)
        return -value if has_neg else value

    def pattern_literal(self, items):
        """Literal pattern (integer, boolean, or character)."""
        items = self._filter_tokens(items, keep_types={'INTEGER', 'BOOLEAN', 'CHAR_LITERAL'})
        token = items[0]
        if token.type == 'INTEGER':
            value = parse_integer(token.value)
        elif token.type == 'CHAR_LITERAL':
            raw = token.value
            if raw.startswith('b'):
                raw = raw[1:]
            inner = raw[1:-1]
            value = self._parse_char_content(inner, None)
        else:  # BOOLEAN
            value = token.value == 'true'
        return ast.LiteralPattern(value=value)

    @v_args(tree=True)
    def pattern_neg_literal(self, tree):
        """Negative integer literal pattern (e.g., -1)."""
        int_token = next(
            t for t in tree.children
            if isinstance(t, LarkToken) and t.type == 'INTEGER'
        )
        value = parse_integer(int_token.value)
        return ast.LiteralPattern(value=-value)

    def pattern_enum_payload(self, items):
        """Enum pattern binding a payload: `Shape::C(c)`."""
        items = self._filter_tokens(items)
        return ast.EnumPattern(enum_name=str(items[0]), variant_name=str(items[1]),
                               binding=str(items[2]))

    def pattern_enum(self, items):
        """Enum variant pattern."""
        items = self._filter_tokens(items, keep_types={'IDENT'})
        return ast.EnumPattern(enum_name=items[0].value, variant_name=items[1].value)

    def pattern_wildcard(self, items):
        """Wildcard pattern (_)."""
        return ast.WildcardPattern()

    def pattern_ident(self, items):
        """Identifier pattern."""
        items = self._filter_tokens(items, keep_types={'IDENT'})
        return ast.IdentifierPattern(name=items[0].value)

    def pattern_or(self, items):
        """Or pattern (pattern1 | pattern2 | ...)."""
        items = self._filter_tokens(items)
        if len(items) == 1:
            # Single pattern, not an or-pattern
            return items[0]
        return ast.OrPattern(patterns=items)

    # ========================================================================
    # Operation Handler Factories
    # ========================================================================

    @staticmethod
    def _make_binary_op_handler(operator: str):
        """
        Create a binary operation handler for a given operator.

        Args:
            operator: The operator string ('+', '-', '*', etc.)

        Returns:
            A handler function for Lark Transformer
        """
        @v_args(tree=True)
        def handler(self, tree):
            items = self._filter_tokens(tree.children)
            return ast.BinaryOp(op=operator, left=items[0], right=items[1],
                                source_loc=self._make_source_loc(tree.meta))
        return handler

    @staticmethod
    def _make_unary_op_handler(operator: str):
        """
        Create a unary operation handler for a given operator.

        Args:
            operator: The operator string ('!', '~', '-')

        Returns:
            A handler function for Lark Transformer
        """
        @v_args(tree=True)
        def handler(self, tree):
            items = self._filter_tokens(tree.children)
            return ast.UnaryOp(op=operator, operand=items[0],
                               source_loc=self._make_source_loc(tree.meta))
        return handler

    # ========================================================================
    # Binary Operations (generated via factory)
    # ========================================================================

    # Arithmetic operators
    add = _make_binary_op_handler('+')
    sub = _make_binary_op_handler('-')
    mul = _make_binary_op_handler('*')
    div = _make_binary_op_handler('/')
    mod = _make_binary_op_handler('%')
    # Bitwise operators
    bitand = _make_binary_op_handler('&')
    bitor = _make_binary_op_handler('|')
    bitxor = _make_binary_op_handler('^')
    lshift = _make_binary_op_handler('<<')
    rshift = _make_binary_op_handler('>>')

    # Comparison operators
    eq = _make_binary_op_handler('==')
    ne = _make_binary_op_handler('!=')
    lt = _make_binary_op_handler('<')
    le = _make_binary_op_handler('<=')
    gt = _make_binary_op_handler('>')
    ge = _make_binary_op_handler('>=')

    # Logical operators
    and_expr = _make_binary_op_handler('&&')
    or_expr = _make_binary_op_handler('||')

    # ========================================================================
    # Unary Operations (generated via factory)
    # ========================================================================

    not_expr = _make_unary_op_handler('!')
    bitnot = _make_unary_op_handler('~')
    neg = _make_unary_op_handler('-')

    def deref(self, items):
        items = self._filter_tokens(items)
        return ast.Dereference(pointer=items[0])

    def addressof(self, items):
        items = self._filter_tokens(items)
        return ast.AddressOf(operand=items[0])

    def cfg_function_call(self, items):
        """cfg!(condition) compile-time check."""
        items = self._filter_tokens(items)
        # The argument is a cfg condition (identifier)
        condition = items[0] if items else None
        return ast.CfgFunctionCall(condition=condition)

    # Postfix operations
    def call(self, items):
        """Function call."""
        items = self._filter_tokens(items)
        func = items[0]
        args = items[1] if len(items) > 1 and isinstance(items[1], list) else []
        return ast.FunctionCall(func=func, args=args)

    def arg_list(self, items):
        """Argument list."""
        # Filter out commas, parentheses - keep only expressions
        return [item for item in items if not isinstance(item, LarkToken)]

    def array_index(self, items):
        """Array indexing."""
        items = self._filter_tokens(items)
        return ast.ArrayIndex(array=items[0], index=items[1])

    def field_access(self, items):
        """Field access."""
        items = self._filter_tokens(items, keep_types={'IDENT'})
        base = items[0]
        field = _tok(items[1])
        return ast.FieldAccess(base=base, field=field)

    @v_args(tree=True)
    def method_macro(self, tree):
        """Method macro invocation: receiver.name!(args)"""
        items = tree.children
        receiver = None
        name = None
        args = []

        for item in items:
            if receiver is None and isinstance(item, ast.Expression):
                receiver = item
            elif isinstance(item, LarkToken) and item.type == 'IDENT':
                name = item.value
            elif isinstance(item, list):  # macro_args result
                args = item

        return ast.MethodMacro(
            receiver=receiver,
            name=name,
            args=args,
            source_loc=self._make_source_loc(tree.meta)
        )

    @v_args(tree=True)
    def assoc_macro(self, tree):
        """Associated macro invocation: Type::name!(args)"""
        idents = [i for i in tree.children
                  if isinstance(i, LarkToken) and i.type == 'IDENT']
        args = next((i for i in tree.children if isinstance(i, list)), [])

        return ast.MethodMacro(
            receiver=None,
            name=idents[1].value,
            args=args,
            type_name=idents[0].value,
            source_loc=self._make_source_loc(tree.meta)
        )

    @v_args(tree=True)
    def type_cast(self, tree):
        """Type cast."""
        items = self._filter_tokens(tree.children)
        return ast.TypeCast(
            expr=items[0],
            target_type=items[1],
            source_loc=self._make_source_loc(tree.meta),
        )

    # Assignment
    @v_args(tree=True)
    def assign(self, tree):
        """Assignment."""
        items = self._filter_tokens(tree.children)
        target = self._place(items[0], tree.meta)
        value = items[1]
        source_loc = self._make_source_loc(tree.meta)

        return ast.Assignment(target=target, value=value, source_loc=source_loc)

    # Compound assignment operator token -> binary operator
    _COMPOUND_OPS = {
        'PLUSEQUAL': '+',
        'MINUSEQUAL': '-',
        'STAREQUAL': '*',
        'SLASHEQUAL': '/',
        'PERCENTEQUAL': '%',
        'AMPEREQUAL': '&',
        'VBAREQUAL': '|',
        'CIRCUMFLEXEQUAL': '^',
        'LSHIFTEQUAL': '<<',
        'RSHIFTEQUAL': '>>',
    }

    @v_args(tree=True)
    def compound_assign(self, tree):
        """Compound assignment (+=, -=, etc.)."""
        # compound_op has already reduced to its operator token
        items = self._filter_tokens(tree.children, keep_types=self._COMPOUND_OPS.keys())
        target, op_token, value = items
        return ast.CompoundAssignment(target=self._place(target, tree.meta),
                                      operator=self._COMPOUND_OPS[op_token.type],
                                      value=value, source_loc=self._make_source_loc(tree.meta))

    def compound_op(self, items):
        """Compound operator: the operator token itself."""
        (token,) = self._filter_tokens(items, keep_types=self._COMPOUND_OPS.keys())
        return token

    # What an assignment may write: a variable, a register, an array element,
    # a field, or through a pointer.
    _PLACE_TYPES = (ast.Identifier, ast.Register, ast.ArrayIndex, ast.FieldAccess,
                    ast.Dereference)

    def _place(self, target, meta):
        """*target* if an assignment may write it, else a ParseError."""
        if isinstance(target, self._PLACE_TYPES):
            return target
        source_loc = self._make_source_loc(meta)
        if isinstance(target, ast.NewtypeFieldAccess):
            base = target.base
            name = base.name if isinstance(base, ast.Identifier) else "value"
            raise ParseError(
                f"a newtype payload is read-only, so '.{target.index}' cannot be assigned",
                source_loc=source_loc,
                hint=f"assign the whole value instead: '{name} = TypeName(x);'"
            )
        raise ParseError(
            "invalid assignment target",
            source_loc=source_loc,
            hint="assign to a variable, a register, an array element, a field or '*pointer'"
        )

    @v_args(tree=True)
    def tuple_assign_stmt(self, tree):
        """Multi-value assignment: `a, b = expr;` (two or more identifier/register targets, no parentheses)."""
        items = self._filter_tokens(tree.children)
        # All items except the last are targets, the last is the value
        targets = list(items[:-1])
        value = items[-1]
        source_loc = self._make_source_loc(tree.meta)
        multi_assign = ast.MultiAssignment(targets=targets, value=value, source_loc=source_loc)
        return ast.ExprStmt(expr=multi_assign, source_loc=source_loc)

    def tuple_target_ident(self, items):
        """Tuple assignment target - identifier."""
        items = self._filter_tokens(items)
        token = items[0]
        identifier = _tok(token)
        return ast.Identifier(name=identifier)

    def tuple_target_register(self, items):
        """Tuple assignment target - register."""
        items = self._filter_tokens(items, keep_types={'REGISTER'})
        return ast.Register(name=_tok(items[0]))

    # ========================================================================
    # Types
    # ========================================================================

    def type_basic(self, items):
        """Basic type (built-in types or user-defined type names)."""
        items = self._filter_tokens(items, keep_types={'TYPE_NAME', 'IDENT'})
        return ast.BasicType(name=_tok(items[0]))

    def type_array(self, items):
        """Array type."""
        items = self._filter_tokens(items)
        element_type = items[0]
        size = items[1]
        return ast.ArrayType(element_type=element_type, size=size)


    def type_pointer(self, items):
        """Pointer type: *type (implied near) or far *type or near *type.
        With optional dyn keyword for trait pointers: *dyn Trait or far *dyn Trait."""
        items = self._filter_tokens(items, keep_types={'FAR', 'NEAR', 'DYN', 'TYPE_NAME'})

        idx = 0
        is_far = False
        is_dyn = False

        # Check for far/near modifier
        if idx < len(items) and isinstance(items[idx], LarkToken):
            if items[idx].type == 'FAR':
                is_far = True
                idx += 1
            elif items[idx].type == 'NEAR':
                is_far = False  # Explicit near
                idx += 1

        # Check for dyn keyword
        if idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'DYN':
            is_dyn = True
            idx += 1

        # The pointee type is the remaining item
        pointee_type = items[idx] if idx < len(items) else items[-1]
        return ast.PointerType(is_far=is_far, pointee_type=pointee_type, is_dyn=is_dyn)

    def type_fn(self, items):
        """Function type: far by default, `near fn(...)` for a near pointer."""
        items = self._filter_tokens(items, keep_types={'FAR', 'NEAR'})
        is_far = True
        idx = 0

        if idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type in ('FAR', 'NEAR'):
            is_far = items[idx].type == 'FAR'
            idx += 1

        param_types = []
        return_type = None

        if idx < len(items):
            if isinstance(items[idx], list):
                param_types = items[idx]
                idx += 1

            if idx < len(items):
                return_type = items[idx]

        return ast.FunctionType(
            is_far=is_far,
            param_types=param_types,
            return_type=return_type
        )

    def multi_return_type(self, items):
        """Multi-return type: a comma-separated type list, e.g. `u8, u16`.

        Registers are assigned to each value during MIR lowering
        (codegen/constants.get_return_registers) from the types and the
        function's mode, so no register names appear in the source.
        """
        element_types = [item for item in items if not isinstance(item, LarkToken)]
        return ast.MultiReturnType(element_types=element_types)

    def type_list(self, items):
        """Type list."""
        # Filter out commas - keep only type nodes
        return [item for item in items if not isinstance(item, LarkToken)]


class Parser:
    """Parser for R65 source code."""

    # Cache compiled Lark parser - grammar is static so this is shared across instances
    _lark_cache = None

    def __init__(self):
        """Initialize the parser."""
        if Parser._lark_cache is None:
            Parser._lark_cache = _build_lark()
        self.lark = Parser._lark_cache

    def parse(self, source: str, filename: str = "<input>",
              included_from=None, origin_package=None) -> ast.Program:
        """
        Parse source code into an AST.

        Args:
            source: Source code to parse
            filename: Name of the source file (for error messages)
            included_from: SourceLocation of include! statement if parsing included file

        Returns:
            Program AST node

        Raises:
            ParseError: If parsing fails
        """
        try:
            tree = self.lark.parse(source)
            transformer = ASTBuilder(filename=filename, included_from=included_from,
                                     origin_package=origin_package)
            program = transformer.transform(tree)
            return program

        except UnexpectedToken as e:
            # Handle unexpected token errors with detailed context
            token = e.token
            # Note: Token with empty value (like $END) is falsy, so use 'is not None'
            line = token.line if token is not None else 0
            column = token.column if token is not None else 0

            # Build descriptive message
            if token is not None and token.type == '$END':
                message = "unexpected end of file"
            elif token is not None:
                message = f"unexpected token '{token.value}'"
            else:
                message = "unexpected token"

            # Check for Rust feature hints first (most specific)
            rust_hint = self._check_rust_feature_hints(source, token, line)
            if rust_hint:
                message, hint = rust_hint
            # Check for macro syntax hints - these are important enough to be in the message
            elif (macro_hint := self._check_macro_syntax_hints(source, str(e))):
                # For macro errors, include the full hint in the message
                message = f"{message}\n\n{macro_hint}"
                hint = None
            else:
                # Add what was expected (limit to reasonable number)
                expected = getattr(e, 'expected', None)
                hint = None
                if expected:
                    # Translate internal token names to friendly names
                    friendly = self._translate_expected_tokens(expected)
                    if len(friendly) <= 5:
                        hint = f"expected: {', '.join(friendly)}"
                    else:
                        hint = f"expected one of: {', '.join(friendly[:5])}..."

            # Create source location with context
            source_line = get_source_line(source, line)
            source_loc = SourceLocation(
                file_path=filename,
                line=line,
                column=column,
                source_line=source_line,
                included_from=included_from,
                origin_package=origin_package,
            )

            error = ParseError(message, source_loc)
            error.hint = hint
            raise error from e

        except UnexpectedCharacters as e:
            # Handle unexpected characters (lexer-level errors during parsing)
            line = getattr(e, 'line', 0)
            column = getattr(e, 'column', 0)
            char = getattr(e, 'char', None)

            if char:
                message = f"unexpected character '{char}'"
            else:
                message = "unexpected character"

            # Check for common unsupported characters with helpful hints
            hint = None
            if char == '?':
                message = "the '?' operator is not supported in R65"
                hint = "R65 has no Result/Option types; use return codes or error flags"

            source_line = get_source_line(source, line)
            source_loc = SourceLocation(
                file_path=filename,
                line=line,
                column=column,
                source_line=source_line,
                included_from=included_from,
                origin_package=origin_package,
            )

            error = ParseError(message, source_loc)
            error.hint = hint
            raise error from e

        except UnexpectedEOF as e:
            # Handle unexpected end of file
            # Find the last line of the source
            lines = source.splitlines()
            line = len(lines) if lines else 1
            column = len(lines[-1]) + 1 if lines else 1

            message = "unexpected end of file"

            expected = getattr(e, 'expected', None)
            hint = None
            if expected:
                friendly = self._translate_expected_tokens(expected)
                if len(friendly) <= 5:
                    hint = f"expected: {', '.join(friendly)}"
                else:
                    hint = f"expected one of: {', '.join(friendly[:5])}..."

            source_line = get_source_line(source, line)
            source_loc = SourceLocation(
                file_path=filename,
                line=line,
                column=column,
                source_line=source_line,
                included_from=included_from,
                origin_package=origin_package,
            )

            error = ParseError(message, source_loc)
            error.hint = hint
            raise error from e

        except VisitError as e:
            # Handle errors during AST transformation
            # Extract the original exception if it's a ParseError with source_loc
            orig = e.orig_exc
            if isinstance(orig, ParseError) and orig.source_loc is not None:
                raise orig from e

            # Otherwise wrap it with context
            # VisitError has 'rule' and 'obj' attributes
            obj = getattr(e, 'obj', None)

            # Try to get position from the object (token or tree)
            line = 0
            column = 0
            if obj is not None and hasattr(obj, 'line'):
                line = obj.line
                column = getattr(obj, 'column', 0)
            elif obj is not None and hasattr(obj, 'meta') and obj.meta is not None:
                line = getattr(obj.meta, 'line', 0)
                column = getattr(obj.meta, 'column', 0)

            source_line = get_source_line(source, line) if line > 0 else None
            source_loc = SourceLocation(
                file_path=filename,
                line=line,
                column=column,
                source_line=source_line,
                included_from=included_from,
                origin_package=origin_package,
            ) if line > 0 else None

            raise ParseError(str(orig), source_loc) from e

        except Exception as e:
            # Fallback for any other exceptions
            # Check for common Rust macro syntax mistakes
            error_msg = self._check_macro_syntax_hints(source, str(e))
            if error_msg:
                raise ParseError(f"{error_msg}") from e
            raise ParseError(f"{e}") from e

    def _translate_expected_tokens(self, expected: set) -> list:
        """Translate internal Lark token names to user-friendly names."""
        translations = {
            # Delimiters
            'SEMI': ';',
            'COLON': ':',
            'COMMA': ',',
            'LPAR': '(',
            'RPAR': ')',
            'LBRACE': '{',
            'RBRACE': '}',
            'LSQB': '[',
            'RSQB': ']',
            'RARROW': '->',
            'AT': '@',
            'HASH': '#',
            'DOT': '.',
            # Assignment and comparison
            'EQUAL': '=',
            'EQEQUAL': '==',
            'NOTEQUAL': '!=',
            'LESS': '<',
            'GREATER': '>',
            'LESSEQUAL': '<=',
            'GREATEREQUAL': '>=',
            # Arithmetic operators
            'PLUS': '+',
            'MINUS': '-',
            'STAR': '*',
            'SLASH': '/',
            'PERCENT': '%',
            # Bitwise operators
            'AMPER': '&',
            'VBAR': '|',
            'CIRCUMFLEX': '^',
            'TILDE': '~',
            'LSHIFT': '<<',
            'RSHIFT': '>>',
            # Logical operators
            'AND': '&&',
            'OR': '||',
            'EXCLAMATION': '!',
            # Compound assignment operators
            'PLUSEQUAL': '+=',
            'MINUSEQUAL': '-=',
            'STAREQUAL': '*=',
            'SLASHEQUAL': '/=',
            'PERCENTEQUAL': '%=',
            'AMPEREQUAL': '&=',
            'VBAREQUAL': '|=',
            'CIRCUMFLEXEQUAL': '^=',
            'LSHIFTEQUAL': '<<=',
            'RSHIFTEQUAL': '>>=',
            # Increment/decrement
            'PLUSPLUS': '++',
            'MINUSMINUS': '--',
            # Literals and identifiers
            'IDENT': 'identifier',
            'INTEGER': 'number',
            'DEC_INTEGER': 'number',
            'HEX_INTEGER': 'hex number',
            'BIN_INTEGER': 'binary number',
            'BOOLEAN': 'true/false',
            'STRING': 'string',
            'TYPE_NAME': 'type',
            'REGISTER': 'register',
            # Keywords
            'FN': 'fn',
            'LET': 'let',
            'MUT': 'mut',
            'IF': 'if',
            'ELSE': 'else',
            'LOOP': 'loop',
            'WHILE': 'while',
            'BREAK': 'break',
            'CONTINUE': 'continue',
            'RETURN': 'return',
            'STRUCT': 'struct',
            'ENUM': 'enum',
            'STATIC': 'static',
            'CONST': 'const',
            'AS': 'as',
            'FAR': 'far',
            'NEAR': 'near',
            'TYPE': 'type',
            'ASM': 'asm',
            'INCLUDE': 'include',
            'INCLUDE_BYTES': 'include_bytes',
            'INCLUDE_ASM': 'include_asm',
            'MACRO_RULES': 'macro_rules',
            'MATCH': 'match',
            'FOR': 'for',
            'IN': 'in',
            'IMPL': 'impl',
            'TRAIT': 'trait',
            'SELF': 'self',
            'DYN': 'dyn',
            'EXTERN': 'extern',
            'UNION': 'union',
            # Special
            '$END': 'end of file',
        }

        result = []
        for token in expected:
            if token in translations:
                result.append(translations[token])
            elif token == 'KEYWORD':
                # The reserved-only Rust keywords (where, use, pub, ...)
                continue
            elif token.startswith('__'):
                # Skip internal rule names
                continue
            else:
                # Use token as-is but lowercase
                result.append(token.lower().replace('_', ' '))

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for item in result:
            if item not in seen:
                seen.add(item)
                unique.append(item)

        return unique

    def _check_rust_feature_hints(self, source, token, line):
        """Check for common Rust features and return (message, hint) or None.

        Only runs on error paths — zero cost on successful compilation.
        """
        if token is None or token.type == '$END':
            return None

        source_line = get_source_line(source, line) or ''
        # Lark columns are 1-based; slice up to (but not including) the token
        col = token.column - 1 if token.column > 0 else 0
        before = source_line[:col]

        import re

        # Generics: '<' after 'fn name' or after a type name
        if token.value == '<':
            if re.search(r'\bfn\s+\w+\s*$', before):
                return ("generics are not supported in R65",
                        "use concrete types (u8, u16) instead of type parameters")
            # Turbofish: foo::<u8>()
            if re.search(r'::\s*$', before):
                return ("turbofish syntax (::<T>) is not supported in R65",
                        "R65 has no generics; call functions directly without type parameters")
            # Also catch Type<T> patterns
            if re.search(r'\b[A-Z]\w*\s*$', before):
                return ("generics are not supported in R65",
                        "use concrete types (u8, u16) instead of type parameters")

        # Closures: '|' after '=' in a let/assignment context
        if token.value == '|':
            if before.rstrip().endswith('='):
                return ("closures are not supported in R65",
                        "use function pointers (fn() or far fn())")

        # 'use' statement: 'use' was consumed as _reserved_keyword, error on following tokens
        # Detect by checking if 'use' appears earlier on the same line
        if re.search(r'\buse\s+\w+\s*$', before) or re.search(r'\buse\s*$', before):
            return ("'use' imports are not supported in R65",
                    "use include!(\"file.r65\") for file inclusion")

        # 'mod foo': `mod` is not a keyword terminal, so the parser fails on the
        # module name after it
        if re.search(r'(?:^|[;}])\s*mod\s+$', before):
            return ("'mod' modules are not supported in R65",
                    "use include!(\"file.r65\") for file inclusion")

        # Struct literal with register name: S { x: 1 } where S is a register
        if token.value == '{' or (hasattr(token, 'type') and token.type == 'LBRACE'):
            if re.search(r'=\s*[ABSXYD]\s*$', before):
                reg = before.rstrip()[-1]
                return (f"'{reg}' is a hardware register name, not a struct identifier",
                        f"rename the struct to avoid conflict with register {reg} "
                        f"(reserved: A, B, S, X, Y, D, DBR, PBR, STATUS)")

        # Tuple literals: (1, 2) — comma inside parenthesized expression
        if token.value == ',':
            if re.search(r'\(\s*(?:\d+|0x[0-9a-fA-F]+|\w+)\s*$', before):
                return ("tuple literals are not supported in R65",
                        "R65 has no tuple type; use separate variables or a struct")

        # Procedural macro / derive attributes on declarations that don't accept
        # attributes in the grammar (e.g. `#[derive(Debug)] enum E { ... }`;
        # structs/unions/newtypes take attributes and are handled by
        # `_reject_derive_on_type`).
        # The attribute parses as a stray `#`, and the next decl token is what
        # trips the parser. Scan the preceding source for an unmatched
        # #[derive(...)] / #[proc_macro...] and emit a targeted hint.
        lines = source.splitlines(keepends=True)
        prefix_offset = sum(len(l) for l in lines[:line - 1]) + max(0, token.column - 1)
        prefix = source[:prefix_offset]
        m = re.search(r'#\[\s*(derive|proc_macro|proc_macro_derive|proc_macro_attribute)\b([^\]]*)\]\s*\Z',
                      prefix, re.DOTALL)
        if m:
            name = m.group(1)
            if name == 'derive':
                traits = re.findall(r'\w+', m.group(2) or '')
                # The declaration the attribute was meant for follows the error token
                decl = re.search(r'\b(?:struct|union|enum|trait|type)\s+(\w+)',
                                 source[prefix_offset:])
                return ("#[derive(...)] is not supported in R65",
                        derive_hint(traits, decl.group(1) if decl else None))
            return (f"#[{name}] is not supported in R65",
                    "procedural macros are not supported; use `macro_rules!` "
                    "for declarative macros (see docs/macros.md)")

        return None

    def _check_macro_syntax_hints(self, source: str, error_str: str) -> str:
        """Check for common Rust macro syntax mistakes and return helpful error message."""
        import re

        # Check for unsupported fragment types
        unsupported_fragments = ['stmt', 'block', 'item', 'meta', 'pat', 'path', 'vis', 'lifetime']
        for frag in unsupported_fragments:
            if f':{frag}' in source or f': {frag}' in source:
                return (
                    f"Fragment type '${frag}' is not supported in R65.\n"
                    "  Supported fragment types: expr, ident, literal, ty, reg, tt\n"
                    "See docs/macros.md for details on each fragment type."
                )

        # Check for + repetition (we only support *)
        if re.search(r'\$\([^)]+\)\s*[,;]?\s*\+', source):
            return (
                "R65 only supports '*' repetition (zero or more), not '+' (one or more).\n"
                "  Unsupported:  $($x:expr),+\n"
                "  Supported:    $($x:expr),*\n"
                "See docs/macros.md for repetition syntax."
            )

        # Check for semicolon separator in repetition
        if re.search(r'\$\([^)]+\)\s*;\s*\*', source):
            return (
                "R65 only supports comma separator in repetitions, not semicolon.\n"
                "  Unsupported:  $($x:expr);*\n"
                "  Supported:    $($x:expr),*\n"
                "See docs/macros.md for repetition syntax."
            )

        return None


# Module-level parser instance - Lark LALR table construction is expensive,
# so we reuse the same parser for all parse calls
_parser = Parser()


def parse(source: str, filename: str = "<input>", included_from=None,
          origin_package=None) -> ast.Program:
    """
    Convenience function to parse source code.

    Args:
        source: Source code to parse
        filename: Name of the source file
        included_from: SourceLocation of include! statement if parsing included file

    Returns:
        Program AST node
    """
    return _parser.parse(source, filename, included_from=included_from,
                         origin_package=origin_package)
