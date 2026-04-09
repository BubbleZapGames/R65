# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
"""
Parser for R65 using Lark.

Transforms Lark parse trees into our custom AST.
"""
from pathlib import Path
from lark import Lark, Transformer, Token as LarkToken, Tree, v_args
from lark.exceptions import UnexpectedToken, UnexpectedCharacters, UnexpectedEOF, VisitError
from r65.compiler.frontend import ast
from r65.compiler.errors import ParseError, SourceLocation, get_source_line
from typing import List, Union, Optional


# Load the grammar
GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"
with open(GRAMMAR_PATH) as f:
    GRAMMAR = f.read()


class ASTBuilder(Transformer):
    """
    Lark Transformer that builds our AST from the parse tree.

    Each method corresponds to a grammar rule and transforms
    the matched nodes into AST nodes.
    """

    def __init__(self, filename: str = "<input>", included_from=None):
        """
        Initialize the AST builder.

        Args:
            filename: Current source file name
            included_from: SourceLocation of include! statement if this is included
        """
        super().__init__()
        self.filename = filename
        self.included_from = included_from

    def _make_source_loc(self, meta):
        """Create a SourceLocation from Lark meta info."""
        from r65.compiler.hir.errors import SourceLocation
        if meta is None:
            return None
        return SourceLocation(
            file_path=self.filename,
            line=getattr(meta, 'line', 0),
            column=getattr(meta, 'column', 0),
            included_from=self.included_from
        )

    _SUFFIXES = ('u16', 'i16', 'u8', 'i8')

    def _strip_integer_suffix(self, raw: str):
        """Strip type suffix from integer literal string. Returns (digits, suffix)."""
        for s in self._SUFFIXES:
            if raw.endswith(s):
                return raw[:-len(s)], s
        return raw, None

    def _parse_integer(self, value: str) -> int:
        """Parse an integer literal."""
        clean_value = value.replace('_', '')
        try:
            if clean_value.startswith('0x') or clean_value.startswith('0X'):
                if len(clean_value) <= 2:
                    raise ValueError(f"missing digits after {clean_value[:2]}")
                return int(clean_value, 16)
            elif clean_value.startswith('0b') or clean_value.startswith('0B'):
                if len(clean_value) <= 2:
                    raise ValueError(f"missing digits after {clean_value[:2]}")
                return int(clean_value, 2)
            else:
                return int(clean_value, 10)
        except ValueError as e:
            # Raise ValueError, not ParseError - let the VisitError handler add source location
            raise ValueError(f"invalid integer literal '{value}': {e}")

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

    def _validate_identifier_not_register(self, identifier: str, token: LarkToken):
        """
        Validate identifier (currently a no-op).

        Lowercase and mixed-case identifiers are valid variable names, even if they
        match register names case-insensitively. Only uppercase names (A, X, Y,
        STATUS, D, DBR, PBR, S) are treated as registers by the grammar.

        Examples of valid variable names:
        - let x: u8 = 0;      // 'x' is a variable, not register X
        - let status: u8 = 1; // 'status' is a variable, not register STATUS
        - let Status: u8 = 2; // Mixed case also valid
        """
        # No validation needed - the grammar's REGISTER terminal only matches
        # uppercase register names. All other identifiers are valid variable names.
        pass

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
        elif text.startswith('/**'):
            # Block doc comment: strip /** and */
            inner = text[3:-2]
            # Strip leading * from each line (common formatting)
            lines = inner.split('\n')
            stripped = []
            for line in lines:
                s = line.lstrip()
                if s.startswith('* '):
                    stripped.append(s[2:])
                elif s.startswith('*') and (len(s) == 1 or not s[1:].strip()):
                    stripped.append('')
                else:
                    stripped.append(line)
            return '\n'.join(stripped)
        elif text.startswith('/*!'):
            inner = text[3:-2]
            lines = inner.split('\n')
            stripped = []
            for line in lines:
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
        'extern':   ("'extern' is not supported in R65", "use #[hw(addr)] for hardware registers or asm!() for assembly"),
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
        items = self._filter_tokens(tree.children, keep_types={'IDENT', 'MUT', 'FAR', 'CONST', 'INTEGER', 'STRING', 'BOOLEAN', 'REGISTER'})

        # Collect doc comments
        doc, idx = self._collect_doc_comments(items, 0)

        # Collect attributes
        attrs, idx = self._collect_attributes(items, idx)

        # Check for const
        is_const = False
        if idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'CONST':
            is_const = True
            idx += 1

        # Check for far
        is_far = False
        if idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'FAR':
            is_far = True
            idx += 1

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
            is_far=is_far,
            name=name.value if isinstance(name, LarkToken) else name,
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
        is_far = False
        is_near = False
        if idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'FAR':
            is_far = True
            idx += 1
        elif idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'NEAR':
            is_near = True
            idx += 1

        # Check for pointer (*) prefix
        is_pointer = False
        if idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'STAR':
            is_pointer = True
            idx += 1

        # Name
        name = items[idx].value if isinstance(items[idx], LarkToken) else items[idx]
        idx += 1

        # Check for binding (@ register or variable)
        binding = None
        if idx < len(items) - 1:  # There's something between name and type
            binding_node = items[idx]
            if isinstance(binding_node, ast.Register):
                binding = binding_node
            elif isinstance(binding_node, ast.Identifier):
                binding = binding_node
            else:
                binding = binding_node.value if isinstance(binding_node, LarkToken) else binding_node
            idx += 1

        # Type is always last (this is the pointee type if is_pointer is True)
        param_type = items[-1]

        # If this is a pointer parameter, wrap the type in PointerType
        if is_pointer:
            param_type = ast.PointerType(is_far=is_far, pointee_type=param_type)

        return ast.Parameter(name=name, binding=binding, param_type=param_type)

    @v_args(tree=True)
    def param_safe_ptr_error(self, tree):
        """Error handler for safe pointer syntax in parameters."""
        # Extract the name for a helpful error message
        items = self._filter_tokens(tree.children, keep_types={'IDENT', 'AMPER'})
        name = None
        for item in items:
            if isinstance(item, LarkToken) and item.type == 'IDENT':
                name = item.value
                break
        name_str = name if name else "name"
        source_loc = self._make_source_loc(tree.meta) if hasattr(tree, 'meta') else None
        raise ParseError(
            f"safe pointers are not supported in R65",
            source_loc=source_loc,
            hint=f"use '{name_str}: *type' instead of '&{name_str}: type' for pointer parameters"
        )

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
        is_far = False
        is_near = False
        if idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'FAR':
            is_far = True
            idx += 1
        elif idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'NEAR':
            is_near = True
            idx += 1

        # Check for mut token
        is_mut = False
        if idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'MUT':
            is_mut = True
            idx += 1

        # Check for pointer (*) prefix
        is_pointer = False
        if idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'STAR':
            is_pointer = True
            idx += 1

        # Name
        name = items[idx].value if isinstance(items[idx], LarkToken) else items[idx]
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
        return token.value if isinstance(token, LarkToken) else token

    @v_args(tree=True)
    def static_decl_safe_ptr_error(self, tree):
        """Error handler for safe pointer syntax in static declarations."""
        # Extract the name for a helpful error message
        items = self._filter_tokens(tree.children, keep_types={'IDENT', 'AMPER', 'MUT'})
        name = None
        is_mut = False
        for item in items:
            if isinstance(item, LarkToken):
                if item.type == 'MUT':
                    is_mut = True
                elif item.type == 'IDENT':
                    name = item.value
                    break
        name_str = name if name else "name"
        mut_str = "mut " if is_mut else ""
        source_loc = self._make_source_loc(tree.meta) if hasattr(tree, 'meta') else None
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
        name = items[idx].value if isinstance(items[idx], LarkToken) else items[idx]
        const_type = items[idx + 1]
        value = items[idx + 2]
        return ast.ConstDecl(name=name, const_type=const_type, value=value,
                             doc=doc, source_loc=self._make_source_loc(tree.meta))

    def struct_decl(self, items):
        """Struct declaration."""
        items = self._filter_tokens(items)
        doc, idx = self._collect_doc_comments(items, 0)
        name = items[idx].value if isinstance(items[idx], LarkToken) else items[idx]
        fields = [item for item in items[idx + 1:] if isinstance(item, ast.StructField)]
        return ast.StructDecl(name=name, fields=fields, doc=doc)

    def struct_field(self, items):
        """Struct field."""
        items = self._filter_tokens(items, keep_types={'IDENT', 'FAR', 'NEAR', 'STAR'})

        idx = 0

        # Check for far/near modifier (for pointer fields)
        is_far = False
        is_near = False
        if idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'FAR':
            is_far = True
            idx += 1
        elif idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'NEAR':
            is_near = True
            idx += 1

        # Check for pointer (*) prefix
        is_pointer = False
        if idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'STAR':
            is_pointer = True
            idx += 1

        # Name
        name = items[idx].value if isinstance(items[idx], LarkToken) else items[idx]
        idx += 1

        # Type (this is the pointee type if is_pointer is True)
        field_type = items[idx]

        # If this is a pointer field, wrap the type in PointerType
        if is_pointer:
            field_type = ast.PointerType(is_far=is_far, pointee_type=field_type)

        return ast.StructField(name=name, field_type=field_type)

    @v_args(tree=True)
    def struct_field_safe_ptr_error(self, tree):
        """Error handler for safe pointer syntax in struct fields."""
        # Extract the name for a helpful error message
        items = self._filter_tokens(tree.children, keep_types={'IDENT', 'AMPER'})
        name = None
        for item in items:
            if isinstance(item, LarkToken) and item.type == 'IDENT':
                name = item.value
                break
        name_str = name if name else "name"
        source_loc = self._make_source_loc(tree.meta) if hasattr(tree, 'meta') else None
        raise ParseError(
            f"safe pointers are not supported in R65",
            source_loc=source_loc,
            hint=f"use '{name_str}: *type' instead of '&{name_str}: type' for struct fields"
        )

    def enum_decl(self, items):
        """Enum declaration."""
        items = self._filter_tokens(items)
        doc, idx = self._collect_doc_comments(items, 0)
        name = items[idx].value if isinstance(items[idx], LarkToken) else items[idx]
        variants = [item for item in items[idx + 1:] if isinstance(item, ast.EnumVariant)]
        return ast.EnumDecl(name=name, variants=variants, doc=doc)

    def enum_variant(self, items):
        """Enum variant."""
        items = self._filter_tokens(items)
        name = items[0].value if isinstance(items[0], LarkToken) else items[0]
        value = items[1] if len(items) > 1 else None
        return ast.EnumVariant(name=name, value=value)

    @v_args(tree=True)
    def impl_decl(self, tree):
        """Impl block declaration: impl [far] StructName { methods and constants }"""
        items = self._filter_tokens(tree.children, keep_types={'IDENT', 'FAR'})

        # Collect doc comments
        doc, idx = self._collect_doc_comments(items, 0)

        # Check for far modifier
        is_far = False
        if idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'FAR':
            is_far = True
            idx += 1

        # Struct name
        struct_name = items[idx].value if isinstance(items[idx], LarkToken) else items[idx]
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
            elif isinstance(item, ast.ImplMacro):
                macros.append(item)

        return ast.ImplDecl(
            struct_name=struct_name,
            is_far=is_far,
            methods=methods,
            constants=constants,
            macros=macros,
            doc=doc,
            source_loc=self._make_source_loc(tree.meta)
        )

    @v_args(tree=True)
    def impl_method(self, tree):
        """Method declaration in impl block."""
        items = self._filter_tokens(tree.children, keep_types={'IDENT', 'FAR', 'CONST'})

        # Collect attributes
        attrs, idx = self._collect_attributes(items, 0)

        # Check for const
        is_const = False
        if idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'CONST':
            is_const = True
            idx += 1

        # Check for far fn
        is_far = False
        if idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'FAR':
            is_far = True
            idx += 1

        # Method name
        name = items[idx].value if isinstance(items[idx], LarkToken) else items[idx]
        idx += 1

        # Parameters (from impl_param_list)
        self_is_far = False
        params = []
        if idx < len(items):
            param_result = items[idx]
            if isinstance(param_result, tuple) and param_result[0] == 'impl_params':
                self_is_far = param_result[1]
                params = param_result[2]
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
            is_far=is_far,
            name=name,
            self_is_far=self_is_far,
            params=params,
            return_type=return_type,
            body=body,
            is_const=is_const,
            source_loc=self._make_source_loc(tree.meta)
        )

    def impl_param_list(self, items):
        """Parameter list for impl methods - may include self parameter."""
        # Check if first item is a self_param tuple
        if items and isinstance(items[0], tuple) and items[0][0] == 'self_param':
            self_is_far = items[0][1]
            # Rest are regular parameters
            params = [item for item in items[1:] if isinstance(item, ast.Parameter)]
            return ('impl_params', self_is_far, params)
        else:
            # No self param - just regular parameters
            params = [item for item in items if isinstance(item, ast.Parameter)]
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

        return ('self_param', is_far)

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
        name = items[0].value if isinstance(items[0], LarkToken) else items[0]
        const_type = items[1]
        value = items[2]
        return ast.ImplConst(name=name, const_type=const_type, value=value)

    @v_args(tree=True)
    def impl_macro(self, tree):
        """Macro definition inside impl block: macro_rules! name($param:type, ...) { body }"""
        items = tree.children
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

        return ast.ImplMacro(
            name=name,
            params=params,
            body_tokens=body_tokens,
            source_loc=self._make_source_loc(tree.meta)
        )

    @v_args(tree=True)
    def impl_trait_decl(self, tree):
        """Trait impl declaration: impl TraitName for StructName { methods and constants }"""
        items = self._filter_tokens(tree.children, keep_types={'IDENT'})

        # Collect doc comments
        doc, idx = self._collect_doc_comments(items, 0)

        # Trait name (first IDENT)
        trait_name = items[idx].value if isinstance(items[idx], LarkToken) else items[idx]
        idx += 1

        # Struct name (second IDENT, after FOR which is filtered out)
        struct_name = items[idx].value if isinstance(items[idx], LarkToken) else items[idx]
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
            is_far=False,
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
        name = items[idx].value if isinstance(items[idx], LarkToken) else items[idx]
        idx += 1

        # Collect methods and constants
        methods = []
        constants = []
        for item in items[idx:]:
            if isinstance(item, ast.TraitMethod):
                methods.append(item)
            elif isinstance(item, ast.TraitConst):
                constants.append(item)

        return ast.TraitDecl(
            name=name,
            methods=methods,
            constants=constants,
            doc=doc,
            source_loc=self._make_source_loc(tree.meta)
        )

    @v_args(tree=True)
    def trait_method(self, tree):
        """Method signature in trait declaration (no body)."""
        items = self._filter_tokens(tree.children, keep_types={'IDENT', 'FAR'})

        idx = 0

        # Check for far fn
        is_far = False
        if idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'FAR':
            is_far = True
            idx += 1

        # Method name
        name = items[idx].value if isinstance(items[idx], LarkToken) else items[idx]
        idx += 1

        # Parameters (from trait_param_list)
        self_is_far = False
        params = []
        if idx < len(items):
            param_result = items[idx]
            if isinstance(param_result, tuple) and param_result[0] == 'impl_params':
                self_is_far = param_result[1]
                params = param_result[2]
                idx += 1
            elif isinstance(param_result, list):
                params = param_result
                idx += 1

        # Return type (optional)
        return_type = None
        if idx < len(items):
            return_type = items[idx]
            idx += 1

        return ast.TraitMethod(
            is_far=is_far,
            name=name,
            self_is_far=self_is_far,
            params=params,
            return_type=return_type,
            source_loc=self._make_source_loc(tree.meta)
        )

    def trait_param_list(self, items):
        """Parameter list for trait methods - self parameter required."""
        # Reuse impl_param_list logic
        if items and isinstance(items[0], tuple) and items[0][0] == 'self_param':
            self_is_far = items[0][1]
            params = [item for item in items[1:] if isinstance(item, ast.Parameter)]
            return ('impl_params', self_is_far, params)
        else:
            params = [item for item in items if isinstance(item, ast.Parameter)]
            return params

    def trait_const(self, items):
        """Associated constant declaration in trait: const NAME: type;"""
        items = self._filter_tokens(items)
        name = items[0].value if isinstance(items[0], LarkToken) else items[0]
        const_type = items[1]
        return ast.TraitConst(name=name, const_type=const_type)

    def type_alias(self, items):
        """Type alias."""
        items = self._filter_tokens(items)
        name = items[0].value if isinstance(items[0], LarkToken) else items[0]
        aliased_type = items[1]
        return ast.TypeAlias(name=name, aliased_type=aliased_type)

    @v_args(tree=True)
    def include_stmt(self, tree):
        """Include statement."""
        items = self._filter_tokens(tree.children, keep_types={'STRING'})
        path = items[0].value.strip('"')  # Remove quotes
        return ast.IncludeStmt(path=path, source_loc=self._make_source_loc(tree.meta))

    def stack_directive(self, items):
        """Stack directive: #[stack(lower, upper)]"""
        items = self._filter_tokens(items, keep_types={'INTEGER'})
        lower = int(items[0].value, 0)  # Parse with base detection (0x prefix)
        upper = int(items[1].value, 0)
        return ast.StackDirective(lower=lower, upper=upper)

    def bank_directive(self, items):
        """Bank directive: #[bank(n)] or #[bank(auto)] - sets current ROM bank for following declarations."""
        # Grammar uses AUTO_KEYWORD for "auto"
        items = self._filter_tokens(items, keep_types={'INTEGER', 'AUTO_KEYWORD'})
        if items:
            token = items[0]
            if isinstance(token, LarkToken):
                if token.type == 'AUTO_KEYWORD':
                    # #[bank(auto)] - automatic placement
                    return ast.BankDirective(bank_number=None)
                elif token.type == 'INTEGER':
                    # #[bank(n)] - explicit bank number
                    bank_number = int(token.value, 0)  # Parse with base detection (0x prefix)
                    return ast.BankDirective(bank_number=bank_number)
        # Default to bank 0 if something is wrong
        return ast.BankDirective(bank_number=0)

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
    def macro_decl(self, tree):
        """Macro definition: macro! name($param:type, ...) { body }"""
        items = tree.children
        # Filter to get name and params
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
            params=params,
            body_tokens=body_tokens,
            source_loc=self._make_source_loc(tree.meta)
        )

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

    @v_args(tree=True)
    def macro_invocation_stmt(self, tree):
        """Top-level macro invocation: name!(args);"""
        items = tree.children
        name = None
        args = []

        for item in items:
            if isinstance(item, LarkToken) and item.type == 'IDENT':
                name = item.value
            elif isinstance(item, list):  # macro_args result
                args = item

        return ast.MacroInvocationStmt(
            name=name,
            args=args,
            source_loc=self._make_source_loc(tree.meta)
        )

    @v_args(tree=True)
    def macro_invocation_stmt_inner(self, tree):
        """Statement-level macro invocation: name!(args);"""
        items = tree.children
        name = None
        args = []

        for item in items:
            if isinstance(item, LarkToken) and item.type == 'IDENT':
                name = item.value
            elif isinstance(item, list):  # macro_args result
                args = item

        return ast.MacroInvocationStmtInner(
            name=name,
            args=args,
            source_loc=self._make_source_loc(tree.meta)
        )

    @v_args(tree=True)
    def macro_invocation_expr(self, tree):
        """Expression-level macro invocation: name!(args)"""
        items = tree.children
        name = None
        args = []

        for item in items:
            if isinstance(item, LarkToken) and item.type == 'IDENT':
                name = item.value
            elif isinstance(item, list):  # macro_args result
                args = item

        return ast.MacroInvocation(
            name=name,
            args=args,
            source_loc=self._make_source_loc(tree.meta)
        )

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

    def attribute_inner(self, items):
        """Attribute inner."""
        items = self._filter_tokens(items)
        name = items[0]  # Will be string from attr_name
        # args will be a list from attribute_args if present
        args = items[1] if len(items) > 1 and isinstance(items[1], list) else []
        return ast.Attribute(name=name, args=args)

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
        return ast.Attribute(name='cfg', args=[ast.AttributeArg(name=None, value=condition)])

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
        return ast.Block(statements=statements)

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
        return ast.Block(statements=statements)

    def block(self, items):
        """Block statement."""
        # Filter out brace tokens, keep only statement nodes
        statements = [item for item in items if not isinstance(item, LarkToken)]
        return ast.Block(statements=statements)

    def single_pattern(self, items):
        """Single binding pattern: IDENT or IDENT @ binding"""
        items = self._filter_tokens(items)
        name = items[0].value
        binding = items[1] if len(items) > 1 else None
        return ('single', name, binding)

    def pointer_pattern(self, items):
        """Pointer pattern: *IDENT or *IDENT @ binding"""
        items = self._filter_tokens(items)
        name = items[0].value if isinstance(items[0], LarkToken) else items[0]
        binding = items[1] if len(items) > 1 else None
        return ('pointer', name, binding)

    def tuple_pattern(self, items):
        """Tuple pattern: (a, b, c)"""
        items = self._filter_tokens(items)
        names = [item.value for item in items if isinstance(item, LarkToken) and item.type == 'IDENT']
        return ('tuple', names)

    @v_args(tree=True)
    def let_stmt(self, tree):
        """Let statement."""
        items = self._filter_tokens(tree.children, keep_types={'IDENT', 'MUT', 'FAR', 'NEAR', 'INTEGER', 'STRING', 'BOOLEAN', 'REGISTER'})

        idx = 0

        # Check for far/near modifier (for pointer declarations)
        is_far = False
        is_near = False
        if idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'FAR':
            is_far = True
            idx += 1
        elif idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'NEAR':
            is_near = True
            idx += 1

        # Check for mut
        is_mut = False
        if idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'MUT':
            is_mut = True
            idx += 1

        # Get pattern (single, pointer, or tuple)
        pattern_item = items[idx]
        idx += 1

        name = None
        binding = None
        tuple_pattern = None
        is_pointer = False

        if isinstance(pattern_item, tuple):
            if pattern_item[0] == 'single':
                name = pattern_item[1]
                binding = pattern_item[2]
            elif pattern_item[0] == 'pointer':
                name = pattern_item[1]
                binding = pattern_item[2]
                is_pointer = True
            elif pattern_item[0] == 'tuple':
                tuple_pattern = ast.TuplePattern(names=pattern_item[1])
        else:
            # Fallback for direct token (shouldn't happen with new grammar)
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
            var_name = name if name else (tuple_pattern.names[0] if tuple_pattern else "variable")
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
            pattern=tuple_pattern
        )

    @v_args(tree=True)
    def let_stmt_safe_ptr_error(self, tree):
        """Error handler for safe pointer syntax in let statements."""
        items = self._filter_tokens(tree.children, keep_types={'IDENT', 'AMPER'})
        name = None
        for item in items:
            if isinstance(item, LarkToken) and item.type == 'IDENT':
                name = item.value
                break
            elif isinstance(item, tuple) and len(item) >= 2:
                # safe_ptr_pattern returns a tuple
                name = item[1]
                break
        name_str = name if name else "name"
        source_loc = self._make_source_loc(tree.meta) if hasattr(tree, 'meta') else None
        raise ParseError(
            f"safe pointers are not supported in R65",
            source_loc=source_loc,
            hint=f"use 'let {name_str}: *type' instead of 'let &{name_str}: type' for pointer variables"
        )

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
        """Return tuple: (expr, expr, ...)."""
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
        lvalue = items[0]
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
        lvalue = items[0]
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
        then_block = items[1]
        else_block = items[2] if len(items) > 2 else None
        return ast.IfStmt(condition=condition, then_block=then_block, else_block=else_block,
                         source_loc=self._make_source_loc(tree.meta))

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
        digits, suffix = self._strip_integer_suffix(raw)
        value = self._parse_integer(digits)
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

    _CHAR_ESCAPES = {
        '\\n': 10, '\\t': 9, '\\r': 13, '\\0': 0,
        '\\\\': 92, "\\'": 39, '\\"': 34,
    }

    def _parse_char_content(self, inner: str, src_loc) -> int:
        """Parse character literal content, returning the byte value (0-255)."""
        if inner.startswith('\\'):
            # Hex escape: \xNN
            if inner.startswith('\\x'):
                return int(inner[2:], 16)
            # Simple escape
            if inner in self._CHAR_ESCAPES:
                return self._CHAR_ESCAPES[inner]
            raise ParseError(
                f"invalid escape sequence in character literal: '{inner}'",
                source_loc=src_loc,
                hint=r"supported escapes: \n \t \r \0 \\ \' \" \xNN"
            )
        # Single character — must be 7-bit ASCII. Use \xNN escape for 128-255.
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
        return code

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
        # Validate that this isn't a wrong-case register name
        self._validate_identifier_not_register(identifier, token)
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
        struct_name = items[0].value if isinstance(items[0], LarkToken) else items[0]
        fields = [f for f in items[1:] if isinstance(f, ast.StructFieldInit)]
        return ast.StructLiteralExpr(struct_name=struct_name, fields=fields)

    def struct_field_init(self, items):
        """Struct field initializer (e.g., x: 10)."""
        items = self._filter_tokens(items)
        name = items[0].value if isinstance(items[0], LarkToken) else items[0]
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
        final_expr = items[-1]
        statements = list(items[:-1])
        return ast.BlockExpression(
            statements=statements,
            final_expr=final_expr,
            source_loc=self._make_source_loc(tree.meta)
        )

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
            else_block=else_block,
            source_loc=self._make_source_loc(tree.meta)
        )

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
        return ast.Block(statements=list(items), source_loc=self._make_source_loc(tree.meta))

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
                filtered.append(self._parse_integer(item.value))
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
        value = self._parse_integer(int_token.value)
        return -value if has_neg else value

    def pattern_literal(self, items):
        """Literal pattern (integer, boolean, or character)."""
        items = self._filter_tokens(items, keep_types={'INTEGER', 'BOOLEAN', 'CHAR_LITERAL'})
        token = items[0]
        if token.type == 'INTEGER':
            value = self._parse_integer(token.value)
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
        value = self._parse_integer(int_token.value)
        return ast.LiteralPattern(value=-value)

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
        
        # Check for built-in stringify function
        if (isinstance(func, LarkToken) and func.type == 'IDENT' and func.value == 'stringify'):
            # Handle stringify! as special built-in function
            return ast.StringifyCall(func=func, args=args)
        
        # Check for built-in cfg function
        if (isinstance(func, LarkToken) and func.type == 'CFG_FUNCTION'):
            # Handle cfg! as special built-in function
            return ast.FunctionCall(func=ast.Identifier(name='cfg'), args=args)
        
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
        field = items[1].value if isinstance(items[1], LarkToken) else items[1]
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

    def type_cast(self, items):
        """Type cast."""
        items = self._filter_tokens(items)
        return ast.TypeCast(expr=items[0], target_type=items[1])

    # Assignment
    @v_args(tree=True)
    def assign(self, tree):
        """Assignment."""
        items = self._filter_tokens(tree.children)
        lvalue = items[0]
        value = items[1]
        source_loc = self._make_source_loc(tree.meta)

        # Convert lvalue to appropriate target
        if isinstance(lvalue, ast.Identifier):
            target = lvalue
        else:
            target = lvalue

        return ast.Assignment(target=target, value=value, source_loc=source_loc)

    @v_args(tree=True)
    def compound_assign(self, tree):
        """Compound assignment (+=, -=, etc.)."""
        # Keep compound operator tokens
        items = self._filter_tokens(tree.children, keep_types={
            'PLUSEQUAL', 'MINUSEQUAL', 'STAREQUAL', 'SLASHEQUAL',
            'AMPEREQUAL', 'VBAREQUAL', 'CIRCUMFLEXEQUAL', 'LSHIFTEQUAL', 'RSHIFTEQUAL'
        })
        lvalue = items[0]
        # items[1] is the compound_op result (a token like PLUSEQUAL)
        compound_op_token = items[1]
        value = items[2]
        source_loc = self._make_source_loc(tree.meta)

        # Map compound operator token to binary operator
        op_map = {
            'PLUSEQUAL': '+',
            'MINUSEQUAL': '-',
            'STAREQUAL': '*',
            'SLASHEQUAL': '/',
            'AMPEREQUAL': '&',
            'VBAREQUAL': '|',
            'CIRCUMFLEXEQUAL': '^',
            'LSHIFTEQUAL': '<<',
            'RSHIFTEQUAL': '>>'
        }

        # Get the operator type
        if isinstance(compound_op_token, LarkToken):
            op_type = compound_op_token.type
        else:
            # It might be wrapped; extract the actual token type
            op_type = str(compound_op_token)

        operator = op_map.get(op_type, '+')  # Default to '+' if unknown

        return ast.CompoundAssignment(target=lvalue, operator=operator, value=value, source_loc=source_loc)

    def multi_assign(self, items):
        """Multiple assignment for multiple return values (e.g., lo, hi = func())."""
        items = self._filter_tokens(items)
        # All items except the last are lvalues, the last is the value
        targets = list(items[:-1])
        value = items[-1]

        return ast.MultiAssignment(targets=targets, value=value)

    @v_args(tree=True)
    def multi_assign_stmt(self, tree):
        """Multi-assignment statement: (a, b) = tuple_expr;"""
        items = self._filter_tokens(tree.children)
        # All items except the last are targets, the last is the expression
        targets = list(items[:-1])
        value = items[-1]
        source_loc = self._make_source_loc(tree.meta)

        multi_assign = ast.MultiAssignment(targets=targets, value=value, source_loc=source_loc)
        return ast.ExprStmt(expr=multi_assign, source_loc=source_loc)

    def multi_assign_ident(self, items):
        """Multi-assignment target identifier."""
        items = self._filter_tokens(items)
        token = items[0]
        identifier = token.value if isinstance(token, LarkToken) else token
        if isinstance(token, LarkToken):
            self._validate_identifier_not_register(identifier, token)
        return ast.Identifier(name=identifier)

    def multi_assign_register(self, items):
        """Multi-assignment target register."""
        items = self._filter_tokens(items, keep_types={'REGISTER'})
        return ast.Register(name=items[0].value if isinstance(items[0], LarkToken) else items[0])

    def compound_op(self, items):
        """Compound operator."""
        # Return the first token which is the compound operator
        # Keep all compound operator tokens
        items = self._filter_tokens(items, keep_types={
            'PLUSEQUAL', 'MINUSEQUAL', 'STAREQUAL', 'SLASHEQUAL',
            'AMPEREQUAL', 'VBAREQUAL', 'CIRCUMFLEXEQUAL', 'LSHIFTEQUAL', 'RSHIFTEQUAL'
        })
        return items[0]

    def lvalue_ident(self, items):
        """Lvalue identifier."""
        items = self._filter_tokens(items)
        token = items[0]
        identifier = token.value if isinstance(token, LarkToken) else token
        # Validate that this isn't a wrong-case register name
        if isinstance(token, LarkToken):
            self._validate_identifier_not_register(identifier, token)
        return ast.Identifier(name=identifier)

    def lvalue_register(self, items):
        """Lvalue register."""
        items = self._filter_tokens(items, keep_types={'REGISTER'})
        return ast.Register(name=items[0].value if isinstance(items[0], LarkToken) else items[0])

    def lvalue_array(self, items):
        """Lvalue array index."""
        items = self._filter_tokens(items)
        return ast.ArrayIndex(array=items[0], index=items[1])

    def lvalue_field(self, items):
        """Lvalue field access."""
        items = self._filter_tokens(items, keep_types={'IDENT'})
        return ast.FieldAccess(base=items[0], field=items[1].value if isinstance(items[1], LarkToken) else items[1])

    def lvalue_deref(self, items):
        """Lvalue pointer dereference."""
        items = self._filter_tokens(items)
        return ast.Dereference(pointer=items[0])

    @v_args(tree=True)
    def tuple_assign_stmt(self, tree):
        """Tuple assignment statement: (a, b) = expr; or (a) = expr;"""
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
        identifier = token.value if isinstance(token, LarkToken) else token
        if isinstance(token, LarkToken):
            self._validate_identifier_not_register(identifier, token)
        return ast.Identifier(name=identifier)

    def tuple_target_register(self, items):
        """Tuple assignment target - register."""
        items = self._filter_tokens(items, keep_types={'REGISTER'})
        return ast.Register(name=items[0].value if isinstance(items[0], LarkToken) else items[0])

    # ========================================================================
    # Types
    # ========================================================================

    def type_basic(self, items):
        """Basic type (built-in types or user-defined type names)."""
        items = self._filter_tokens(items, keep_types={'TYPE_NAME', 'IDENT'})
        return ast.BasicType(name=items[0].value if isinstance(items[0], LarkToken) else items[0])

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
        """Function type."""
        items = self._filter_tokens(items, keep_types={'FAR'})
        is_far = False
        idx = 0

        if idx < len(items) and isinstance(items[idx], LarkToken) and items[idx].type == 'FAR':
            is_far = True
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

    def type_tuple(self, items):
        """Tuple type for multiple return values: (u8, u8)."""
        # Filter out punctuation tokens (parentheses, commas)
        element_types = [item for item in items if not isinstance(item, LarkToken)]
        return ast.TupleType(element_types=element_types)

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
            Parser._lark_cache = Lark(
                GRAMMAR,
                parser='lalr',  # LALR is much faster than Earley
                lexer='contextual',  # Contextual lexer for LALR
                start='start',
                keep_all_tokens=True,
                propagate_positions=True  # Enable source location tracking
            )
        self.lark = Parser._lark_cache

    def parse(self, source: str, filename: str = "<input>",
              included_from=None) -> ast.Program:
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
            transformer = ASTBuilder(filename=filename, included_from=included_from)
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
                included_from=included_from
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
            elif char == '%':
                message = "the '%' modulo operator is not supported in R65"
                hint = "use mod8(a, b) for u8 or mod16(a, b) for u16 (from stdlib math.r65)"

            source_line = get_source_line(source, line)
            source_loc = SourceLocation(
                file_path=filename,
                line=line,
                column=column,
                source_line=source_line,
                included_from=included_from
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
                included_from=included_from
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
                included_from=included_from
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
            # Special
            '$END': 'end of file',
        }

        # Reserved Rust keywords that shouldn't appear in suggestions
        # (they're in the grammar as reserved but aren't valid R65 syntax)
        reserved_rust_keywords = {
            'MATCH', 'FOR', 'IN', 'IMPL', 'TRAIT', 'WHERE', 'USE', 'PUB',
            'CRATE', 'SELF', 'SELF_TYPE', 'SUPER', 'ASYNC', 'AWAIT', 'MOVE',
            'REF', 'DYN', 'EXTERN', 'UNSAFE', 'ABSTRACT', 'BECOME', 'BOX',
            'DO', 'FINAL', 'MACRO', 'OVERRIDE', 'PRIV', 'TYPEOF', 'UNSIZED',
            'VIRTUAL', 'YIELD', 'TRY',
        }

        result = []
        for token in expected:
            if token in translations:
                result.append(translations[token])
            elif token in reserved_rust_keywords:
                # Skip reserved Rust keywords
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

        # 'if let' / 'while let': unexpected token after 'if let' or 'while let'
        if re.search(r'\bif\s+let\s*$', before):
            return ("'if let' pattern matching is not supported in R65",
                    "use 'if' with comparison operators or 'match'")
        if re.search(r'\bwhile\s+let\s*$', before):
            return ("'while let' pattern matching is not supported in R65",
                    "use 'while' with comparison operators")

        # 'use' statement: 'use' was consumed as _reserved_keyword, error on following tokens
        # Detect by checking if 'use' appears earlier on the same line
        if re.search(r'\buse\s+\w+\s*$', before) or re.search(r'\buse\s*$', before):
            return ("'use' imports are not supported in R65",
                    "use include!(\"file.r65\") for file inclusion")

        # 'mod' keyword (it's an IDENT, not KEYWORD, so start() won't catch it)
        if token.type == 'IDENT' and token.value == 'mod':
            stripped = before.strip()
            if stripped == '' or stripped.endswith(';') or stripped.endswith('}'):
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

        return None

    def _check_macro_syntax_hints(self, source: str, error_str: str) -> str:
        """Check for common Rust macro syntax mistakes and return helpful error message."""
        import re

        # Check for Rust's => syntax in macros
        if '=>' in source and 'macro_rules!' in source:
            if re.search(r'macro_rules!\s*\w+\s*\{', source) or re.search(r'\)\s*=>\s*\{', source):
                return (
                    "R65 uses simplified macro syntax without '=>' and without pattern matching.\n"
                    "  Rust syntax:   macro_rules! name { ($x:expr) => { ... }; }\n"
                    "  R65 syntax:    macro_rules! name($x:expr) { ... }\n"
                    "See docs/macros.md for the complete macro syntax."
                )

        # Check for unsupported fragment types
        unsupported_fragments = ['stmt', 'block', 'item', 'meta', 'pat', 'path', 'vis', 'lifetime']
        for frag in unsupported_fragments:
            if f':{frag}' in source or f': {frag}' in source:
                return (
                    f"Fragment type '${frag}' is not supported in R65.\n"
                    "  Supported fragment types: expr, ident, literal, ty, reg, tt\n"
                    "See docs/macros.md for details on each fragment type."
                )

        # Check for multiple macro arms (Rust uses { } with multiple arms)
        if 'macro_rules!' in source and re.search(r'macro_rules!\s*\w+\s*\{[^}]*;\s*\(', source):
            return (
                "R65 macros support only a single pattern (no multiple arms).\n"
                "  Rust syntax:   macro_rules! name { (pat1) => {...}; (pat2) => {...}; }\n"
                "  R65 syntax:    macro_rules! name(params) { body }\n"
                "See docs/macros.md for the complete macro syntax."
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


# ParseError is now imported from r65.compiler.errors

# Module-level parser instance - Lark LALR table construction is expensive,
# so we reuse the same parser for all parse calls
_parser = Parser()


def parse(source: str, filename: str = "<input>", included_from=None) -> ast.Program:
    """
    Convenience function to parse source code.

    Args:
        source: Source code to parse
        filename: Name of the source file
        included_from: SourceLocation of include! statement if parsing included file

    Returns:
        Program AST node
    """
    return _parser.parse(source, filename, included_from=included_from)
