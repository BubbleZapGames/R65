"""
Write cc65-compatible .dbg files for Mesen debugger.

Generates debug info files in the cc65/ld65 format that Mesen can import
for source-level debugging support.
"""

from typing import TextIO

from r65.compiler.codegen.debug_info import DebugInfoCollector


class Cc65DebugWriter:
    """
    Writes debug info in cc65 .dbg format.

    The cc65 debug format is a text-based format with tab-separated key=value
    pairs. Each line starts with a record type identifier followed by fields.

    Format:
        version major=2,minor=0
        info csym=0,file=N,...
        file id=0,name="file.r65",size=0,mtime=0
        seg id=0,name="CODE",start=0x8000,...
        line id=0,file=0,line=10,type=0,span=0
        span id=0,seg=0,start=0,size=3
        sym id=0,name="main",addrsize=absolute,val=0x8000,seg=0,type=lab
        scope id=0,name="main",type=scope,size=16,span=0+1+2
    """

    VERSION_MAJOR = 2
    VERSION_MINOR = 0

    def __init__(self, debug_info: DebugInfoCollector):
        """
        Initialize debug writer.

        Args:
            debug_info: Collected debug information
        """
        self.info = debug_info

    def write(self, f: TextIO):
        """
        Write debug info to file object.

        Args:
            f: File object to write to
        """
        self._write_version(f)
        self._write_info(f)
        self._write_files(f)
        self._write_segments(f)
        self._write_lines(f)
        self._write_spans(f)
        self._write_symbols(f)
        self._write_scopes(f)

    def write_to_file(self, path: str):
        """
        Write debug info to file path.

        Args:
            path: Output file path
        """
        with open(path, 'w') as f:
            self.write(f)

    def _write_version(self, f: TextIO):
        """Write version record."""
        f.write(f"version\tmajor={self.VERSION_MAJOR},minor={self.VERSION_MINOR}\n")

    def _write_info(self, f: TextIO):
        """Write info record with counts."""
        f.write(
            f"info\t"
            f"csym=0,"
            f"file={len(self.info.files)},"
            f"lib=0,"
            f"line={len(self.info.lines)},"
            f"mod=1,"
            f"scope={len(self.info.scopes)},"
            f"seg={len(self.info.segments)},"
            f"span={len(self.info.spans)},"
            f"sym={len(self.info.symbols)},"
            f"type=0\n"
        )

    def _write_files(self, f: TextIO):
        """Write file records."""
        for dbg_file in sorted(self.info.files.values(), key=lambda x: x.id):
            # Escape backslashes and quotes in filename
            escaped_name = dbg_file.name.replace('\\', '/').replace('"', '\\"')
            f.write(f"file\tid={dbg_file.id},name=\"{escaped_name}\",size=0,mtime=0\n")

    def _write_segments(self, f: TextIO):
        """Write segment records."""
        for seg in self.info.segments:
            f.write(
                f"seg\t"
                f"id={seg.id},"
                f"name=\"{seg.name}\","
                f"start=0x{seg.start:06X},"
                f"size=0x{seg.size:04X},"
                f"addrsize={seg.addrsize},"
                f"type={seg.seg_type},"
                f"ooffs={seg.ooffs}\n"
            )

    def _write_lines(self, f: TextIO):
        """Write line records."""
        for line_entry in self.info.lines:
            spans = "+".join(str(s) for s in line_entry.span_ids) if line_entry.span_ids else "0"
            f.write(
                f"line\t"
                f"id={line_entry.id},"
                f"file={line_entry.file_id},"
                f"line={line_entry.line},"
                f"type={line_entry.line_type},"
                f"span={spans}\n"
            )

    def _write_spans(self, f: TextIO):
        """Write span records."""
        for span in self.info.spans:
            f.write(
                f"span\t"
                f"id={span.id},"
                f"seg={span.seg_id},"
                f"start={span.start},"
                f"size={span.size}\n"
            )

    def _write_symbols(self, f: TextIO):
        """Write symbol records."""
        for sym in self.info.symbols:
            parts = [f"id={sym.id}", f"name=\"{sym.name}\""]
            parts.append("addrsize=absolute")
            parts.append(f"val=0x{sym.value:06X}")
            if sym.seg_id is not None:
                parts.append(f"seg={sym.seg_id}")
            if sym.size > 0:
                parts.append(f"size={sym.size}")
            parts.append(f"type={sym.sym_type}")
            if sym.scope_id is not None:
                parts.append(f"scope={sym.scope_id}")
            f.write(f"sym\t{','.join(parts)}\n")

    def _write_scopes(self, f: TextIO):
        """Write scope records."""
        for scope in self.info.scopes:
            parts = [f"id={scope.id}", f"name=\"{scope.name}\""]
            parts.append(f"type={scope.scope_type}")
            parts.append(f"size={scope.size}")
            if scope.parent_id is not None:
                parts.append(f"parent={scope.parent_id}")
            if scope.sym_id is not None:
                parts.append(f"sym={scope.sym_id}")
            if scope.span_ids:
                spans = "+".join(str(s) for s in scope.span_ids)
                parts.append(f"span={spans}")
            f.write(f"scope\t{','.join(parts)}\n")
