# -*- coding: utf-8 -*-

from r65.tools.compression import aplib
from r65.tools.compression import byte_rle
from r65.tools.compression import rle1
from r65.tools.compression import rle2
from r65.tools.compression import lz1
from r65.tools.compression import lz2
from r65.tools.compression import lz3
from r65.tools.compression import lz4
from r65.tools.compression import lz5
from r65.tools.compression import lz19
from r65.tools.compression import lz77
from r65.tools.compression import hal

_ENCODINGS = frozenset([
	'aplib', 'byte_rle', 'rle1', 'rle2',
	'lz1', 'lz2', 'lz3', 'lz4', 'lz5', 'lz19', 'lz77', 'hal',
])

def get_names():
	import sys
	from inspect import getmembers, ismodule
	return [m[0] for m in getmembers(sys.modules[__name__], ismodule) if m[0] in _ENCODINGS]

def get_encoding(encoding):
	import sys
	return getattr(sys.modules[__name__], encoding)

def compress(encoding, data):
	return get_encoding(encoding).compress(data)

def decompress(encoding, data):
	return get_encoding(encoding).decompress(data)
