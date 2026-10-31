# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
# -*- coding: utf-8 -*-

from r65.tools.compression.lz import lz_compress, lz_decompress

def compress(data):
	return lz5_compress(data).do()

def decompress(data):
	return lz5_decompress(data).do()

class lz5_compress(lz_compress):

	REPEAT_INV = 5
	REPEAT_REL = 6

	def __init__(self, data):
		lz_compress.__init__(self, data)
		self._functions = [self._rle16,self._rle8,self._increment_fill,self._repeat_le,self._repeat_inverse,self._repeat_rel]

	def _repeat_inverse(self):
		length, index = self._search_inverse()
		return (self.REPEAT_INV, length, bytearray([index >> 8, index & 0xFF]))

	def _repeat_rel(self):
		length, index = self._search()
		relative_index = self._offset - index
		if relative_index > 255:
			return (self.REPEAT_REL, 0, bytearray())
		else:
			return (self.REPEAT_REL, length, bytearray([relative_index]))

class lz5_decompress(lz_decompress):
	def __init__(self, data):
		lz_decompress.__init__(self, data)
		self._functions = [self._direct_copy, self._fill_byte, self._fill_word, self._inc_fill, self._repeat_le, self._repeat_inverse, self._repeat_rel, self._long_command]

	def _repeat_inverse(self):
		self._copy((self._in[self._offset] << 8) | self._in[self._offset+1], lambda b: b ^ 0xFF)
		self._offset += 2

	def _repeat_rel(self):
		start = len(self._out) - self._in[self._offset]
		self._offset += 1
		self._copy(start)
