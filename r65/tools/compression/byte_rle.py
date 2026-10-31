# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
# -*- coding: utf-8 -*-

from itertools import groupby

def compress(data):
	# The tag is the first byte value the data never holds
	tag = next((d for d in range(256) if d not in data), None)
	if tag is None:
		raise ValueError("byte_rle needs a byte value the data never holds; this data holds all 256")
	out = bytearray([tag])

	# Process grouped data runs
	for val, item in groupby(data):
		count = len(list(item))
		if count > 2:
			# The byte, then tag/count pairs repeating it; a count is 1-255
			out.append(val)
			count -= 1
			while count > 0:
				n = min(count, 255)
				out += bytearray([tag, n])
				count -= n
		else:
			out += bytearray([val]*count)

	# Mark end of rle
	out += bytearray([tag,0])
	return out

def decompress(data):
	out = bytearray()
	tag = data[0]
	last = tag
	i = 1
	while i < len(data):
		c = data[i]
		if c == tag:
			i += 1
			count = data[i]
			# End of rle found
			if count == 0:
				break
			out += bytearray([last] * count)
		else:
			out.append(c)
			last = c
		i += 1
	return out
