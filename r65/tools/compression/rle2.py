# Copyright (c) 2026 Neutron Emulation, LLC. MIT licensed.
# -*- coding: utf-8 -*-

from r65.tools.compression import rle1

def compress(data):
	data1 = bytearray([data[i] for i in range(0,len(data),2)])
	data2 = bytearray([data[i] for i in range(1,len(data),2)])
	out1 = rle1.compress(data1,False)
	out2 = rle1.compress(data2,False)
	return out1 + out2

def decompress(data):
	out = bytearray()
	decomp = rle1.decompress(data)
	# The even plane holds the extra byte of an odd length
	half = (len(decomp) + 1) // 2
	data1 = decomp[:half]
	data2 = decomp[half:]
	for i in range(0,half):
		out.append(data1[i])
		if i < len(data2):
			out.append(data2[i])
	return out
