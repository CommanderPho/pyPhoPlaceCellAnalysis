from dataclasses import dataclass


@dataclass
class PositionDecoder(object):
	"""Basically just the result from a specific decoding. Meant to be a cleaner way  for PositionDecoder.

	See also: DecoderResultDisplayingBaseClass
	"""
	property: type
	