import numpy as np
from pyphoplacecellanalysis.External.pyqtgraph.exporters.ImageExporter import ImageExporter
from pyphoplacecellanalysis.External.pyqtgraph import functions as fn
from pathlib import Path
import sys
from pyphocorehelpers.programming_helpers import metadata_attributes
from pyphocorehelpers.function_helpers import function_attributes

@metadata_attributes(short_name=None, tags=['pyqtgraph', 'export'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2026-04-01 06:48', related_items=[])
class ExportHelpers:
	"""
	from pyphoplacecellanalysis.External.pyqtgraph_extensions.export_helpers import ExportHelpers
	
	"""
	# Helper to convert QImage to BGR array for OpenCV (contiguous uint8 for compatibility)
	@classmethod
	def qimage_to_bgr(cls, qimage):
		img_array = fn.ndarray_from_qimage(qimage)
		# Handle ARGB32 format conversion based on byte order
		if img_array.shape[2] == 4:
			# ARGB32 format - extract RGB channels based on byte order
			if sys.byteorder == 'little':
				# Little-endian: channels are [B, G, R, A] in memory
				bgr = img_array[:, :, :3]  # B, G, R (first 3 channels)
			else:
				# Big-endian: channels are [A, R, G, B] in memory
				bgr = img_array[:, :, [3, 2, 1]]  # B, G, R from indices 3,2,1
		elif img_array.shape[2] == 3:
			# Already RGB format, convert to BGR for OpenCV
			bgr = img_array[:, :, ::-1]
		else:
			raise ValueError(f"Unexpected image format with {img_array.shape[2]} channels")
		# Ensure contiguous uint8 array for OpenCV compatibility
		return np.ascontiguousarray(bgr, dtype=np.uint8)
	
	@classmethod
	def qimage_to_rgb(cls, qimage):
		return np.ascontiguousarray(cls.qimage_to_bgr(qimage)[:, :, ::-1], dtype=np.uint8)
        