from typing import List, Dict, Tuple, Optional, Any


class ReprPrintableItemMixin:
	""" Implementors provide a better repr than the default Pyqt5-based widget one that shows the class name and the object instance memory address.
	
	from pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.Mixins.ReprPrintableWidgetMixin import ReprPrintableItemMixin
	
	"""
	def __repr__(self):
		class_parts: List[str] = str(self.__class__).strip('<class').strip('>').strip(' ').strip("'").split('.')
		class_name: str = class_parts[-1] # 'IntervalRectsItem'
		class_path: str = '.'.join(class_parts[:-1]) # 'pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.GraphicsObjects.IntervalRectsItem'
		out_str: str = f"{class_name}"
		out_str_list: List[str] = []
		try:
			view_range_str = f"viewRange: [[xmin, xmax], [ymin, ymax]]: {self.viewRange()}"
			out_str_list.append(view_range_str)
		except (AttributeError, TypeError, ValueError, KeyError) as e:
			## missing propertry or something
			pass
		except Exception as e:
			raise e

		if len(out_str_list) > 0:
			out_str = f"{out_str}[" + ', '.join(out_str_list) + ']'
			
		return out_str
		# return super().__repr__()

