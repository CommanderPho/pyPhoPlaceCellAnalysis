import plotly.graph_objs as go
import ipywidgets as widgets
from ipywidgets import interact, interactive, interactive_output, Output
from IPython.display import display, clear_output  # Imports for display and clearing output  # Import for the display function

class PlotlyInteractivePlacementWidget:
	""" an interactive widget with sliders and controls that allow you to test Plotly annotations parameters in real time

	Usage:
		from pyphoplacecellanalysis.Pho2D.plotly.Extensions.plotly_interactive_placement_widget import PlotlyInteractivePlacementWidget

		# Instantiate and run the interactive plot
		interactive_plot = PlotlyInteractivePlacementWidget()
		interactive_plot.run()
		
	"""
	def __init__(self):
		# Set up widgets for interactive adjustment
		self.x_widget = widgets.FloatSlider(min=0, max=1, step=0.01, value=0.5, description="X Position")
		self.y_widget = widgets.FloatSlider(min=-0.5, max=1.5, step=0.01, value=-0.1, description="Y Position")
		self.xref_widget = widgets.Dropdown(options=['paper', 'x'], value='paper', description='X Reference')
		self.yref_widget = widgets.Dropdown(options=['paper', 'y'], value='paper', description='Y Reference')
		self.font_size_widget = widgets.IntSlider(min=5, max=30, step=1, value=10, description="Font Size")
		self.text_widget = widgets.Text(value="This is a footer label", description="Text")
		self.showarrow_widget = widgets.Checkbox(value=False, description="Show Arrow")
		self.textangle_widget = widgets.IntSlider(min=-180, max=180, step=1, value=0, description="Text Angle")
		self.xanchor_widget = widgets.Dropdown(options=['auto', 'left', 'center', 'right'], value='center', description='X Anchor')
		self.yanchor_widget = widgets.Dropdown(options=['auto', 'top', 'middle', 'bottom'], value='middle', description='Y Anchor')
		self.margin_top_widget = widgets.IntSlider(min=0, max=200, step=10, value=50, description="Top Margin")
		self.margin_bottom_widget = widgets.IntSlider(min=0, max=200, step=10, value=100, description="Bottom Margin")
		self.margin_left_widget = widgets.IntSlider(min=0, max=200, step=10, value=50, description="Left Margin")
		self.margin_right_widget = widgets.IntSlider(min=0, max=200, step=10, value=50, description="Right Margin")
		self.print_button = widgets.Button(description="Print Annotation Kwargs")
		self.print_button.on_click(self.print_annotations_kwargs)
		# New button widget
		self.reset_button = widgets.Button(description="Reset Values")
		self.reset_button.on_click(self.reset_values)
	
		# Output widget to capture the plot
		self.output = Output(layout={'border': '3px solid red'})
	

	def update_plot(self, x=0.5, y=-0.1, xref='paper', yref='paper', font_size=10, text="This is a footer label",
					showarrow=False, textangle=0, xanchor='center', yanchor='middle',
					margin_top=50, margin_bottom=100, margin_left=50, margin_right=50):
		fig = go.Figure()
		fig.add_scatter(x=[1, 2, 3], y=[4, 5, 6])
		fig.update_layout(
			annotations=[
				dict(
					text=text,
					x=x,
					y=y,
					xref=xref,
					yref=yref,
					showarrow=showarrow,
					font=dict(size=font_size, color='gray'),
					textangle=textangle,
					xanchor=xanchor,
					yanchor=yanchor
				)
			],
			margin=dict(t=margin_top, b=margin_bottom, l=margin_left, r=margin_right)  # Set margins dynamically
		)
		with self.output:
			clear_output(wait=True)
			fig.show()


	def run(self):
		# Define a dictionary with all the widgets
		widget_dict = {
			'x': self.x_widget,
			'y': self.y_widget,
			'xref': self.xref_widget,
			'yref': self.yref_widget,
			'font_size': self.font_size_widget,
			'text': self.text_widget,
			'showarrow': self.showarrow_widget,
			'textangle': self.textangle_widget,
			'xanchor': self.xanchor_widget,
			'yanchor': self.yanchor_widget,
			'margin_top': self.margin_top_widget,
			'margin_bottom': self.margin_bottom_widget,
			'margin_left': self.margin_left_widget,
			'margin_right': self.margin_right_widget
		}

		# Create interactive output
		out = interactive_output(self.update_plot, widget_dict)
		
		# Manually creating VBox layout with all widgets and control buttons
		ui = widgets.VBox([
			self.x_widget,
			self.y_widget,
			self.xref_widget,
			self.yref_widget,
			self.font_size_widget,
			self.text_widget,
			self.showarrow_widget,
			self.textangle_widget,
			self.xanchor_widget,
			self.yanchor_widget,
			self.margin_top_widget,
			self.margin_bottom_widget,
			self.margin_left_widget,
			self.margin_right_widget,
			self.print_button,
			self.reset_button,
			self.output  # Output widget for displaying the plot
		])
		
		display(ui)
		
		# Display the output of the interactive
		display(out)


	def print_annotations_kwargs(self, b=None):
		kwargs = {
			'text': self.text_widget.value,
			'x': self.x_widget.value,
			'y': self.y_widget.value,
			'xref': self.xref_widget.value,
			'yref': self.yref_widget.value,
			'showarrow': self.showarrow_widget.value,
			'font': {'size': self.font_size_widget.value, 'color': 'gray'},
			'textangle': self.textangle_widget.value,
			'xanchor': self.xanchor_widget.value,
			'yanchor': self.yanchor_widget.value
		}
		print(kwargs)
	
	def reset_values(self, b=None):
		"""Reset all widgets to their default values."""
		self.x_widget.value = 0.5
		self.y_widget.value = -0.1
		self.xref_widget.value = 'paper'
		self.yref_widget.value = 'paper'
		self.font_size_widget.value = 10
		self.text_widget.value = "This is a footer label"
		self.showarrow_widget.value = False
		self.textangle_widget.value = 0
		self.xanchor_widget.value = 'center'
		self.yanchor_widget.value = 'middle'
		self.margin_top_widget.value = 50
		self.margin_bottom_widget.value = 100
		self.margin_left_widget.value = 50
		self.margin_right_widget.value = 50
				


# # Instantiate and run the interactive plot
# interactive_plot = PlotlyInteractivePlacementWidget()
# interactive_plot.run()