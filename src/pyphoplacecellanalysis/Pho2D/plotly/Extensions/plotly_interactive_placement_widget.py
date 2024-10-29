from copy import deepcopy
import plotly.graph_objs as go
import ipywidgets as widgets
from ipywidgets import interact, interactive, interactive_output, Output
from IPython.display import display, clear_output  # Imports for display and clearing output  # Import for the display function
from pyphocorehelpers.programming_helpers import copy_to_clipboard


class PlotlyInteractivePlacementWidget:
    """ an interactive widget with sliders and controls that allow you to test Plotly annotations parameters in real time

    Usage:
        from pyphoplacecellanalysis.Pho2D.plotly.Extensions.plotly_interactive_placement_widget import PlotlyInteractivePlacementWidget

        # Instantiate and run the interactive plot
        interactive_plot = PlotlyInteractivePlacementWidget()
        interactive_plot.run()
        
    """    
    def __init__(self, base_fig=None):
        # Initialize the figure widget
        
        # Initialize default values
        annotation_defaults = {
            'text': "This is a footer label",
            'x': 0.5,
            'y': -0.1,
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 10, 'color': 'gray'},
            'textangle': 0,
            'xanchor': 'center',
            'yanchor': 'middle'
        }
        margin_defaults = {'t': 50, 'b': 100, 'l': 50, 'r': 50}
        
        if base_fig is not None:
            print(f'using passed figure')
            # self.fig = go.FigureWidget(data=base_fig.data, layout=base_fig.layout.to_plotly_json()) # , frames=base_fig.frames	
            self.fig = go.FigureWidget(deepcopy(base_fig).to_plotly_json()) # , frames=base_fig.frames

            #TODO 2024-10-29 17:47: - [ ] Get default values for widgets from the values in the passed figure.
            # Get the first annotation if it exists
            if self.fig.layout.annotations and len(self.fig.layout.annotations) > 0:
                # annotation = self.fig.layout.annotations[0]
                annotation = self.fig.layout.annotations[-1]
                # Update defaults with values from annotation
                for key in annotation_defaults.keys():
                    if key in annotation:
                        annotation_defaults[key] = annotation[key]
                # For font size
                if 'font' in annotation and 'size' in annotation.font:
                    annotation_defaults['font']['size'] = annotation.font.size
            else:
                print("No annotations found in the passed figure. Using default values.")

            # Get margins
            if self.fig.layout.margin:
                for side in ['t', 'b', 'l', 'r']:
                    margin_value = getattr(self.fig.layout.margin, side, None)
                    if margin_value is not None:
                        margin_defaults[side] = margin_value
                        
        else:
            print(f'no passed figure, generating a default figure.')
            self.fig = go.FigureWidget()
            self.fig.add_scatter(x=[1, 2, 3], y=[4, 5, 6])
            # Use default values


        # Store defaults for reset
        self.annotation_defaults = annotation_defaults
        self.margin_defaults = margin_defaults

        # Set up widgets for interactive adjustment with default values
        self.x_widget = widgets.FloatSlider(min=0, max=1, step=0.01, value=annotation_defaults['x'], description="X Position")
        self.y_widget = widgets.FloatSlider(min=-0.5, max=1.5, step=0.01, value=annotation_defaults['y'], description="Y Position")
        self.xref_widget = widgets.Dropdown(options=['paper', 'x'], value=annotation_defaults['xref'], description="X Reference")
        self.yref_widget = widgets.Dropdown(options=['paper', 'y'], value=annotation_defaults['yref'], description="Y Reference")
        self.font_size_widget = widgets.IntSlider(min=5, max=30, step=1, value=annotation_defaults['font']['size'], description="Font Size")
        self.text_widget = widgets.Text(value=annotation_defaults['text'], description="Text")
        self.showarrow_widget = widgets.Checkbox(value=annotation_defaults['showarrow'], description="Show Arrow")
        self.textangle_widget = widgets.IntSlider(min=-180, max=180, step=1, value=annotation_defaults['textangle'], description="Text Angle")
        self.xanchor_widget = widgets.Dropdown(options=['auto', 'left', 'center', 'right'], value=annotation_defaults['xanchor'], description="X Anchor")
        self.yanchor_widget = widgets.Dropdown(options=['auto', 'top', 'middle', 'bottom'], value=annotation_defaults['yanchor'], description="Y Anchor")
        self.margin_top_widget = widgets.IntSlider(min=0, max=200, step=10, value=margin_defaults['t'], description="Top Margin")
        self.margin_bottom_widget = widgets.IntSlider(min=0, max=200, step=10, value=margin_defaults['b'], description="Bottom Margin")
        self.margin_left_widget = widgets.IntSlider(min=0, max=200, step=10, value=margin_defaults['l'], description="Left Margin")
        self.margin_right_widget = widgets.IntSlider(min=0, max=200, step=10, value=margin_defaults['r'], description="Right Margin")
        self.print_button = widgets.Button(description="Print Annotation Kwargs")
        self.print_button.on_click(self.print_annotations_kwargs)
        self.reset_button = widgets.Button(description="Reset Values")
        self.reset_button.on_click(self.reset_values)
        
        # Set initial layout
        self.update_plot()
        
        # Observe changes in the widgets
        self.observe_widgets()
        

    def observe_widgets(self):
        # List of all widgets to observe
        widget_list = [
            self.x_widget, self.y_widget, self.xref_widget, self.yref_widget,
            self.font_size_widget, self.text_widget, self.showarrow_widget,
            self.textangle_widget, self.xanchor_widget, self.yanchor_widget,
            self.margin_top_widget, self.margin_bottom_widget,
            self.margin_left_widget, self.margin_right_widget
        ]
        for widget in widget_list:
            widget.observe(self.on_widget_change, names='value')
    
    def on_widget_change(self, change):
        self.update_plot()
    
    def update_plot(self):
        # Extract current widget values
        x = self.x_widget.value
        y = self.y_widget.value
        xref = self.xref_widget.value
        yref = self.yref_widget.value
        font_size = self.font_size_widget.value
        text = self.text_widget.value
        showarrow = self.showarrow_widget.value
        textangle = self.textangle_widget.value
        xanchor = self.xanchor_widget.value
        yanchor = self.yanchor_widget.value
        margin_top = self.margin_top_widget.value
        margin_bottom = self.margin_bottom_widget.value
        margin_left = self.margin_left_widget.value
        margin_right = self.margin_right_widget.value

        # self.fig.update_annotations(row=1, col=1, text=text,
        #         x=x,
        #         y=y,
        #         xref=xref,
        #         yref=yref,
        #         showarrow=showarrow,
        #         font=dict(size=font_size, color='gray'),
        #         textangle=textangle,
        #         xanchor=xanchor,
        #         yanchor=yanchor)
        
    
        # Update the figure's annotations
        self.fig.layout.annotations = [
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
        ]
        # Update the figure's margins
        self.fig.layout.margin = dict(t=margin_top, b=margin_bottom, l=margin_left, r=margin_right)
    
    def run(self):
        # Arrange widgets in a vertical box
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
            self.reset_button
        ])
        # Display the UI and the figure
        display(ui, self.fig)
    
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
        copy_to_clipboard(str(kwargs)) ## to clipboard
        

    def reset_values(self, b=None):
        """Reset all widgets to their default values."""
        # self.x_widget.value = 0.5
        # self.y_widget.value = -0.1
        # self.xref_widget.value = 'paper'
        # self.yref_widget.value = 'paper'
        # self.font_size_widget.value = 10
        # self.text_widget.value = "This is a footer label"
        # self.showarrow_widget.value = False
        # self.textangle_widget.value = 0
        # self.xanchor_widget.value = 'center'
        # self.yanchor_widget.value = 'middle'
        # self.margin_top_widget.value = 50
        # self.margin_bottom_widget.value = 100
        # self.margin_left_widget.value = 50
        # self.margin_right_widget.value = 50    
        self.x_widget.value = self.annotation_defaults['x']
        self.y_widget.value = self.annotation_defaults['y']
        self.xref_widget.value = self.annotation_defaults['xref']
        self.yref_widget.value = self.annotation_defaults['yref']
        self.font_size_widget.value = self.annotation_defaults['font']['size']
        self.text_widget.value = self.annotation_defaults['text']
        self.showarrow_widget.value = self.annotation_defaults['showarrow']
        self.textangle_widget.value = self.annotation_defaults['textangle']
        self.xanchor_widget.value = self.annotation_defaults['xanchor']
        self.yanchor_widget.value = self.annotation_defaults['yanchor']
        self.margin_top_widget.value = self.margin_defaults['t']
        self.margin_bottom_widget.value = self.margin_defaults['b']
        self.margin_left_widget.value = self.margin_defaults['l']
        self.margin_right_widget.value = self.margin_defaults['r']
        


if __name__ == "__main__":
    # Instantiate and run the interactive plot
    interactive_plot = PlotlyInteractivePlacementWidget()
    interactive_plot.run()

