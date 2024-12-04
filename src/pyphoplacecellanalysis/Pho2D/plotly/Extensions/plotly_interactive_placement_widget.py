from copy import deepcopy
import plotly.graph_objs as go
import ipywidgets as widgets
from ipywidgets import interact, interactive, interactive_output, Output
from IPython.display import display, clear_output  # Imports for display and clearing output  # Import for the display function
from pyphocorehelpers.programming_helpers import copy_to_clipboard
from attrs import field, Factory, define


@define(slots=False)
class PlotlyInteractivePlacementWidget:
    """ an interactive widget with sliders and controls that allow you to test Plotly annotations parameters in real time

    Usage:
        from pyphoplacecellanalysis.Pho2D.plotly.Extensions.plotly_interactive_placement_widget import PlotlyInteractivePlacementWidget

        # Instantiate and run the interactive plot
        interactive_plot = PlotlyInteractivePlacementWidget()
        interactive_plot.run()
        
    """
    base_fig: go.FigureWidget = field(default=None)
    annotation_defaults: dict = field(default=Factory(lambda: {
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
    }))
    margin_defaults: dict = field(default=Factory(lambda: {'t': 50, 'b': 100, 'l': 50, 'r': 50}))
    
    # Widgets
    x_widget: widgets.FloatSlider = field(init=False)
    y_widget: widgets.FloatSlider = field(init=False)
    xref_widget: widgets.Dropdown = field(init=False)
    yref_widget: widgets.Dropdown = field(init=False)
    font_size_widget: widgets.IntSlider = field(init=False)
    text_widget: widgets.Text = field(init=False)
    showarrow_widget: widgets.Checkbox = field(init=False)
    textangle_widget: widgets.IntSlider = field(init=False)
    xanchor_widget: widgets.Dropdown = field(init=False)
    yanchor_widget: widgets.Dropdown = field(init=False)
    margin_top_widget: widgets.IntSlider = field(init=False)
    margin_bottom_widget: widgets.IntSlider = field(init=False)
    margin_left_widget: widgets.IntSlider = field(init=False)
    margin_right_widget: widgets.IntSlider = field(init=False)
    print_button: widgets.Button = field(init=False)
    reset_button: widgets.Button = field(init=False)
    
    active_annotation_index_widget: widgets.IntSlider = field(init=False)

    @property
    def num_annotations(self) -> int:
        """The num_annotations property."""
        if not self.base_fig.layout.annotations:
            return 0
        else:
            return len(self.base_fig.layout.annotations)

    @property
    def active_annotation_index(self) -> int:
        """The active_annotation_index property."""
        return self.active_annotation_index_widget.value
    @active_annotation_index.setter
    def active_annotation_index(self, value):
        self.active_annotation_index_widget.value = value        

    @property
    def active_annotation(self):
        """The num_annotations property."""
        active_annotation_index = self.active_annotation_index_widget.value
        if active_annotation_index > 0:
            if self.base_fig is not None:
                if self.base_fig.layout.annotations and (len(self.base_fig.layout.annotations) > 0):
                    active_annotation = self.base_fig.layout.annotations[active_annotation_index]
                    # ## update defaults when annotation selection changes:
                    # for key in self.annotation_defaults.keys():
                    #     if key in active_annotation:
                    #         self.annotation_defaults[key] = active_annotation[key]
                    # if 'font' in active_annotation and 'size' in active_annotation.font:
                    #     self.annotation_defaults['font']['size'] = active_annotation.font.size
                    return active_annotation                        
            else:
                return None
        else:
            return None           


    @classmethod
    def init_with_figure_and_annotation_index(cls, base_fig, annotation_index: int):
        _obj = cls(base_fig=base_fig)
        _obj.active_annotation_index = annotation_index
        return _obj
    
    
    def __attrs_post_init__(self):
        # Initialize the figure widget
        if self.base_fig is not None:
            self.base_fig = go.FigureWidget(deepcopy(self.base_fig).to_plotly_json())
            if self.base_fig.layout.margin:
                for side in ['t', 'b', 'l', 'r']:
                    margin_value = getattr(self.base_fig.layout.margin, side, None)
                    if margin_value is not None:
                        self.margin_defaults[side] = margin_value
        else:
            self.base_fig = go.FigureWidget()
            self.base_fig.add_scatter(x=[1, 2, 3], y=[4, 5, 6])
            

        # Initialize widgets
        self.x_widget = widgets.FloatSlider(min=0, max=1, step=0.01, value=self.annotation_defaults['x'], description="X Position")
        self.y_widget = widgets.FloatSlider(min=-0.5, max=1.5, step=0.01, value=self.annotation_defaults['y'], description="Y Position")
        self.xref_widget = widgets.Dropdown(options=['paper', 'x'], value=self.annotation_defaults['xref'], description="X Reference")
        self.yref_widget = widgets.Dropdown(options=['paper', 'y'], value=self.annotation_defaults['yref'], description="Y Reference")
        self.font_size_widget = widgets.IntSlider(min=5, max=30, step=1, value=self.annotation_defaults['font']['size'], description="Font Size")
        self.text_widget = widgets.Text(value=self.annotation_defaults['text'], description="Text")
        self.showarrow_widget = widgets.Checkbox(value=self.annotation_defaults['showarrow'], description="Show Arrow")
        self.textangle_widget = widgets.IntSlider(min=-180, max=180, step=1, value=self.annotation_defaults['textangle'], description="Text Angle")
        self.xanchor_widget = widgets.Dropdown(options=['auto', 'left', 'center', 'right'], value=self.annotation_defaults['xanchor'], description="X Anchor")
        self.yanchor_widget = widgets.Dropdown(options=['auto', 'top', 'middle', 'bottom'], value=self.annotation_defaults['yanchor'], description="Y Anchor")
        self.margin_top_widget = widgets.IntSlider(min=0, max=200, step=10, value=self.margin_defaults['t'], description="Top Margin")
        self.margin_bottom_widget = widgets.IntSlider(min=0, max=200, step=10, value=self.margin_defaults['b'], description="Bottom Margin")
        self.margin_left_widget = widgets.IntSlider(min=0, max=200, step=10, value=self.margin_defaults['l'], description="Left Margin")
        self.margin_right_widget = widgets.IntSlider(min=0, max=200, step=10, value=self.margin_defaults['r'], description="Right Margin")
        self.print_button = widgets.Button(description="Print Annotation Kwargs")
        self.print_button.on_click(self.print_annotations_kwargs)
        self.reset_button = widgets.Button(description="Reset Values")
        self.reset_button.on_click(self.reset_values)
        self.active_annotation_index_widget = widgets.IntSlider(min=0, max=(self.num_annotations-1), step=1, value=0, description="Annotation Label Index")
        
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
            self.margin_left_widget, self.margin_right_widget, self.active_annotation_index_widget,
        ]
        for widget in widget_list:
            widget.observe(self.on_widget_change, names='value')
        ## register special annotation widget one:
        self.active_annotation_index_widget.observe(self.on_annotation_index_changed, names='value')
        
    
    def on_widget_change(self, change):
        self.update_plot()
    

    def on_annotation_index_changed(self, change):
        """ todo: not yet called """
        print(f'on_annotation_index_changed(change: {change})')
        active_annotation_index = self.active_annotation_index
        
        if self.base_fig is not None:
            if self.base_fig.layout.annotations and (len(self.base_fig.layout.annotations) > 0):
                active_annotation = self.base_fig.layout.annotations[active_annotation_index]
                for key in self.annotation_defaults.keys():
                    if key in active_annotation:
                        self.annotation_defaults[key] = active_annotation[key]
                if 'font' in active_annotation and 'size' in active_annotation.font:
                    self.annotation_defaults['font']['size'] = active_annotation.font.size


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
        active_annotation_index = self.active_annotation_index_widget.value
        active_annotation = self.active_annotation

        # Convert annotations to a list for mutable operations
        annotations = list(self.base_fig.layout.annotations)


        # Update a specific annotation without overwriting the entire list
        # Replace with the index of the annotation you want to update
        new_annotation = dict(
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

        # Update only the target annotation if it exists; otherwise, add it to the list
        if len(annotations) > active_annotation_index:
            annotations[active_annotation_index] = new_annotation
        else:
            annotations.append(new_annotation)
                    
        # Reassign the modified list back as a tuple to layout.annotations
        self.base_fig.layout.annotations = tuple(annotations)


        # Update the figure's margins
        self.base_fig.layout.margin = dict(t=margin_top, b=margin_bottom, l=margin_left, r=margin_right)
    

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
            self.reset_button,
            self.active_annotation_index_widget,
        ])
        # Display the UI and the figure
        display(ui, self.base_fig)
    
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
        self.active_annotation_index_widget.disabled = (self.num_annotations == 0)
        self.active_annotation_index_widget.value = 0
        self.active_annotation_index_widget.max = (self.num_annotations-1)


if __name__ == "__main__":
    # Instantiate and run the interactive plot
    interactive_plot = PlotlyInteractivePlacementWidget()
    interactive_plot.run()

