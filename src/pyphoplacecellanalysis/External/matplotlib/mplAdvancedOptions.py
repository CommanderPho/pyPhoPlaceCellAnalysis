
# Original Copyright © 2009 Pierre Raybaut
# Licensed under the terms of the MIT License
# see the Matplotlib licenses directory for a copy of the license
# Modified by Seth Roffe 2020


"""Module that provides a GUI-based editor for Matplotlib's figure options."""

import re

from matplotlib import cbook, cm, colors as mcolors, markers, image as mimage
from matplotlib.backends.qt_compat import QtGui
from matplotlib.backends.qt_editor import _formlayout
import matplotlib.transforms as mtransforms
import matplotlib
import numpy as np


LINESTYLES = {'-': 'Solid',
              '--': 'Dashed',
              '-.': 'DashDot',
              ':': 'Dotted',
              'None': 'None',
              }

DRAWSTYLES = {
    'default': 'Default',
    'steps-pre': 'Steps (Pre)', 'steps': 'Steps (Pre)',
    'steps-mid': 'Steps (Mid)',
    'steps-post': 'Steps (Post)'}

MARKERS = markers.MarkerStyle.markers



def figure_edit(axes, parent=None):
    """Edit matplotlib figure options"""
    sep = (None, None)  # separator

    # Get / General
    # Cast to builtin floats as they have nicer reprs.
    xmin, xmax = map(float, axes.get_xlim())
    ymin, ymax = map(float, axes.get_ylim())
    general = [('Title', axes.get_title()),
                ("Title Fontsize", axes.title.get_fontsize()),
               sep,
               (None, "<b>X-Axis</b>"),
               ("Auto xlim", False),
               ('Left', xmin), ('Right', xmax),
               ('Label', axes.get_xlabel()),
               ('Scale', [axes.get_xscale(), 'linear', 'log', 'logit']),
               ("X Fontsize", axes.xaxis.label.get_size()),
               sep,
               (None, "<b>Y-Axis</b>"),
               ("Auto ylim", False),
               ('Bottom', ymin), ('Top', ymax),
               ('Label', axes.get_ylabel()),
               ('Scale', [axes.get_yscale(), 'linear', 'log', 'logit']),
               ("Y FontSize", axes.yaxis.label.get_size()),
               sep,
               (None, "<b>Legend</b>"),
               ('(Re-)Generate automatic legend', False),
               ("Legend Location", [matplotlib.rcParams['legend.loc'],"best",'upper right','upper left','lower left','lower right','right','center left','center right','lower center','upper center','center']),
               ("Legend Draggable", axes.legend().get_draggable()),
               #("Legend Fontsize", [matplotlib.rcParams['legend.fontsize'],'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']),
               ("Legend Fontsize", [matplotlib.rcParams['legend.fontsize'],'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large']),
               ]

    # Save the unit data
    xconverter = axes.xaxis.converter
    yconverter = axes.yaxis.converter
    xunits = axes.xaxis.get_units()
    yunits = axes.yaxis.get_units()

    # Sorting for default labels (_lineXXX, _imageXXX).
    def cmp_key(label):
        match = re.match(r"(_line|_image)(\d+)", label)
        if match:
            return match.group(1), int(match.group(2))
        else:
            return label, 0

    # Get / Curves
    linedict = {}
    for line in axes.get_lines():
        label = line.get_label()
        if label == '_nolegend_':
            continue
        linedict[label] = line
    curves = []

    def prepare_data(d, init):
        """Prepare entry for FormLayout.

        *d* is a mapping of shorthands to style names (a single style may
        have multiple shorthands, in particular the shorthands `None`,
        `"None"`, `"none"` and `""` are synonyms); *init* is one shorthand
        of the initial style.

        This function returns an list suitable for initializing a
        FormLayout combobox, namely `[initial_name, (shorthand,
        style_name), (shorthand, style_name), ...]`.
        """
        if init not in d:
            d = {**d, init: str(init)}
        # Drop duplicate shorthands from dict (by overwriting them during
        # the dict comprehension).
        name2short = {name: short for short, name in d.items()}
        # Convert back to {shorthand: name}.
        short2name = {short: name for name, short in name2short.items()}
        # Find the kept shorthand for the style specified by init.
        canonical_init = name2short[d[init]]
        # Sort by representation and prepend the initial value.
        return ([canonical_init] +
                sorted(short2name.items(),
                       key=lambda short_and_name: short_and_name[1]))

    curvelabels = sorted(linedict, key=cmp_key)
    for label in curvelabels:
        line = linedict[label]
        color = mcolors.to_hex(
            mcolors.to_rgba(line.get_color(), line.get_alpha()),
            keep_alpha=True)
        ec = mcolors.to_hex(
            mcolors.to_rgba(line.get_markeredgecolor(), line.get_alpha()),
            keep_alpha=True)
        fc = mcolors.to_hex(
            mcolors.to_rgba(line.get_markerfacecolor(), line.get_alpha()),
            keep_alpha=True)
        curvedata = [
            ('Label', label),
            sep,
            (None, '<b>Line</b>'),
            ('Line style', prepare_data(LINESTYLES, line.get_linestyle())),
            ('Draw style', prepare_data(DRAWSTYLES, line.get_drawstyle())),
            ('Width', line.get_linewidth()),
            ('Color (RGBA)', color),
            sep,
            (None, '<b>Marker</b>'),
            ('Style', prepare_data(MARKERS, line.get_marker())),
            ('Size', line.get_markersize()),
            ('Face color (RGBA)', fc),
            ('Edge color (RGBA)', ec),
            ("Visible", line.get_visible()),
                ]
        curves.append([curvedata, label, ""])
    # Is there a curve displayed?
    has_curve = bool(curves)

    # Get ScalarMappables.
    mappabledict = {}
    for mappable in [*axes.images, *axes.collections]:
        label = mappable.get_label()
        if label == '_nolegend_' or mappable.get_array() is None:
            continue
        mappabledict[label] = mappable
    mappablelabels = sorted(mappabledict, key=cmp_key)
    mappables = []
    cmaps = [(cmap, name) for name, cmap in sorted(cm.cmap_d.items())]
    for label in mappablelabels:
        mappable = mappabledict[label]
        cmap = mappable.get_cmap()
        if cmap not in cm.cmap_d.values():
            cmaps = [(cmap, cmap.name), *cmaps]
        low, high = mappable.get_clim()
        mappabledata = [
            ('Label', label),
            ('Colormap', [cmap.name] + cmaps),
            ('Min. value', low),
            ('Max. value', high),
        ]
        if hasattr(mappable, "get_interpolation"):  # Images.
            interpolations = [
                (name, name) for name in sorted(mimage.interpolations_names)]
            mappabledata.append((
                'Interpolation',
                [mappable.get_interpolation(), *interpolations]))
        mappables.append([mappabledata, label, ""])
    # Is there a scalarmappable displayed?
    has_sm = bool(mappables)

    datalist = [(general, "Axes", "")]
    if curves:
        datalist.append((curves, "Curves", ""))
    if mappables:
        datalist.append((mappables, "Images, etc.", ""))

    def autoscale_based_on(ax, lines):
        """ Apply autoscaling based on visible lines"""
        #ax.dataLim.update_from_data_xy(mtransforms.Bbox.unit(),ignore=True)
        ax.dataLim = mtransforms.Bbox.unit()
        for line in lines:
            xy = np.vstack(line.get_data()).T
            ax.dataLim.update_from_data_xy(xy, ignore=False)
        ax.autoscale_view()

    def apply_callback(data):
        """This function will be called to apply changes"""
        orig_xlim = axes.get_xlim()
        orig_ylim = axes.get_ylim()

        general = data.pop(0)
        curves = data.pop(0) if has_curve else []
        mappables = data.pop(0) if has_sm else []
        if data:
            raise ValueError("Unexpected field")

        # Set / General
        (title, titlefontsize, autoxlim, xmin, xmax, xlabel, xscale, xfontsize, autoylim, ymin, ymax, ylabel, yscale, yfontsize,
         generate_legend,legendloc,legenddrag,legendfontsize) = general

        if axes.get_xscale() != xscale:
            axes.set_xscale(xscale)
        if axes.get_yscale() != yscale:
            axes.set_yscale(yscale)

        axes.set_title(title)
        axes.title.set_fontsize(titlefontsize)
        axes.set_xlim(xmin, xmax)
        if autoxlim:
            axes.set_xlim(auto=True)
        axes.set_xlabel(xlabel)
        axes.xaxis.label.set_size(xfontsize)
        axes.set_ylim(ymin, ymax)
        if autoylim:
            axes.set_ylim(auto=True)
        axes.set_ylabel(ylabel)
        axes.yaxis.label.set_size(yfontsize)
        #matplotlib.rcParams['legend.loc'] = legendloc
        axes.legend().set_draggable(legenddrag)

        # Restore the unit data
        axes.xaxis.converter = xconverter
        axes.yaxis.converter = yconverter
        axes.xaxis.set_units(xunits)
        axes.yaxis.set_units(yunits)
        axes.xaxis._update_axisinfo()
        axes.yaxis._update_axisinfo()

        # Set / Curves
        visible_lines = []
        for index, curve in enumerate(curves):
            line = linedict[curvelabels[index]]
            (label, linestyle, drawstyle, linewidth, color, marker, markersize,
             markerfacecolor, markeredgecolor,visible) = curve
            line.set_label(label)
            line.set_linestyle(linestyle)
            line.set_drawstyle(drawstyle)
            line.set_linewidth(linewidth)
            rgba = mcolors.to_rgba(color)
            line.set_alpha(None)
            line.set_color(rgba)
            line.set_visible(visible)
            if not line.get_visible():
                xy = np.vstack(line.get_data()).T
                axes.dataLim.update_from_data_xy(xy,ignore=True)
                axes.autoscale_view()
            else:
                visible_lines.append(line)

            if marker != 'none':
                line.set_marker(marker)
                line.set_markersize(markersize)
                line.set_markerfacecolor(markerfacecolor)
                line.set_markeredgecolor(markeredgecolor)
        autoscale_based_on(axes,visible_lines)
        # Set ScalarMappables.
        for index, mappable_settings in enumerate(mappables):
            mappable = mappabledict[mappablelabels[index]]
            if len(mappable_settings) == 5:
                label, cmap, low, high, interpolation = mappable_settings
                mappable.set_interpolation(interpolation)
            elif len(mappable_settings) == 4:
                label, cmap, low, high = mappable_settings
            mappable.set_label(label)
            mappable.set_cmap(cm.get_cmap(cmap))
            mappable.set_clim(*sorted([low, high]))

        # re-generate legend, if checkbox is checked
        if generate_legend:
            draggable = None
            ncol = 1
            if axes.legend_ is not None:
                old_legend = axes.get_legend()
                draggable = old_legend._draggable is not None
                ncol = old_legend._ncol
            new_legend = axes.legend(ncol=ncol,fontsize=legendfontsize,loc=legendloc)
            if new_legend:
                new_legend.set_draggable(draggable)
        else:
            axes.get_legend().remove()

        # Redraw
        figure = axes.get_figure()
        figure.canvas.draw()
        if not (axes.get_xlim() == orig_xlim and axes.get_ylim() == orig_ylim):
            figure.canvas.toolbar.push_current()

    data = _formlayout.fedit(
        datalist, title="Figure options", parent=parent,
        icon=QtGui.QIcon(
            str(cbook._get_data_path('images', 'qt4_editor_options.svg'))),
        apply=apply_callback)
    if data is not None:
        apply_callback(data)


from matplotlib.backends.qt_editor import figureoptions
figureoptions.figure_edit = figure_edit
