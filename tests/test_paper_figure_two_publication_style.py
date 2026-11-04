"""Figure 2 bar publication styling via PhoPublicationFigureHelper.get_figure2_bar_style."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

tests_folder = Path(os.path.dirname(__file__))
root_project_folder = tests_folder.parent
src_folder = root_project_folder.joinpath('src')
if str(src_folder) not in sys.path:
    sys.path.insert(0, str(src_folder))

from neuropy.utils.result_context import IdentifyingContext
from pyphoplacecellanalysis.SpecificResults.PhoDiba2023Paper import PaperFigureTwo, PhoPublicationFigureHelper


class TestPaperFigureTwoPublicationStyle(unittest.TestCase):


    def test_get_figure2_bar_style_axes_linewidth_matches_artboard3(self):
        style = PhoPublicationFigureHelper.get_figure2_bar_style(prepare_for_publication=True)
        expected_lw = float(PhoPublicationFigureHelper.artboard3_fig3_style['axes']['linewidth'])
        self.assertEqual(style.axes_linewidth, expected_lw)
        self.assertEqual(style.bar_linewidth, expected_lw)
        self.assertEqual(style.errorbar_linewidth, expected_lw)
        self.assertEqual(style.mpl_rc['axes.linewidth'], expected_lw)
        self.assertEqual(style.mpl_rc['lines.linewidth'], expected_lw)
        self.assertEqual(style.mpl_rc['patch.linewidth'], expected_lw)


    def test_create_plot_prepare_for_publication_applies_spine_linewidth(self):
        style = PhoPublicationFigureHelper.get_figure2_bar_style(prepare_for_publication=True)
        y_values = [np.array([1.0, 2.0, 3.0]), np.array([0.5, 1.5]), np.array([2.0, 2.5, 3.5]), np.array([1.0, 4.0])]
        scatter_props = [{}, {}, {}, {}]
        x_labels = ['a', 'b', 'c', 'd']
        ctx = IdentifyingContext(format_name='kdiba', fig='2', display_fn_name='test_pub_style')

        with style.rc_context():
            render = PaperFigureTwo.create_plot(
                x_labels, y_values, scatter_props, 'Lap-averaged Firing Rates (Hz)', 'Lap', 'test_fig2_pub_style',
                ctx, defer_show=True, prepare_for_publication=True,
                **style.create_plot_kwargs(),
            )

        ax = render.ax
        expected_lw = float(style.axes_linewidth)
        self.assertEqual(ax.spines['left'].get_linewidth(), expected_lw)
        self.assertEqual(ax.spines['bottom'].get_linewidth(), expected_lw)
        plt.close(render.fig)


    def test_get_figure2_bar_style_non_publication_is_passthrough(self):
        style = PhoPublicationFigureHelper.get_figure2_bar_style(prepare_for_publication=False)
        self.assertFalse(style.prepare_for_publication)
        self.assertIsNone(style.axes_linewidth)
        self.assertEqual(style.create_plot_kwargs(), {})


if __name__ == '__main__':
    unittest.main()
