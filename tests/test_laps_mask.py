import pytest
import numpy as np
import sys
import unittest.mock as mock

# In this environment, full imports fail due to missing PyVista/Qt and core dependencies.
# We will mock pyvista, pyphocorehelpers, and neuropy out before importing the target module,
# or simply test the target module by ensuring it can be imported after mocking the problematic imports.

import sys
sys.modules['pyvista'] = mock.MagicMock()
sys.modules['pyvistaqt'] = mock.MagicMock()
sys.modules['pyphocorehelpers'] = mock.MagicMock()
sys.modules['pyphocorehelpers.indexing_helpers'] = mock.MagicMock()
sys.modules['pyphocorehelpers.function_helpers'] = mock.MagicMock()
sys.modules['pyphocorehelpers.gui'] = mock.MagicMock()
sys.modules['pyphocorehelpers.gui.PyVista'] = mock.MagicMock()
sys.modules['pyphocorehelpers.gui.PyVista.PhoCustomVtkWidgets'] = mock.MagicMock()
sys.modules['pyphocorehelpers.DataStructure'] = mock.MagicMock()
sys.modules['pyphocorehelpers.DataStructure.RenderPlots'] = mock.MagicMock()
sys.modules['pyphocorehelpers.DataStructure.RenderPlots.MatplotLibRenderPlots'] = mock.MagicMock()
sys.modules['neuropy'] = mock.MagicMock()
sys.modules['neuropy.utils'] = mock.MagicMock()
sys.modules['neuropy.utils.mixins'] = mock.MagicMock()
sys.modules['neuropy.utils.mixins.dict_representable'] = mock.MagicMock()
sys.modules['neuropy.utils.matplotlib_helpers'] = mock.MagicMock()
sys.modules['matplotlib'] = mock.MagicMock()
sys.modules['matplotlib.pyplot'] = mock.MagicMock()
sys.modules['matplotlib.collections'] = mock.MagicMock()
sys.modules['matplotlib.colors'] = mock.MagicMock()
sys.modules['pyphoplacecellanalysis.GUI.PyQtPlot.Widgets.ContainerBased.PhoContainerTool'] = mock.MagicMock()
sys.modules['pyphoplacecellanalysis.GUI.PyVista.InteractivePlotter.Mixins.LapsVisualizationMixin'] = mock.MagicMock()
sys.modules['pyphoplacecellanalysis.General.Model.Configs.LongShortDisplayConfig'] = mock.MagicMock()
sys.modules['pyphoplacecellanalysis.PhoPositionalData.plotting.mixins'] = mock.MagicMock()
sys.modules['pyphoplacecellanalysis.PhoPositionalData.plotting.mixins.decoder_plotting_mixins'] = mock.MagicMock()

from pyphoplacecellanalysis.PhoPositionalData.plotting.laps import _build_included_mask

def test_build_included_mask_basic():
    """Test basic non-overlapping ranges."""
    mask_shape = (10,)
    crossing_beginings = [2, 7]
    crossing_endings = [5, 9]

    mask = _build_included_mask(mask_shape, crossing_beginings, crossing_endings)

    expected_mask = np.array([False, False, True, True, True, False, False, True, True, False])

    np.testing.assert_array_equal(mask, expected_mask)

def test_build_included_mask_empty():
    """Test with empty input lists."""
    mask_shape = (5,)
    crossing_beginings = []
    crossing_endings = []

    mask = _build_included_mask(mask_shape, crossing_beginings, crossing_endings)

    expected_mask = np.array([False, False, False, False, False])
    np.testing.assert_array_equal(mask, expected_mask)

def test_build_included_mask_overlapping():
    """Test with overlapping ranges."""
    mask_shape = (10,)
    crossing_beginings = [1, 3]
    crossing_endings = [5, 6]

    mask = _build_included_mask(mask_shape, crossing_beginings, crossing_endings)

    expected_mask = np.array([False, True, True, True, True, True, False, False, False, False])
    np.testing.assert_array_equal(mask, expected_mask)

def test_build_included_mask_single_point():
    """Test with ranges that contain a single point (begin + 1 == end)."""
    mask_shape = (5,)
    crossing_beginings = [1, 3]
    crossing_endings = [2, 4]

    mask = _build_included_mask(mask_shape, crossing_beginings, crossing_endings)

    expected_mask = np.array([False, True, False, True, False])
    np.testing.assert_array_equal(mask, expected_mask)
