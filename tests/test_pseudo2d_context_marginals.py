import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

tests_folder = Path(__file__).parent
root_project_folder = tests_folder.parent
src_folder = root_project_folder.joinpath('src')
if str(src_folder) not in sys.path:
    sys.path.insert(0, str(src_folder))

_DIRECTIONAL_PSEUDO2D_MODULE = None


def _get_directional_pseudo2d_decoders_result():
    global _DIRECTIONAL_PSEUDO2D_MODULE
    if _DIRECTIONAL_PSEUDO2D_MODULE is None:
        for mod_name in ('PyQt5Singleton', 'plotly', 'plotly.express', 'plotly.graph_objects', 'awkward'):
            sys.modules.setdefault(mod_name, MagicMock())
        from neuropy.analyses.placefields import PfND  # noqa: F401 — neuropy import order
        module_path = src_folder.joinpath('pyphoplacecellanalysis/General/Pipeline/Stages/ComputationFunctions/MultiContextComputationFunctions/DirectionalPlacefieldGlobalComputationFunctions.py')
        spec = importlib.util.spec_from_file_location('dpf_gcf_for_tests', module_path)
        _DIRECTIONAL_PSEUDO2D_MODULE = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(_DIRECTIONAL_PSEUDO2D_MODULE)
    return _DIRECTIONAL_PSEUDO2D_MODULE.DirectionalPseudo2DDecodersResult


def _build_synthetic_kdiba_posterior(n_pos: int = 10, n_time_bins: int = 5) -> np.ndarray:
    """Posterior with distinct spatial peaks per context so grouped marginals are sensitive to summation order."""
    rng = np.random.default_rng(42)
    p_x_given_n = np.zeros((n_pos, 4, n_time_bins), dtype=float)
    context_peak_positions = [1, 3, 6, 8]
    for context_idx, peak_pos in enumerate(context_peak_positions):
        for time_bin_idx in range(n_time_bins):
            spatial_profile = np.exp(-0.5 * np.square(np.arange(n_pos) - (peak_pos + 0.3 * time_bin_idx)))
            p_x_given_n[:, context_idx, time_bin_idx] = spatial_profile * (1.0 + 0.2 * context_idx + 0.1 * rng.random())
    for context_idx in range(4):
        for time_bin_idx in range(n_time_bins):
            col_sum = np.sum(p_x_given_n[:, context_idx, time_bin_idx])
            p_x_given_n[:, context_idx, time_bin_idx] /= col_sum
    return p_x_given_n


def _old_kdiba_marginal_over_long_short(a_p_x_given_n: np.ndarray) -> np.ndarray:
    """Literal port of pre-generalization KDiba long/short marginalization."""
    curr_array_shape = np.shape(a_p_x_given_n)
    out_p_x_given_n = np.zeros((curr_array_shape[0], 2, curr_array_shape[-1]))
    out_p_x_given_n[:, 0, :] = a_p_x_given_n[:, 0, :] + a_p_x_given_n[:, 1, :]
    out_p_x_given_n[:, 1, :] = a_p_x_given_n[:, 2, :] + a_p_x_given_n[:, 3, :]
    marginal = np.squeeze(np.sum(out_p_x_given_n, axis=0))
    marginal = marginal / np.sum(marginal, axis=0, keepdims=True)
    if marginal.ndim == 1:
        marginal = marginal[:, np.newaxis]
    return marginal


def _old_kdiba_marginal_over_direction(a_p_x_given_n: np.ndarray) -> np.ndarray:
    """Literal port of pre-generalization KDiba LR/RL marginalization."""
    curr_array_shape = np.shape(a_p_x_given_n)
    out_p_x_given_n = np.zeros((curr_array_shape[0], 2, curr_array_shape[-1]))
    out_p_x_given_n[:, 0, :] = a_p_x_given_n[:, 0, :] + a_p_x_given_n[:, 2, :]
    out_p_x_given_n[:, 1, :] = a_p_x_given_n[:, 1, :] + a_p_x_given_n[:, 3, :]
    marginal = np.squeeze(np.sum(out_p_x_given_n, axis=0))
    marginal = marginal / np.sum(marginal, axis=0, keepdims=True)
    if marginal.ndim == 1:
        marginal = marginal[:, np.newaxis]
    return marginal


def _broken_generalized_marginal(a_p_x_given_n: np.ndarray, group_indices_list, directional_cls) -> np.ndarray:
    """Previous generalized path: spatial sum per context, normalize 4-way, then group and re-normalize."""
    context_marginal = directional_cls._marginalize_p_x_given_n_to_context_probs(a_p_x_given_n, (0,))
    return directional_cls._group_context_marginal(context_marginal, group_indices_list)


@pytest.fixture
def directional_cls():
    return _get_directional_pseudo2d_decoders_result()


@pytest.fixture
def synthetic_posterior() -> np.ndarray:
    return _build_synthetic_kdiba_posterior()


def test_long_short_marginal_matches_old_kdiba_implementation(directional_cls, synthetic_posterior: np.ndarray):
    expected = _old_kdiba_marginal_over_long_short(synthetic_posterior)
    actual = directional_cls._marginalize_p_x_given_n_over_context_groups_in_position_space(synthetic_posterior, context_dim_idx=1, spatial_sum_axes=(0,), group_indices_list=[[0, 1], [2, 3]])
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_direction_marginal_matches_old_kdiba_implementation(directional_cls, synthetic_posterior: np.ndarray):
    expected = _old_kdiba_marginal_over_direction(synthetic_posterior)
    actual = directional_cls._marginalize_p_x_given_n_over_context_groups_in_position_space(synthetic_posterior, context_dim_idx=1, spatial_sum_axes=(0,), group_indices_list=[[0, 2], [1, 3]])
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_long_short_marginal_differs_from_broken_generalized_path(directional_cls, synthetic_posterior: np.ndarray):
    fixed = directional_cls._marginalize_p_x_given_n_over_context_groups_in_position_space(synthetic_posterior, context_dim_idx=1, spatial_sum_axes=(0,), group_indices_list=[[0, 1], [2, 3]])
    broken = _broken_generalized_marginal(synthetic_posterior, [[0, 1], [2, 3]], directional_cls)
    assert not np.allclose(fixed, broken, rtol=1e-6, atol=1e-6)


def test_direction_marginal_differs_from_broken_generalized_path(directional_cls, synthetic_posterior: np.ndarray):
    fixed = directional_cls._marginalize_p_x_given_n_over_context_groups_in_position_space(synthetic_posterior, context_dim_idx=1, spatial_sum_axes=(0,), group_indices_list=[[0, 2], [1, 3]])
    broken = _broken_generalized_marginal(synthetic_posterior, [[0, 2], [1, 3]], directional_cls)
    assert not np.allclose(fixed, broken, rtol=1e-6, atol=1e-6)


def test_build_custom_marginal_over_long_short_integration(directional_cls, synthetic_posterior: np.ndarray):
    marginals = directional_cls.build_custom_marginal_over_long_short([synthetic_posterior])
    expected = _old_kdiba_marginal_over_long_short(synthetic_posterior)
    np.testing.assert_allclose(marginals[0].p_x_given_n, expected, rtol=1e-12, atol=1e-12)


def test_build_custom_marginal_over_direction_integration(directional_cls, synthetic_posterior: np.ndarray):
    marginals = directional_cls.build_custom_marginal_over_direction([synthetic_posterior])
    expected = _old_kdiba_marginal_over_direction(synthetic_posterior)
    np.testing.assert_allclose(marginals[0].p_x_given_n, expected, rtol=1e-12, atol=1e-12)
