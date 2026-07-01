import ast
import numpy as np
import pytest
from pathlib import Path
from typing import Optional

from neuropy.utils.dynamic_container import DynamicContainer


def _load_resolver_under_test():
    source_path = Path(__file__).resolve().parents[1] / "src" / "pyphoplacecellanalysis" / "SpecificResults" / "PendingNotebookCode.py"
    source_tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function_names = {"_computed_data_get", "_resolve_bapun_position_decoder"}
    helper_nodes = [node for node in source_tree.body if isinstance(node, ast.FunctionDef) and node.name in function_names]
    namespace = {"Optional": Optional, "BasePositionDecoder": object}
    exec(compile(ast.Module(body=helper_nodes, type_ignores=[]), filename=str(source_path), mode="exec"), namespace)
    return namespace["_resolve_bapun_position_decoder"]


def _load_context_stacker_under_test():
    source_path = Path(__file__).resolve().parents[1] / "src" / "pyphoplacecellanalysis" / "SpecificResults" / "PendingNotebookCode.py"
    source_tree = ast.parse(source_path.read_text(encoding="utf-8"))
    helper_nodes = [node for node in source_tree.body if isinstance(node, ast.FunctionDef) and node.name == "_stack_context_posteriors_for_epoch"]
    namespace = {"List": list, "np": np}
    exec(compile(ast.Module(body=helper_nodes, type_ignores=[]), filename=str(source_path), mode="exec"), namespace)
    return namespace["_stack_context_posteriors_for_epoch"]


def _computed_data(unit_1d=None, unit_2d=None, clusterless_1d=None, clusterless_2d=None):
    return DynamicContainer(pf1D_Decoder=unit_1d, pf2D_Decoder=unit_2d, pf1D_ClusterlessDecoder=clusterless_1d, pf2D_ClusterlessDecoder=clusterless_2d)


def test_resolve_bapun_position_decoder_defaults_to_standard_when_available():
    resolve_bapun_position_decoder = _load_resolver_under_test()
    unit_decoder = object()
    clusterless_decoder = object()
    computed_data = _computed_data(unit_2d=unit_decoder, clusterless_2d=clusterless_decoder)

    assert resolve_bapun_position_decoder(computed_data, decoder_dim="2D", use_clusterless_decoders=None, context_name="maze1") is unit_decoder


def test_resolve_bapun_position_decoder_auto_uses_clusterless_when_standard_missing():
    resolve_bapun_position_decoder = _load_resolver_under_test()
    clusterless_decoder = object()
    computed_data = _computed_data(unit_2d=None, clusterless_2d=clusterless_decoder)

    assert resolve_bapun_position_decoder(computed_data, decoder_dim="2D", use_clusterless_decoders=None, context_name="maze1") is clusterless_decoder


def test_resolve_bapun_position_decoder_explicit_clusterless_requires_clusterless_decoder():
    resolve_bapun_position_decoder = _load_resolver_under_test()
    computed_data = _computed_data(unit_2d=object(), clusterless_2d=None)

    with pytest.raises(ValueError, match="pf2D_ClusterlessDecoder"):
        resolve_bapun_position_decoder(computed_data, decoder_dim="2D", use_clusterless_decoders=True, context_name="maze1")


def test_resolve_bapun_position_decoder_explicit_standard_requires_standard_decoder():
    resolve_bapun_position_decoder = _load_resolver_under_test()
    computed_data = _computed_data(unit_1d=None, clusterless_1d=object())

    with pytest.raises(ValueError, match="pf1D_Decoder"):
        resolve_bapun_position_decoder(computed_data, decoder_dim="1D", use_clusterless_decoders=False, context_name="maze1")


def test_stack_context_posteriors_for_epoch_flattens_and_normalizes_context_axis():
    stack_context_posteriors_for_epoch = _load_context_stacker_under_test()
    context_a = np.ones((2, 3, 4))
    context_b = np.full((5, 4), 2.0)

    stacked = stack_context_posteriors_for_epoch([context_a, context_b])

    assert stacked.shape == (6, 2, 4)
    np.testing.assert_allclose(np.nansum(stacked, axis=(0, 1)), np.ones(4))
    assert np.isfinite(stacked[:, 0, :]).sum() == context_a.size
    assert np.isfinite(stacked[:, 1, :]).sum() == context_b.size
