"""Export a sliced Bayesian / DST decoder payload for the browser equation debugger.

Writes a Zarr v2 directory store that the static webapp under
``Spike3D/webapps/bayesian_2d_eqn_debugger`` can load via zarrita.

Usage:

    from pathlib import Path
    from pyphoplacecellanalysis.Analysis.Decoder.eqn_debugger_export import export_bayesian_2d_eqn_debugger

    export_bayesian_2d_eqn_debugger(
        a_dst_decoder2D,
        out_path=Path("webapps/bayesian_2d_eqn_debugger/data/bayesian_2d_eqn_debugger.zarr"),
        group_key="JS15_cells_27_29_31",
        neuron_ids=(27, 29, 31),
    )
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import zarr

from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter


FORMAT_ID = "bayesian_2d_eqn_debugger/v1"


def resolve_neuron_ids_for_eqn_debugger(decoder, neuron_ids: Optional[Union[List[int], Tuple[int, ...], int]] = None) -> Tuple[int, ...]:
    """Resolve ``None`` / int pair-index / explicit ids the same way as ``InteractiveBayesian2DEquationDebugger``."""
    from pyphoplacecellanalysis.SpecificResults.PendingNotebookCode import DisjointPlacefieldsExploration

    if neuron_ids is None:
        pairs = DisjointPlacefieldsExploration.compute_unit_pair_least_overlapping(ratemap=decoder.ratemap)
        neuron_ids = (int(pairs[0][0]), int(pairs[0][1]))
        print(f'Using most-disjoint pair: {neuron_ids} (overlap={pairs[0][2]:.4g})')
    elif isinstance(neuron_ids, int):
        pairs = DisjointPlacefieldsExploration.compute_unit_pair_least_overlapping(ratemap=decoder.ratemap)
        pair_idx: int = neuron_ids
        neuron_ids = (int(pairs[pair_idx][0]), int(pairs[pair_idx][1]))
        print(f'Using pair_idx {pair_idx}: neuron_ids: {neuron_ids} (overlap={pairs[pair_idx][2]:.4g})')
    ## END if neuron_ids is None...

    return tuple(int(x) for x in neuron_ids)


def build_eqn_debugger_seed_n(decoder, sliced, aclu_list: Sequence[int], all_epochs_decoding_result=None, seed_epoch_idx: int = 0, seed_t_bin_idx: int = 0) -> np.ndarray:
    """Seed spike counts from an optional epoch decode result, else ones."""
    n_cells = len(aclu_list)
    if all_epochs_decoding_result is None:
        return np.ones(n_cells, dtype=np.int32)

    spk = all_epochs_decoding_result.spkcount[seed_epoch_idx]
    try:
        full_ids = np.asarray(decoder.neuron_IDs)
        row_idx = [int(np.where(full_ids == a)[0][0]) for a in aclu_list]
        n0 = np.asarray(spk[row_idx, seed_t_bin_idx], dtype=np.int32)
        print(f'Seeded n from epoch={seed_epoch_idx}, t_bin={seed_t_bin_idx}: {dict(zip(list(aclu_list), n0.tolist()))}')
        return n0
    except Exception as e:
        print(f'Could not seed from decoding result ({e}); starting at n=1 for each cell.')
        return np.ones(n_cells, dtype=np.int32)


def export_bayesian_2d_eqn_debugger(decoder, out_path: Union[str, Path], group_key: str, neuron_ids: Optional[Union[List[int], Tuple[int, ...], int]] = None,
                                    all_epochs_decoding_result=None, seed_epoch_idx: int = 0, seed_t_bin_idx: int = 0,
                                    max_spikes_per_cell: int = 15, show_log_likelihood: bool = True,
                                    drop_negative_contributing_terms_mode: bool = True, overwrite: bool = True) -> Path:
    """Slice ``decoder`` to ``neuron_ids`` and write a Zarr group for the browser equation debugger.

    Args:
        decoder: ``BayesianPlacemapPositionDecoder`` or DST subclass.
        out_path: Path to a Zarr directory store (created if missing).
        group_key: Name of the subgroup under the store root (e.g. ``JS15_cells_27_29_31``).
        neuron_ids: Explicit ACLUs, ``None`` for most-disjoint pair, or int pair-index.
        all_epochs_decoding_result: Optional seed source for ``seed_n``.
        seed_epoch_idx / seed_t_bin_idx: Which epoch/time-bin to seed from.
        max_spikes_per_cell / show_log_likelihood / drop_negative_contributing_terms_mode: UI attrs.
        overwrite: If True, replace an existing subgroup with the same ``group_key``.

    Returns:
        Path to the Zarr store.
    """
    from pyphoplacecellanalysis.Analysis.Decoder.reconstruction_dst import BayesianPlacemapPositionDecoderDST

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    neuron_ids = resolve_neuron_ids_for_eqn_debugger(decoder, neuron_ids=neuron_ids)
    sliced = decoder.get_by_id(list(neuron_ids), defer_compute_all=True)
    is_dst: bool = isinstance(sliced, BayesianPlacemapPositionDecoderDST)
    if is_dst and (getattr(sliced, 'reliability_active', None) is None):
        sliced._compute_reliability_metrics()

    tau: float = float(sliced.time_bin_size)
    tc = np.asarray(sliced.ratemap.tuning_curves, dtype=np.float32)  # (n_cells, nx, ny)
    n_cells, nx, ny = tc.shape
    aclu_list = list(map(int, sliced.ratemap.neuron_ids))
    xbin = np.asarray(sliced.xbin, dtype=np.float32)
    ybin = np.asarray(sliced.ybin, dtype=np.float32)
    seed_n = build_eqn_debugger_seed_n(decoder, sliced, aclu_list, all_epochs_decoding_result=all_epochs_decoding_result, seed_epoch_idx=seed_epoch_idx, seed_t_bin_idx=seed_t_bin_idx)

    with ProgressMessagePrinter(out_path, 'Writing', f'eqn debugger group {group_key}'):
        root = zarr.open_group(str(out_path), mode='a')
        keys: List[str] = list(root.attrs.get('keys', []) or [])
        if group_key in root and overwrite:
            del root[group_key]
        if group_key not in keys:
            keys.append(group_key)
        ## END if group_key not in keys...

        root.attrs['format'] = FORMAT_ID
        root.attrs['keys'] = keys

        # Sidecar catalog + per-group JSON for the webapp (avoids ES-module / zarrita MIME issues)
        json_dir = out_path.parent
        json_dir.mkdir(parents=True, exist_ok=True)
        catalog = {'format': FORMAT_ID, 'keys': keys}
        (json_dir / 'groups.json').write_text(json.dumps(catalog, indent=2), encoding='utf-8')
        (out_path / 'groups.json').write_text(json.dumps(catalog, indent=2), encoding='utf-8')

        g = root.require_group(group_key)
        # Clear leftover arrays if group already existed without full delete
        for arr_name in list(g.array_keys()):
            del g[arr_name]
        ## END for arr_name in list(g.array_keys())....

        chunks = (1, int(nx), int(ny))
        g.create_dataset('tuning_curves', data=tc, chunks=chunks, overwrite=True)
        g.create_dataset('neuron_ids', data=np.asarray(aclu_list, dtype=np.int32), overwrite=True)
        g.create_dataset('xbin', data=xbin, overwrite=True)
        g.create_dataset('ybin', data=ybin, overwrite=True)
        g.create_dataset('seed_n', data=np.asarray(seed_n, dtype=np.int32), overwrite=True)

        g.attrs['format'] = FORMAT_ID
        g.attrs['tau'] = tau
        g.attrs['is_dst'] = bool(is_dst)
        g.attrs['n_cells'] = int(n_cells)
        g.attrs['neuron_ids'] = aclu_list
        g.attrs['max_spikes_per_cell'] = int(max_spikes_per_cell)
        g.attrs['show_log_likelihood'] = bool(show_log_likelihood)
        g.attrs['drop_negative_contributing_terms_mode'] = bool(drop_negative_contributing_terms_mode)
        g.attrs['tuning_curves_shape'] = [int(n_cells), int(nx), int(ny)]

        rel_active = None
        rel_silent = None
        if is_dst:
            rel_active = _as_per_cell_reliability(sliced.reliability_active, n_cells)
            rel_silent = _as_per_cell_reliability(sliced.reliability_silent, n_cells)
            g.create_dataset('reliability_active', data=rel_active, overwrite=True)
            g.create_dataset('reliability_silent', data=rel_silent, overwrite=True)
            g.attrs['should_discount_silence'] = bool(getattr(sliced, 'should_discount_silence', False))
        ## END if is_dst...

        web_payload = {
            'format': FORMAT_ID,
            'group_key': group_key,
            'tau': tau,
            'is_dst': bool(is_dst),
            'n_cells': int(n_cells),
            'nx': int(nx),
            'ny': int(ny),
            'neuron_ids': aclu_list,
            'xbin': xbin.astype(float).tolist(),
            'ybin': ybin.astype(float).tolist(),
            'seed_n': [int(x) for x in np.asarray(seed_n).tolist()],
            'tuning_curves': tc.astype(float).tolist(),
            'max_spikes_per_cell': int(max_spikes_per_cell),
            'show_log_likelihood': bool(show_log_likelihood),
            'drop_negative_contributing_terms_mode': bool(drop_negative_contributing_terms_mode),
        }
        if is_dst and (rel_active is not None) and (rel_silent is not None):
            web_payload['reliability_active'] = rel_active.astype(float).tolist()
            web_payload['reliability_silent'] = rel_silent.astype(float).tolist()
            web_payload['should_discount_silence'] = bool(getattr(sliced, 'should_discount_silence', False))
        ## END if is_dst and (rel_active is not None) and (rel_silent is not None)...

        (json_dir / f'{group_key}.json').write_text(json.dumps(web_payload), encoding='utf-8')

    return out_path


def _as_per_cell_reliability(arr, n_cells: int) -> np.ndarray:
    """Normalize reliability to shape ``(n_cells,)`` float32 for the eqn debugger."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        assert arr.shape[0] == n_cells, f'reliability length {arr.shape[0]} != n_cells {n_cells}'
        return arr
    if arr.ndim == 2:
        # (n_flat, n_cells) → mean over position (matches scalar α usage in eqn debugger)
        out = np.nanmean(arr, axis=0).astype(np.float32)
        assert out.shape[0] == n_cells
        return out
    raise ValueError(f'Unsupported reliability ndim={arr.ndim}; expected 1 or 2.')


def list_eqn_debugger_groups(out_path: Union[str, Path]) -> List[str]:
    """Return exported group keys from a Zarr store (empty list if missing)."""
    out_path = Path(out_path)
    if not out_path.exists():
        return []
    root = zarr.open_group(str(out_path), mode='r')
    keys = root.attrs.get('keys', None)
    if keys is not None:
        return list(keys)
    return list(root.group_keys())
