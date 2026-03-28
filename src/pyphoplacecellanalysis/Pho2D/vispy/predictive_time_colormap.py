"""Sequential predictive-decoding time colors: optional cyan→magenta stops or matplotlib cmaps (e.g. plasma, cool)."""

from __future__ import annotations

import math
from typing import Literal, cast
import colorsys

import numpy as np

PredictiveTimeColormapName = Literal["cyan_magenta", "plasma", "cool"]

PREDICTIVE_TIME_COLORMAP_NAMES: tuple[PredictiveTimeColormapName, ...] = ("cyan_magenta", "plasma", "cool")

# Matplotlib colormap name per predictive-time key (None = custom piecewise-linear stops in this file).
_PREDICTIVE_TIME_MPL_NAME: dict[PredictiveTimeColormapName, str | None] = {
    "cyan_magenta": None,
    "plasma": "plasma",
    "cool": "cool",
}

# _active_predictive_time_colormap: PredictiveTimeColormapName = "cyan_magenta"
# _active_predictive_time_colormap: PredictiveTimeColormapName = "plasma"
_active_predictive_time_colormap: PredictiveTimeColormapName = "cool"


def set_predictive_time_colormap(name: PredictiveTimeColormapName | str) -> None:
    """Set the module default used by ``predictive_time_*`` when ``colormap`` is omitted."""
    global _active_predictive_time_colormap
    key = str(name)
    if key not in PREDICTIVE_TIME_COLORMAP_NAMES:
        raise ValueError(f"Unknown predictive time colormap {name!r}; choose one of {PREDICTIVE_TIME_COLORMAP_NAMES}.")
    _active_predictive_time_colormap = cast(PredictiveTimeColormapName, key)


def get_predictive_time_colormap() -> PredictiveTimeColormapName:
    return _active_predictive_time_colormap


def _hex_to_rgb01(hex_str: str) -> tuple[float, float, float]:
    h = hex_str.lstrip('#')
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


# Control stops u in [0,1]: start bright cyan, mid periwinkle, end magenta-purple (cyan_magenta only).
_PREDICTIVE_TIME_STOPS_U = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
_PREDICTIVE_TIME_STOPS_RGB = np.array(
    [
        _hex_to_rgb01('#58D4E8'),
        _hex_to_rgb01('#6BC0E8'),
        _hex_to_rgb01('#7886C8'),
        _hex_to_rgb01('#986BB4'),
        _hex_to_rgb01('#B850A0'),
    ],
    dtype=np.float32,
)


def _predictive_time_rgb_for_key(key: PredictiveTimeColormapName, u: float) -> tuple[float, float, float]:
    uu = float(u)
    if not math.isfinite(uu):
        uu = 0.0
    uu = float(np.clip(uu, 0.0, 1.0))
    mpl = _PREDICTIVE_TIME_MPL_NAME[key]
    if mpl is None:
        r_g_b = np.empty(3, dtype=np.float32)
        for c in range(3):
            r_g_b[c] = np.float32(np.interp(uu, _PREDICTIVE_TIME_STOPS_U, _PREDICTIVE_TIME_STOPS_RGB[:, c]))
        return (float(r_g_b[0]), float(r_g_b[1]), float(r_g_b[2]))
    import matplotlib

    c = matplotlib.colormaps[mpl](uu)
    return (float(c[0]), float(c[1]), float(c[2]))


def _resolve_colormap_key(colormap: PredictiveTimeColormapName | str | None) -> PredictiveTimeColormapName:
    if colormap is None:
        return _active_predictive_time_colormap
    key = str(colormap)
    if key not in PREDICTIVE_TIME_COLORMAP_NAMES:
        raise ValueError(f"Unknown predictive time colormap {colormap!r}; choose one of {PREDICTIVE_TIME_COLORMAP_NAMES}.")
    return cast(PredictiveTimeColormapName, key)


def predictive_time_rgb(u: float, s: int | float=0.8, v: int | float=0.9, colormap: PredictiveTimeColormapName | str | None = None) -> tuple[float, float, float]:
    """Sample the sequential time colormap at u in [0, 1] (clamp). Returns RGB in [0, 1].
    hue = (t_idx / max(n_tbins, 1)) % 1.0

    """
    # return _predictive_time_rgb_for_key(_resolve_colormap_key(colormap), u)
    return colorsys.hsv_to_rgb(u, s=s, v=v)



def predictive_time_rgba_u(u: float, alpha: float, colormap: PredictiveTimeColormapName | str | None = None) -> tuple[float, float, float, float]:
    """RGBA sample at u with given alpha."""
    r, g, b = predictive_time_rgb(u, colormap=colormap)
    a = float(alpha)
    if not math.isfinite(a):
        a = 1.0
    a = float(np.clip(a, 0.0, 1.0))
    return (r, g, b, a)


def predictive_time_bin_rgba(n_bins: int, alpha: float = 0.9, colormap: PredictiveTimeColormapName | str | None = None) -> np.ndarray:
    """(n_bins, 4) float32 RGBA; u = t_idx / max(n_bins, 1) per legacy HSV bin spacing."""
    # key = _resolve_colormap_key(colormap)
    # n = int(n_bins)
    # out = np.zeros((max(n, 0), 4), dtype=np.float32)
    # if n <= 0:
    #     return out
    # denom = float(max(n, 1))
    # for t_idx in range(n):
    #     u = t_idx / denom
    #     r, g, b = _predictive_time_rgb_for_key(key, u)
    #     out[t_idx] = (r, g, b, float(alpha))
    # return out
    """Return (n_bins, 4) float32 array of RGBA colors for time bins (hue cycled)."""
    out = np.zeros((n_bins, 4), dtype=np.float32)
    for t_idx in range(n_bins):
        hue = (t_idx / max(n_bins, 1)) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        out[t_idx] = (rgb[0], rgb[1], rgb[2], alpha)
    return out

