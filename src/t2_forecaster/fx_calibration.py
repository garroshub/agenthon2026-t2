from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
import pandas as pd

FX_SCALE = 0.975
BACKTEST_TARGET = 8
MIN_BACKTESTS = 6
WIN_RATE_MIN = 0.625
MEAN_DELTA_MAX = -0.002
WORST_DELTA_MAX = 0.05
G10 = frozenset({"AUD","CAD","CHF","DKK","EUR","GBP","JPY","NOK","NZD","SEK"})
HISTORICAL_DRAWS = 1000
TAIL_LEVELS_DEFAULT = (0.01,0.05,0.95,0.99)


@dataclass(frozen=True)
class CalibrationResult:
    draws: np.ndarray
    applied: bool
    reason: str
    n_backtests: int
    mean_delta: float | None
    win_rate: float | None
    worst_delta: float | None


def scale_about_mean(draws: np.ndarray, scale: float = FX_SCALE) -> np.ndarray:
    x = np.asarray(draws, dtype=float)
    c = np.mean(x, axis=0, keepdims=True)
    y = c + float(scale) * (x - c)
    if not np.all(np.isfinite(y)):
        raise FloatingPointError("non-finite calibrated draws")
    return y


def _pairwise_abs_mean_sorted(samples: np.ndarray) -> np.ndarray:
    x = np.sort(np.asarray(samples, dtype=float), axis=0)
    m = x.shape[0]
    i = np.arange(1, m + 1, dtype=float)
    coef = (2.0 * i - m - 1.0)
    if x.ndim == 2:
        coef = coef[:, None]
    total = 2.0 * np.sum(coef * x, axis=0)
    denom = m * (m - 1) if m > 1 else m * m
    return total / denom


def _crps_marginal(samples: np.ndarray, y: np.ndarray) -> float:
    s = np.asarray(samples, dtype=float)
    yy = np.asarray(y, dtype=float)
    term1 = np.abs(s - yy).mean(axis=0)
    spread = _pairwise_abs_mean_sorted(s)
    return float(np.mean(term1 - 0.5 * spread))


def _energy_score(samples: np.ndarray, y: np.ndarray) -> float:
    s = np.asarray(samples, dtype=float)
    yy = np.asarray(y, dtype=float)
    m = s.shape[0]
    term1 = np.linalg.norm(s - yy[None, :], axis=1).mean()
    # Chunked exact pairwise computation to bound memory.
    total = 0.0
    chunk = 128
    for start in range(0, m, chunk):
        a = s[start:start+chunk]
        diff = a[:, None, :] - s[None, :, :]
        total += float(np.linalg.norm(diff, axis=2).sum())
    denom = m * (m - 1) if m > 1 else m * m
    return float(term1 - 0.5 * total / denom)


def _variogram_score(samples: np.ndarray, y: np.ndarray, p: float = 0.5) -> float:
    s = np.asarray(samples, dtype=float)
    yy = np.asarray(y, dtype=float)
    y_vg = np.abs(yy[:, None] - yy[None, :]) ** p
    x_vg = (np.abs(s[:, :, None] - s[:, None, :]) ** p).mean(axis=0)
    return float(np.sum((y_vg - x_vg) ** 2))


def _tail_pinball(samples: np.ndarray, y: np.ndarray, levels: tuple[float, ...]) -> float:
    s = np.asarray(samples, dtype=float)
    yy = np.asarray(y, dtype=float)
    total = 0.0
    for a in levels:
        q = np.quantile(s, a, axis=0)
        loss = np.where(yy >= q, a * (yy - q), (1.0 - a) * (q - yy))
        total += float(np.mean(loss))
    return total / len(levels)


def _tail_coverage(samples: np.ndarray, y: np.ndarray, levels: tuple[float, ...]) -> float:
    s = np.asarray(samples, dtype=float)
    yy = np.asarray(y, dtype=float)
    err = 0.0
    for a in levels:
        q = np.quantile(s, a, axis=0)
        err += abs(float(np.mean(yy <= q)) - a)
    return float(err)


def _weights_and_params(card: dict[str, Any], cell_count: int):
    params = card.get("scoring", {}).get("params", {})
    weights = params.get("weights", {"marginal":0.5,"joint":0.3,"tail":0.2})
    wt = (float(weights["marginal"]), float(weights["joint"]), float(weights["tail"]))
    joint = str(params.get("joint", "variogram"))
    if cell_count == 1 and joint == "variogram":
        live = wt[0] + wt[2]
        if live <= 0:
            raise ValueError("nonpositive single-cell live weight")
        wt = (wt[0]/live, 0.0, wt[2]/live)
    levels = tuple(float(x) for x in params.get("tail_levels", TAIL_LEVELS_DEFAULT))
    tail_metric = str(params.get("tail_metric", "pinball"))
    return wt, levels, joint, tail_metric


def score_components(samples: np.ndarray, y: np.ndarray, card: dict[str, Any], ref_scale: dict[str,float] | None = None) -> dict[str,float]:
    s = np.asarray(samples, dtype=float)
    yy = np.asarray(y, dtype=float)
    if s.ndim != 2 or yy.shape != (s.shape[1],):
        raise ValueError("score shape mismatch")
    wt, levels, joint, tail_metric = _weights_and_params(card, yy.size)
    marginal = _crps_marginal(s, yy)
    if joint == "variogram":
        jnt = _variogram_score(s, yy, p=0.5)
    elif joint == "energy":
        jnt = _energy_score(s, yy)
    else:
        raise ValueError(f"unknown joint statistic {joint!r}")
    if tail_metric == "pinball":
        tail = _tail_pinball(s, yy, levels)
    elif tail_metric == "coverage":
        tail = _tail_coverage(s, yy, levels)
    else:
        raise ValueError(f"unknown tail metric {tail_metric!r}")
    if ref_scale:
        mn = marginal / float(ref_scale["marginal"])
        jn = jnt / float(ref_scale["joint"])
        tn = tail / float(ref_scale["tail"])
    else:
        mn, jn, tn = marginal, jnt, tail
    comp = wt[0]*mn + wt[1]*jn + wt[2]*tn
    return {"marginal":float(marginal),"joint":float(jnt),"tail":float(tail),"composite":float(comp)}


def _common_dates(history: dict[str,pd.Series]) -> pd.DatetimeIndex:
    common = None
    for s in history.values():
        idx = pd.DatetimeIndex(s.index).sort_values().unique()
        common = idx if common is None else common.intersection(idx)
    return pd.DatetimeIndex([]) if common is None else common.sort_values()


def _recent_origins(history: dict[str,pd.Series], current: pd.Timestamp, steps: np.ndarray) -> list[pd.Timestamp]:
    mx = int(np.max(steps))
    common = _common_dates(history)
    common = common[common < current]
    valid: list[pd.Timestamp] = []
    for dt in common:
        ok = True
        for s in history.values():
            ss = s.sort_index()
            try:
                loc = int(ss.index.get_loc(dt))
            except Exception:
                ok = False
                break
            if loc < 320 or loc + mx >= len(ss) or pd.Timestamp(ss.index[loc+mx]) > current:
                ok = False
                break
        if ok:
            valid.append(pd.Timestamp(dt))
    out: list[pd.Timestamp] = []
    spacing = max(21, mx)
    for dt in reversed(valid):
        pos = int(common.get_loc(dt))
        if all(abs(pos - int(common.get_loc(x))) >= spacing for x in out):
            out.append(dt)
        if len(out) >= BACKTEST_TARGET:
            break
    return sorted(out)


def _realized_level(history: dict[str,pd.Series], cutoff: pd.Timestamp, steps: np.ndarray, assets: tuple[str,...], horizons: tuple[int,...]) -> np.ndarray:
    vals: list[float] = []
    for ai, asset in enumerate(assets):
        s = history[asset].sort_index()
        loc = int(s.index.get_loc(cutoff))
        for hi, _h in enumerate(horizons):
            end = loc + int(steps[ai,hi])
            if end >= len(s):
                raise ValueError("insufficient historical future")
            v = float(s.iloc[end])
            if not np.isfinite(v):
                raise ValueError("nonfinite historical realization")
            vals.append(v)
    return np.asarray(vals, dtype=float)


def is_eligible(spec: Any) -> bool:
    return (
        getattr(spec, "target_type", None) == "level"
        and getattr(spec, "target_frequency", None) == "daily"
        and getattr(spec, "target_panel", None) == "g10_fx_daily"
        and set(getattr(spec, "assets", ())).issubset(G10)
    )


def maybe_calibrate(
    *,
    card: dict[str,Any],
    spec: Any,
    history: dict[str,pd.Series],
    steps: np.ndarray,
    current_draws: np.ndarray,
    v2_factory: Callable[[Any,dict[str,pd.Series],np.ndarray,int],np.ndarray],
    m0_factory: Callable[[Any,dict[str,pd.Series],np.ndarray,int],np.ndarray],
) -> CalibrationResult:
    base = np.asarray(current_draws, dtype=float)
    if not is_eligible(spec):
        return CalibrationResult(base, False, "ineligible", 0, None, None, None)
    try:
        current = pd.Timestamp(spec.asof)
        origins = _recent_origins(history, current, steps)
        deltas: list[float] = []
        for origin in origins:
            try:
                sliced = {a:s.sort_index().loc[:origin].copy() for a,s in history.items()}
                hspec = replace(spec, asof=origin.date().isoformat())
                y = _realized_level(history, origin, steps, spec.assets, spec.horizons)
                m0 = m0_factory(hspec, sliced, steps, HISTORICAL_DRAWS)
                ref_raw = score_components(m0.reshape(m0.shape[0], -1), y, card, None)
                ref = {
                    "marginal": float(ref_raw["marginal"]),
                    "joint": float(ref_raw["joint"]) if float(ref_raw["joint"]) > 0 else 1.0,
                    "tail": float(ref_raw["tail"]),
                }
                if min(ref.values()) <= 1e-12 or not all(np.isfinite(list(ref.values()))):
                    continue
                v2 = v2_factory(hspec, sliced, steps, HISTORICAL_DRAWS)
                cal = scale_about_mean(v2, FX_SCALE)
                sv = score_components(v2.reshape(v2.shape[0], -1), y, card, ref)
                sc = score_components(cal.reshape(cal.shape[0], -1), y, card, ref)
                delta = float(np.clip(sc["composite"],0.0,4.0) - np.clip(sv["composite"],0.0,4.0))
                if np.isfinite(delta):
                    deltas.append(delta)
            except Exception:
                continue
        arr = np.asarray(deltas, dtype=float)
        if len(arr) < MIN_BACKTESTS:
            return CalibrationResult(base, False, "insufficient_backtests", int(len(arr)), None, None, None)
        mean_delta = float(np.mean(arr))
        win_rate = float(np.mean(arr < 0.0))
        worst_delta = float(np.max(arr))
        gate = (
            win_rate >= WIN_RATE_MIN
            and mean_delta <= MEAN_DELTA_MAX
            and worst_delta <= WORST_DELTA_MAX
        )
        if not gate:
            return CalibrationResult(base, False, "gate_fail", int(len(arr)), mean_delta, win_rate, worst_delta)
        calibrated = scale_about_mean(base, FX_SCALE)
        return CalibrationResult(calibrated, True, "gate_pass", int(len(arr)), mean_delta, win_rate, worst_delta)
    except Exception:
        return CalibrationResult(base, False, "calibration_error", 0, None, None, None)
