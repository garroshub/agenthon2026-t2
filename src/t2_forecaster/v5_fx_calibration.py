
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import numpy as np
import pandas as pd

FX_SCALE = 0.99
GATE_MIN_BACKTESTS = 6
GATE_BACKTEST_COUNT = 8
GATE_WIN_RATE_MIN = 0.625
GATE_MEAN_DELTA_MAX = -0.002
GATE_WORST_DELTA_MAX = 0.05
TAIL_LEVELS_DEFAULT = (0.01, 0.05, 0.95, 0.99)
G10 = frozenset({"AUD","CAD","CHF","DKK","EUR","GBP","JPY","NOK","NZD","SEK"})


def eligible_full_history_fx(card: dict[str, Any], spec: Any) -> bool:
    panel_ids = {str(x) for x in card.get("panels", {}).get("panel_ids", ())}
    return (
        getattr(spec, "target_type", None) == "level"
        and getattr(spec, "target_frequency", None) == "daily"
        and getattr(spec, "target_panel", None) == "g10_fx_daily"
        and "g10_fx_daily" in panel_ids
        and "em_transfer_early" not in panel_ids
        and set(getattr(spec, "assets", ())).issubset(G10)
    )


def shrink_dispersion(draws: np.ndarray, scale: float = FX_SCALE) -> np.ndarray:
    x = np.asarray(draws, dtype=float)
    center = np.mean(x, axis=0, keepdims=True)
    out = center + float(scale) * (x - center)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("nonfinite calibrated draws")
    return out


def _mean_abs_pairwise_sorted(x_sorted: np.ndarray, fair: bool = True) -> np.ndarray:
    m = x_sorted.shape[0]
    i = np.arange(1, m + 1, dtype=np.float64)
    coef = 2.0 * i - m - 1.0
    if x_sorted.ndim == 2:
        coef = coef[:, None]
    s = 2.0 * np.sum(coef * x_sorted, axis=0)
    denom = m * (m - 1) if (fair and m > 1) else m * m
    return s / denom


def crps_marginal(samples: np.ndarray, y: np.ndarray) -> float:
    s = np.asarray(samples, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    term1 = np.abs(s - yy).mean(axis=0)
    spread = _mean_abs_pairwise_sorted(np.sort(s, axis=0), True)
    return float(np.mean(term1 - 0.5 * spread))


def energy_score(samples: np.ndarray, y: np.ndarray) -> float:
    s = np.asarray(samples, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    m = s.shape[0]
    term1 = np.linalg.norm(s - yy[None, :], axis=1).mean()
    diff = s[:, None, :] - s[None, :, :]
    pn = np.linalg.norm(diff, axis=2)
    denom = m * (m - 1) if m > 1 else m * m
    term2 = pn.sum() / denom
    return float(term1 - 0.5 * term2)


def variogram_score(samples: np.ndarray, y: np.ndarray, p: float = 0.5) -> float:
    s = np.asarray(samples, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    d = yy.shape[0]
    w = np.ones((d, d))
    y_vg = np.abs(yy[:, None] - yy[None, :]) ** p
    x_vg = (np.abs(s[:, :, None] - s[:, None, :]) ** p).mean(axis=0)
    return float(np.sum(w * (y_vg - x_vg) ** 2))


def tail_pinball(samples: np.ndarray, y: np.ndarray, levels: tuple[float, ...]) -> float:
    s = np.asarray(samples, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    total = 0.0
    for a in levels:
        q = np.quantile(s, a, axis=0)
        loss = np.where(yy >= q, a * (yy - q), (1.0 - a) * (q - yy))
        total += float(np.mean(loss))
    return total / len(levels)


def _weights_and_params(card: dict[str, Any], cell_count: int):
    params = card.get("scoring", {}).get("params", {})
    weights = params.get("weights", {"marginal": 0.5, "joint": 0.3, "tail": 0.2})
    wt = (float(weights["marginal"]), float(weights["joint"]), float(weights["tail"]))
    joint = str(params.get("joint", "variogram"))
    if cell_count == 1:
        live = wt[0] + wt[2]
        if live <= 0:
            raise ValueError("single-cell live weights are nonpositive")
        wt = (wt[0] / live, 0.0, wt[2] / live)
    levels = tuple(float(x) for x in params.get("tail_levels", TAIL_LEVELS_DEFAULT))
    tail_metric = str(params.get("tail_metric", "pinball"))
    if tail_metric != "pinball":
        raise ValueError("V5 FX gate supports the current pinball tail metric only")
    return wt, levels, joint


def score_components(
    draws: np.ndarray,
    y: np.ndarray,
    card: dict[str, Any],
    ref_scale: dict[str, float] | None,
) -> dict[str, float]:
    flat = np.asarray(draws, dtype=float).reshape(draws.shape[0], -1)
    yy = np.asarray(y, dtype=float)
    wt, levels, joint = _weights_and_params(card, yy.size)
    marg = crps_marginal(flat, yy)
    if wt[1] == 0.0:
        jnt = 0.0
    elif joint == "energy":
        jnt = energy_score(flat, yy)
    elif joint == "variogram":
        jnt = variogram_score(flat, yy, p=0.5)
    else:
        raise ValueError(f"unknown joint score {joint!r}")
    tail = tail_pinball(flat, yy, levels)
    m_n = marg / ref_scale["marginal"] if ref_scale else marg
    j_n = jnt / ref_scale["joint"] if (ref_scale and wt[1] > 0) else jnt
    t_n = tail / ref_scale["tail"] if ref_scale else tail
    return {
        "marginal": float(marg),
        "joint": float(jnt),
        "tail": float(tail),
        "composite": float(wt[0] * m_n + wt[1] * j_n + wt[2] * t_n),
        "joint_weight": float(wt[1]),
    }


def _common_dates(history: dict[str, pd.Series]) -> pd.DatetimeIndex:
    common: pd.DatetimeIndex | None = None
    for s in history.values():
        idx = pd.DatetimeIndex(s.index).sort_values().unique()
        common = idx if common is None else common.intersection(idx)
    return pd.DatetimeIndex([]) if common is None else common.sort_values()


def _recent_origins(
    history: dict[str, pd.Series],
    current_cutoff: pd.Timestamp,
    steps: np.ndarray,
    n: int = GATE_BACKTEST_COUNT,
) -> list[pd.Timestamp]:
    max_step = int(np.max(steps))
    common = _common_dates(history)
    common = common[common < current_cutoff]
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
            if loc < 320 or loc + max_step >= len(ss):
                ok = False
                break
            if pd.Timestamp(ss.index[loc + max_step]) > current_cutoff:
                ok = False
                break
        if ok:
            valid.append(pd.Timestamp(dt))
    out: list[pd.Timestamp] = []
    spacing = max(21, max_step)
    for dt in reversed(valid):
        pos = int(common.get_loc(dt))
        if all(abs(pos - int(common.get_loc(x))) >= spacing for x in out):
            out.append(dt)
        if len(out) >= int(n):
            break
    return sorted(out)


def _slice_history(history: dict[str, pd.Series], cutoff: pd.Timestamp) -> dict[str, pd.Series]:
    return {a: s.sort_index().loc[:cutoff].copy() for a, s in history.items()}


def _realized_levels(
    history: dict[str, pd.Series],
    cutoff: pd.Timestamp,
    steps: np.ndarray,
    assets: tuple[str, ...],
    horizons: tuple[int, ...],
) -> np.ndarray:
    values: list[float] = []
    for ai, asset in enumerate(assets):
        s = history[asset].sort_index()
        loc = int(s.index.get_loc(cutoff))
        for hi, _h in enumerate(horizons):
            end_loc = loc + int(steps[ai, hi])
            if end_loc >= len(s):
                raise ValueError("insufficient future observations")
            value = float(s.iloc[end_loc])
            if not np.isfinite(value):
                raise ValueError("nonfinite realization")
            values.append(value)
    return np.asarray(values, dtype=float)


def evaluate_gate(
    *,
    card: dict[str, Any],
    spec: Any,
    history: dict[str, pd.Series],
    steps: np.ndarray,
    v2_draws_fn: Callable[[Any, dict[str, pd.Series], np.ndarray], np.ndarray],
    m0_draws_fn: Callable[[Any, dict[str, pd.Series], np.ndarray], np.ndarray],
) -> tuple[bool, dict[str, float | int | str]]:
    if not eligible_full_history_fx(card, spec):
        return False, {"reason": "ineligible"}
    cutoff = pd.Timestamp(spec.asof)
    origins = _recent_origins(history, cutoff, steps, GATE_BACKTEST_COUNT)
    deltas: list[float] = []
    for origin in origins:
        try:
            sliced = _slice_history(history, origin)
            past_spec = replace(spec, asof=origin.date().isoformat())
            y = _realized_levels(history, origin, steps, spec.assets, spec.horizons)
            m0 = m0_draws_fn(past_spec, sliced, steps)
            m0_raw = score_components(m0, y, card, None)
            ref = {
                "marginal": float(m0_raw["marginal"]),
                "joint": float(m0_raw["joint"]) if float(m0_raw["joint_weight"]) > 0 else 1.0,
                "tail": float(m0_raw["tail"]),
            }
            if ref["marginal"] <= 0 or ref["tail"] <= 0 or (
                float(m0_raw["joint_weight"]) > 0 and ref["joint"] <= 0
            ):
                continue
            v2 = v2_draws_fn(past_spec, sliced, steps)
            cal = shrink_dispersion(v2, FX_SCALE)
            sv = score_components(v2, y, card, ref)["composite"]
            sc = score_components(cal, y, card, ref)["composite"]
            deltas.append(float(np.clip(sc, 0, 4) - np.clip(sv, 0, 4)))
        except Exception:
            continue

    arr = np.asarray(deltas, dtype=float)
    n_bt = int(len(arr))
    if n_bt == 0:
        return False, {"reason": "no_backtests", "n_backtests": 0}
    mean_delta = float(arr.mean())
    win_rate = float(np.mean(arr < 0))
    worst_delta = float(arr.max())
    activate = bool(
        n_bt >= GATE_MIN_BACKTESTS
        and win_rate >= GATE_WIN_RATE_MIN
        and mean_delta <= GATE_MEAN_DELTA_MAX
        and worst_delta <= GATE_WORST_DELTA_MAX
    )
    return activate, {
        "reason": "pass" if activate else "gate_fail",
        "n_backtests": n_bt,
        "mean_delta": mean_delta,
        "win_rate": win_rate,
        "worst_delta": worst_delta,
    }
