from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import house_scenario

N_DRAWS = 1000
STUDENT_DF = 7.0
PARTICIPANT_OUTPUT_FILES = ("forecast.parquet", "forecast_meta.json", "forecast_rationale.md")


@dataclass(frozen=True)
class ForecastSpec:
    unit_id: str
    asof: str
    assets: tuple[str, ...]
    horizons: tuple[int, ...]
    target_type: str
    target_frequency: str
    value_unit: str
    target_panel: str | None


@dataclass(frozen=True)
class InputContext:
    unit_root: Path
    panel_dir: Path


def _resolve_input_context(panels: str | Path) -> InputContext:
    panel_dir = Path(panels)
    if not panel_dir.is_dir():
        raise FileNotFoundError(f"panels directory does not exist: {panel_dir}")
    if (panel_dir / "card.toml").is_file():
        return InputContext(unit_root=panel_dir, panel_dir=panel_dir)
    parent = panel_dir.parent
    if (parent / "card.toml").is_file():
        return InputContext(unit_root=parent, panel_dir=panel_dir)
    raise FileNotFoundError(
        f"could not locate card.toml in supplied panels directory or its parent: {panel_dir}"
    )


def _stable_seed(unit_id: str, asof: str) -> int:
    raw = os.environ.get("QFBENCH_SEED")
    if raw:
        try:
            return int(raw) % (2**32)
        except ValueError:
            pass
    digest = hashlib.sha256(f"t2-v1|{unit_id}|{asof}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _read_card(root: Path, asof: str) -> tuple[dict[str, Any], ForecastSpec]:
    card_path = root / "card.toml"
    if not card_path.is_file():
        raise FileNotFoundError(f"missing card.toml under {root}")
    with card_path.open("rb") as fh:
        card = tomllib.load(fh)
    targets = card.get("targets", {})
    task = card.get("task", {})
    meta = card.get("metadata", {})
    assets = tuple(str(x) for x in targets.get("asset_ids", ()))
    horizons = tuple(int(x) for x in targets.get("horizons", ()))
    if not assets or not horizons:
        raise ValueError("card targets must declare non-empty asset_ids and horizons")
    if len(set(assets)) != len(assets) or len(set(horizons)) != len(horizons):
        raise ValueError("duplicate asset or horizon in card")
    target_type = str(targets.get("target_type", meta.get("target_type", "level")))
    target_frequency = str(targets.get("target_frequency", meta.get("target_frequency", "daily")))
    if target_type not in {"level", "log_return"}:
        raise ValueError(f"unsupported target_type={target_type!r}")
    return card, ForecastSpec(
        unit_id=str(task.get("id", root.name)),
        asof=asof,
        assets=assets,
        horizons=horizons,
        target_type=target_type,
        target_frequency=target_frequency,
        value_unit=str(targets.get("value_unit", "native")),
        target_panel=str(meta.get("asset_panel")) if meta.get("asset_panel") else None,
    )


def _candidate_panel_paths(panel_dir: Path, target_panel: str | None) -> list[Path]:
    paths: list[Path] = []
    if target_panel:
        direct = panel_dir / f"{target_panel}.parquet"
        if direct.is_file():
            paths.append(direct)
    for path in sorted(panel_dir.glob("*.parquet")):
        if path not in paths:
            paths.append(path)
    return paths


def _load_target_history(panel_dir: Path, spec: ForecastSpec) -> dict[str, pd.Series]:
    frames: list[pd.DataFrame] = []
    for path in _candidate_panel_paths(panel_dir, spec.target_panel):
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        asset_col = "asset" if "asset" in frame.columns else "asset_id" if "asset_id" in frame.columns else None
        if asset_col is None or "date" not in frame.columns or "value" not in frame.columns:
            continue
        sub = frame[["date", asset_col, "value"]].rename(columns={asset_col: "asset"}).copy()
        sub = sub[sub["asset"].astype(str).isin(spec.assets)]
        if not sub.empty:
            frames.append(sub)
        if set(sub["asset"].astype(str).unique()) >= set(spec.assets):
            break
    if not frames:
        raise ValueError("no target assets found in root parquet panels")
    data = pd.concat(frames, ignore_index=True)
    data["asset"] = data["asset"].astype(str)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    cutoff = pd.Timestamp(spec.asof)
    data = data[(data["date"].notna()) & (data["date"] <= cutoff) & np.isfinite(data["value"].to_numpy(float))]
    out: dict[str, pd.Series] = {}
    for asset in spec.assets:
        part = data[data["asset"] == asset].sort_values("date").drop_duplicates("date", keep="last")
        if part.empty:
            raise ValueError(f"target asset {asset!r} has no usable observations through asof")
        series = pd.Series(part["value"].to_numpy(float), index=pd.DatetimeIndex(part["date"]), name=asset)
        out[asset] = series
    return out


def _clean_participant_outputs(out_path: Path) -> None:
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in PARTICIPANT_OUTPUT_FILES:
        path = out_dir / name
        if path.exists():
            path.unlink()


def _load_best_effort_history(panel_dir: Path, spec: ForecastSpec) -> dict[str, pd.Series]:
    buckets: dict[str, list[pd.DataFrame]] = {asset: [] for asset in spec.assets}
    for path in _candidate_panel_paths(panel_dir, spec.target_panel):
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        asset_col = "asset" if "asset" in frame.columns else "asset_id" if "asset_id" in frame.columns else None
        if asset_col is None or "date" not in frame.columns or "value" not in frame.columns:
            continue
        try:
            sub = frame[["date", asset_col, "value"]].rename(columns={asset_col: "asset"}).copy()
            sub["asset"] = sub["asset"].astype(str)
            sub = sub[sub["asset"].isin(spec.assets)]
            if sub.empty:
                continue
            sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
            sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
            sub = sub[(sub["date"].notna()) & (sub["date"] <= pd.Timestamp(spec.asof))]
            for asset in spec.assets:
                part = sub[sub["asset"] == asset]
                if not part.empty:
                    buckets[asset].append(part[["date", "value"]])
        except Exception:
            continue
    out: dict[str, pd.Series] = {}
    for asset in spec.assets:
        if not buckets[asset]:
            out[asset] = pd.Series(dtype=float, name=asset)
            continue
        part = pd.concat(buckets[asset], ignore_index=True)
        part = part.sort_values("date").drop_duplicates("date", keep="last")
        values = pd.to_numeric(part["value"], errors="coerce").to_numpy(float)
        dates = pd.DatetimeIndex(part["date"])
        mask = np.isfinite(values)
        out[asset] = pd.Series(values[mask], index=dates[mask], name=asset)
    return out


def _valid_level_increments(series: pd.Series, monthly: bool) -> pd.Series:
    if len(series) < 2:
        return pd.Series(dtype=float, name=series.name)
    dates = pd.DatetimeIndex(series.index)
    gaps = dates.to_series().diff().dt.days.to_numpy(float)
    positive = gaps[np.isfinite(gaps) & (gaps > 0)]
    median_gap = float(np.median(positive)) if len(positive) else (30.0 if monthly else 1.0)
    gap_limit = max(75.0 if monthly else 5.0, 2.5 * median_gap)
    diff = series.diff()
    ok = np.isfinite(gaps) & (gaps <= gap_limit)
    vals = diff.to_numpy(float)
    mask = np.isfinite(vals) & ok
    return pd.Series(vals[mask], index=dates[mask], name=series.name)


def _innovation_series(history: dict[str, pd.Series], spec: ForecastSpec) -> dict[str, pd.Series]:
    monthly = spec.target_frequency == "monthly"
    out: dict[str, pd.Series] = {}
    for asset, series in history.items():
        if spec.target_type == "log_return":
            vals = series[np.isfinite(series.to_numpy(float))].astype(float)
            if (vals <= -1.0).any():
                raise ValueError("log_return history requires simple returns greater than -1")
            out[asset] = pd.Series(np.log1p(vals.to_numpy(float)), index=vals.index, name=asset)
        else:
            out[asset] = _valid_level_increments(series, monthly=monthly)
    return out


def _m0_level_step_frame(history: dict[str, pd.Series], assets: tuple[str, ...]) -> pd.DataFrame:
    columns: list[pd.Series] = []
    for asset in assets:
        series = history[asset].sort_index().iloc[-300:]
        if len(series) < 2:
            columns.append(pd.Series(dtype=float, name=asset))
            continue
        dates = pd.DatetimeIndex(series.index)
        gaps = dates.to_series().diff().dt.days.to_numpy(float)
        positive = gaps[np.isfinite(gaps) & (gaps > 0)]
        median_gap = float(np.median(positive)) if positive.size else 1.0
        gap_limit = max(10.0 * median_gap, 5.0)
        diffs = series.diff().to_numpy(float)
        mask = np.isfinite(diffs) & np.isfinite(gaps) & (gaps <= gap_limit)
        columns.append(pd.Series(diffs[mask], index=dates[mask], name=asset))
    if not columns:
        return pd.DataFrame()
    return pd.concat(columns, axis=1, join="inner").dropna()


def _estimate_level_m0_parameters(
    history: dict[str, pd.Series], spec: ForecastSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = _m0_level_step_frame(history, spec.assets)
    n_assets = len(spec.assets)
    if len(frame) < 2:
        innov = _innovation_series(history, spec)
        drifts: list[float] = []
        scales: list[float] = []
        for asset in spec.assets:
            x = innov[asset].to_numpy(float)
            x = x[np.isfinite(x)]
            drifts.append(float(np.mean(x)) if x.size else 0.0)
            scales.append(float(np.std(x, ddof=1)) if x.size > 1 else 1e-6)
        return np.asarray(drifts), np.maximum(np.asarray(scales), 1e-9), np.eye(n_assets)

    values = frame.loc[:, list(spec.assets)].to_numpy(float)
    drifts = np.mean(values, axis=0)
    if n_assets == 1:
        scale = float(np.std(values[:, 0], ddof=1))
        return np.asarray(drifts), np.asarray([max(scale, 1e-9)]), np.ones((1, 1))

    cov = np.asarray(np.cov(values, rowvar=False, ddof=1), dtype=float)
    cov[~np.isfinite(cov)] = 0.0
    scales = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    corr = cov / np.outer(scales, scales)
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, 1.0)
    return np.asarray(drifts), np.maximum(scales, 1e-9), _nearest_correlation(corr)


def _robust_location_scale(x: np.ndarray, monthly: bool, target_type: str) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1e-6
    window = min(x.size, 60 if monthly else 300)
    y = x[-window:]
    med = float(np.median(y))
    mad = float(1.4826 * np.median(np.abs(y - med)))
    sample = float(np.std(y, ddof=1)) if y.size > 1 else 0.0
    alpha = 0.90 if monthly else 0.97
    w = alpha ** np.arange(y.size - 1, -1, -1, dtype=float)
    w /= w.sum()
    ew_mean = float(np.sum(w * y))
    ew_var = float(np.sum(w * (y - ew_mean) ** 2))
    ew_scale = math.sqrt(max(ew_var, 0.0))
    positive_scales = [v for v in (mad, sample, ew_scale) if np.isfinite(v) and v > 0]
    scale = float(np.median(positive_scales)) if positive_scales else max(abs(med) * 0.05, 1e-6)
    floor = max(np.finfo(float).eps, float(np.median(np.abs(y))) * 1e-6, 1e-9)
    scale = max(scale, floor)
    raw_center = 0.5 * ew_mean + 0.5 * med
    if monthly:
        shrink = 0.75
    elif target_type == "log_return":
        shrink = 0.20
    else:
        shrink = 0.10
    return shrink * raw_center, scale


def _nearest_correlation(corr: np.ndarray) -> np.ndarray:
    corr = np.asarray(corr, dtype=float)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    vals, vecs = np.linalg.eigh(corr)
    vals = np.clip(vals, 1e-6, None)
    psd = (vecs * vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(psd), 1e-12, None))
    psd = psd / np.outer(d, d)
    np.fill_diagonal(psd, 1.0)
    return psd


def _safe_log_innovations(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float, name=series.name)
    raw = pd.to_numeric(series, errors="coerce").to_numpy(float)
    mask = np.isfinite(raw) & (raw > -1.0)
    if not np.any(mask):
        return pd.Series(dtype=float, name=series.name)
    return pd.Series(np.log1p(raw[mask]), index=series.index[mask], name=series.name)


def _safe_scale(x: np.ndarray, default: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float(max(default, 1e-9))
    tail = x[-300:]
    med = float(np.median(tail))
    mad = float(1.4826 * np.median(np.abs(tail - med)))
    std = float(np.std(tail, ddof=1))
    candidates = [v for v in (mad, std) if np.isfinite(v) and v > 0]
    scale = float(np.median(candidates)) if candidates else float(default)
    return max(scale, float(default), 1e-9)


def _factor_safe_parameters(
    history: dict[str, pd.Series], spec: ForecastSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    innovations: list[pd.Series] = []
    drifts: list[float] = []
    scales: list[float] = []
    anchors = np.zeros(len(spec.assets), dtype=float)
    for asset in spec.assets:
        series = history.get(asset, pd.Series(dtype=float, name=asset))
        x = _safe_log_innovations(series)
        arr = x.to_numpy(float)
        scale = _safe_scale(arr, 0.01)
        center = float(np.median(arr[-120:])) if arr.size else 0.0
        drifts.append(0.05 * center)
        scales.append(scale)
        innovations.append(((x - drifts[-1]) / scale).rename(asset))

    n = len(spec.assets)
    corr = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            pair = pd.concat([innovations[i], innovations[j]], axis=1, join="inner").dropna().tail(500)
            value = 0.0
            if len(pair) >= 20:
                a = pair.iloc[:, 0].to_numpy(float)
                b = pair.iloc[:, 1].to_numpy(float)
                if np.std(a) > 1e-12 and np.std(b) > 1e-12:
                    candidate = float(np.corrcoef(a, b)[0, 1])
                    if np.isfinite(candidate):
                        value = candidate
            corr[i, j] = corr[j, i] = 0.40 * value
    corr = _nearest_correlation(corr)
    return np.asarray(drifts), np.asarray(scales), corr, anchors


def _universal_safe_parameters(
    history: dict[str, pd.Series], spec: ForecastSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    drifts: list[float] = []
    scales: list[float] = []
    anchors: list[float] = []
    for asset in spec.assets:
        series = history.get(asset, pd.Series(dtype=float, name=asset))
        if spec.target_type == "log_return":
            x = _safe_log_innovations(series).to_numpy(float)
            drifts.append(0.0)
            scales.append(_safe_scale(x, 0.01))
            anchors.append(0.0)
        else:
            finite = series[np.isfinite(series.to_numpy(float))] if not series.empty else series
            anchor = float(finite.iloc[-1]) if len(finite) else 0.0
            increments = _valid_level_increments(finite, monthly=spec.target_frequency == "monthly")
            x = increments.to_numpy(float)
            drift = float(np.mean(x[-300:])) if x.size else 0.0
            default = max(abs(anchor) * 0.005, 0.01)
            drifts.append(drift if np.isfinite(drift) else 0.0)
            scales.append(_safe_scale(x, default))
            anchors.append(anchor)
    n = len(spec.assets)
    return np.asarray(drifts), np.asarray(scales), np.eye(n), np.asarray(anchors)


def _estimate_parameters(innov: dict[str, pd.Series], spec: ForecastSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    monthly = spec.target_frequency == "monthly"
    drifts: list[float] = []
    scales: list[float] = []
    normalized: list[pd.Series] = []
    for asset in spec.assets:
        series = innov[asset]
        drift, scale = _robust_location_scale(series.to_numpy(float), monthly, spec.target_type)
        if spec.target_type == "log_return":
            scale *= 0.90
        drifts.append(drift)
        scales.append(scale)
        normalized.append(((series - drift) / scale).rename(asset))
    n = len(spec.assets)
    if n == 1:
        corr = np.ones((1, 1), dtype=float)
    else:
        aligned = pd.concat(normalized, axis=1).sort_index().tail(500)
        corr_df = aligned.corr(min_periods=20).reindex(index=spec.assets, columns=spec.assets)
        corr = corr_df.to_numpy(float)
        corr[~np.isfinite(corr)] = 0.0
        np.fill_diagonal(corr, 1.0)
        corr = 0.65 * corr + 0.35 * np.eye(n)
        corr = _nearest_correlation(corr)
    return np.asarray(drifts), np.asarray(scales), corr


def _month_index(value: str | pd.Timestamp) -> int:
    ts = pd.Timestamp(value)
    return int(ts.year * 12 + ts.month - 1)


def _load_forecast_spec(root: Path) -> dict[str, Any]:
    path = root / "forecast_spec.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        obj = json.load(fh)
    return obj if isinstance(obj, dict) else {}


def _monthly_steps(root: Path, spec: ForecastSpec, history: dict[str, pd.Series]) -> np.ndarray:
    meta = _load_forecast_spec(root)
    with (root / "card.toml").open("rb") as fh:
        card = tomllib.load(fh)
    periods: dict[tuple[str, int], int] = {}

    def add(asset: str, horizon: int, value: Any) -> None:
        key = (asset, horizon)
        period = _month_index(str(value))
        if key in periods and periods[key] != period:
            raise ValueError("conflicting monthly target periods")
        periods[key] = period

    for source in (meta, card):
        targets = source.get("targets", {}) if isinstance(source.get("targets", {}), dict) else {}
        declared_h = targets.get("horizons")
        declared_a = targets.get("asset_ids", list(spec.assets))
        for field in ("observation_periods", "target_dates"):
            values = targets.get(field)
            if values is None:
                continue
            if (
                not isinstance(values, list)
                or not isinstance(declared_h, list)
                or not isinstance(declared_a, list)
                or len(values) != len(declared_h)
            ):
                raise ValueError("monthly target metadata must align with declared horizons")
            for horizon, value in zip(declared_h, values, strict=True):
                for asset in declared_a:
                    if asset in spec.assets and int(horizon) in spec.horizons:
                        add(str(asset), int(horizon), value)

        questions = source.get("questions", [])
        if questions is None:
            questions = []
        if not isinstance(questions, list):
            raise ValueError("monthly questions metadata must be a list")
        for row in questions:
            if not isinstance(row, dict):
                raise ValueError("monthly question must be an object")
            asset = row.get("asset")
            horizon = row.get("horizon")
            if asset not in spec.assets or horizon not in spec.horizons:
                continue
            value = row.get("observation_period", row.get("target_date"))
            if value is not None:
                add(str(asset), int(horizon), value)

    expected = {(asset, horizon) for asset in spec.assets for horizon in spec.horizons}
    if set(periods) != expected:
        raise ValueError("monthly target periods are incomplete")

    steps = np.empty((len(spec.assets), len(spec.horizons)), dtype=int)
    for ai, asset in enumerate(spec.assets):
        anchor = _month_index(history[asset].index[-1])
        for hi, horizon in enumerate(spec.horizons):
            target = periods.get((asset, horizon))
            assert target is not None
            delta = target - anchor
            if delta < 1:
                raise ValueError("monthly target period must follow final panel observation month")
            steps[ai, hi] = delta
    return steps


def _step_matrix(root: Path, spec: ForecastSpec, history: dict[str, pd.Series]) -> np.ndarray:
    if spec.target_frequency == "monthly":
        return _monthly_steps(root, spec, history)
    arr = np.asarray(spec.horizons, dtype=int)
    if np.any(arr < 1):
        raise ValueError("horizons must be positive")
    return np.tile(arr, (len(spec.assets), 1))


def _emergency_step_matrix(root: Path, spec: ForecastSpec, history: dict[str, pd.Series]) -> np.ndarray:
    try:
        return _step_matrix(root, spec, history)
    except Exception:
        if spec.target_frequency == "monthly":
            base = np.asarray([max(1, int(round(h / 21.0))) for h in spec.horizons], dtype=int)
        else:
            base = np.asarray([max(1, int(h)) for h in spec.horizons], dtype=int)
        return np.tile(base, (len(spec.assets), 1))


def _simulate_draws(
    spec: ForecastSpec,
    history: dict[str, pd.Series],
    drifts: np.ndarray,
    scales: np.ndarray,
    corr: np.ndarray,
    steps: np.ndarray,
    n_draws: int,
) -> np.ndarray:
    max_steps = int(np.max(steps))
    n_assets = len(spec.assets)
    rng = np.random.default_rng(_stable_seed(spec.unit_id, spec.asof))
    try:
        chol = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        chol = np.linalg.cholesky(_nearest_correlation(corr) + 1e-8 * np.eye(n_assets))
    z = rng.standard_normal((n_draws, max_steps, n_assets))
    z = z @ chol.T
    if spec.target_type == "log_return" and all(len(history[a]) >= 30 for a in spec.assets):
        chi = rng.chisquare(STUDENT_DF, size=(n_draws, max_steps, 1))
        z = z * np.sqrt((STUDENT_DF - 2.0) / np.maximum(chi, 1e-12))
    shocks = drifts.reshape(1, 1, -1) + scales.reshape(1, 1, -1) * z
    csum = np.cumsum(shocks, axis=1)
    out = np.empty((n_draws, n_assets, len(spec.horizons)), dtype=float)
    for ai, asset in enumerate(spec.assets):
        anchor = float(history[asset].iloc[-1]) if spec.target_type == "level" else 0.0
        for hi, _h in enumerate(spec.horizons):
            s = int(steps[ai, hi])
            out[:, ai, hi] = anchor + csum[:, s - 1, ai]
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("non-finite simulated forecast")
    return out


def _simulate_gaussian_fallback(
    spec: ForecastSpec,
    drifts: np.ndarray,
    scales: np.ndarray,
    corr: np.ndarray,
    anchors: np.ndarray,
    steps: np.ndarray,
    n_draws: int,
    seed_suffix: str,
) -> np.ndarray:
    max_steps = max(1, int(np.max(steps)))
    n_assets = len(spec.assets)
    digest = hashlib.sha256(f"{_stable_seed(spec.unit_id, spec.asof)}|{seed_suffix}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big") % (2**32))
    try:
        chol = np.linalg.cholesky(_nearest_correlation(corr))
    except Exception:
        chol = np.eye(n_assets)
    z = rng.standard_normal((n_draws, max_steps, n_assets)) @ chol.T
    shocks = drifts.reshape(1, 1, -1) + scales.reshape(1, 1, -1) * z
    csum = np.cumsum(shocks, axis=1)
    out = np.empty((n_draws, n_assets, len(spec.horizons)), dtype=float)
    for ai in range(n_assets):
        for hi in range(len(spec.horizons)):
            s = max(1, int(steps[ai, hi]))
            out[:, ai, hi] = float(anchors[ai]) + csum[:, s - 1, ai]
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("non-finite fallback forecast")
    return out


def _write_outputs(
    out_path: Path,
    spec: ForecastSpec,
    draws: np.ndarray,
    method: str = "primary_v1_hybrid",
    rationale_body: str | None = None,
) -> None:
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    n_draws, n_assets, n_horizons = draws.shape
    rows = n_draws * n_assets * n_horizons
    draw_col = np.repeat(np.arange(n_draws, dtype=np.int32), n_assets * n_horizons)
    asset_pattern = np.repeat(np.asarray(spec.assets, dtype=object), n_horizons)
    asset_col = np.tile(asset_pattern, n_draws)
    horizon_col = np.tile(np.asarray(spec.horizons, dtype=np.int32), n_draws * n_assets)
    value_col = draws.reshape(rows)
    frame = pd.DataFrame({"draw": draw_col, "asset": asset_col, "horizon": horizon_col, "value": value_col})
    frame.to_parquet(out_path, index=False)
    meta = {
        "unit_id": spec.unit_id,
        "asof": spec.asof,
        "representation": "samples",
        "asset_ids": list(spec.assets),
        "horizons": list(spec.horizons),
        "n_draws": int(n_draws),
        "target": spec.target_type,
        "units": spec.value_unit,
        "rationale": {
            "file": "forecast_rationale.md",
            "method": method,
        },
    }
    (out_dir / "forecast_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if rationale_body is None:
        rationale_body = (
            "This submission uses only panel observations available through the supplied as-of date. "
            "For level targets it uses a trailing-300 joint Gaussian random walk with unshrunk mean "
            "steps, sample covariance, gap filtering, and a shared path across horizons. For cumulative "
            "log-return targets it converts decimal simple returns with log1p, shrinks recent drift, "
            "uses a 0.90 robust/EWMA scale multiplier and shrunk cross-asset correlation, then simulates "
            "finite-variance Student-t innovations from a zero anchor. No text or House-model call is used."
        )
    rationale = "# Forecast rationale\n\n" + rationale_body.strip() + "\n"
    (out_dir / "forecast_rationale.md").write_text(rationale, encoding="utf-8")


def forecast_unit(panels: str | Path, asof: str, out: str | Path, n_draws: int = N_DRAWS, text_dir: str | Path | None = None) -> Path:
    ctx = _resolve_input_context(panels)
    root = ctx.unit_root
    out_path = Path(out)
    if not (200 <= int(n_draws) <= 20000):
        raise ValueError("n_draws must be in [200, 20000]")
    _card, spec = _read_card(root, asof)
    _clean_participant_outputs(out_path)

    try:
        history = _load_target_history(ctx.panel_dir, spec)
        innov = _innovation_series(history, spec)
        if spec.target_type == "level":
            drifts, scales, corr = _estimate_level_m0_parameters(history, spec)
        else:
            drifts, scales, corr = _estimate_parameters(innov, spec)
        steps = _step_matrix(root, spec, history)
        draws = _simulate_draws(spec, history, drifts, scales, corr, steps, int(n_draws))
        hs = house_scenario.maybe_apply(
            draws,
            target_type=spec.target_type,
            target_frequency=spec.target_frequency,
            assets=spec.assets,
            horizons=spec.horizons,
            target_panel=spec.target_panel,
            asof=spec.asof,
            text_dir=text_dir if text_dir is not None else (ctx.unit_root / "text"),
        )
        if hs.applied:
            draws = hs.draws
            _write_outputs(
                out_path,
                spec,
                draws,
                method="primary_v1_hybrid_house_scenario_graph",
                rationale_body=(
                    "The numerical V2 forecast generated the marginal samples. Cutoff-safe supplied text was "
                    "read once by the organizer House model to construct an explicitly hypothetical low-rank "
                    "cross-asset scenario graph. The deterministic dependence transport preserves every "
                    "asset-by-horizon marginal sample multiset and changes only joint dependence."
                ),
            )
        else:
            _write_outputs(out_path, spec, draws, method="primary_v1_hybrid")
        return out_path
    except Exception:
        _clean_participant_outputs(out_path)

    is_factor_multiasset = (
        spec.target_type == "log_return"
        and spec.target_frequency == "daily"
        and len(spec.assets) > 1
        and spec.target_panel == "factors_daily"
    )
    if is_factor_multiasset:
        try:
            history = _load_best_effort_history(ctx.panel_dir, spec)
            drifts, scales, corr, anchors = _factor_safe_parameters(history, spec)
            steps = _emergency_step_matrix(root, spec, history)
            draws = _simulate_gaussian_fallback(
                spec, drifts, scales, corr, anchors, steps, int(n_draws), "factor-safe"
            )
            _write_outputs(
                out_path,
                spec,
                draws,
                method="factor_safe_joint",
                rationale_body=(
                    "The primary numerical path could not complete, so this unit used the crash-safe "
                    "multi-asset factor fallback. It filters invalid simple returns before log1p, uses "
                    "strongly shrunk drift and pairwise-complete correlations, repairs dependence to a "
                    "PSD correlation matrix, and simulates coherent Gaussian paths across horizons."
                ),
            )
            return out_path
        except Exception:
            _clean_participant_outputs(out_path)

    try:
        history = _load_best_effort_history(ctx.panel_dir, spec)
        drifts, scales, corr, anchors = _universal_safe_parameters(history, spec)
        steps = _emergency_step_matrix(root, spec, history)
        draws = _simulate_gaussian_fallback(
            spec, drifts, scales, corr, anchors, steps, int(n_draws), "universal-safe"
        )
        _write_outputs(
            out_path,
            spec,
            draws,
            method="universal_safe_fallback",
            rationale_body=(
                "The primary numerical path could not complete, so this unit used the universal crash-safe "
                "fallback. It preserves the declared asset and horizon grid, uses finite best-effort historical "
                "anchors and scales, and simulates independent Gaussian paths with deterministic randomness."
            ),
        )
        return out_path
    except Exception:
        _clean_participant_outputs(out_path)

    n_assets = len(spec.assets)
    n_horizons = len(spec.horizons)
    anchors = np.zeros(n_assets, dtype=float)
    if spec.target_type == "level":
        try:
            last_history = _load_best_effort_history(ctx.panel_dir, spec)
            for ai, asset in enumerate(spec.assets):
                series = last_history.get(asset, pd.Series(dtype=float))
                if len(series):
                    value = float(series.iloc[-1])
                    if np.isfinite(value):
                        anchors[ai] = value
        except Exception:
            pass
    digest = hashlib.sha256(f"{_stable_seed(spec.unit_id, spec.asof)}|contract-last-resort".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big") % (2**32))
    draws = np.empty((int(n_draws), n_assets, n_horizons), dtype=float)
    for ai in range(n_assets):
        base_scale = 0.01 if spec.target_type == "log_return" else max(abs(anchors[ai]) * 0.0025, 0.01)
        for hi, horizon in enumerate(spec.horizons):
            effective_steps = max(1, int(round(horizon / 21.0))) if spec.target_frequency == "monthly" else max(1, int(horizon))
            draws[:, ai, hi] = anchors[ai] + base_scale * math.sqrt(effective_steps) * rng.standard_normal(int(n_draws))
    _write_outputs(
        out_path,
        spec,
        draws,
        method="contract_last_resort",
        rationale_body=(
            "All richer numerical paths failed, so this unit used the final contract-preserving fallback. "
            "It uses only the parsed task contract, finite protected anchors when available, and a small "
            "deterministic independent Gaussian dispersion to guarantee a valid forecast grid."
        ),
    )
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forecast")
    parser.add_argument("--panels", required=True)
    parser.add_argument("--text", required=False, default=None)
    parser.add_argument("--asof", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    forecast_unit(args.panels, args.asof, args.out, args.draws, args.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
