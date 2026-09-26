from __future__ import annotations

import argparse
from pathlib import Path

from . import m0_core
from . import v2_safe

N_DRAWS = 500


def forecast_unit(panels: str | Path, asof: str, out: str | Path, n_draws: int = N_DRAWS) -> Path:
    if not (200 <= int(n_draws) <= 20000):
        raise ValueError("n_draws must be in [200, 20000]")

    ctx = v2_safe._resolve_input_context(panels)
    root = ctx.unit_root
    out_path = Path(out)
    _card, spec = v2_safe._read_card(root, asof)
    v2_safe._clean_participant_outputs(out_path)

    try:
        draws, diag = m0_core.generate(
            root,
            spec.unit_id,
            spec.asof,
            list(spec.assets),
            list(spec.horizons),
            spec.target_type,
            spec.target_frequency,
            step_matrix=None,
            order="sorted",
            base_draws=500,
            output_draws=int(n_draws),
        )
        v2_safe._write_outputs(
            out_path,
            spec,
            draws,
            method="v5_m0_faithful",
            rationale_body=(
                "This forecast implements the published Track 2 M0 text-blind baseline procedure. "
                "It uses the trailing 300 observations per asset, gap-filtered and date-aligned step "
                "series, unshrunk step means and sample covariance, a shared Gaussian random-walk path "
                "across assets and horizons with covariance min(s_i,s_j)*Sigma, the published CRC32 "
                "unit seed, and the published monthly-panel step overrides. Level targets anchor at "
                "the last panel observation. Cumulative log-return targets use log1p of the supplied "
                "simple-return rows and a zero anchor. No text or House-model call is used."
            ),
        )
        return out_path
    except Exception:
        v2_safe._clean_participant_outputs(out_path)
        return v2_safe.forecast_unit(panels, asof, out, int(n_draws))


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
    forecast_unit(args.panels, args.asof, args.out, args.draws)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
