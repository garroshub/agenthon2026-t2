# Agenthon 2026 Track 2 V3-House-Probe

Development-only black-box probe built on the frozen V2-safe numerical baseline.

The V2-safe primary forecast is generated first. House is queried only for eligible single-asset daily U.S. Treasury level contracts (UST_2Y or UST_10Y), using one recent policy document and one request. Only strictly grounded future policy-path claims can shift the center. The fixed shift is 0.12 predictive standard deviations times the bounded semantic signal. Dispersion and tails are unchanged.

Any missing House configuration, request failure, parse/schema/span failure, stale/no policy document, conflicting semantics, or ineligible contract returns the exact V2-safe draws. Factor, FX, log-return, multi-asset, and fallback paths never use the House overlay.

Release tag: `ghcr.io/garroshub/agenthon2026-t2:v3-house-probe`.
