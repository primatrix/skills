"""TPU peak compute / HBM bandwidth table and CLI override resolver.

Per spec §8.1: per-device peaks = per-chip / 2 (v7x has 2 TensorCores;
`/device:TPU:N` is one TensorCore).
"""

# Per-device peaks. fp32 / fp16 are not officially listed → None.
BUILTIN_PEAKS = {
    "v7x": {
        "peak_tflops_bf16": 1153.5,
        "peak_tflops_fp8":  2307.0,
        "peak_tflops_fp32": None,
        "peak_tflops_fp16": None,
        "peak_hbm_gibps":   3690.0,
    },
}


def resolve_peaks(chip: str, *,
                  override_tflops_bf16: float | None = None,
                  override_tflops_fp8: float | None = None,
                  override_tflops_fp32: float | None = None,
                  override_tflops_fp16: float | None = None,
                  override_hbm_gibps: float | None = None) -> dict:
    """Return a peaks dict for `chip`, applying any CLI overrides.

    Returns shape:
        {
          "chip": str,
          "peak_tflops_bf16": float | None,
          "peak_tflops_fp8":  float | None,
          "peak_tflops_fp32": float | None,
          "peak_tflops_fp16": float | None,
          "peak_hbm_gibps":   float,
          "ridge_points":     {dtype: AI_at_ridge},
          "source":           "builtin v7x table" | "cli override",
          "unit":             "GiB/s (base-1024) per device",
        }
    """
    if chip not in BUILTIN_PEAKS:
        raise KeyError(f"unknown chip: {chip!r}")
    base = dict(BUILTIN_PEAKS[chip])

    overrides = {
        "peak_tflops_bf16": override_tflops_bf16,
        "peak_tflops_fp8":  override_tflops_fp8,
        "peak_tflops_fp32": override_tflops_fp32,
        "peak_tflops_fp16": override_tflops_fp16,
        "peak_hbm_gibps":   override_hbm_gibps,
    }
    any_override = False
    for key, val in overrides.items():
        if val is not None:
            base[key] = val
            any_override = True

    hbm = base["peak_hbm_gibps"]
    ridge_points = {}
    for dtype in ("bf16", "fp8", "fp32", "fp16"):
        peak = base[f"peak_tflops_{dtype}"]
        if peak is not None and hbm is not None and hbm > 0:
            ridge_points[dtype] = (peak * 1e12) / (hbm * (1024 ** 3))

    return {
        "chip":             chip,
        "peak_tflops_bf16": base["peak_tflops_bf16"],
        "peak_tflops_fp8":  base["peak_tflops_fp8"],
        "peak_tflops_fp32": base["peak_tflops_fp32"],
        "peak_tflops_fp16": base["peak_tflops_fp16"],
        "peak_hbm_gibps":   base["peak_hbm_gibps"],
        "ridge_points":     ridge_points,
        "source":           "cli override" if any_override else "builtin v7x table",
        "unit":             "GiB/s (base-1024) per device",
    }
