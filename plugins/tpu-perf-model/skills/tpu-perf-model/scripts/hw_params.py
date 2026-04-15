#!/usr/bin/env python3
"""TPU v7x hardware parameters.

All values are for a single TPU v7x chip.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TPUParams:
    """Hardware parameters for a TPU generation."""
    name: str
    hbm_capacity_bytes: int
    hbm_bw_bytes_per_sec: float
    vmem_capacity_bytes: int
    spr_count: int
    vpr_count: int
    vpr_size_bytes: int
    mxu_peak_flops: float
    mxu_count: int
    dma_engine_count: int
    vpu_count: int
    alignment: int

    @property
    def vpr_total_bytes(self) -> int:
        return self.vpr_count * self.vpr_size_bytes

    @property
    def vpr_lane_count(self) -> int:
        return self.vpr_size_bytes // 2

    @property
    def reg_group_count(self) -> int:
        return self.vpr_count

    @property
    def ridge_point(self) -> float:
        return self.mxu_peak_flops / self.hbm_bw_bytes_per_sec


TPU_V7X = TPUParams(
    name="v7x",
    hbm_capacity_bytes=192 * 1024**3,
    hbm_bw_bytes_per_sec=3690e9,
    vmem_capacity_bytes=64 * 1024**2,
    spr_count=4096,
    vpr_count=32,
    vpr_size_bytes=8 * 128 * 4,
    mxu_peak_flops=2307e12,
    mxu_count=2,
    dma_engine_count=1,
    vpu_count=1,
    alignment=128,
)


DTYPE_BYTES = {
    "bf16": 2,
    "f16": 2,
    "f32": 4,
    "int8": 1,
    "int32": 4,
}


def dtype_bytes(dtype: str) -> int:
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"Unknown dtype: {dtype}. Supported: {list(DTYPE_BYTES.keys())}")
    return DTYPE_BYTES[dtype]
