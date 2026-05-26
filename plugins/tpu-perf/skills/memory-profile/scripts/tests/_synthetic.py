"""Builders for synthetic xplane_pb2.XSpace fixtures used in unit tests.

Mirrors the helper pattern in compute-breakdown/tests but tailored to
allocator events on /host:CPU."""
from __future__ import annotations

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_PROFILE_ANATOMY_PROTO = (
    _HERE.parent.parent.parent / "profile-anatomy" / "scripts" / "_proto"
)
sys.path.insert(0, str(_PROFILE_ANATOMY_PROTO))
import xplane_pb2  # noqa: E402


# id allocators: we manage stat/event metadata ids per-plane, never globally.
class PlaneBuilder:
    def __init__(self, xs: xplane_pb2.XSpace, name: str):
        self.plane = xs.planes.add()
        self.plane.name = name
        self._next_stat_id = 1
        self._next_event_id = 1
        self._stat_ids: dict[str, int] = {}
        self._event_ids: dict[str, int] = {}

    def stat_id(self, name: str) -> int:
        if name not in self._stat_ids:
            sm = self.plane.stat_metadata[self._next_stat_id]
            sm.id = self._next_stat_id
            sm.name = name
            self._stat_ids[name] = self._next_stat_id
            self._next_stat_id += 1
        return self._stat_ids[name]

    def event_id(self, name: str) -> int:
        if name not in self._event_ids:
            em = self.plane.event_metadata[self._next_event_id]
            em.id = self._next_event_id
            em.name = name
            self._event_ids[name] = self._next_event_id
            self._next_event_id += 1
        return self._event_ids[name]

    def add_line(self, name: str, timestamp_ns: int = 0) -> "LineBuilder":
        return LineBuilder(self, name, timestamp_ns)


class LineBuilder:
    def __init__(self, pb: PlaneBuilder, name: str, timestamp_ns: int):
        self.pb = pb
        self.line = pb.plane.lines.add()
        self.line.name = name
        self.line.timestamp_ns = timestamp_ns

    def add_event(self, name: str, *, offset_ps: int, duration_ps: int = 0,
                  stats: dict[str, int | float | str | bytes] | None = None) -> None:
        ev = self.line.events.add()
        ev.metadata_id = self.pb.event_id(name)
        ev.offset_ps = offset_ps
        ev.duration_ps = duration_ps
        for sname, val in (stats or {}).items():
            st = ev.stats.add()
            st.metadata_id = self.pb.stat_id(sname)
            if isinstance(val, bool):  # bool is int subclass; skip
                raise TypeError("bool stats not supported by xplane")
            elif isinstance(val, int):
                st.int64_value = val
            elif isinstance(val, float):
                st.double_value = val
            elif isinstance(val, str):
                st.str_value = val
            elif isinstance(val, bytes):
                st.bytes_value = val
            else:
                raise TypeError(f"unsupported stat type for {sname}: {type(val)}")


def make_alloc_event(line: LineBuilder, *, offset_ps: int, addr: int,
                     requested: int, allocation: int, pool_id: int = 0,
                     bytes_allocated: int, peak_bytes_in_use: int,
                     bytes_reserved: int, bytes_available: int = 0,
                     fragmentation: float = 0.0,
                     shape: str = "", tf_op: str = "",
                     data_type: int = 0) -> None:
    line.add_event(
        "MemoryAllocation", offset_ps=offset_ps,
        stats={
            "addr": addr, "id": pool_id,
            "requested_bytes": requested, "allocation_bytes": allocation,
            "bytes_allocated": bytes_allocated,
            "peak_bytes_in_use": peak_bytes_in_use,
            "bytes_reserved": bytes_reserved,
            "bytes_available": bytes_available,
            "fragmentation": fragmentation,
            "shape": shape, "tf_op": tf_op, "data_type": data_type,
        },
    )


def make_dealloc_event(line: LineBuilder, *, offset_ps: int, addr: int,
                       bytes_allocated: int, peak_bytes_in_use: int,
                       bytes_reserved: int, bytes_available: int = 0,
                       fragmentation: float = 0.0) -> None:
    line.add_event(
        "MemoryDeallocation", offset_ps=offset_ps,
        stats={
            "addr": addr,
            "bytes_allocated": bytes_allocated,
            "peak_bytes_in_use": peak_bytes_in_use,
            "bytes_reserved": bytes_reserved,
            "bytes_available": bytes_available,
            "fragmentation": fragmentation,
        },
    )


def make_xspace() -> xplane_pb2.XSpace:
    return xplane_pb2.XSpace()
