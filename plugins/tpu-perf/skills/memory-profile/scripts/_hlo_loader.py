"""
HLO buffer-assignment loader for tpu-perf:memory-profile.

Reads `*.hlo_proto.pb` files from the profile directory. These carry the
compile-time `BufferAssignmentProto` that TensorBoard's Memory Viewer
displays, and they are the authoritative source of HBM peak: the runtime
allocator events on /host:CPU are routinely truncated by the trace window
and miss long-lived buffers (weights / optimizer state) entirely.

Public surface:
    HloAnalysis              dataclass returned by analyse(profile_dir)
    analyse(profile_dir)     find largest hlo_proto.pb, parse, sweep
                             entry-computation schedule, return analysis
"""
from __future__ import annotations

import dataclasses
import pathlib
import sys
from collections import Counter, defaultdict
from typing import Optional

# Reuse comm-analysis's vendored hlo.proto — same schema across skills.
_HERE = pathlib.Path(__file__).resolve().parent
_HLO_PROTO = (
    _HERE.parent.parent / "comm-analysis" / "scripts" / "_proto"
)
sys.path.insert(0, str(_HLO_PROTO))
import hlo_pb2  # noqa: E402


@dataclasses.dataclass(slots=True)
class HloAliveBuffer:
    """One logical buffer alive at the static peak moment."""
    logical_buffer_id: int
    size_bytes: int
    allocation_index: int
    offset_in_allocation: int
    instruction_id: int
    instruction_name: str
    opcode: str
    op_name: str  # OpMetadata.op_name (JAXPR call site)
    shape_index: list[int]


@dataclasses.dataclass(slots=True)
class HloAnalysis:
    hlo_proto_path: str
    module_name: str

    # --- compile-time totals (these are what Memory Viewer shows) ----------
    static_peak_bytes: int          # sum of buffer_allocations[*].size
    entry_param_bytes: int          # entry params + tuple flag
    constant_bytes: int
    thread_local_bytes: int
    temp_pool_bytes: int            # the largest non-thread-local non-param non-const allocation
    temp_pool_alloc_index: int
    n_logical_buffers: int
    n_buffer_allocations: int

    # --- entry-schedule sweep (peak moment) --------------------------------
    # When the entry schedule is available we sweep instructions in order and
    # locate the moment where Σ live-buffer-sizes is maximal. This is the
    # static analogue of the runtime "alive_at_peak" set.
    schedule_present: bool
    entry_schedule_length: int
    peak_schedule_pos: int
    peak_instruction_id: int
    peak_instruction_name: str
    peak_instruction_opcode: str
    peak_instruction_op_name: str
    peak_alive_bytes: int           # Σ over alive lbs at peak pos (entry-level)
    peak_alive_buffers: list[HloAliveBuffer]
    n_subcomputation_lbs_skipped: int  # lbs whose def is inside while/fusion bodies

    # --- always-alive (offset-sweep over the temp pool) --------------------
    # Bytes inside the temp pool that are owned by exactly one logical buffer
    # in the address space — those bytes are alive at every schedule position.
    always_alive_bytes: int
    always_alive_buffers: list[HloAliveBuffer]   # top owners

    # --- top allocations (informational) -----------------------------------
    top_allocations: list[dict]


def _find_largest_hlo_proto(profile_dir: pathlib.Path) -> Optional[pathlib.Path]:
    """Return the largest *.hlo_proto.pb in the directory, or None.

    XLA emits one hlo_proto.pb per compiled jit. The largest is almost always
    `jit_train_step(*).hlo_proto.pb` — the small ones are reshard/eval/utility
    jits whose buffer assignment is uninteresting. memory-profile only needs
    the train-step module.
    """
    pbs = sorted(profile_dir.glob("*.hlo_proto.pb"), key=lambda p: p.stat().st_size)
    return pbs[-1] if pbs else None


def _build_alive_buffer(
    lb_id: int,
    lb_size: int,
    alloc_index: int,
    offset_in_alloc: int,
    instructions: dict[int, "hlo_pb2.HloInstructionProto"],
    location: "hlo_pb2.LogicalBufferProto.Location",
) -> HloAliveBuffer:
    iid = location.instruction_id
    ins = instructions.get(iid)
    return HloAliveBuffer(
        logical_buffer_id=lb_id,
        size_bytes=lb_size,
        allocation_index=alloc_index,
        offset_in_allocation=offset_in_alloc,
        instruction_id=iid,
        instruction_name=ins.name if ins else "",
        opcode=ins.opcode if ins else "",
        op_name=(ins.metadata.op_name if ins else "") or "",
        shape_index=list(location.shape_index),
    )


def _entry_schedule_sweep(
    big: "hlo_pb2.BufferAllocationProto",
    instructions: dict[int, "hlo_pb2.HloInstructionProto"],
    lb_by_id: dict[int, "hlo_pb2.LogicalBufferProto"],
    lb_alloc_offset: dict[int, tuple[int, int]],
    entry_seq: list[int],
) -> tuple[int, int, set[int], int]:
    """Sweep the entry computation schedule and find peak among alloc[big]'s lbs.

    Each logical buffer's lifetime is [def_pos, last_use_pos]; def_pos is the
    schedule index of its defining instruction, last_use_pos is the maximum
    schedule index of any instruction that takes the defining instruction's
    output as an operand. Buffers whose defining instruction lives in a
    sub-computation (while body / fusion / scan body) cannot be placed on
    the entry schedule and are reported via n_skipped.

    Returns (peak_bytes, peak_pos, peak_alive_lb_ids, n_skipped).
    """
    pos_of = {iid: i for i, iid in enumerate(entry_seq)}

    # use_positions[def_inst_id] = list of consumer schedule positions
    uses_of: dict[int, list[int]] = defaultdict(list)
    for iid in entry_seq:
        ins = instructions.get(iid)
        if ins is None:
            continue
        p = pos_of[iid]
        for op_id in ins.operand_ids:
            uses_of[op_id].append(p)

    intervals = []
    n_skipped = 0
    for asgn in big.assigned:
        lb = lb_by_id.get(asgn.logical_buffer_id)
        if lb is None or not lb.HasField("defined_at"):
            n_skipped += 1
            continue
        def_inst_id = lb.defined_at.instruction_id
        if def_inst_id not in pos_of:
            # defined inside a sub-computation; can't schedule at entry level
            n_skipped += 1
            continue
        dp = pos_of[def_inst_id]
        us = uses_of.get(def_inst_id)
        lp = max(us) if us else dp
        intervals.append((dp, lp, asgn.logical_buffer_id, lb.size))

    events = []
    for dp, lp, lid, sz in intervals:
        events.append((dp, 0, lid, sz))
        events.append((lp + 1, 1, lid, sz))
    events.sort(key=lambda e: (e[0], e[1]))

    stack = 0
    peak = 0
    peak_pos = 0
    peak_alive: set[int] = set()
    alive: dict[int, int] = {}
    for x, kind, lid, sz in events:
        if kind == 0:
            alive[lid] = sz
            stack += sz
            if stack > peak:
                peak = stack
                peak_pos = x
                peak_alive = set(alive.keys())
        else:
            stack -= alive.pop(lid, 0)

    return peak, peak_pos, peak_alive, n_skipped


def _always_alive_sweep(
    big: "hlo_pb2.BufferAllocationProto",
    lb_by_id: dict[int, "hlo_pb2.LogicalBufferProto"],
) -> tuple[int, dict[int, int]]:
    """Find offset regions inside `big` owned by exactly one logical buffer.

    Inside a single allocation, two logical buffers can share an (offset,size)
    range only if XLA's scheduler proved their lifetimes are disjoint. So a
    region of the address space that is assigned to exactly one logical
    buffer is necessarily alive at every schedule position — these are the
    static-residency bytes a remat policy cannot eliminate.

    Returns (total_unique_bytes, {lb_id: unique_bytes_owned}).
    """
    events: list[tuple[int, int, int]] = []  # (x, kind, lb_id)
    for asgn in big.assigned:
        events.append((asgn.offset, 0, asgn.logical_buffer_id))
        events.append((asgn.offset + asgn.size, 1, asgn.logical_buffer_id))
    events.sort(key=lambda e: (e[0], e[1]))

    active: set[int] = set()
    last_x = 0
    unique_bytes: dict[int, int] = defaultdict(int)
    for x, kind, lid in events:
        if last_x < x and len(active) == 1:
            (only,) = active
            unique_bytes[only] += (x - last_x)
        if kind == 0:
            active.add(lid)
        else:
            active.discard(lid)
        last_x = x

    total = sum(unique_bytes.values())
    return total, dict(unique_bytes)


def analyse(profile_dir: str | pathlib.Path) -> Optional[HloAnalysis]:
    """Run static buffer-assignment analysis on the largest train-step HLO.

    Returns None if no `*.hlo_proto.pb` is present.
    """
    profile_path = pathlib.Path(profile_dir)
    hlo_path = _find_largest_hlo_proto(profile_path)
    if hlo_path is None:
        return None

    hp = hlo_pb2.HloProto()
    with open(hlo_path, "rb") as f:
        hp.ParseFromString(f.read())

    ba = hp.buffer_assignment
    mod = hp.hlo_module

    instructions = {ins.id: ins for c in mod.computations for ins in c.instructions}
    lb_by_id = {lb.id: lb for lb in ba.logical_buffers}

    # Classify allocations.
    entry_param_bytes = 0
    constant_bytes = 0
    thread_local_bytes = 0
    temp_pool_bytes = 0
    temp_pool_idx = -1
    static_peak_bytes = 0
    for a in ba.buffer_allocations:
        static_peak_bytes += a.size
        if a.is_entry_computation_parameter:
            entry_param_bytes += a.size
        elif a.is_constant:
            constant_bytes += a.size
        elif a.is_thread_local:
            thread_local_bytes += a.size
        else:
            if a.size > temp_pool_bytes:
                temp_pool_bytes = a.size
                temp_pool_idx = a.index

    # Map lb_id -> (allocation_index, offset_in_alloc) for the temp pool.
    big = ba.buffer_allocations[temp_pool_idx] if temp_pool_idx >= 0 else None
    lb_alloc_offset: dict[int, tuple[int, int]] = {}
    if big is not None:
        for asgn in big.assigned:
            lb_alloc_offset[asgn.logical_buffer_id] = (big.index, asgn.offset)

    # --- entry-schedule sweep --------------------------------------------
    entry_id = mod.entry_computation_id
    schedule_present = (
        mod.HasField("schedule") if mod.DESCRIPTOR.fields_by_name["schedule"].has_presence
        else len(mod.schedule.sequences) > 0
    )
    entry_seq: list[int] = []
    if schedule_present and entry_id in mod.schedule.sequences:
        entry_seq = list(mod.schedule.sequences[entry_id].instruction_ids)

    peak_alive_bytes = 0
    peak_pos = 0
    peak_alive_ids: set[int] = set()
    n_skipped = 0
    if big is not None and entry_seq:
        peak_alive_bytes, peak_pos, peak_alive_ids, n_skipped = _entry_schedule_sweep(
            big, instructions, lb_by_id, lb_alloc_offset, entry_seq,
        )

    peak_inst_id = entry_seq[peak_pos] if entry_seq and peak_pos < len(entry_seq) else 0
    peak_inst = instructions.get(peak_inst_id)

    peak_alive_buffers = []
    for lid in peak_alive_ids:
        lb = lb_by_id[lid]
        ao = lb_alloc_offset.get(lid, (-1, -1))
        peak_alive_buffers.append(_build_alive_buffer(
            lid, lb.size, ao[0], ao[1], instructions, lb.defined_at,
        ))
    peak_alive_buffers.sort(key=lambda b: -b.size_bytes)

    # --- always-alive sweep ----------------------------------------------
    always_alive_bytes = 0
    always_alive_buffers: list[HloAliveBuffer] = []
    if big is not None:
        always_alive_bytes, unique_per_lb = _always_alive_sweep(big, lb_by_id)
        for lid, bts in sorted(unique_per_lb.items(), key=lambda x: -x[1])[:64]:
            lb = lb_by_id.get(lid)
            if lb is None or not lb.HasField("defined_at"):
                continue
            ao = lb_alloc_offset.get(lid, (-1, -1))
            buf = _build_alive_buffer(lid, lb.size, ao[0], ao[1], instructions, lb.defined_at)
            buf.size_bytes = bts  # overwrite size with "always-alive bytes for this lb"
            always_alive_buffers.append(buf)

    # --- top allocations (informational) ---------------------------------
    top_allocs = []
    for a in sorted(ba.buffer_allocations, key=lambda a: -a.size)[:15]:
        flags = []
        if a.is_entry_computation_parameter:
            flags.append(f"entry_param[{a.parameter_number}]")
        if a.is_constant:
            flags.append("const")
        if a.is_thread_local:
            flags.append("thread_local")
        if a.is_tuple:
            flags.append("tuple")
        if a.maybe_live_out:
            flags.append("live_out")
        top_allocs.append({
            "index": a.index,
            "size_bytes": a.size,
            "n_assigned_lbs": len(a.assigned),
            "color": a.color,
            "flags": flags,
        })

    return HloAnalysis(
        hlo_proto_path=str(hlo_path),
        module_name=mod.name,
        static_peak_bytes=static_peak_bytes,
        entry_param_bytes=entry_param_bytes,
        constant_bytes=constant_bytes,
        thread_local_bytes=thread_local_bytes,
        temp_pool_bytes=temp_pool_bytes,
        temp_pool_alloc_index=temp_pool_idx,
        n_logical_buffers=len(ba.logical_buffers),
        n_buffer_allocations=len(ba.buffer_allocations),
        schedule_present=bool(entry_seq),
        entry_schedule_length=len(entry_seq),
        peak_schedule_pos=peak_pos,
        peak_instruction_id=peak_inst_id,
        peak_instruction_name=peak_inst.name if peak_inst else "",
        peak_instruction_opcode=peak_inst.opcode if peak_inst else "",
        peak_instruction_op_name=(peak_inst.metadata.op_name if peak_inst else "") or "",
        peak_alive_bytes=peak_alive_bytes,
        peak_alive_buffers=peak_alive_buffers,
        n_subcomputation_lbs_skipped=n_skipped,
        always_alive_bytes=always_alive_bytes,
        always_alive_buffers=always_alive_buffers,
        top_allocations=top_allocs,
    )


def rollups_for_alive(buffers: list[HloAliveBuffer], total_bytes: int, top_k: int = 20) -> dict:
    """Build by_opcode / by_op_name rollups over a list of HloAliveBuffer.

    `total_bytes` is the denominator for `pct`. Pass peak_alive_bytes for
    peak-moment rollups, always_alive_bytes for static-residency rollups.
    """
    by_opcode = Counter()
    by_op_name = Counter()
    for b in buffers:
        by_opcode[b.opcode or "<no opcode>"] += b.size_bytes
        by_op_name[(b.op_name or "<no op_name>")[:120]] += b.size_bytes

    def _to_list(c: Counter, k: int) -> list[dict]:
        items = sorted(c.items(), key=lambda x: -x[1])
        head = items[:k]
        tail = items[k:]
        out = [
            {"key": key, "total_bytes": v,
             "pct": (100.0 * v / total_bytes) if total_bytes else 0.0}
            for key, v in head
        ]
        if tail:
            out.append({
                "key": "<other>", "total_bytes": sum(v for _, v in tail),
                "n_keys": len(tail),
                "pct": (100.0 * sum(v for _, v in tail) / total_bytes) if total_bytes else 0.0,
            })
        return out

    return {
        "by_opcode": _to_list(by_opcode, top_k),
        "by_op_name": _to_list(by_op_name, top_k),
    }
