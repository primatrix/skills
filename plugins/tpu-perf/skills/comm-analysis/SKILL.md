---
name: comm-analysis
description: Use when analyzing communication on a TPU pretraining profile — extracts every comm primitive (async + sync, TC + SparseCore), attributes axes via HLO replica_groups, computes per-row NCCL bus BW vs per-axis peak ICI BW (peak_link × k_torus_dims × directions_per_dim; TPUv7x: 200 GB/s bidir per link on a 3D torus; util% requires `--mesh-spec` with topology), and reports per-step compute/comm overlap. Builds on profile-anatomy.
---

# Communication Analysis

**回答语言要求：调用此 skill 时，所有面向用户的回答必须使用中文。**

Three reference scripts for analyzing the communication portion of a TPU
pretraining profile. Each script accepts a profile directory as `argv[1]`
(or `--profile-dir DIR`) and runs standalone with stdlib + `protobuf` +
optional `pyyaml` (only for `--mesh-spec`).

This skill builds on [`profile-anatomy`](../profile-anatomy/SKILL.md);
read that first for the xplane.pb / xplane.proto schema.

## 0. Reading rules (do these BEFORE quoting any number)

These are anti-foot-gun rules. They are NOT optional — every previous
analysis that ignored them produced a wrong attribution.

1. **First, check `unpaired_ratio` in the script header.** It's printed by
   `list_comm_primitives.py` and embedded in `overlap_report.py` table
   titles. If `unpaired_ratio > 50%`, the capture is "unpaired-dominated"
   and per-row `stall_ps` / `hidden_ps` are SENTINEL values (stall ≈ wall,
   hidden = 0). They are not data.
2. **Check the topology line at the top of `axis_bandwidth.py` output.**
   If it shows `topology=?`, the script ran without `--mesh-spec` (or the
   mesh-spec had no `topology:` entry). In that case `k_dims` defaults to
   1, `peak` and `util%` columns are SUPPRESSED, and the script prints
   `[error] topology unknown — util% is suppressed`. Do not work around
   this — re-run with `--mesh-spec` pointing at a YAML that has
   `topology: [X, Y, Z]`. **A util% computed against k_dims=1 would be
   wrong by 2-3× on any multi-dim collective.**
3. **Never quote a per-op `hidden%` or `exposed%` in unpaired-dominated
   captures.** The only authoritative per-op critical-path metric there is
   `NOT_cov_by_compute` (sweep-derived; in the new column on every table,
   and the sole sort key in `critical_path_comm.py`).
4. **Before recommending an optimization off a top-N list, check the same
   op's `cov%`** (= `1 − NOT_cov / wall`). If `cov% > 90%`, the op is NOT
   on the critical path regardless of how big its `wall_ps` looks.
5. **§8's "stall is degenerate" warning applies to EVERY stall-based
   table**, not just step totals. If §8 says stall is sentinel, every
   per-row stall column in every table in this skill is sentinel.
6. **TC and SC are separate timelines.** TC comm overlap must be computed
   against TC compute, SC comm overlap against SC compute. Cross-core
   overlap is meaningless because TC and SC don't compete for resources.
   The helpers `cc.merged_compute_by_core` / `cc.not_covered_by_compute`
   enforce this; never mix the two by hand.

## 1. What's covered

| Capability | Script |
|---|---|
| List every comm primitive (async + sync, TC + SC) with rich attributes | [`scripts/list_comm_primitives.py`](scripts/list_comm_primitives.py) |
| Per-axis bandwidth utilization (NCCL bus BW vs peak ICI link BW) | [`scripts/axis_bandwidth.py`](scripts/axis_bandwidth.py) |
| Per-step compute/comm overlap (sweep-line union) | [`scripts/overlap_report.py`](scripts/overlap_report.py) |
| Per-op critical-path attribution (TC and SC kept strictly separate) | [`scripts/critical_path_comm.py`](scripts/critical_path_comm.py) |

ICI only. DCN/megascale collectives are deferred to a future skill.

## 2. Per-primitive row schema

`list_comm_primitives.py` builds rows with these fields (also the `--json`
payload):

| Field | Source |
|---|---|
| `op_name` | `hlo_op` stat (canonicalized — `.call-start` / `.call-done` / `.start` / `.done` stripped) |
| `kind` | `AllReduce` / `AllGather` / `ReduceScatter` / `AllToAll` / `CollectivePermute` / `P2P` / `Copy` / `Unknown` |
| `mode` | `async` (Async XLA Ops) or `sync` (XLA Ops) |
| `core` | `TC`, `SC0`, or `SC1` |
| `axis` | logical or physical mesh axis (set by `axis_bandwidth.py`); `—` when unattributed |
| `group_size` | size of the first replica group (walks `collective_device_list` / `iota_collective_device_list` / legacy `replica_groups`) |
| `bidir` | `yes` / `no` heuristic from `(opcode, shape, replica_groups, sharding)` cluster having ≥2 distinct channel_ids |
| `bytes` | `bytes_accessed` from `XEventMetadata.stats` |
| `wall_ps` | `done.offset_ps + done.duration_ps − start.offset_ps` for paired async; `duration_ps` for sync |
| `stall_ps` | `done.device_duration_ps` for async; full `duration_ps` for sync (sync = always exposed). **Sentinel when unpaired** — see §0 rules. |
| `hidden_ps` | `wall_ps − stall_ps`. **Sentinel = 0 when unpaired** — see §0 rules. |
| `not_cov_ps` | Time the op was running while NO same-core compute was running. Computed by sweep against `merged_compute_by_core[row.core]`. **This is the only authoritative per-op critical-path metric on unpaired-dominated captures.** TC vs TC compute, SC vs SC compute — never crossed. |
| `bus_bw_gbps` | NCCL bus BW for this row's `kind` and `group_size`, computed by `axis_bandwidth.py` |
| `effective_bus_bw_gbps` | `2 × bus_bw_gbps` when `bidir == yes` (both ICI directions carry traffic), else `bus_bw_gbps` |
| `phys_dims` | Physical torus dims this collective contracts over (subset of `XYZ`), computed from replica_ids vs `topology` |
| `k_dims` | `len(phys_dims)` — number of physical torus dims contracted (1 for X-only, 2 for XY, 3 for XYZ) |
| `peak_axis_gbps` | Per-row theoretical peak: `peak_link × k_dims × directions` |
| `source` | `XEventMetadata.stats.source` / `source_stack`; falls back to HLO `OpMetadata.source_file:line` |
| `flow` | the `flow` XStat used to pair async events |
| `program_id` | `XEventMetadata.stats.program_id` |
| `channel_id` | from joined HLO instruction (int64 scalar; `0` ⇒ unset) |

## 3. Aggregation views

`list_comm_primitives.py --by {kind,source,op} [--sort-by stall|wall|not_cov|auto]`:

- `kind` (default): roll up by `(kind, axis, core)` with count, Σwall,
  Σstall, ΣNOT_cov, p50/p99 stall.
- `source`: roll up by source `file:line` — answers "which line of the
  model is causing comm?". Includes ΣNOT_cov.
- `op`: per individual `op_name`, with ΣNOT_cov.

**Sort key default is adaptive (`auto`)**: uses `Σstall` when
`unpaired_ratio ≤ 50%` (in that regime stall reliably equals exposed
time), and auto-switches to `Σ NOT_cov` when above. The script header
prints the active sort key and a `[warn]` if you forced `--sort-by stall`
on an unpaired-dominated capture. **Always read the header line first**
to know which key the table actually uses.

For pure per-op critical-path attribution (NOT_cov-only, with three
top-N tables), use `critical_path_comm.py` instead — see §1.

## 4. Bus-bandwidth formulas (NCCL/XLA convention)

| Kind | Bus BW |
|---|---|
| AllReduce | `2 × (N−1)/N × message_bytes / time` |
| AllGather | `(N−1)/N × output_bytes / time` |
| ReduceScatter | `(N−1)/N × input_bytes / time` |
| AllToAll | `(N−1)/N × message_bytes / time` |
| CollectivePermute / P2P | `message_bytes / time` |

`N = group_size`; `time = wall_ps` (in-flight, not stall). Bus_BW from
this formula is wire-level — it already reflects whatever directions of
the ring carried traffic. **There is NO additional ×2 multiplier for
"bidir" rows.** `bidir` in the per-collective table is a structural
label only (does the cluster have ≥2 distinct channel_ids?), not a
factor in the BW or util computation.

Per-axis peak is computed PER ROW from the physical torus dims the
collective contracts over:

```
peak_axis = peak_link_unidir × k_dims × directions_per_dim
```

- `peak_link_unidir` is the single-direction per-link peak (e.g. 100 GB/s
  on TPUv7x, since 200 GB/s bidirectional ÷ 2 directions = 100 GB/s/dir).
- `k_dims` ∈ {1,2,3} is the number of physical torus dims the collective's
  replica group spans (derived from replica_ids vs `topology`). A
  collective on `X` only has `k_dims=1`; one spanning `XY` has `k_dims=2`.
- `directions_per_dim` = 2 (every torus dim has two ring directions and
  both are available to a single collective). Configurable via
  mesh-spec `links_per_axis` for non-torus topologies.

Equivalently on TPUv7x: peak = 200 GB/s × `k_dims` (always — regardless
of the bidir label).

If `topology` is unknown, util% is suppressed (see §0 rule 2). Util% > 100
indicates a bug — please file it.

## 5. Peak-BW resolution order

Util% requires **both** a peak_link value and a topology. They are
resolved separately:

`peak_link` (unidirectional GB/s per ICI link):

1. xprof XStat (`peak_ici_*` / `peak_link_*` scanned across device, host, and Task Environment planes via `cc.peak_ici_link_gbps_from_xprof`).
2. `--mesh-spec` YAML `peak_link_gbps:`.
3. `--peak-ici-link-gbps N` flag (unidirectional GB/s per link).
4. `--tpu-version v7x` flag or `tpu_version: v7x` in mesh-spec ⇒ defaults
   `peak_link_gbps = 100` (unidir, = 200 GB/s bidir per link).
5. None ⇒ peak/util columns blank, `[warn] peak ICI BW unknown` printed.

`topology`:

1. `--mesh-spec` YAML `topology: [X, Y, Z]`.
2. None ⇒ peak/util columns blank, `[error] topology unknown` printed.
   This is a hard gate — there is no fallback, because k_dims=1 would
   silently undercount peak by 2-3× on multi-dim collectives.

`*op_stats.pb` does NOT carry ICI peak BW: its
`PerfEnv.peak_bws_giga_bytes_per_second` list is keyed by upstream
`MemBwType` (HBM_RW / SRAM_* / CMEM_* / VMEM_*) and has no ICI entry.

## 6. Mesh-spec YAML (required for util%)

```yaml
tpu_version: v7x                 # ⇒ peak_link defaults to 100 GB/s unidir
topology: [4, 4, 4]              # physical chip dims (X, Y, Z) — REQUIRED
                                 # for util%. TPUv7x is 3D torus.
axes:
  fsdp:  {dims: [Y, Z], size: 32}
  dp:    {dims: [X],    size: 4}
peak_link_gbps: 100              # unidirectional; overrides tpu_version default
links_per_axis: 2                # = directions per torus dim (2 for any torus)
```

Without `topology`, util% is suppressed (see §0 rule 2 and §5). Without a
mesh-spec, axes are still attributed via Shardy mesh metadata
(`axis_0=128`, `axis_1=128`, etc.) when present in HLO, but no peak/util
is computed and physical-dim attribution falls back to `—`.

### TPUv7x specifics

- 3D torus interconnect.
- 200 GB/s **bidirectional** per ICI link (= 100 GB/s per direction).
- 2 directions per torus dim ⇒ `links_per_axis = 2` (the default).
- A collective contracting over `k` of the 3 torus dims gets a
  per-axis peak of `100 × k × 2 = 200 × k` GB/s when bidir.

## 7. Sample invocations

```bash
python3 plugins/tpu-perf/skills/comm-analysis/scripts/list_comm_primitives.py \
  /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128

python3 plugins/tpu-perf/skills/comm-analysis/scripts/list_comm_primitives.py \
  /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 --by source

python3 plugins/tpu-perf/skills/comm-analysis/scripts/axis_bandwidth.py \
  /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 --mesh-spec mesh.yaml

# Without a mesh-spec, util% is suppressed (you'll only see bus_BW absolute
# values). To get util% you must pass --mesh-spec with a `topology:` entry.
python3 plugins/tpu-perf/skills/comm-analysis/scripts/axis_bandwidth.py \
  /tmp/tensorboard/tensorboard/plugins/profile/run --tpu-version v7x

python3 plugins/tpu-perf/skills/comm-analysis/scripts/overlap_report.py \
  /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128

# Per-op critical-path attribution (NOT_cov-sorted; TC and SC kept separate):
python3 plugins/tpu-perf/skills/comm-analysis/scripts/critical_path_comm.py \
  /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128

# Force a specific sort key on list_comm_primitives (override --sort-by auto):
python3 plugins/tpu-perf/skills/comm-analysis/scripts/list_comm_primitives.py \
  /tmp/tensorboard/tensorboard/plugins/profile/dp8_fsdp128 --by op --sort-by not_cov
```

## 8. Common gotchas

- **Async pairing uses `flow` (uint64), not `is_root`.** See profile-anatomy.
- **`pair_size=1` is observed in current captures.** Treated as fully
  exposed (`wall = stall`, `hidden = 0`); the row's `bidir` is "no" and
  it still appears in the Top-N exposed contributors table.
- **HLO module is optional but recommended.** Without `*.hlo_proto.pb`,
  axis attribution and the bidir heuristic degrade gracefully — `axis`
  stays `—` and a `[warn] N collective rows have no HLO counterpart`
  line is emitted.
- **HLO axis-attribution requires resolver chasing for two opcodes:**
  1. `call` wrappers (e.g. `all-reduce.3008.cloned.1`) — `_resolve_instr`
     follows `called_computation_ids` to the underlying collective and
     reads replica info from there.
  2. `*-done` events (e.g. `collective-permute-done.66`) — replica info
     lives on the matched `*-start`, not the `*-done` event xprof emits.
     `_resolve_instr` flips the suffix
     (`collective-permute-done.66` → `collective-permute-start.66`)
     and reads replica info from there.

  Rows that fall through both paths (e.g. `*-start` instructions whose
  `source_target_pairs` field is missing from the vendored HLO proto)
  are classified by the operation kind:
   - `collective-permute` / `send` / `recv` → axis is set to `p2p`.
     This is a structural label, not a peer-counting attribution.
   - Everything else is counted in the
     `[warn] N collective rows could not be axis-attributed` line.
- **Modern HLO uses `collective_device_list` / `iota_collective_device_list`,
  not legacy `replica_groups` (field 49).** The vendored helpers walk
  all three locations; the legacy field is empty in current captures.
- **SparseCore comm is reported in a separate sub-table** in
  `overlap_report.py` because SC and TC compute don't compete; mixing
  them would muddle the math.
- **`overlap_report.py` excludes wrapper / control-flow categories from
  the compute set.** The "XLA Ops" line on the TC device plane carries
  events whose `hlo_category` is `while`, `call`, `conditional`,
  `async-start`, `async-done`, `copy-start`, `copy-done`, etc. These are
  CONTAINERS or comm-completion wrappers, not real compute:
  - `while` / `call` / `conditional` cover their entire body — in
    MaxText captures the outer `jax.lax.while_loop` is one event whose
    `duration_ps` ≈ full step time (~hundreds of seconds).
  - `async-done` carries the full async-collective wall (mirror of
    Async XLA Ops) — the same time also appears on the comm side.
  Counting any of these as compute makes `compute_busy ≈ step_time` and
  forces every comm interval to overlap ⇒ fake 100% overlap and
  exposed≈0. The script now drops them in `_compute_intervals` and
  prints an `[info] excluded from compute (wrapper/container ops): …`
  line so the reader can sanity-check the fix on their own capture.
  When the per-step `compute(us)` column is suspiciously close to
  `step(us)`, look at that `[info]` line first — a missing wrapper
  category is the usual culprit.
- **The sweep-derived `exposed_comm` is authoritative** when it disagrees
  with `Σ done.device_duration_ps` by >5%; the metadata sum doesn't
  account for parallel streams.
- **`Σ stall_ps` (from `list_comm_primitives.py`) ≠ `Σ exposed_comm_ps` (from
  `overlap_report.py`) on captures where async events are flow-singletons.**
  When every Async XLA Op is unpaired (no flow start/done pair), each row's
  `stall_ps` falls back to `device_duration_ps`, which measures the comm
  engine's busy time — NOT the exposed (un-overlapped) slice. Multiple ICI
  links and SC lanes can be busy in parallel and overlap with TC compute,
  so summed `stall_ps` legitimately exceeds wall-clock. Always trust
  `overlap_report.py`'s sweep-derived `exposed_comm` for true exposed time.

  **Corollary — per-op rankings degrade too.** When `unpaired_ratio > 50%`,
  any "Top-N by stall" or "Top-N exposed contributors" table sourced from
  per-row `stall_ps` / `hidden_ps` is meaningless: every row will show
  `hidden = 0`, and the rank ordering reflects engine-busy time (which
  can run on parallel ICI links / SC lanes and overlap with TC compute),
  not critical-path exposure. To answer "which ops are on the critical
  path", use the `NOT_cov_by_compute` column — present on every table
  produced by the updated `list_comm_primitives.py` and
  `overlap_report.py`, and the only metric used by `critical_path_comm.py`.
  In this regime, `overlap_report.py` automatically renames its top-N
  table to "comm engine busy contributors (NOT exposed)" and re-sorts by
  NOT_cov. `list_comm_primitives.py --sort-by auto` (the default)
  auto-switches the sort key to `not_cov` and prints a `[warn]` line.

  This corollary applies to ALL stall-based reasoning, not just step
  totals — the same diagnosis (sentinel value) drives both. If you find
  yourself about to attribute a bottleneck to "this op has the largest
  Σstall", first read the `unpaired_ratio` header line and §0 rule 4.
- **`xplane_pb2.py` is reused from profile-anatomy** via
  `sys.path.insert`. Don't re-vendor it.
