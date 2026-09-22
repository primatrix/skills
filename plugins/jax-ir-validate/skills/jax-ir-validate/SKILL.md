---
name: jax-ir-validate
description: Validate JAX compiler source and IR from files or logs with deterministic scripts. Use for JAX/Pallas Python source, Jaxpr text, StableHLO and Mosaic TPU MLIR syntax, SSA/type errors, and parser diagnostics; not numerical correctness or performance analysis.
---

# JAX source and IR validation

Each script is a complete standalone file. Copy only the matching `.py` file
to any directory and run it there; no other skill files, sibling modules or
repository checkout are needed. The chosen interpreter still needs the external
dependencies listed in the relevant contract. Do not send Python or Jaxpr text
to an MLIR parser.

| Input | Script |
|---|---|
| Jaxpr text | `scripts/validate_jaxpr.py` |
| Pallas kernel source | `scripts/validate_pallas_kernel.py` |
| JAX Python source | `scripts/validate_jax_code.py` |
| StableHLO textual module | `scripts/validate_stablehlo_ir.py` |
| Mosaic TPU textual module | `scripts/validate_mosaic_ir.py` |

Run `python /path/to/copied_script.py INPUT --json`; use `-` for stdin. Preserve
source input and report the script's exit code, checked stages, original source
locations and tool versions. An unsupported format or missing dependency is not
evidence of invalid IR. Never report an unperformed check as passed.

The default `--input-format raw` validates the entire input. For logs, explicitly
select fenced blocks with `--input-format log`, or a complete unit with
`--lines START:END` (inclusive, 1-based). Never silently discard malformed input
to obtain a passing fragment. In log/range mode, success covers only the selected
units; content outside them is not validated.

Read [code provenance and upstream APIs](references/upstream.md) when explaining
where the implementation comes from. These are project-authored wrappers around
existing APIs, not copied official validator scripts. The Jaxpr text adapter and
Pallas target discovery remain bounded custom code.

Read [the input and MLIR contract](references/mlir.md) for dependencies, log
prefix removal, supported formats and examples. Parse + verifier success covers
public IR constraints only; it does not establish backend compilation, execution,
numerical correctness, race freedom or performance.

For Python syntax or explicit trusted-local tracing, read [the Python contract](references/python.md). Default syntax validation never executes source. Tracing requires an explicit request and complete calling context.

For Pallas target selection and real pallas_call tracing, read [the Pallas contract](references/pallas.md). Reuse the supplied wrapper rather than guessing Ref/grid/BlockSpec configuration.

For Jaxpr, read [the versioned grammar and verifier contract](references/jaxpr.md). This is a bounded text parser; report unsupported syntax/parameters as unsupported, never as a successful check.
