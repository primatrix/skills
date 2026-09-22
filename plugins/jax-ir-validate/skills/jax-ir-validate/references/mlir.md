# MLIR input contract

Each listed script can be copied and run as a single file. Paths below use the
skill directory for convenience; no sibling Python modules or reference documents
are required at runtime. External Python/JAX dependencies still apply.

Tested with Python 3.12 and CPU `jax==0.10.0`, `jaxlib==0.10.0`.
Install these in a dedicated environment with `python -m pip install 'jax[cpu]==0.10.0'`.
The scripts use the matched JAX MLIR bindings, `Module.parse`, and
`Operation.parse` for empty-module disambiguation, and `operation.verify`;
Mosaic additionally registers the TPU dialect. Unknown
unregistered dialects are rejected. No transformation passes or input computation
are run. Other JAX versions are not guaranteed compatible; reports record the
actual versions and binding failures are environment errors.

```sh
python <skill>/scripts/validate_stablehlo_ir.py dump.txt --json
cat run.log | python <skill>/scripts/validate_mosaic_ir.py - --input-format log --json
python <skill>/scripts/validate_stablehlo_ir.py run.log --lines 20:45 --strip-prefix '^\[INFO\] '
```

Inputs are UTF-8, at most 20 MiB. The default `--input-format raw` passes the
entire input to MLIR, including trailing text. Generic quoted module syntax,
aliases and multiple operations are handled by the upstream parser; no custom
brace or alias scanner rewrites them. Binary MLIR/base64 custom-call payloads
are outside this text interface.

CRLF and CR line endings are normalized to LF before selection and validation.
Line ranges and source mapping use physical newline boundaries; form-feed and
Unicode line-separator characters inside source are not treated as new lines.

For mixed logs, choose one of these explicit selection modes:

- `--input-format log`: validate every matching fenced block (at most 256).
  StableHLO accepts `mlir`/`stablehlo`; Mosaic accepts `mlir`/`mosaic`/`tpu`.
  Use backtick or tilde fences of at least three characters. Each block must be
  self-contained, including its aliases. Use a longer outer fence if its content
  includes fence lines. Empty matching blocks, missing matches and unclosed
  fences fail. Text outside matching blocks is not validated.
- `--lines START:END`: validate exactly this inclusive 1-based line range.
  This option requires raw mode. The caller must select a complete input unit;
  content outside that range is not validated.

Prefix stripping is
explicit: `--strip-prefix REGEX` matches at the start of each line and removes one
match; choose a regex that consumes exactly the logging prefix, not IR indentation.
Diagnostics map stripped columns and extracted lines back to the original input.
The bottom-level MLIR message retains its fragment-relative locations as well.

Exit 0 means every selected input unit parsed and verified. Exit 1 means selection,
input, environment, parse or verification failure; exit 2 is a CLI usage error.
`--json` includes input mode/range, per-fragment status, validator and diagnostics.
Empty, comment-only and truncated inputs fail; an explicit empty `module {}` is
valid. MLIR may wrap non-module operations in an implicit module. Validation is
against the registered dialects, not a claim that only
one dialect occurs in a module: mixed `func`/`arith`/StableHLO is normal.

See [code provenance and upstream APIs](upstream.md). The scripts call the
installed MLIR bindings; they do not embed the `stablehlo-opt` executable.

When MLIR reports a location embedded in the IR rather than a textual input offset, the summary points to the fragment start and retains that upstream location verbatim.
