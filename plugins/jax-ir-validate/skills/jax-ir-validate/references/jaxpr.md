# Jaxpr text validation

Each listed script can be copied and run as a single file. Paths below use the
skill directory for convenience; no sibling Python modules or reference documents
are required at runtime. External Python/JAX dependencies still apply.

```sh
python <skill>/scripts/validate_jaxpr.py dump.txt --json
cat run.log | python <skill>/scripts/validate_jaxpr.py - --input-format log --strip-prefix '^\[IR\] ' --json
```

The text adapter requires **jax==0.10.0 and jaxlib==0.10.0** in an isolated CPU
environment. It rejects other versions rather than silently accepting a different
printer/primitive contract. This pin is specific to Jaxpr text; the other scripts
report their actual versions and have separate contracts.

## Supported language

A project-authored lexer and recursive-descent adapter consume complete, unaliased
`{ lambda consts; inputs. let equations in (outputs,) }` programs. In raw mode the
entire input must contain exactly one program; trailing junk fails. For logs,
use explicit `jaxpr` fenced blocks with `--input-format log`, or select a complete
program with `--lines START:END` in raw mode. Content outside selected units is
not validated. Preserve one complete program per block; nested programs are
parsed recursively, not counted as separate top-level successes. The adapter
accepts `#` and `//` line comments as extensions to the printer output.

- Static shape annotations: bool, bf16, f16/f32/f64, signed/unsigned integer
  8/16/32/64, c64/c128. Scalar numeric and boolean literals have explicit types;
  complex literal spelling, dynamic dimensions, Ref/key/extended avals and meshes
  are not supported. Finite numeric literals that overflow their declared type
  fail; explicitly spelled `inf`/`nan` are retained.
- Constants are typed lambda bindings. Their runtime values are not present in a
  normal dump and remain **not checked**. No dummy constant values are fabricated.
- SSA binding/use order, duplicate names, types, tuple outputs and `_` discard
  results are checked. Nested cond/scan/while programs are recursively verified.
- Arithmetic/comparison/elementwise primitives, reductions, reshape, broadcast,
  squeeze, transpose, conversion, concatenate, slice, select_n, iota, rev,
  stop_gradient, integer_pow, default-precision dot_general, cond, scan and while
  are supported. The exact allowlist and accepted parameter keys are `PRIMITIVES`
  and `PARAMETERS` in `scripts/validate_jaxpr.py`.
- Parameters may be scalar values, None, dtype names, tuples/lists and nested
  Jaxprs. Unknown keys are rejected even if the upstream abstract evaluator would
  ignore them. Non-None sharding and dot precision are unsupported.

After text parsing, the script builds actual JAX Vars/Literals/Jaxprs, invokes the
upstream `jax._src.core.check_eqn` (including primitive abstract evaluation and
nested Jaxpr checks), then `jax._src.core.check_jaxpr`. It never
executes the represented computation. Only printer-elided defaults with known
0.10 rules are restored: conversion sharding=None; broadcast result shape,
empty broadcast dimensions and sharding=None; default dot precision/preferred
dtype/out_sharding. Sharded type syntax is rejected, so no mesh is invented.

## Unsupported and incomplete inputs

Shared top-level pretty-printer aliases, jit/pjit wrapper objects, custom or
effectful primitives, opaque Python parameters, nested closed-Jaxpr constants,
truncated/abbreviated dumps and unsupported versions fail with a nonzero exit.
Unknown syntax is not repaired. Unsupported does not mean the original Jaxpr is
invalid; obtain an unaliased supported dump or extend the versioned adapter after
checking real printer output and malformed input diagnostics.

Reports distinguish syntax, structural/type verification, and unvalidated constant
values. Diagnostics include original line/column and a nested equation path where
available. Verifier failures not attributable to an individual equation point to
that nested program's start. All candidates must pass; one failure fails the run.
Limits: common 20 MiB input limit, 256 top-level fragments, 100000 tokens per fragment, 5000 equations per
program, and 32 nested Jaxpr levels. Exceeding limits is an unsupported input.

## Sources

- https://docs.jax.dev/en/latest/601/jaxpr.html
- https://github.com/jax-ml/jax/blob/jax-v0.10.0/jax/_src/core.py
- https://github.com/jax-ml/jax/blob/jax-v0.10.0/jax/_src/lax/lax.py
  (`_convert_elt_type_pp_rule`, `_broadcast_in_dim_pp_rule`, `_dot_general_pp_rule`)

`check_jaxpr` checks an object, not a string. No general text-to-Jaxpr parser was
found in the inspected JAX 0.10.0 sources. This bounded adapter is custom code,
not an upstream general-purpose Jaxpr deserializer. See [provenance](upstream.md).
