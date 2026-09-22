# JAX IR validation skill review and publication

Source: `primatrix/pallas-kernel` PR #604, commit
`5ddf19d9d43e4e55239d81f34c8ea1c4ca770239`.
Review fixes synchronized to source commit `bae3ee05ecf082e20d1f28b7b0f299feabfb0cff`.
Destination baseline: `primatrix/skills` main
`ce0882ed22d111bddf8868eb9733668edd06ed38`.

Scope: review and publish the current `jax-ir-validate` skill, including all five
standalone scripts and their contracts. Keep every script independently copyable;
do not add test files or a CI workflow. Other plugins are outside this review.

## Review and delivery plan

1. Inspect all implementation paths and upstream API contracts, prioritizing
   false validation success, source execution boundaries and malformed input.
2. Reproduce concrete failures in temporary directories, fix them in the source
   skill and verify the corrected behavior against upstream checks.
3. Package the reviewed skill as `plugins/jax-ir-validate/skills/jax-ir-validate/`,
   register the plugin and update the marketplace inventory/install examples.
4. Validate manifests, relative references, standalone copies and representative
   real JAX/Pallas/MLIR inputs. Record evidence and remaining limitations here.
5. Push a feature branch and open one destination PR. Keep the source PR's
   implementation synchronized with any review fixes; do not merge either PR.

## Findings

- High, correctness, fixed — [Python compilation](../../plugins/jax-ir-validate/skills/jax-ir-validate/scripts/validate_jax_code.py#L337)
  and the corresponding Pallas path inherited the validator's
  `from __future__ import annotations` flag. Valid source containing a named
  expression in a nested function annotation passed direct CPython compilation
  but failed the validator. Isolate the input's compiler flags with
  `dont_inherit=True` in both Python-based validators.
- High, correctness, fixed — [entry-file tracing](../../plugins/jax-ir-validate/skills/jax-ir-validate/scripts/validate_jax_code.py#L276)
  and the corresponding Pallas path used `spec.loader.exec_module`, which could trace stale entry-file
  bytecode even though fresh source had just been validated. A same-size edit
  with an unchanged cached timestamp made an incompatible-shape function pass
  tracing. Execute the already-validated source snapshot with normal module
  metadata, bypassing the entry file's `.pyc` cache.
- High, correctness, fixed — [source line handling](../../plugins/jax-ir-validate/skills/jax-ir-validate/scripts/validate_jax_code.py#L127)
  in all five scripts used `str.splitlines`, which interpreted form-feed and Unicode line
  separators as source line boundaries. For example, `--lines 1:1` could accept
  `x=1\fthis broken` by discarding the invalid suffix of the same physical line.
  Use universal-newline input handling and split selected input only at physical
  LF boundaries, preserving source characters inside each line.
- Medium, diagnostics, fixed — [Jaxpr string parameters](../../plugins/jax-ir-validate/skills/jax-ir-validate/scripts/validate_jaxpr.py#L347)
  such as `y='\xZZ'`
  raised an uncaught `SyntaxError` in `ast.literal_eval`, producing a traceback
  instead of the requested JSON report. Translate that exception to a located
  syntax diagnostic.

## Validation and conclusion

All four reproduced findings are fixed. No other reproducible high-severity
finding remained in the reviewed paths. This is ready for human review, not a
claim of a proof for all possible inputs.

- 190 temporary checks passed with Python 3.12.3 and CPU JAX/jaxlib 0.10.0.
  The 99 prior scenarios were replayed from separately copied/renamed scripts
  with isolated Python imports. Additional checks covered all 60 allowlisted
  primitive forms from real `jax.make_jaxpr` output, six unsupported generic
  reductions containing opaque Python functions, compiler flags, stale bytecode,
  non-execution in syntax mode, CR/CRLF input, physical line selection, malformed
  string literals and oversized line-number arguments.
- Claude plugin and marketplace validation passed with `--strict`; the Codex
  plugin schema validator and skill frontmatter validator also passed.
- Focused Ruff lint/format, Python compilation, local reference checks, matching
  plugin names/versions, source-to-destination file equality and `git diff --check`
  passed. No test cases or CI workflow were added to either repository.
- Temporary detailed inputs/reports are at
  `/tmp/ir-upload-check-ub0_s4t5/results.json` on the review machine; that path is
  local evidence, not a resource needed to use the published skill.

The Jaxpr text adapter remains a custom, version-pinned subset. Upstream object
verification cannot prove faithful reconstruction of arbitrary text. Pallas
target discovery is conservative; tracing applies only to the supplied wrapper
and signature. Backend compilation, accelerator execution, numerical correctness
and performance were not checked. Plugin validation establishes package/schema
structure; installation in every supported agent UI was not exercised.

The marketplace entry is appended without changing other plugins. README
inventory and install examples are updated. The published skill contains exactly
five independently copyable scripts and local references; no source repository
checkout is required at runtime.
