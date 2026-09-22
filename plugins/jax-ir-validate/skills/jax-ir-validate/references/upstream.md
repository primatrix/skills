# Code provenance and validation boundaries

These five files are project-authored standalone wrappers, not copies of
official command-line scripts. Existing functions are called directly from the
installed Python/JAX libraries; no upstream source is vendored. Every file embeds
its own CLI, explicit input selection, source mapping and report code so it can
be copied alone.

| Script | Existing validation functions | Remaining project-authored behavior |
|---|---|---|
| `validate_stablehlo_ir.py` | JAX `mlir.make_ir_context`; MLIR `ir.Module.parse`, `ir.Operation.parse`, `Operation.verify` | Input selection and reports |
| `validate_mosaic_ir.py` | The same MLIR APIs, plus `tpu.register_dialect(context)` | Input selection and reports |
| `validate_jax_code.py` | CPython `ast.parse`, `compile`; explicit trace with `jax.make_jaxpr`, `core.check_jaxpr` | Function selection, signature decoding, import/tracing wrapper and reports |
| `validate_pallas_kernel.py` | The same Python/JAX APIs; Pallas tracing follows the supplied `pallas_call` | Conservative AST target discovery and traced kernel source matching |
| `validate_jaxpr.py` | JAX `core.check_eqn`, `core.typematch`, `core.check_jaxpr` | Bounded lexer/grammar, allowlists, object construction and documented printer defaults |

The JAX APIs under `_src` are private and version-dependent. Jaxpr text is pinned
to JAX/jaxlib 0.10.0. Other wrappers record installed versions and were manually
checked with Python 3.12 and CPU JAX/jaxlib 0.10.0. Pallas/JAX syntax alone does
not check API availability or Ref/shape legality. Requested tracing checks a
particular frontend signature; it does not prove device compilation, execution
or numerical correctness.

Jaxpr is the main remaining custom parser: upstream object verification cannot
prove that an arbitrary text dump was reconstructed faithfully. Unknown formats
and parameters fail as unsupported. The inspected JAX 0.10.0 source did not
provide a general text deserializer to substitute for this bounded adapter.
CPython's tokenizer is not used for Jaxpr: its documented contract covers valid
Python source, while Jaxpr has a different grammar.

## Source locations

- [CPython AST parsing and compilation](https://docs.python.org/3/library/ast.html#ast.parse).
- [CPython tokenization contract](https://docs.python.org/3/library/tokenize.html): explains why it is not a general Jaxpr lexer.
- [LLVM MLIR Python bindings](https://mlir.llvm.org/docs/Bindings/Python/) and
  [IRCore.cpp implementation](https://mlir.llvm.org/doxygen/IRCore_8cpp_source.html).
- [JAX 0.10.0 MLIR context setup](https://github.com/jax-ml/jax/blob/jax-v0.10.0/jax/_src/interpreters/mlir.py): `make_ir_context` registers available dialects including StableHLO.
- [JAX 0.10.0 Mosaic TPU bindings](https://github.com/jax-ml/jax/blob/jax-v0.10.0/jaxlib/mosaic/python/tpu.py): TPU dialect registration.
- [JAX 0.10.0 core](https://github.com/jax-ml/jax/blob/jax-v0.10.0/jax/_src/core.py): `check_eqn`, `typematch`, `check_jaxpr`, and Jaxpr object types.
- [JAX 0.10.0 frontend API](https://github.com/jax-ml/jax/blob/jax-v0.10.0/jax/_src/api.py): `make_jaxpr`.
- [JAX 0.10.0 lax printer rules](https://github.com/jax-ml/jax/blob/jax-v0.10.0/jax/_src/lax/lax.py): `_convert_elt_type_pp_rule`, `_broadcast_in_dim_pp_rule`, `_dot_general_pp_rule` explain restored defaults.
- [OpenXLA stablehlo-opt entrypoint](https://github.com/openxla/stablehlo/blob/main/stablehlo/tools/StablehloOptMain.cpp): upstream CLI reference; these wrappers use Python bindings instead of copying that executable's implementation.

## Audit findings addressed

- Removed automatic module/brace extraction that could accept `module {}` while
  discarding a following broken operation. Raw mode now validates all input.
- Removed automatic Python fence detection that confused triple-quoted strings
  with log blocks. Log selection is explicit, and its scope is reported.
- Delegated Jaxpr equation checking to `core.check_eqn` and retained an explicitly
  bounded text adapter. Finite literal overflow is rejected,
  including bfloat16 scalar casts that do not honor NumPy's error-state setting.

Local manual checks cover independent copied scripts, valid and malformed raw
input, explicit log/range selection, source locations, actual JAX-generated
Jaxprs and trusted JAX/Pallas traces. They are evidence for those inputs, not a
claim of a complete grammar proof. No automated test files or dedicated CI are
shipped, as requested.
