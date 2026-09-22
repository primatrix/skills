# Python source and tracing

Each listed script can be copied and run as a single file. Paths below use the
skill directory for convenience; no sibling Python modules or reference documents
are required at runtime. External Python/JAX dependencies still apply.

`validate_jax_code.py` runs on Python 3.12+ with no dependencies in its default
syntax mode. It uses AST parsing and Python compilation without importing or
executing the input. Compilation uses the input's own future imports and does
not inherit compiler flags from the validator. Undefined globals, API availability and dynamic type
constraints are not syntax checks. Select a source function with `--function`;
qualified nested definitions can be selected statically.

```sh
python <skill>/scripts/validate_jax_code.py model.py --json
cat snippet.txt | python <skill>/scripts/validate_jax_code.py - --json
```

Pure source is accepted from .py/.txt/stdin; raw mode checks the entire input,
including strings that contain Markdown fences. Use `--input-format log` to
select complete fenced `python`/`py` blocks from logs, or `--lines START:END` in
raw mode to select an explicit inclusive line range. Content outside selected
units is not validated. Each block is checked as an independent source unit; do not
split one function across blocks. `--strip-prefix` uses the common prefix and
source mapping contract. Unclosed fences and comment-only inputs fail. The
validator cannot detect a semantically missing line if the remaining text is
still syntactically complete. Python's legitimate `x[...]` is retained.

## Explicit tracing of trusted local files

```sh
python <skill>/scripts/validate_jax_code.py model.py --trace --function forward \
  --signature signature.json --json
```

Requires matched JAX/jaxlib (CPU 0.10.0 tested). The file must be a local `.py`
file in raw mode without prefix stripping or line selection. Import and tracing execute Python code,
including decorators and host-side effects; this is an explicitly requested
frontend invocation, not a sandbox. The script calls `make_jaxpr` with abstract
inputs and `jax._src.core.check_jaxpr` on the resulting object, never calls
`.compile()` or an input executable. Use a pure frontend
wrapper and provide any mesh/axis context there. Transforms are explicit in that
wrapper; tracing an ordinary function says nothing about unrequested transforms.
The entry file executes from the same source snapshot that passed syntax checks,
with its normal module metadata; cached `.pyc` code for that entry is not loaded.
Imported dependencies still follow Python's normal import behavior.

Signature JSON:

```json
{
  "args": [{"shape": [4], "dtype": "float32"}, {"value": 4}],
  "static_argnums": [1],
  "kwargs": {},
  "static_kwargs": {"axis": 0}
}
```

- Shape/dtype leaves become `ShapeDtypeStruct`; `{"value": ...}` supplies a JSON
  scalar/value (mark its positional index static if the function needs it concrete).
- Lists/dictionaries compose PyTrees; `{"tuple": [...]}` creates a tuple.
- `static_kwargs` are bound into the callable before tracing and cannot overlap
  dynamic `kwargs`. Static positional values must satisfy JAX's hashability rules.
- Relative sibling imports resolve from the source file's directory. Package
  relative imports require an appropriate standalone wrapper.
- Missing dependencies/context or invalid signature fails the requested trace.

Reports keep syntax and tracing separate, record the signature/entry and runtime
versions, and capture up to the final 8192 characters of Python frontend output.
Success is limited to this input signature and transformation context. Backend
lowering, device compilation/execution and numerical correctness remain unchecked.

References: https://docs.python.org/3/library/ast.html#ast.parse,
https://docs.jax.dev/en/latest/_autosummary/jax.make_jaxpr.html.
