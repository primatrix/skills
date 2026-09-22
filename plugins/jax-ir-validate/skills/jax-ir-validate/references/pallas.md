# Pallas source validation

Each listed script can be copied and run as a single file. Paths below use the
skill directory for convenience; no sibling Python modules or reference documents
are required at runtime. External Python/JAX dependencies still apply.

Uses the shared [Python input/tracing contract](python.md). Default validation
checks Python syntax and identifies a source-level kernel, without importing JAX
or executing the source. No shape/Ref legality is claimed in this mode.

```sh
python <skill>/scripts/validate_pallas_kernel.py kernel.py --kernel compute --json
cat kernel.log | python <skill>/scripts/validate_pallas_kernel.py - --input-format log --kernel compute --json
python <skill>/scripts/validate_pallas_kernel.py kernel.py --kernel compute \
  --trace --function call_kernel --signature signature.json --json
```

`--kernel` can name a qualified nested function for static checks. When omitted,
the script resolves simple `pallas_call` references, import aliases, keyword
`kernel=...`, and `functools.partial(kernel, ...)`. Exactly one locally defined
candidate is required. Dynamic aliases/factories or multiple candidates require
an explicit kernel selection. Name-based static discovery is conservative and
is not Python execution or proof of actual runtime binding.

Tracing uses a trusted local wrapper, never calls the kernel with ordinary arrays:

```python
import jax
from jax.experimental import pallas as pl

def compute(x_ref, out_ref):
    out_ref[...] = x_ref[...] + 1

def call_kernel(x):
    return pl.pallas_call(
        compute, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
    )(x)
```

Supply `{"args": [{"shape": [4], "dtype": "float32"}]}` in signature.json.
`make_jaxpr` follows the real Pallas Ref/grid/BlockSpec configuration supplied by
the wrapper; `core.check_jaxpr` checks the resulting object. This does not compile
or execute the resulting kernel. The trace must
contain a `pallas_call` whose kernel debug name matches the selected source kernel,
and source file/line, including calls nested inside other Jaxprs. An ordinary
JAX-only wrapper fails. Target discovery and debug-source matching are
project-authored checks, not an official Pallas source parser.
Kernel wrappers that change debug names may require selecting the actual wrapped
source function; this is a versioned frontend check (JAX 0.10.0 tested).

A successful trace applies only to its signature/configuration. Backend layout
constraints, compilation, dynamic bounds, race freedom, values and performance
remain outside this validation.
