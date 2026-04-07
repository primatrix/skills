# Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `SyntaxError` on `*` unpacking | Python < 3.12 | Use Docker image with Python >= 3.12 |
| `BooleanOptionalAction` error | xpk on Python 3.14 | `pipx reinstall xpk --python python3.13` |
| JAX TPU init hangs > 60s | Not all containers started | Must start all containers simultaneously |
| Sharded computation hangs | Worker not running same code | ALL processes must execute same jitted code paths |
| `Shutdown barrier DEADLINE_EXCEEDED` | One process crashed | Check crashed process logs, restart all |
| `ModuleNotFoundError` | Missing deps or PYTHONPATH | Ensure paths in sys.path |
| `gcloud auth` errors | Token expired | `gcloud auth login` |
| No LLO rows in profile | LIBTPU flags not set before JAX import | Use `profile_launcher.py` |
| `kubectl cp` truncated | Large file > 50 MB | Use GCS as intermediate |
| TensorBoard plugin error | Running on macOS | Run on Linux pod + port-forward |
| `pkg_resources` missing | setuptools >= 82 | `pip install 'setuptools<81'` |
