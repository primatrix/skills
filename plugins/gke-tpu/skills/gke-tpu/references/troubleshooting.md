# Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `SyntaxError` on `*` unpacking | Python < 3.12 | Use Docker image with Python >= 3.12 |
| JAX TPU init hangs > 60s | Not all containers started | Must start all containers simultaneously |
| Sharded computation hangs | Worker not running same code | ALL processes must execute same jitted code paths |
| `Shutdown barrier DEADLINE_EXCEEDED` | One process crashed | Check crashed process logs, restart all |
| `ModuleNotFoundError` | Missing deps or PYTHONPATH | Ensure paths in sys.path |
| `gcloud auth` errors | Token expired | `gcloud auth login` |
| `kubectl cp` truncated | Large file > 50 MB | Use GCS as intermediate |
| `pkg_resources` missing | setuptools >= 82 | `pip install 'setuptools<81'` |
