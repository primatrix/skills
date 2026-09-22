#!/usr/bin/env python3
"""Standalone JAX source validator. Syntax mode needs only Python; explicit tracing needs JAX.

Copy this file alone; no skill files or repository modules are needed.
Python 3.12+ is required. Run with --help for arguments.

Syntax uses CPython ast.parse/compile. Explicit frontend validation calls
jax.make_jaxpr and jax._src.core.check_jaxpr. Selection/reporting code is
project-authored. Upstream: jax-ml/jax, jax-v0.10.0, jax/_src/api.py and core.py.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import ast
import importlib.util
import io
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from functools import partial


MAX_BYTES = 20 * 1024 * 1024


@dataclass
class Diagnostic:
  stage: str
  message: str
  line: int = 1
  column: int = 1
  status: str = "failed"


@dataclass
class Source:
  name: str
  text: str
  offsets: list[int]

  def position(self, line: int, column: int = 1) -> tuple[int, int]:
    return line, column + (
      self.offsets[line - 1] if 0 < line <= len(self.offsets) else 0
    )


@dataclass
class Fragment:
  text: str
  start_line: int
  lines: list[int]

  def position(self, source: Source, line: int = 1, column: int = 1):
    original = self.lines[line - 1] if 0 < line <= len(self.lines) else self.start_line
    return source.position(original, column)


@dataclass
class Report:
  kind: str
  source: str
  input_mode: str = "raw"
  selection: str | None = None
  fragments: list[dict] = field(default_factory=list)
  diagnostics: list[Diagnostic] = field(default_factory=list)
  versions: dict = field(default_factory=lambda: {"python": platform.python_version()})

  @property
  def ok(self):
    return (
      bool(self.fragments)
      and not self.diagnostics
      and all(item["status"] == "passed" for item in self.fragments)
    )

  def emit(self, as_json: bool) -> int:
    data = asdict(self)
    data["status"] = "passed" if self.ok else "failed"
    if as_json:
      print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
      print(f"{self.kind}: {data['status']} ({self.source})")
      print("versions: " + ", ".join(f"{k}={v}" for k, v in self.versions.items()))
      for index, item in enumerate(self.fragments, 1):
        print(f"  [{index}] line {item['line']}: {item['status']} — {item['message']}")
      for d in self.diagnostics:
        print(f"{self.source}:{d.line}:{d.column}: {d.status} [{d.stage}] {d.message}")
    return 0 if self.ok else 1


def parser(description: str) -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description=description)
  p.add_argument("input", help="UTF-8 file, or - for stdin")
  p.add_argument("--json", action="store_true", help="emit a structured report")
  p.add_argument("--strip-prefix", help="anchored regex removed once per matching line")
  p.add_argument(
    "--input-format",
    choices=("raw", "log"),
    default="raw",
    help="raw: validate the complete input; log: select explicitly fenced code blocks",
  )
  p.add_argument(
    "--lines",
    metavar="START:END",
    help="validate only this explicit inclusive line range",
  )
  return p


def read_source(args) -> Source:
  if args.input == "-":
    raw = sys.stdin.buffer.read(MAX_BYTES + 1)
  else:
    with Path(args.input).open("rb") as stream:
      raw = stream.read(MAX_BYTES + 1)
  if len(raw) > MAX_BYTES:
    raise ValueError("input exceeds the 20 MiB limit")
  text = raw.decode("utf-8")
  pattern = re.compile(args.strip_prefix) if args.strip_prefix else None
  lines, offsets = [], []
  # Use physical source lines; str.splitlines also splits inside form-feed/Unicode text.
  for line in io.StringIO(text, newline=None):
    match = pattern.match(line.rstrip("\r\n")) if pattern else None
    offset = match.end() if match else 0
    lines.append(line[offset:])
    offsets.append(offset)
  if not "".join(lines).strip():
    raise ValueError("empty input")
  return Source(args.input, "".join(lines), offsets)


def input_report(args, kind):
  report = Report(kind, args.input, input_mode=args.input_format, selection=args.lines)
  try:
    return report, read_source(args)
  except (OSError, UnicodeError, ValueError, re.error) as exc:
    report.diagnostics.append(Diagnostic("input", str(exc)))
    return report, None


LOG_LANGUAGES = ("py", "python")


def extract(source, args):
  """Select explicit input units; leave syntax to the upstream parser."""
  lines = list(io.StringIO(source.text))
  if args.lines:
    if args.input_format != "raw":
      return [], [Diagnostic("configuration", "--lines requires --input-format raw")]
    match = re.fullmatch(r"([1-9][0-9]{0,8}):([1-9][0-9]{0,8})", args.lines)
    if not match:
      return [], [
        Diagnostic("configuration", "--lines must be START:END (inclusive, 1-based)")
      ]
    first, last = map(int, match.groups())
    if first > last or last > len(lines):
      return [], [Diagnostic("configuration", "--lines is outside the input")]
    return [
      Fragment("".join(lines[first - 1 : last]), first, list(range(first, last + 1)))
    ], []
  if args.input_format == "raw":
    return [Fragment(source.text, 1, list(range(1, len(lines) + 1)))], []
  fragments, errors = [], []
  start = None
  fence = None
  for index, line in enumerate(lines):
    if start is None:
      opening = re.fullmatch(
        r"[ \t]*(`{3,}|~{3,})([A-Za-z0-9_-]*)[ \t]*(?:\r?\n)?", line
      )
      if opening:
        fence, language = opening.groups()
        start = index + 1
    elif re.fullmatch(
      r"[ \t]*" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}[ \t]*(?:\r?\n)?",
      line,
    ):
      if language in LOG_LANGUAGES:
        text = "".join(lines[start:index])
        if not text.strip():
          errors.append(Diagnostic("extract", "empty target code block", start + 1))
        else:
          fragments.append(Fragment(text, start + 1, list(range(start + 1, index + 1))))
      start = None
      if len(fragments) > 256:
        return [], [Diagnostic("extract", "more than 256 input blocks")]
  if start is not None:
    errors.append(Diagnostic("extract", "unclosed fenced code block", start))
  if not fragments:
    errors.append(
      Diagnostic(
        "extract", "no target code blocks; use explicit fences or --lines START:END"
      )
    )
  return fragments, errors


def functions(tree):
  """Return qualified source-level function names, including nested definitions."""
  found = {}

  def visit(node, path):
    for child in ast.iter_child_nodes(node):
      name = path
      if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        name = (*path, child.name)
        if not isinstance(child, ast.ClassDef):
          found[".".join(name)] = child
      visit(child, name)

  visit(tree, ())
  return found


def decode_spec(value, jax, numpy):
  if isinstance(value, dict):
    if set(value) == {"shape", "dtype"}:
      shape = value["shape"]
      if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("shape must be a list of non-negative integer dimensions")
      return jax.ShapeDtypeStruct(tuple(shape), numpy.dtype(value["dtype"]))
    if set(value) == {"value"}:
      return value["value"]
    if set(value) == {"tuple"}:
      return tuple(decode_spec(v, jax, numpy) for v in value["tuple"])
    return {k: decode_spec(v, jax, numpy) for k, v in value.items()}
  if isinstance(value, list):
    return [decode_spec(v, jax, numpy) for v in value]
  raise ValueError(
    'signature leaves require {"shape": [...], "dtype": "..."} or {"value": ...}'
  )


def trace_file(args, report, source_text, trace_check=None):
  """Trace a caller-authorized source snapshot without backend compilation."""
  path = Path(args.input).resolve()
  config = json.loads(Path(args.signature).read_text())
  if not isinstance(config, dict) or set(config) - {
    "args",
    "kwargs",
    "static_argnums",
    "static_kwargs",
  }:
    raise ValueError(
      "signature accepts args, kwargs, static_argnums, static_kwargs only"
    )
  import jax
  import jaxlib
  import numpy as np

  report.versions.update(jax=jax.__version__, jaxlib=jaxlib.__version__)
  positional = [decode_spec(v, jax, np) for v in config.get("args", [])]
  keywords = {k: decode_spec(v, jax, np) for k, v in config.get("kwargs", {}).items()}
  static = config.get("static_argnums", [])
  if not isinstance(static, list) or any(
    type(i) is not int or not 0 <= i < len(positional) for i in static
  ):
    raise ValueError("static_argnums must index positional arguments")
  static_kwargs = config.get("static_kwargs", {})
  if not isinstance(static_kwargs, dict) or set(static_kwargs) & set(keywords):
    raise ValueError("static_kwargs must be an object disjoint from kwargs")
  spec = importlib.util.spec_from_file_location("_ir_validation_user_source", path)
  if spec is None or spec.loader is None:
    raise ValueError("cannot load source file")
  module = importlib.util.module_from_spec(spec)
  previous = sys.modules.get(spec.name)
  sys.modules[spec.name] = module
  sys.path.insert(0, str(path.parent))
  try:
    # Trace exactly the source snapshot that passed syntax checks, not cached bytecode.
    exec(compile(source_text, str(path), "exec", dont_inherit=True), module.__dict__)
    entry = module
    for part in args.function.split("."):
      entry = getattr(entry, part)
    closed = jax.make_jaxpr(
      partial(entry, **static_kwargs), static_argnums=tuple(static)
    )(*positional, **keywords)
    # Use JAX's existing verifier rather than duplicating equation type rules.
    from jax._src import core

    core.check_jaxpr(closed.jaxpr)
    if trace_check:
      trace_check(closed, args)
    return {
      "entry": args.function,
      "signature": config,
      "equations": len(closed.jaxpr.eqns),
      "validator": "jax.make_jaxpr + jax._src.core.check_jaxpr",
    }
  finally:
    sys.path.pop(0)
    if previous is None:
      sys.modules.pop(spec.name, None)
    else:
      sys.modules[spec.name] = previous


def main(argv=None, *, kind="jax-code", configure=None, analyze=None, trace_check=None):
  p = parser(
    "Check Python syntax without execution; optionally trace trusted local JAX code."
  )
  p.add_argument("--function", help="qualified function name (required for --trace)")
  p.add_argument(
    "--trace",
    action="store_true",
    help="execute the trusted local Python frontend with abstract inputs",
  )
  p.add_argument(
    "--signature", help="JSON args/kwargs and static argument specification"
  )
  if configure:
    configure(p)
  args = p.parse_args(argv)
  report, source = input_report(args, kind)
  if source is None:
    return report.emit(args.json)
  fragments, errors = extract(source, args)
  report.diagnostics.extend(errors)
  for fragment in fragments:
    result = {
      "line": fragment.start_line,
      "status": "failed",
      "message": "Python syntax failed",
      "syntax": "failed",
      "validator": "CPython ast.parse + compile",
      "tracing": "not_checked",
    }
    try:
      tree = ast.parse(fragment.text, filename=source.name)
      # The validator's own future imports must not alter the input language.
      compile(tree, source.name, "exec", dont_inherit=True)
      if not tree.body:
        raise ValueError("no Python statements found")
      targets = functions(tree)
      if args.function and args.function not in targets:
        raise ValueError(
          f"function {args.function!r} not found; available: {', '.join(targets)}"
        )
      if analyze:
        result.update(analyze(tree, args))
      result.update(
        status="passed",
        syntax="passed",
        message="Python syntax passed; JAX tracing not checked",
      )
    except RecursionError as exc:
      line, col = fragment.position(source)
      result.update(
        syntax="not_checked", message="Python parser/compiler recursion limit exceeded"
      )
      report.diagnostics.append(
        Diagnostic("unsupported", str(exc), line, col, "unsupported")
      )
    except (SyntaxError, ValueError) as exc:
      line, col = fragment.position(
        source, getattr(exc, "lineno", None) or 1, getattr(exc, "offset", None) or 1
      )
      report.diagnostics.append(
        Diagnostic(
          "syntax" if isinstance(exc, SyntaxError) else "source", str(exc), line, col
        )
      )
    report.fragments.append(result)
  if args.signature and not args.trace:
    report.diagnostics.append(
      Diagnostic("configuration", "--signature requires --trace")
    )
  if args.trace:
    if not args.function or not args.signature:
      report.diagnostics.append(
        Diagnostic("configuration", "--trace requires --function and --signature")
      )
    elif (
      args.input == "-"
      or Path(args.input).suffix != ".py"
      or args.strip_prefix
      or args.lines
      or args.input_format != "raw"
      or len(fragments) != 1
      or fragments[0].text != source.text
    ):
      report.diagnostics.append(
        Diagnostic(
          "configuration",
          "tracing requires an unmodified trusted local .py file; stdin/log blocks cannot be executed",
        )
      )
    elif report.ok:
      output = io.StringIO()
      try:
        with redirect_stdout(output), redirect_stderr(output):
          details = trace_file(args, report, source.text, trace_check)
        report.fragments[0].update(
          tracing="passed",
          trace=details,
          message="Python syntax and requested JAX trace passed (no backend compilation)",
        )
      except (Exception, SystemExit) as exc:
        frames = traceback.extract_tb(exc.__traceback__)
        frames = [
          f for f in frames if Path(f.filename).resolve() == Path(args.input).resolve()
        ]
        line = frames[-1].lineno if frames else 1
        report.diagnostics.append(
          Diagnostic(
            "environment" if isinstance(exc, ImportError) else "tracing",
            f"{type(exc).__name__}: {exc}",
            line,
          )
        )
        report.fragments[0].update(
          status="failed",
          tracing="failed",
          message="Python syntax passed; requested JAX trace failed",
        )
      if output.getvalue():
        report.fragments[0]["frontend_output"] = output.getvalue()[-8192:]
  return report.emit(args.json)


if __name__ == "__main__":
  raise SystemExit(main())
