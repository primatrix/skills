#!/usr/bin/env python3
"""Standalone StableHLO text/log validator. Requires matched JAX/jaxlib (0.10.0 validated).

Copy this file alone; no skill files or repository modules are needed.
Python 3.12+ is required. Run with --help for arguments.

Syntax/type validation calls LLVM MLIR through JAX/jaxlib; this file owns only
input selection, CLI and diagnostic handling. Upstream: jax-ml/jax,
jax-v0.10.0, jax/_src/interpreters/mlir.py and jaxlib/mosaic/python/tpu.py.
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


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


LOG_LANGUAGES = ("mlir", "stablehlo")


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


def main(argv=None):
  kind = "stablehlo"
  args = parser(f"Validate {kind} text using registered JAX MLIR dialects.").parse_args(
    argv
  )
  report, source = input_report(args, kind)
  if source is None:
    return report.emit(args.json)
  fragments, errors = extract(source, args)
  report.diagnostics.extend(errors)
  try:
    import jax
    import jaxlib
    from jax._src.interpreters import mlir
    from jax._src.lib.mlir import ir

    context = mlir.make_ir_context()
    if kind == "mosaic":
      from jax._src.lib import tpu

      tpu.register_dialect(context)
    context.allow_unregistered_dialects = False
    report.versions.update(jax=jax.__version__, jaxlib=jaxlib.__version__)
  except (ImportError, AttributeError, RuntimeError) as exc:
    report.diagnostics.append(
      Diagnostic("environment", f"JAX MLIR bindings unavailable: {exc}")
    )
    return report.emit(args.json)
  for fragment in fragments:
    result = {
      "line": fragment.start_line,
      "status": "failed",
      "message": "parse/verify failed",
      "validator": "jaxlib MLIR Module.parse + Operation.verify",
    }
    try:
      with context:
        # JAX 0.10.0 jax/_src/interpreters/mlir.py: make_ir_context;
        # LLVM MLIR IRCore.cpp: Module.parse / Operation.verify.
        module = ir.Module.parse(fragment.text)
        if not list(module.body.operations):
          # Module.parse inserts an empty implicit module for comment-only input.
          # The upstream single-operation parser distinguishes explicit module {}.
          ir.Operation.parse(fragment.text)
        if not module.operation.verify():
          raise ValueError("upstream verifier returned false")
      result.update(
        status="passed", message="MLIR parse + registered dialect verification passed"
      )
    except Exception as exc:
      detail = str(exc)
      locations = re.findall(r'(?:"-"|<string>):(\d+):(\d+)', detail)
      line, col = (
        fragment.position(source, *map(int, locations[0]))
        if locations
        else fragment.position(source)
      )
      report.diagnostics.append(Diagnostic("parse/verify", detail, line, col))
    report.fragments.append(result)
  return report.emit(args.json)


if __name__ == "__main__":
  raise SystemExit(main())
