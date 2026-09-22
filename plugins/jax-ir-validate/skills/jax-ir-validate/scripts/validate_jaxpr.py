#!/usr/bin/env python3
"""Standalone Jaxpr text validator. Requires jax==0.10.0 and jaxlib==0.10.0.

Copy this file alone; no skill files or repository modules are needed.
Python 3.12+ is required. Run with --help for arguments.

The bounded Jaxpr lexer and text adapter are project-authored; equation/type
checks call existing JAX core.check_eqn/check_jaxpr functions.
Upstream: jax-ml/jax, jax-v0.10.0, jax/_src/core.py.
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import re
import sys
import ast
import math
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


# Printer notation is versioned independently of MLIR. Reject unknown tokens,
# primitives and parameters instead of reconstructing missing runtime objects.
TOKEN = re.compile(
  r"""\s+|//[^\n]*|\#[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:inf|nan)\b|[A-Za-z_][A-Za-z_0-9]*|[{}\[\]():;=,.]"""
)
DTYPES = {
  "bool": "bool",
  "bf16": "bfloat16",
  **{
    f"{prefix}{bits}": f"{name}{bits}"
    for prefix, name, widths in [
      ("f", "float", (16, 32, 64)),
      ("i", "int", (8, 16, 32, 64)),
      ("u", "uint", (8, 16, 32, 64)),
      ("c", "complex", (64, 128)),
    ]
    for bits in widths
  },
}
PRIMITIVES = """abs neg sign sin cos tan tanh exp expm1 log log1p sqrt rsqrt floor ceil
logistic is_finite add sub mul div rem pow max min eq ne gt ge lt le and or xor not
shift_left shift_right_arithmetic shift_right_logical reduce_sum reduce_prod
reduce_max reduce_min reduce_or reduce_and reshape broadcast_in_dim squeeze
transpose convert_element_type concatenate slice select_n iota rev stop_gradient
integer_pow dot_general cond scan while""".split()

# Unlike the upstream abstract evaluator, this text adapter rejects unknown
# kwargs: several standard primitives otherwise silently accept arbitrary keys.
PARAMETERS = {
  "integer_pow": {"y"},
  "reshape": {"new_sizes", "dimensions", "sharding"},
  "broadcast_in_dim": {"shape", "broadcast_dimensions", "sharding"},
  "squeeze": {"dimensions"},
  "transpose": {"permutation"},
  "convert_element_type": {"new_dtype", "weak_type", "sharding"},
  "concatenate": {"dimension"},
  "slice": {"start_indices", "limit_indices", "strides"},
  "iota": {"dtype", "shape", "dimension", "sharding"},
  "rev": {"dimensions"},
  "dot_general": {
    "dimension_numbers",
    "precision",
    "preferred_element_type",
    "out_sharding",
  },
  "cond": {"branches"},
  "scan": {"jaxpr", "length", "num_carry", "num_consts", "reverse", "unroll"},
  "while": {"body_jaxpr", "cond_jaxpr", "body_nconsts", "cond_nconsts"},
  **{
    name: {"axes", "out_sharding"} for name in PRIMITIVES if name.startswith("reduce_")
  },
}


@dataclass
class Token:
  value: str
  offset: int


class Problem(Exception):
  def __init__(self, stage, message, offset=0, path="root"):
    super().__init__(message)
    self.stage, self.offset, self.path = stage, offset, path


@dataclass
class Binding:
  name: str
  dtype: str
  shape: tuple
  offset: int


@dataclass
class Atom:
  token: Token
  annotation: Binding | None = None


@dataclass
class Equation:
  outputs: list
  primitive: Token
  params: dict
  inputs: list


@dataclass
class Program:
  consts: list
  inputs: list
  equations: list
  outputs: list
  offset: int


def lex(text):
  """Tokenize the documented subset; do not treat Jaxpr as Python syntax."""
  tokens, pos = [], 0
  while pos < len(text):
    match = TOKEN.match(text, pos)
    if not match:
      raise Problem(
        "unsupported", f"unsupported token near {text[pos : pos + 30]!r}", pos
      )
    value = match[0]
    if not value.isspace() and not value.startswith(("#", "//")):
      tokens.append(Token(value, pos))
    pos = match.end()
    if len(tokens) > 100000:
      raise Problem("unsupported", "token limit exceeded (100000)", pos)
  tokens.append(Token("<eof>", len(text)))
  return tokens


class Parser:
  def __init__(self, text):
    self.tokens, self.i, self.depth = lex(text), 0, 0

  def peek(self, offset=0):
    return self.tokens[min(self.i + offset, len(self.tokens) - 1)].value

  def take(self, expected=None):
    token = self.tokens[self.i]
    if expected is not None and token.value != expected:
      raise Problem(
        "syntax", f"expected {expected!r}, got {token.value!r}", token.offset
      )
    if token.value == "<eof>":
      raise Problem("syntax", "truncated Jaxpr", token.offset)
    self.i += 1
    return token

  def name(self):
    token = self.take()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", token.value):
      raise Problem("syntax", f"expected identifier, got {token.value!r}", token.offset)
    return token

  def annotation(self, name):
    self.take(":")
    dtype = self.take()
    if dtype.value not in DTYPES:
      raise Problem("unsupported", f"unsupported dtype {dtype.value!r}", dtype.offset)
    self.take("[")
    shape = []
    while self.peek() != "]":
      dimension = self.take()
      if not re.fullmatch(r"\d+", dimension.value):
        raise Problem(
          "unsupported",
          "only static non-negative dimensions are supported",
          dimension.offset,
        )
      shape.append(int(dimension.value))
      if self.peek() != "]":
        self.take(",")
    self.take("]")
    return Binding(name.value, DTYPES[dtype.value], tuple(shape), name.offset)

  def binding(self):
    return self.annotation(self.name())

  def atom(self):
    token = self.take()
    if self.peek() == ":":
      if not re.fullmatch(
        r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|inf|nan)|True|False",
        token.value,
      ):
        raise Problem(
          "unsupported",
          "only typed scalar numeric/boolean literals are supported",
          token.offset,
        )
      return Atom(token, self.annotation(token))
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", token.value):
      raise Problem(
        "syntax", "expected a bound variable or typed scalar literal", token.offset
      )
    return Atom(token)

  def value(self):
    token = self.tokens[self.i]
    if self.peek() == "{":
      return self.program()
    if self.peek() in ("(", "["):
      closing = ")" if self.take().value == "(" else "]"
      values = []
      while self.peek() != closing:
        values.append(self.value())
        if self.peek() == ",":
          self.take(",")
        elif self.peek() != closing and not isinstance(values[-1], Program):
          raise Problem(
            "syntax", "expected comma in parameter sequence", self.tokens[self.i].offset
          )
      self.take(closing)
      return tuple(values) if closing == ")" else values
    value = self.take().value
    if value in ("True", "False", "None"):
      return {"True": True, "False": False, "None": None}[value]
    if value.startswith(('"', "'")):
      try:
        return ast.literal_eval(value)
      except (SyntaxError, ValueError) as exc:
        raise Problem(
          "syntax", f"invalid string parameter: {exc}", token.offset
        ) from exc
    if re.fullmatch(r"[-+]?\d+", value):
      return int(value)
    if re.fullmatch(
      r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|inf|nan)", value
    ):
      return float(value)
    if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value):
      return value
    raise Problem("unsupported", f"unsupported parameter value {value!r}", token.offset)

  def program(self):
    offset = self.take("{").offset
    self.depth += 1
    if self.depth > 32:
      raise Problem("unsupported", "nested Jaxpr depth exceeds 32", offset)
    self.take("lambda")
    consts, inputs, equations = [], [], []
    while self.peek() != ";":
      consts.append(self.binding())
    self.take(";")
    while self.peek() != ".":
      inputs.append(self.binding())
    self.take(".")
    self.take("let")
    while self.peek() != "in":
      outputs = []
      while self.peek() != "=":
        outputs.append(self.binding())
      self.take("=")
      primitive = self.name()
      params = {}
      if self.peek() == "[":
        self.take("[")
        while self.peek() != "]":
          key = self.name()
          if key.value in params:
            raise Problem("syntax", f"duplicate parameter {key.value!r}", key.offset)
          self.take("=")
          params[key.value] = self.value()
        self.take("]")
      args = []
      while self.peek() not in ("in", ";", "<eof>"):
        if (
          self.peek(1) == ":"
          and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", self.peek())
          and self.peek() not in ("True", "False", "inf", "nan")
        ):
          break
        args.append(self.atom())
      if self.peek() == ";":
        self.take(";")
      equations.append(Equation(outputs, primitive, params, args))
      if len(equations) > 5000:
        raise Problem("unsupported", "equation limit exceeded (5000)", primitive.offset)
    self.take("in")
    self.take("(")
    outputs = []
    while self.peek() != ")":
      outputs.append(self.atom())
      if self.peek() == ",":
        self.take(",")
      elif self.peek() != ")":
        raise Problem(
          "syntax", "expected comma in output tuple", self.tokens[self.i].offset
        )
    self.take(")")
    self.take("}")
    self.depth -= 1
    return Program(consts, inputs, equations, outputs, offset)

  def parse(self):
    program = self.program()
    if self.peek() != "<eof>":
      raise Problem(
        "syntax", "unexpected content after Jaxpr", self.tokens[self.i].offset
      )
    return program


LOG_LANGUAGES = ("jaxpr",)


def extract(source, args):
  """Select explicit input units; pass each complete unit to the text adapter."""
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


def build(program, core, lax, np, path="root"):
  env, equations, effects = {}, [], set()
  registry = {name: getattr(lax, name + "_p", None) for name in PRIMITIVES}

  def aval(binding):
    return core.ShapedArray(binding.shape, np.dtype(binding.dtype))

  def bind(binding, allow_drop=False):
    if binding.name == "_":
      if allow_drop:
        return core.DropVar(aval(binding))
      raise Problem(
        "structure",
        "discard variable cannot bind an input or constant",
        binding.offset,
        path,
      )
    if binding.name in env:
      raise Problem(
        "structure", f"variable {binding.name!r} already bound", binding.offset, path
      )
    var = core.Var(aval(binding))
    env[binding.name] = var
    return var

  def atom(node):
    if node.annotation is None:
      if node.token.value not in env:
        raise Problem(
          "structure",
          f"variable {node.token.value!r} not defined",
          node.token.offset,
          path,
        )
      return env[node.token.value]
    binding = node.annotation
    if binding.shape:
      raise Problem(
        "unsupported", "only scalar literals are supported", binding.offset, path
      )
    value = node.token.value
    dtype = np.dtype(binding.dtype)
    if value in ("True", "False"):
      value = value == "True"
    elif re.fullmatch(r"[-+]?\d+", value):
      value = int(value)
    else:
      value = float(value)
    if isinstance(value, bool) and dtype.kind != "b":
      raise Problem(
        "structure", "boolean literal requires bool dtype", binding.offset, path
      )
    if dtype.kind in "iu":
      limits = np.iinfo(dtype)
      if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not limits.min <= value <= limits.max
      ):
        raise Problem(
          "structure",
          "literal is outside its declared integer type",
          binding.offset,
          path,
        )
    elif dtype.kind == "b" and not isinstance(value, bool):
      raise Problem(
        "structure", "boolean literal must be True or False", binding.offset, path
      )
    if (
      isinstance(value, float)
      and not math.isfinite(value)
      and node.token.value.lstrip("+-") not in {"inf", "nan"}
    ):
      raise Problem(
        "structure", "finite literal overflows the host parser", binding.offset, path
      )
    with np.errstate(over="raise", invalid="raise"):
      try:
        converted = dtype.type(value)
      except (FloatingPointError, OverflowError, ValueError) as exc:
        raise Problem(
          "structure",
          f"literal cannot be represented as {binding.dtype}: {exc}",
          binding.offset,
          path,
        ) from exc
    # ml_dtypes scalar casts (including bfloat16) need not honor np.errstate.
    if (not isinstance(value, float) or math.isfinite(value)) and not np.isfinite(
      converted
    ):
      raise Problem(
        "structure", "finite literal overflows its declared dtype", binding.offset, path
      )
    return core.Literal(converted, aval(binding))

  def parameter(value, nested_path):
    if isinstance(value, Program):
      child = build(value, core, lax, np, nested_path)
      if child.constvars:
        raise Problem(
          "unsupported",
          "nested closed Jaxpr constants are not present in this text",
          value.offset,
          nested_path,
        )
      return core.ClosedJaxpr(child, [])
    if isinstance(value, tuple):
      return tuple(parameter(v, f"{nested_path}/{i}") for i, v in enumerate(value))
    if isinstance(value, list):
      return [parameter(v, f"{nested_path}/{i}") for i, v in enumerate(value)]
    return value

  consts = [bind(b) for b in program.consts]
  inputs = [bind(b) for b in program.inputs]
  for i, eqn in enumerate(program.equations):
    location = f"{path}/eqn[{i}]/{eqn.primitive.value}"
    primitive = registry.get(eqn.primitive.value)
    if primitive is None:
      raise Problem(
        "unsupported",
        f"unsupported primitive {eqn.primitive.value!r}",
        eqn.primitive.offset,
        location,
      )
    unknown = set(eqn.params) - PARAMETERS.get(eqn.primitive.value, set())
    if unknown:
      raise Problem(
        "unsupported",
        f"unknown parameters: {sorted(unknown)}",
        eqn.primitive.offset,
        location,
      )
    args = [atom(a) for a in eqn.inputs]
    params = {
      key: parameter(value, f"{location}/{key}") for key, value in eqn.params.items()
    }
    # These defaults are omitted by the 0.10 printer. Sharded aval syntax is
    # rejected by the grammar, so no mesh/sharding information is invented.
    if eqn.primitive.value == "convert_element_type":
      params.setdefault("sharding", None)
    if eqn.primitive.value == "broadcast_in_dim":
      if len(eqn.outputs) != 1:
        raise Problem(
          "verification",
          "broadcast requires one result",
          eqn.primitive.offset,
          location,
        )
      params.setdefault("shape", eqn.outputs[0].shape)
      params.setdefault("broadcast_dimensions", ())
      params.setdefault("sharding", None)
    if eqn.primitive.value == "dot_general":
      for key in ("precision", "preferred_element_type", "out_sharding"):
        params.setdefault(key, None)
      if params.get("precision") is not None:
        raise Problem(
          "unsupported",
          "non-default dot precision is not supported",
          eqn.primitive.offset,
          location,
        )
      if "dimension_numbers" in params:
        params["dimension_numbers"] = tuple(
          tuple(tuple(v) for v in pair) for pair in params["dimension_numbers"]
        )
    for key in ("sharding", "out_sharding"):
      if params.get(key) is not None:
        raise Problem(
          "unsupported",
          "sharded Jaxpr parameters are not supported",
          eqn.primitive.offset,
          location,
        )
    for key in ("new_dtype", "dtype", "preferred_element_type"):
      if params.get(key) is not None:
        try:
          params[key] = np.dtype(params[key])
        except TypeError as exc:
          raise Problem(
            "unsupported", str(exc), eqn.primitive.offset, location
          ) from exc
    try:
      # Upstream JAX core.check_eqn recursively validates nested Jaxprs and
      # delegates abstract evaluation to the registered primitive implementation.
      output_avals, eqn_effects = core.check_eqn(
        primitive, [v.aval for v in args], params
      )
      if len(output_avals) != len(eqn.outputs):
        raise ValueError(
          f"expected {len(output_avals)} results, got {len(eqn.outputs)} bindings"
        )
      for expected, binding in zip(output_avals, eqn.outputs):
        if not core.typematch(expected, aval(binding)):
          raise ValueError(
            f"binding {binding.name!r} declares {aval(binding)}, primitive produces {expected}"
          )
    except Exception as exc:
      raise Problem(
        "verification", f"{type(exc).__name__}: {exc}", eqn.primitive.offset, location
      ) from exc
    outputs = [bind(b, allow_drop=True) for b in eqn.outputs]
    equations.append(core.new_jaxpr_eqn(args, outputs, primitive, params, eqn_effects))
    effects.update(eqn_effects)
  outputs = [atom(a) for a in program.outputs]
  info = core.DebugInfo(
    "ir-validation", "text Jaxpr", tuple(b.name for b in program.inputs), None
  )
  result = core.Jaxpr(
    consts, inputs, outputs, equations, frozenset(effects), debug_info=info
  )
  try:
    core.check_jaxpr(result)
  except Exception as exc:
    raise Problem("verification", str(exc), program.offset, path) from exc
  return result


def main(argv=None):
  p = parser("Validate the supported JAX 0.10 Jaxpr text grammar and upstream types.")
  args = p.parse_args(argv)
  report, source = input_report(args, "jaxpr")
  if source is None:
    return report.emit(args.json)
  fragments, errors = extract(source, args)
  report.diagnostics.extend(errors)
  try:
    import jax
    import jaxlib
    from jax import lax
    from jax._src import core
    import numpy as np

    report.versions.update(jax=jax.__version__, jaxlib=jaxlib.__version__)
    if jax.__version__ != "0.10.0" or jaxlib.__version__ != "0.10.0":
      raise RuntimeError(
        "text adapter currently requires jax==jaxlib==0.10.0; use an isolated matching environment"
      )
  except (ImportError, RuntimeError) as exc:
    report.diagnostics.append(Diagnostic("environment", str(exc)))
    return report.emit(args.json)
  for fragment in fragments:
    result = {
      "line": fragment.start_line,
      "status": "failed",
      "message": "Jaxpr validation failed",
      "validator": "bounded text adapter + JAX core.check_eqn/check_jaxpr",
      "syntax": "failed",
      "verification": "not_checked",
    }
    try:
      tree = Parser(fragment.text).parse()
      result["syntax"] = "passed"
      checked = build(tree, core, lax, np)
      result.update(
        status="passed",
        verification="passed",
        message="Jaxpr text, binding and upstream type verification passed; computation not executed",
        equations=len(checked.eqns),
        constant_values="not_checked",
      )
    except Problem as exc:
      if exc.stage == "unsupported":
        if result["syntax"] == "passed":
          result["verification"] = "unsupported"
        else:
          result["syntax"] = "unsupported"
      elif result["syntax"] == "passed":
        result["verification"] = "failed"
      line = fragment.text.count("\n", 0, exc.offset) + 1
      column = exc.offset - fragment.text.rfind("\n", 0, exc.offset)
      line, column = fragment.position(source, line, column)
      report.diagnostics.append(
        Diagnostic(
          exc.stage,
          f"{exc.path}: {exc}",
          line,
          column,
          "unsupported" if exc.stage == "unsupported" else "failed",
        )
      )
    except (ValueError, TypeError, RecursionError, OverflowError) as exc:
      result["verification" if result["syntax"] == "passed" else "syntax"] = (
        "unsupported"
      )
      line, column = fragment.position(source)
      report.diagnostics.append(
        Diagnostic(
          "unsupported", f"{type(exc).__name__}: {exc}", line, column, "unsupported"
        )
      )
    report.fragments.append(result)
  return report.emit(args.json)


if __name__ == "__main__":
  raise SystemExit(main())
