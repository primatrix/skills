#!/usr/bin/env python3
"""Initialize or inspect the Outline MCP endpoint using the shared memory config."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from outline_request import load_config, redact


def format_sse(body: str) -> str:
    """Redact JSON data frames in an SSE response before printing them."""

    lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("data: "):
            raw_data = line[6:]
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                lines.append(line)
            else:
                lines.append(f"data: {json.dumps(redact(data), ensure_ascii=False)}")
        else:
            lines.append(line)
    return "\n".join(lines)


def request_body(method: str) -> bytes:
    """Build a JSON-RPC request body for the selected MCP method."""

    params: dict[str, Any] = {}
    if method == "initialize":
        params = {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "outline-api-skill", "version": "1.0"},
        }
    return json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        ensure_ascii=False,
    ).encode("utf-8")


def main() -> int:
    """Send one MCP JSON-RPC request and return a non-zero status on failure."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["initialize", "tools/list"], default="initialize")
    parser.add_argument("--transport", help="Configured transport, for example primary or temporary")
    args = parser.parse_args()

    try:
        config = load_config(args.transport)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2

    url = config["url"]
    api_key = config["api_key"]
    if not isinstance(url, str) or not isinstance(api_key, str):
        print("configuration error: invalid URL or API key", file=sys.stderr)
        return 2

    request = Request(
        f"{url.rstrip('/')}/mcp",
        data=request_body(args.method),
        method="POST",
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    if config["host_header"]:
        request.add_header("Host", str(config["host_header"]))
    if config["forwarded_proto"]:
        request.add_header("X-Forwarded-Proto", str(config["forwarded_proto"]))

    try:
        with urlopen(request, timeout=30) as response:
            body = format_sse(response.read().decode("utf-8", errors="replace"))
            print(f"status={response.status} content_type={response.headers.get('Content-Type')}")
            print(body)
            return 0
    except HTTPError as error:
        body = format_sse(error.read().decode("utf-8", errors="replace"))
        print(f"status={error.code} content_type={error.headers.get('Content-Type')}", file=sys.stderr)
        print(body, file=sys.stderr)
        return 1
    except URLError as error:
        print(f"transport error: {error.reason}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
