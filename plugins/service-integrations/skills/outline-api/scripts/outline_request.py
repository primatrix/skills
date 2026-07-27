#!/usr/bin/env python3
"""Send one JSON POST request to an Outline API method."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_CONFIG = Path.home() / ".config" / "outline-api" / "memory.json"
SENSITIVE_KEYS = {
    "access_token",
    "apikey",
    "api_key",
    "authorization",
    "client_secret",
    "collaborationtoken",
    "collaboration_token",
    "password",
    "refreshtoken",
    "refresh_token",
    "secret",
    "token",
}


def load_config(transport: str | None = None) -> dict[str, str | None]:
    """Load endpoint and credential settings without printing the credential."""

    config_path = Path(os.environ.get("OUTLINE_CONFIG_FILE", str(DEFAULT_CONFIG))).expanduser()
    config: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open(encoding="utf-8") as config_file:
            loaded = json.load(config_file)
        if not isinstance(loaded, dict):
            raise ValueError(f"config must be a JSON object: {config_path}")
        config = loaded

    selected_transport = transport or os.environ.get("OUTLINE_TRANSPORT") or config.get("active_transport", "primary")
    if not isinstance(selected_transport, str):
        raise ValueError("active_transport must be a string")

    target = config
    if selected_transport != "primary":
        override = config.get(f"{selected_transport}_transport")
        if not isinstance(override, dict):
            raise ValueError(f"transport is not configured: {selected_transport}")
        target = {**config, **override}

    url = os.environ.get("OUTLINE_URL") or target.get("url")
    api_key = os.environ.get("OUTLINE_API_KEY") or config.get("api_key")
    host_header = os.environ.get("OUTLINE_HOST") or target.get("host_header")
    forwarded_proto = os.environ.get("OUTLINE_FORWARDED_PROTO") or target.get("forwarded_proto")

    if not isinstance(url, str) or not url.strip():
        raise ValueError("missing Outline URL; set OUTLINE_URL or configure memory.json")
    if not isinstance(api_key, str) or not api_key.startswith("ol_api_"):
        raise ValueError("missing or invalid Outline API key; set OUTLINE_API_KEY or configure memory.json")

    return {
        "url": url.rstrip("/"),
        "api_key": api_key,
        "host_header": host_header if isinstance(host_header, str) and host_header else None,
        "forwarded_proto": forwarded_proto if isinstance(forwarded_proto, str) and forwarded_proto else None,
        "transport": selected_transport,
    }


def read_payload(args: argparse.Namespace) -> bytes:
    """Read and validate the JSON request body."""

    if args.data is not None and args.data_file is not None:
        raise ValueError("use only one of --data and --data-file")
    raw = args.data
    if args.data_file is not None:
        raw = Path(args.data_file).read_text(encoding="utf-8")
    if raw is None:
        raw = "{}"
    parsed = json.loads(raw)
    return json.dumps(parsed, ensure_ascii=False).encode("utf-8")


def redact(value: Any) -> Any:
    """Redact credential-like fields before a response is printed."""

    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key.lower() in SENSITIVE_KEYS else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def response_body(response: Any) -> str:
    """Decode and redact a response body for human-readable JSON output."""

    body = response.read().decode("utf-8", errors="replace")
    try:
        return json.dumps(redact(json.loads(body)), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return body


def main() -> int:
    """Send a request and return a non-zero status for API or transport failures."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, help="Outline method, for example documents.list")
    parser.add_argument("--data", help="JSON request body; defaults to {}")
    parser.add_argument("--data-file", help="Read the JSON request body from a file")
    parser.add_argument("--transport", help="Configured transport, for example primary or temporary")
    args = parser.parse_args()

    try:
        config = load_config(args.transport)
        payload = read_payload(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2

    method = args.method.strip().lstrip("/")
    if method.startswith("api/"):
        method = method[4:]
    if not method or "/" in method:
        print("method must be an Outline RPC method such as documents.info", file=sys.stderr)
        return 2

    api_origin = config["url"]
    if not isinstance(api_origin, str):
        print("configuration error: invalid URL", file=sys.stderr)
        return 2
    if not api_origin.endswith("/api"):
        api_origin = f"{api_origin}/api"
    request = Request(
        f"{api_origin}/{method}",
        data=payload,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config['api_key']}",
        },
    )
    if config["host_header"]:
        request.add_header("Host", str(config["host_header"]))
    if config["forwarded_proto"]:
        request.add_header("X-Forwarded-Proto", str(config["forwarded_proto"]))

    try:
        with urlopen(request, timeout=30) as response:
            body = response_body(response)
            print(body)
            try:
                result = json.loads(body)
            except json.JSONDecodeError:
                return 0
            return 0 if isinstance(result, dict) and result.get("ok") is True else 1
    except HTTPError as error:
        body = response_body(error)
        print(body, file=sys.stderr)
        retry_after = error.headers.get("Retry-After")
        if retry_after:
            print(f"retry-after: {retry_after}", file=sys.stderr)
        return 1
    except URLError as error:
        print(f"transport error: {error.reason}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
