---
name: outline-api
description: Use when an agent needs to inspect, create, update, publish, search, move, or manage documents in an Outline knowledge base through its API or MCP endpoint, including self-hosted Outline instances with custom URLs, API key scopes, or proxy/Ingress headers.
---

# Outline API

Use this skill for agent-friendly access to a self-hosted or hosted Outline workspace. Keep the skill shareable: never put a real API key, private URL, or workspace-specific IDs in this directory.

## Configuration

Load credentials in this order:

1. Environment: `OUTLINE_URL`, `OUTLINE_API_KEY`, `OUTLINE_HOST`, `OUTLINE_FORWARDED_PROTO`, `OUTLINE_TRANSPORT`.
2. JSON memory file at `$OUTLINE_CONFIG_FILE`, or by default `$HOME/.config/outline-api/memory.json`.

Expected memory fields:

```json
{
  "url": "https://kb.example.com",
  "api_key": "ol_api_REPLACE_ME",
  "host_header": null,
  "forwarded_proto": null
}
```

`url` is the API origin; the helper appends `/api`. An optional `active_transport` selects a named `*_transport` object, for example `temporary_transport`, when the canonical URL is unavailable. Use `host_header` and `forwarded_proto` only for a documented reverse-proxy or temporary NodePort path. Prefer the canonical HTTPS URL when SNI and certificate routing are healthy. Never print, commit, paste into a document, or include the API key in logs.

## Workflow

1. Load the memory/config and check that the API key starts with `ol_api_`; do not echo it. Check the selected transport before diagnosing the service as unavailable.
2. Run `scripts/outline_request.py --method auth.info --data '{}'` as a preflight.
3. Use the smallest scope needed. Use `read` for inspection, `create` for new documents, and `write` only for mutations that require it. Confirm the target collection/document before writing.
4. Verify the response has `ok: true`; record the returned document ID and URL ID, but not the key.
5. For bulk work, paginate with the returned `pagination.nextPath` or explicit `limit`/`offset`, and back off on `429` using `Retry-After`.

## Changes require current consent

Before every state-changing API or MCP call, obtain an explicit current user instruction that names the target and action. This includes creating, updating, publishing, moving, archiving, deleting, sharing, importing, exporting, or commenting on documents and collections.

Re-read the target immediately before a consequential change. Use the narrowest mutation, confirm the response is `ok: true`, and verify the document or collection state afterwards. Never infer write approval from an earlier read request.

## Common Operations

The helper sends JSON POST requests and redacts authorization from errors:

```bash
python3 /path/to/outline-api/scripts/outline_request.py \
  --method collections.list --data '{"limit":25,"offset":0}'

python3 /path/to/outline-api/scripts/outline_request.py \
  --method documents.create \
  --data '{"title":"Agent note","text":"Markdown body","collectionId":"COLLECTION_UUID","publish":true}'

python3 /path/to/outline-api/scripts/outline_request.py \
  --method documents.update \
  --data '{"id":"DOCUMENT_ID","text":"New Markdown body","editMode":"replace","publish":true}'

# Explicitly select a configured fallback transport when needed.
python3 /path/to/outline-api/scripts/outline_request.py \
  --transport temporary --method auth.info --data '{}'
```

For `documents.update`, use `editMode: "replace"` for a complete body, `append` or `prepend` for additive changes, and `patch` with `findText` for a targeted replacement. Treat document content as Markdown; keep the document title in `title` rather than adding a redundant first-level heading unless the workspace convention requires it.

## Scope and Attribution

API actions are attributed to the user who created the key. A shared admin key makes every write look like that admin; prefer one key per human or agent identity when audit attribution matters. For a typical agent, start with `read` or `documents:read`; add `documents:create` for document creation and only grant broader write access when needed.

## Troubleshooting

- `401`: key is missing, revoked, malformed, or loaded from the wrong memory file.
- `403`: the key scope or the user’s Outline permission does not cover the method/resource.
- `405` on a self-hosted NodePort: the proxy may require `X-Forwarded-Proto: https`; configure `forwarded_proto` only when the route is known to terminate HTTPS upstream.
- TLS/SNI failure: do not disable certificate verification. Check `active_transport` and explicitly test the configured temporary transport before concluding the API is unavailable. Return to `primary` after fixing the CLB/Ingress certificate route.
- `429`: wait for `Retry-After`; do not tight-loop retries.
- A successful API response does not imply a published document; check `publish`, document status, and the document URL separately.

## MCP

When the task is better expressed as tool discovery or conversational document work, use the Outline MCP endpoint at `<selected-url>/mcp` with the same bearer token. Send `Accept: application/json, text/event-stream`, initialize with protocol version `2025-03-26`, then call `tools/list` before `tools/call`. If the selected transport is temporary, preserve its `Host` and `X-Forwarded-Proto` headers. API and MCP share the same user permissions; use the API helper for deterministic CRUD and MCP for tool-oriented workflows.

For a deterministic connectivity check, run `scripts/outline_mcp.py --method initialize`; use `--transport temporary` only when that transport is configured in memory.

Read [references/api.md](references/api.md) for the official API contract and [references/memory.example.json](references/memory.example.json) before configuring another user’s instance.
