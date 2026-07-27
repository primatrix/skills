# Outline API Reference

Source: [Outline API documentation](https://www.getoutline.com/developers#description/policies).

## Contract

- Outline uses RPC-style endpoints. Each method is a JSON `POST` request at `<origin>/api/<method>`.
- Send `Accept: application/json`, `Content-Type: application/json`, and `Authorization: Bearer <API_KEY>`.
- Successful responses use HTTP `200` or `201` and include `ok: true`.
- Errors return an appropriate HTTP status and a JSON body such as `{"ok": false, "error": "Not Found"}`.

## Authentication

API keys are created from Outline settings under **API & Apps**. Treat them like passwords: do not commit or expose them. A revoked key returns `401 Unauthenticated`. OAuth 2.0 is the alternative for an application that should obtain delegated access without sharing a user key.

## Scopes

Scopes can be global, namespaced, or endpoint-specific:

| Scope | Meaning |
| --- | --- |
| `read` | All read actions |
| `write` | All read and write actions |
| `documents:read` | Document read actions |
| `collections:write` | Collection write actions |
| `documents.info` | One API method |
| `documents.*` | All document methods |
| `users.*` | All user methods |

Prefer the narrowest scope that supports the task. The key owner and the user’s workspace permissions still constrain the effective access.

## Documents

The most useful methods for an agent are:

| Method | Typical body | Purpose |
| --- | --- | --- |
| `documents.list` | `{"collectionId":"...","limit":25,"offset":0}` | Browse documents with pagination |
| `documents.info` | `{"id":"UUID-or-urlId"}` | Read one document |
| `documents.search` | Search filters | Find documents by content/metadata |
| `documents.create` | `{"title":"...","text":"...","collectionId":"...","publish":true}` | Create or publish a document |
| `documents.update` | `{"id":"...","text":"...","editMode":"replace","publish":true}` | Modify a document |

`documents.create` can target a collection root or a child via `parentDocumentId`. `documents.update` supports complete replacement and targeted text editing; for a patch, provide `findText` together with the replacement `text`. Use the document UUID or URL ID where accepted by the instance.

## Pagination, rate limits, and policies

List methods accept `limit` and `offset`; responses echo them in `pagination` and may provide `pagination.nextPath`. Mutating endpoints are more restricted than read-only endpoints. On `429 Too Many Requests`, honor the `Retry-After` header. Resource policies describe what the authenticated identity may do for that resource; most clients can rely on authorization responses and only inspect policy data when building a UI or audit tool.

## Self-hosted routing

The official contract assumes HTTPS. For a self-hosted reverse proxy, keep the public URL and certificate/SNI routing correct. A temporary NodePort path may require an explicit `Host` header and `X-Forwarded-Proto: https`; configure those values in local memory only and do not treat that workaround as the production URL.
