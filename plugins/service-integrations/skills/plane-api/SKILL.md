---
name: plane-api
description: Use when reading or managing Plane projects, work items, cycles, modules, pages, members, or workspace settings through the configured Plane MCP tools.
---

# Plane Operations

Use the registered `mcp__plane__*` tools. They keep authentication in Codex configuration; do not use a raw API key or fall back to unauthenticated HTTP.

## Persistent setup

Plane is configured in `~/.codex/config.toml` under `[mcp_servers.plane]`. It supplies `PLANE_BASE_URL`, `PLANE_API_KEY`, and `PLANE_WORKSPACE_SLUG` when Codex starts. Never print, copy, commit, or ask the user to paste the key.

In a fresh session, confirm that `mcp__plane__get_me` and other `mcp__plane__*` tools are available. If they are absent or return authentication errors, report that fact; do not expose the configuration values or construct a curl command containing the key.

## Read first

1. Call `mcp__plane__get_me` to confirm the active identity when access is relevant.
2. Use `mcp__plane__list_projects` to resolve a project name or identifier to its UUID.
3. Use the smallest matching read tool, such as `retrieve_project`, `list_work_items`, `retrieve_work_item_by_identifier`, `list_cycles`, `list_modules`, or `list_pages`.
4. For complex PQL, call `mcp__plane__get_pql_reference` before composing a filter. Do not guess UUIDs or identifiers.

Example: For “summarize open work in INFERENCE,” list projects, match `identifier: INFERENCE`, then list its work items with an explicitly justified PQL filter. Report names, states, owners, and links; keep UUIDs out of the user-facing summary unless they help the task.

## Changes require current consent

Before every external mutation, obtain an explicit current instruction that names the target and action. This includes create, update, archive, delete, linking/unlinking, changes to assignees/labels/states, time logs, and workspace or project feature settings.

Re-read the target immediately before a consequential update or delete. Never infer approval from an earlier read request. Prefer the narrowest operation and verify the returned object afterwards.

## Quick reference

| Task | Preferred tools |
| --- | --- |
| Identify account or projects | `mcp__plane__get_me`, `mcp__plane__list_projects`, `mcp__plane__retrieve_project` |
| Find a work item | `mcp__plane__retrieve_work_item_by_identifier`, `mcp__plane__search_work_items`, `mcp__plane__list_work_items` |
| Inspect planning | `mcp__plane__list_cycles`, `mcp__plane__list_modules`, `mcp__plane__list_milestones` |
| Read discussion or files | `mcp__plane__list_work_item_comments`, `mcp__plane__list_work_item_attachments`, `mcp__plane__read_work_item_attachment` |
| Make an authorized change | Matching `mcp__plane__create_*`, `mcp__plane__update_*`, `mcp__plane__manage_*`, or `mcp__plane__delete_*` tool |

## Common mistakes

- Do not search arbitrary local files for Plane credentials; the MCP configuration is the persistent integration.
- Do not list a whole workspace when a named project or work item is sufficient.
- Do not use a write-capable tool merely to investigate.
- Do not delete, archive, complete a cycle, or alter workspace features without fresh explicit authorization.
