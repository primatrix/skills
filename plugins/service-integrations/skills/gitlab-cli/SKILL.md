---
name: gitlab-cli
description: Use when inspecting or managing GitLab projects, merge requests, CI/CD pipelines, jobs, runners, variables, or permissions through glab CLI or the GitLab REST API, especially on self-hosted instances.
---

# GitLab CLI

Use `glab` and the GitLab REST API for GitLab work. Prefer read-only inspection first; make state changes only when the user explicitly requests them.

## Connect to the correct instance

Set the host and project explicitly. For a self-hosted GitLab instance, do not let `glab` default to `gitlab.com`.

```sh
HOST=gitlab.example.com
PROJECT=group/project
PROJECT_PATH="$(printf '%s' "$PROJECT" | sed 's|/|%2F|g')"

glab auth status --hostname "$HOST"
git remote get-url gitlab
```

If authentication is missing, ask the user to run `glab auth login --hostname "$HOST" --git-protocol ssh`. Never request or paste a personal access token into chat, commands, source files, CI variables, or logs. Stop and report the exact status code if the API returns 401 or 403.

## Read-only status checks

Use `glab api --hostname "$HOST"` with no request fields for GET requests. Check the latest pipeline and its jobs before making a decision:

```sh
glab api --hostname "$HOST" \
  "projects/$PROJECT_PATH/pipelines?ref=main&per_page=1"
glab api --hostname "$HOST" \
  "projects/$PROJECT_PATH/pipelines/<pipeline-id>/jobs?per_page=100"
glab api --hostname "$HOST" \
  "projects/$PROJECT_PATH/jobs/<failed-job-id>/trace"
```

For a failed pipeline, identify the failed job and its error from the trace before proposing any fix. Do not retry, cancel, trigger, merge, or deploy merely to investigate.

## State-changing operations

Obtain explicit user authorization immediately before each external state change: pushing a remote branch, creating or updating an MR, merging, changing CI/CD variables, changing runner settings, retrying jobs, or triggering deployments.

Before a merge, require all of the following:

- The target branch and source branch are correct; check the remote SHA again.
- The MR is not draft and is mergeable.
- The exact MR pipeline for the source SHA is successful.
- Required approvals and discussions satisfy project policy.

Use an expected source SHA to prevent merging a newer revision:

```sh
glab mr merge <mr-iid> --repo "$PROJECT" \
  --sha "$SOURCE_SHA" --auto-merge=false --yes
```

Create MRs explicitly; do not use `--web` or `--fill` (the latter can push implicitly):

```sh
glab mr create --repo "$PROJECT" \
  --source-branch "$BRANCH" --target-branch main \
  --title "$TITLE" --description "$DESCRIPTION" --draft --yes
```

## Production safety

Treat manual jobs named `deploy-production`, `verify-production`, or `rollback-production` as production-impacting actions. Never play, retry, cancel, or modify them without a separate, current user instruction that names the intended environment and action. Image build success does not authorize a production deployment.

When an API or CLI capability is absent, use `glab <command> --help` and the documented REST endpoint. If authorization, endpoint support, or a required value is missing, stop and ask the user; do not substitute GUI interaction or guess destructive defaults.
