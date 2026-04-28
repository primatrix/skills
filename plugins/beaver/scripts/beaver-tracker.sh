#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: beaver-tracker.sh <subcommand> [args]

Subcommands:
  ensure-label <repo> <name> <color> <description>
                                Idempotent label-create (skip on dup)
  find-tracker <repo> <yyyymm>  Search for tracker issue, echo JSON {count, items:[{number,title}]}
  list-carried <prev-number>    Open sub-issues of prior tracker, echo [{number, title, id}]
  create <repo> <title> <body-file>
                                Create tracker issue, echo .number
  add-labels <repo> <number> <label> [<label> ...]
  resolve-issue-id <repo> <number>
                                Echo numeric DB .id of issue
  attach-sub <tracker-number> <child-id>
                                Attach via sub_issues API
  detach-sub <tracker-number> <child-id>
                                Detach via sub_issues API DELETE
  fetch-backlog <repo>          GraphQL fetch Project V2 candidates:
                                Iteration empty AND Status=Triage AND
                                repo matches AND issueType in {Task, Bug}.
                                Echo [{number, title, repo}].
  list-tracker-subs <tracker-number>
                                Echo .[].number of sub-issues
  list-tracker-subs-meta <tracker-number> <repo>
                                Echo [{number, id, iteration_title, repo, repo_match, iteration_match}]
                                where matches are evaluated against expected repo and iteration title prefix YYYY-MM (passed via env BEAVER_EXPECTED_YYYYMM).
  set-tracker-iteration <tracker-number> <yyyymm>
                                Write Iteration field on tracker issue via beaver-lib.sh::set_iteration.
  set-issue-iteration <issue-number> <yyyymm>
                                Write Iteration field on a sub-issue via beaver-lib.sh::set_iteration.
EOF
}

ORG=primatrix
PROJECT_REPO=projects
PROJECT_NUM=14

# Resolve sibling beaver-lib.sh for delegating Project V2 field writes.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEAVER_LIB="${SCRIPT_DIR}/beaver-lib.sh"

case "${1:-}" in
  ensure-label)
    repo=$2; name=$3; color=$4; desc=$5
    gh label create "$name" --repo "${ORG}/${repo}" --color "$color" --description "$desc" 2>/dev/null || true
    ;;
  find-tracker)
    repo=$2; yyyymm=$3
    gh api -X GET search/issues \
      -f q="repo:${ORG}/${PROJECT_REPO} is:issue label:\"tracker/${repo}\" label:\"tracker/${yyyymm}\"" \
      --jq '{count: (.items | length), items: [.items[] | {number, state, title}]}'
    ;;
  list-carried)
    prev=$2
    gh api "repos/${ORG}/${PROJECT_REPO}/issues/${prev}/sub_issues" \
      -H "X-GitHub-Api-Version: 2026-03-10" \
      --jq '[.[] | select(.state=="open") | {number, title, id}]'
    ;;
  create)
    repo=$2; title=$3; body_file=$4
    gh api "repos/${ORG}/${PROJECT_REPO}/issues" --method POST \
      -f title="$title" \
      -F body=@"$body_file" \
      --jq '.number'
    ;;
  add-labels)
    repo=$2; num=$3; shift 3
    args=()
    for label in "$@"; do
      args+=(-f "labels[]=${label}")
    done
    gh api "repos/${ORG}/${PROJECT_REPO}/issues/${num}/labels" --method POST "${args[@]}"
    ;;
  resolve-issue-id)
    repo=$2; num=$3
    gh api "repos/${ORG}/${repo}/issues/${num}" --jq '.id'
    ;;
  attach-sub)
    tracker=$2; child_id=$3
    gh api "repos/${ORG}/${PROJECT_REPO}/issues/${tracker}/sub_issues" --method POST \
      -H "X-GitHub-Api-Version: 2026-03-10" \
      -F sub_issue_id="$child_id"
    ;;
  detach-sub)
    tracker=$2; child_id=$3
    gh api "repos/${ORG}/${PROJECT_REPO}/issues/${tracker}/sub_issue" --method DELETE \
      -H "X-GitHub-Api-Version: 2026-03-10" \
      -F sub_issue_id="$child_id"
    ;;
  fetch-backlog)
    # Backlog candidates: Project V2 #14 items where
    #   - Iteration is unset (fieldValueByName(Iteration) == null)
    #   - Status == "Triage"
    #   - issueType.name ∈ {Task, Bug}
    #   - repository.name == <repo>
    # Output: [{number, title, repo}]
    repo=$2
    gh api -H "GraphQL-Features: issue_types" graphql -f query='
      query($owner: String!, $number: Int!) {
        organization(login: $owner) {
          projectV2(number: $number) {
            items(first: 100) {
              nodes {
                content {
                  ... on Issue {
                    number
                    title
                    issueType { name }
                    repository { name nameWithOwner }
                  }
                }
                iter: fieldValueByName(name: "Iteration") {
                  ... on ProjectV2ItemFieldIterationValue { title }
                }
                stat: fieldValueByName(name: "Status") {
                  ... on ProjectV2ItemFieldSingleSelectValue { name }
                }
              }
            }
          }
        }
      }' -f owner="$ORG" -F number="$PROJECT_NUM" \
      --jq '(.data.organization.projectV2.items.nodes // [])
            | map(select(.content != null
                         and .content.repository.name == "'"$repo"'"
                         and .iter == null
                         and (.stat.name // "") == "Triage"
                         and ((.content.issueType.name // "") == "Task"
                              or (.content.issueType.name // "") == "Bug")))
            | map({number: .content.number,
                   title: .content.title,
                   repo: .content.repository.name})'
    ;;
  list-tracker-subs)
    tracker=$2
    gh api "repos/${ORG}/${PROJECT_REPO}/issues/${tracker}/sub_issues" \
      -H "X-GitHub-Api-Version: 2026-03-10" \
      --jq '.[].number'
    ;;
  list-tracker-subs-meta)
    # For each sub-issue under the tracker, fetch its Project V2 Iteration
    # title and home repo, plus boolean matches against the expected
    # YYYY-MM (env BEAVER_EXPECTED_YYYYMM) and expected <repo> (arg 3).
    tracker=$2
    expected_repo=$3
    expected_yyyymm="${BEAVER_EXPECTED_YYYYMM:-}"
    if [ -z "$expected_yyyymm" ]; then
      echo "list-tracker-subs-meta: BEAVER_EXPECTED_YYYYMM not set" >&2
      exit 1
    fi
    sub_numbers=$(gh api "repos/${ORG}/${PROJECT_REPO}/issues/${tracker}/sub_issues" \
      -H "X-GitHub-Api-Version: 2026-03-10" \
      --jq '[.[] | {number, id, repo: .repository.name}]')
    # For each sub, lookup Iteration title via Project V2.
    echo "$sub_numbers" | jq -c '.[]' | while IFS= read -r row; do
      number=$(echo "$row" | jq -r '.number')
      id=$(echo "$row" | jq -r '.id')
      repo=$(echo "$row" | jq -r '.repo')
      iter=$(gh api graphql -f query='
        query($owner: String!, $repo: String!, $number: Int!) {
          repository(owner: $owner, name: $repo) {
            issue(number: $number) {
              projectItems(first: 20) {
                nodes {
                  project { number }
                  fieldValueByName(name: "Iteration") {
                    ... on ProjectV2ItemFieldIterationValue { title }
                  }
                }
              }
            }
          }
        }' -f owner="$ORG" -f repo="$repo" -F number="$number" \
        --jq ".data.repository.issue.projectItems.nodes
              | map(select(.project.number == ${PROJECT_NUM}))
              | .[0].fieldValueByName.title // \"\"" 2>/dev/null || echo "")
      jq -n \
        --argjson number "$number" \
        --argjson id "$id" \
        --arg iter "$iter" \
        --arg repo "$repo" \
        --arg expected_repo "$expected_repo" \
        --arg expected_yyyymm "$expected_yyyymm" \
        '{number: $number,
          id: $id,
          iteration_title: $iter,
          repo: $repo,
          repo_match: ($repo == $expected_repo),
          iteration_match: ($iter | startswith($expected_yyyymm))}'
    done | jq -s '.'
    ;;
  set-tracker-iteration|set-issue-iteration)
    subcmd=$1; target=$2; yyyymm=$3
    max_retries=3
    for attempt in $(seq 1 $max_retries); do
      if bash "$BEAVER_LIB" set_iteration "$target" "$yyyymm"; then
        # Verify the Iteration was actually set.
        actual=$(bash "$BEAVER_LIB" get_iteration "$target" 2>/dev/null || echo "")
        if [ -n "$actual" ] && echo "$actual" | grep -q "^${yyyymm}"; then
          break
        fi
        echo "$subcmd: attempt $attempt: verification failed (got '$actual'), retrying..." >&2
      else
        echo "$subcmd: attempt $attempt failed, retrying..." >&2
      fi
      if [ "$attempt" = "$max_retries" ]; then
        echo "$subcmd: FAILED after $max_retries attempts for #$target" >&2
        exit 1
      fi
      sleep 2
    done
    ;;
  --help|"")
    usage
    ;;
  *)
    echo "unknown subcommand: $1" >&2
    usage >&2
    exit 1
    ;;
esac
