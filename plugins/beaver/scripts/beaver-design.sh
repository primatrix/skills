#!/usr/bin/env bash
set -euo pipefail

WIKI_REPO="primatrix/wiki"
WIKI_DEFAULT="${HOME}/Code/wiki"

usage() {
  cat <<'EOF'
Usage: beaver-design.sh <subcommand> [args]

Subcommands:
  prepare-wiki [<wiki-dir>]
        Default ~/Code/wiki. If exists: cd && git checkout main && git pull.
        Else: gh repo clone primatrix/wiki <wiki-dir>.
  create-branch <wiki-dir> <branch>
        cd <wiki-dir> && git checkout -b <branch>
  commit-push <wiki-dir> <file> <message> <branch>
        cd <wiki-dir>; git add <file>; git commit -m <message>;
        git push -u origin <branch>
  create-pr <repo> <title> <body>
        gh pr create --repo <repo> --draft --title <title> --body <body>
  comment-issue <org> <repo> <number> <body>
        gh api repos/<org>/<repo>/issues/<number>/comments --method POST
        --raw-field body=<body>
EOF
}

case "${1:-}" in
  prepare-wiki)
    dir="${2:-$WIKI_DEFAULT}"
    if [ -d "$dir" ]; then
      cd "$dir"
      git checkout main
      git pull
    else
      gh repo clone "$WIKI_REPO" "$dir"
    fi
    ;;
  create-branch)
    dir=$2; branch=$3
    cd "$dir"
    git checkout -b "$branch"
    ;;
  commit-push)
    dir=$2; file=$3; message=$4; branch=$5
    cd "$dir"
    git add "$file"
    git commit -m "$message"
    git push -u origin "$branch"
    ;;
  create-pr)
    repo=$2; title=$3; body=$4
    gh pr create --repo "$repo" --draft --title "$title" --body "$body"
    ;;
  comment-issue)
    org=$2; repo=$3; num=$4; body=$5
    gh api "repos/${org}/${repo}/issues/${num}/comments" --method POST \
      --raw-field body="$body"
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
