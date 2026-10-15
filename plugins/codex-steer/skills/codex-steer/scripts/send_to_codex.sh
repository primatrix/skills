#!/bin/bash
# usage: send_to_codex.sh <thread_id> <message_file> [--queue-only]
#
# Deliver a message to a Codex session that is open in the ChatGPT/Codex desktop app.
#  1. `codex queue` adds it to the thread's message queue (the app shows it above the composer).
#  2. The AX helper presses the queued message's 「引导」(steer) button, injecting it into the
#     running turn, exactly like clicking 引导 by hand.
# If the thread is idle (no running turn) there is no 引导 button; the queued message starts the next turn.
set -u
THREAD="$1"; MSG_FILE="$2"; MODE="${3:-}"
DIR="$(cd "$(dirname "$0")" && pwd)"
CODEX=/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex
AX="$DIR/codex_ax_steer"

[ -x "$AX" ] || swiftc -O "$DIR/codex_ax_steer.swift" -o "$AX" || { echo "failed to build AX helper"; exit 1; }

"$CODEX" queue --thread "$THREAD" --message "$(cat "$MSG_FILE")" | tail -1 || exit 1
[ "$MODE" = "--queue-only" ] && exit 0

if [ "$(python3 "$DIR/codex_sessions.py" status "$THREAD" | python3 -c 'import json,sys;print(json.load(sys.stdin)["active_turn"] is not None)')" != "True" ]; then
  echo "thread idle: queued message will start the next turn (no steer needed)"; exit 0
fi

sleep 3
for _ in 1 2 3 4; do
  # Chromium publishes the web AX tree lazily; a fresh process reading AXWindows wakes it.
  [ -x "$DIR/codex_ax_wake" ] || swiftc -O "$DIR/codex_ax_wake.swift" -o "$DIR/codex_ax_wake"
  "$DIR/codex_ax_wake" >/dev/null 2>&1
  out=$("$AX" press 2>&1)
  if echo "$out" | grep -q '^press 0'; then
    echo "STEERED ($(echo "$out" | grep -c '^press 0') queued message(s) injected into the running turn)"; exit 0
  fi
  sleep 3
done
echo "WARNING: 引导 button not found. The app window must be showing this thread (sidebar selection)."
echo "Remaining queue:"
python3 "$DIR/cx_rpc.py" "[[\"thread/queue/list\",{\"threadId\":\"$THREAD\"}]]" | tail -1 | cut -c1-300
exit 2
