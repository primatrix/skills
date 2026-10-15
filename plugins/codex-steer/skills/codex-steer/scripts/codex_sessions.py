"""Find, tail, and check idleness of local Codex sessions (rollout JSONL under ~/.codex/sessions).

usage:
  codex_sessions.py list [N]                 # most recently written sessions: thread id, mtime, size, first user prompt
  codex_sessions.py tail <thread> [--peek]   # new assistant/user messages, key tool calls, reasoning since last tail
  codex_sessions.py status <thread>          # JSON: idle?, active turn id, minutes since last write
  codex_sessions.py successor <thread>       # newest non-subagent thread forked from it (after archive/continue)
State (byte offsets for `tail`) is kept in $CODEX_STEER_STATE_DIR (default ~/.cache/codex-steer/state)/<thread>.json.
"""
import glob, json, os, sys, time

ROOT = os.path.expanduser('~/.codex/sessions')
STATE_DIR = os.path.expanduser(os.environ.get('CODEX_STEER_STATE_DIR', '~/.cache/codex-steer/state'))
KEY_TOOL_WORDS = ('outline doc', 'falcon', 'experiment', 'submit', 'docker run', 'git commit', 'git push', 'aot')


def rollout(thread):
    paths = glob.glob(f'{ROOT}/*/*/*/rollout-*{thread}*.jsonl')
    if not paths:
        archived = glob.glob(os.path.expanduser(f'~/.codex/archived_sessions/rollout-*{thread}*.jsonl'))
        hint = successor(thread)
        sys.exit(f'no live rollout for thread {thread}' + (' (archived)' if archived else '')
                 + (f'; successor (forked from it, newest): {hint}' if hint else ''))
    return max(paths, key=os.path.getmtime)


def meta(path):
    with open(path) as f:
        for _, line in zip(range(30), f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get('type') == 'session_meta':
                return d.get('payload', {})
    return {}


def successor(thread):
    """Newest user (non-subagent) thread forked from `thread`, following the fork chain."""
    paths = sorted(glob.glob(f'{ROOT}/*/*/*/rollout-*.jsonl'), key=os.path.getmtime, reverse=True)[:200]
    for p in paths:
        m = meta(p)
        if m.get('forked_from_id') == thread and m.get('thread_source') != 'subagent':
            return successor(m['id']) or m['id']
    return None


def events(path, start=0, tail_bytes=None):
    size = os.path.getsize(path)
    if tail_bytes is not None:
        start = max(0, size - tail_bytes)
    with open(path, 'rb') as f:
        f.seek(start)
        data = f.read()
    lines = data.splitlines()
    if start and lines:
        lines = lines[1:]  # first line may be partial
    for raw in lines:
        try:
            yield json.loads(raw)
        except Exception:
            continue


def thread_id(path):
    name = os.path.basename(path)[len('rollout-'):-len('.jsonl')]
    return name[20:]  # strip "YYYY-MM-DDTHH-MM-SS-"


def first_prompt(path):
    for d in events(path):
        p = d.get('payload', {})
        if p.get('type') == 'message' and p.get('role') == 'user':
            text = ' '.join(c.get('text', '') for c in p.get('content', []) if isinstance(c, dict))
            if text and not text.lstrip().startswith('<'):
                return text.replace('\n', ' ')[:80]
    return ''


def status(thread):
    path = rollout(thread)
    turn, last = None, None
    for d in events(path, tail_bytes=3_000_000):
        p = d.get('payload', {})
        last = p.get('type') or d.get('type')
        if last == 'task_started':
            turn = p.get('turn_id')
        elif last == 'task_complete' and p.get('turn_id') == turn:
            turn = None
    minutes = (time.time() - os.path.getmtime(path)) / 60
    return {'thread': thread, 'idle': turn is None or minutes > 8, 'active_turn': turn,
            'last_event': last, 'minutes_since_write': round(minutes, 1), 'rollout': path}


def tail(thread, peek):
    path = rollout(thread)
    os.makedirs(STATE_DIR, exist_ok=True)
    state_file = f'{STATE_DIR}/{thread}.json'
    start = json.load(open(state_file)).get('offset', 0) if os.path.exists(state_file) else 0
    if start > os.path.getsize(path):
        start = 0
    out = []
    for d in events(path, start):
        p = d.get('payload', {})
        t = p.get('type') or d.get('type')
        ts = d.get('timestamp', '')[11:19]
        if t == 'message' and p.get('role') in ('assistant', 'user'):
            text = ' '.join(c.get('text', '') for c in p.get('content', []) if isinstance(c, dict))
            if text.strip() and not text.lstrip().startswith('<'):
                out.append(f'{ts} [{p["role"]}] {text[:1500]}')
        elif t in ('function_call', 'custom_tool_call'):
            s = str(p.get('arguments') or p.get('input'))
            if any(k in s for k in KEY_TOOL_WORDS):
                out.append(f'{ts} [tool] {s[:400]}')
        elif t == 'reasoning':
            s = ' '.join(x.get('text', '') for x in p.get('summary', []))
            if s:
                out.append(f'{ts} [think] {s[:200]}')
        elif t == 'task_complete':
            out.append(f'{ts} [turn complete]')
    end = os.path.getsize(path)
    print(f'thread {thread}: bytes {start}->{end}, {len(out)} items')
    print('\n'.join(out[-80:]))
    if not peek:
        json.dump({'offset': end}, open(state_file, 'w'))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'list'
    if cmd == 'list':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        paths = sorted(glob.glob(f'{ROOT}/*/*/*/rollout-*.jsonl'), key=os.path.getmtime, reverse=True)[:n]
        for p in paths:
            age = (time.time() - os.path.getmtime(p)) / 60
            m = meta(p)
            kind = 'subagent' if m.get('thread_source') == 'subagent' else 'MAIN    '
            fork = f' forked_from={m.get("forked_from_id", "")[:13]}' if m.get('forked_from_id') else ''
            print(f'{thread_id(p)}  {kind} {age:6.1f} min ago  {os.path.getsize(p) // 1024:7d} KiB{fork}  {first_prompt(p)[:60]}')
    elif cmd == 'tail':
        tail(sys.argv[2], '--peek' in sys.argv)
    elif cmd == 'successor':
        print(successor(sys.argv[2]) or '')
    elif cmd == 'status':
        print(json.dumps(status(sys.argv[2]), ensure_ascii=False))
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
