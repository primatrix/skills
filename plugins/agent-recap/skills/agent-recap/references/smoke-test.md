# agent-recap — End-to-end smoke test

Run this manually before submitting a PR that changes any agent-recap file. It is
not automated because it requires the Claude Code Agent tool (LLM-driven).

## Setup

1. Have at least one Claude session with code changes from the past 24 hours.
   If not, run a short Claude session that edits a file and commits.

## Steps

1. In Claude Code, run:
   ```
   /agent-recap 1d
   ```

2. Verify Stage 1 (scanner output):
   - The skill calls `scan_sessions.py --since 1d` and reports session count.
   - No exception traceback in the output.

3. Verify Stage 3 (recap printed):
   - The printed Markdown has all 5 sections: ✅ 解决, 🔎 调研, 👀 Review, 🚧 被 Block, 🗒️ 杂项.
   - Each entry shows `[<project> <#issue or ⚠️未匹配issue>] <topic>`, evidence, and source.
   - The ⚠️ 解析失败 section appears at the end (may be empty).

4. Verify Stage 4 (human review):
   - Say "删掉第 1 条" — the next printed recap omits item 1 and re-numbers.
   - Say "第 2 条改成 测试条目" — verify the topic updates.
   - Say "确认无误".

5. Verify Stage 5 (sync prompt):
   - For an unmatched-issue entry, choose option (a) skip.
   - Verify the dry-run list is printed BEFORE any `gh` calls happen.
   - Reply "取消" — verify no `gh` calls were made.

6. Verify intents.json side effects:
   - `ls ~/.agent-recap/*-intents.json` shows one new file with today's timestamp.
   - Open it and verify the `actions[]` shape matches the spec.

7. Verify cleanup:
   - Touch a fake old file: `touch -t 200001010000 ~/.agent-recap/old-intents.json`
   - Run `/agent-recap 1d` again.
   - Verify the old file is gone after Stage 5.0 cleanup.
   - Touch `~/.agent-recap/keep-old-intents.json` with same old date and verify it is NOT deleted.

## Pass criteria

All 7 steps succeed without manual intervention beyond the documented user replies.
