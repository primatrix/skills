---
name: session-recorder
description: Records the complete session content and logs it to a daily work directory with a dynamic filename based on the active CLI agent. Use this for automated progress tracking and documentation.
---

# Session Recorder

This skill enables an AI agent to keep a precise log of its interactions. It records the full history of questions and answers into a date-organized directory structure.

## Workflow

1.  **Identify Yourself**: Determine your active CLI name (e.g., "gemini", "codex", "claude", or "assistant"). This will be used as the prefix for the log file.
2.  **Extract Full Session Content**: For every user question and assistant answer, record the **complete** history. **Do not summarize or paraphrase**; log the exact interaction as provided to the user.
3.  **Filter Content**: Exclude any cancelled operations, tool execution errors that didn't result in an answer, or redundant summaries.
4.  **Log Work**: Execute the `scripts/record_session.py` script to append the content to a file named `{$cli-name}-session.md`.

### Prerequisites

- Default base directory for logs: `~/daily_work` (automatically expands to user's home directory).

### Commands

Run the following command, replacing `<cli-name>` with your actual identity:
```bash
python scripts/record_session.py "<cli-name>" "<one_sentence_title>" "<full_unedited_session_content>"
```

## Example

If the active agent is **Codex** and the session is about a bug fix:
- **CLI Name**: codex
- **Title**: Fixed null pointer exception in auth module
- **Content**: [The complete history of the fix, including tests]
