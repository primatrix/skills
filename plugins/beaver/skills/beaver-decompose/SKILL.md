---
name: beaver-decompose
description: "Decompose a Beaver Goal into Tasks (size/L) or a Task into SubTasks (size/S), guided by a design doc. Trigger when the user wants to split, breakdown, or decompose an issue into sub-issues."
argument-hint: "<owner/repo#issue-number> --design-doc <url>"
---

# Beaver Decompose

Decompose a `Goal` Issue into `Task` sub-issues (size/L), or a `Task` Issue into `SubTask` sub-issues (size/S). Reads a required design doc, drafts the decomposition, confirms each child issue with the user one at a time, then creates them via `gh api`.

**References beaver-engine for:** label ops (Section 4), state machine (Section 2), guardrails G001 (Section 3), transition execution (Section 6).
