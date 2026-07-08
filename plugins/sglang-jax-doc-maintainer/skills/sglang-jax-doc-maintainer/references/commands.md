# 常用命令模板

只在确认路径和权限后运行。

## PR 输入

```bash
gh pr view <pr> --json title,body,files,commits
gh pr diff <pr>
```

## 本地 diff

```bash
git -C <code-root> diff <base>...HEAD
git -C <code-root> log --oneline <range>
git -C <code-root> diff <range>
```

base branch 不明确时先询问用户。

## Workspace 与验证

```bash
git -C <wiki-root> status --short
git -C <wiki-root> diff --name-only
git -C <wiki-root> diff --check
git -C <wiki-root> diff -- <docs-root>
```

## 搜索证据

优先使用 Claude Code 的 Grep / Glob / Read 工具搜索 symbol、config field、文件路径和文档引用。
