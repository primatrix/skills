---
name: codex-steer
description: Supervise and talk to a Codex agent running in the local ChatGPT/Codex desktop app. Find its session, tail what it is doing, detect when it is idle, and send it messages that are injected into its running turn (the app's 「引导」/steer button), not just queued. Use when the user asks to monitor, nudge, correct, review, or coordinate with Codex, or says "引导", "督促 codex", "给 codex 发消息", "codex 停了".
---

# Codex steer：监控并引导本地 Codex 会话

在本机 ChatGPT（Codex）桌面 App 里运行的 Codex，可以用这套工具做四件事：找到会话、跟踪它在做什么、判断它是否空闲、给它发消息。发出去的消息会**插入它正在运行的 turn**（等同于点 App 里的「引导」），而不只是排进队列等这一轮结束。

所有脚本都在本 skill 目录的 `scripts/` 下，下文用 `$S` 指代这个目录（例如 `S=<skill 目录>/scripts`）。`tail` 的读取偏移保存在 `~/.cache/codex-steer/state/`，可用 `CODEX_STEER_STATE_DIR` 覆盖。

**仅支持 macOS**：依赖 ChatGPT/Codex 桌面 App、`swiftc` 和辅助功能（AX）权限。

## 1. 找到目标会话

```bash
python3 $S/codex_sessions.py list 10
```

输出每个会话的 thread id、距离上次写入的时间、文件大小和首条用户 prompt。

- 输出里会标出 `MAIN` 或 `subagent`，以及 `forked_from`。**只选 `MAIN`。** Codex 会派生子 agent 线程，它们的首条 prompt 往往是转发来的消息，文件也比较小（几百 KiB）。主会话通常文件最大、开始时间最早，首条 prompt 是用户原始任务。
- 如果用户提到了具体任务，用 prompt 内容匹配。
- **会话会换 ID**：用户在 App 里点「从聊天中继续」，或者旧会话被归档时，原 thread 会移到 `~/.codex/archived_sessions/`，同时 fork 出一个新的 MAIN 线程，首条 prompt 往往只是“继续”。这时 `tail/status` 会报 `no live rollout ... successor: <新 id>`。用 `codex_sessions.py successor <旧 id>` 拿到新 id，同时更新所有轮询任务里的 thread。

## 2. 看它在做什么

```bash
python3 $S/codex_sessions.py tail <thread>          # 显示上次 tail 之后的新动作，并推进偏移
python3 $S/codex_sessions.py tail <thread> --peek   # 只看，不推进偏移
python3 $S/codex_sessions.py status <thread>        # JSON：idle、active_turn、最后一个事件、几分钟没写入
```

- tail 会输出 Codex 的发言、它收到的用户消息、关键工具调用（outline/falcon/提交/AOT/git）、推理摘要，以及 `[turn complete]` 标记。
- `idle=true` 有两种情况：没有正在运行的 turn（最后一个 turn 已经 task_complete），或者超过 8 分钟没有写入。

## 3. 给它发消息（默认用引导）

```bash
cat > /tmp/msg.md <<'EOF'
【Claude → Codex】……
EOF
$S/send_to_codex.sh <thread> /tmp/msg.md            # 先排队，再自动点「引导」
$S/send_to_codex.sh <thread> /tmp/msg.md --queue-only
```

脚本的执行流程：
1. 用 `codex queue` 把消息加到线程的队列里。App 会在输入框上方显示这条排队消息，带「引导」按钮。
2. 如果线程有正在运行的 turn，用 Swift 写的 AX 辅助工具 `codex_ax_steer` 在 App 窗口的辅助功能树里找到「引导」按钮并按下。看到 `STEERED` 就说明消息已经插进当前 turn。
3. 如果线程空闲，就不需要引导，排队的消息会自动开启下一个 turn。

发送后用 `tail` 确认：应当能看到一行 `[user] 你的消息`，时间戳在发送之后。

Chromium 只会懒加载网页部分的 AX 树。App 不在前台时，窗口通常只暴露约 14 个节点。脚本每次按按钮之前，会先运行 `codex_ax_wake`：在一个新进程里读一次 `AXWindows`，读完树就会展开（约 1300 个节点）。

退出码 `2` 表示没找到「引导」按钮。最常见的原因是 App 窗口当前显示的不是这个线程。左侧栏选中的会话才会渲染它的排队消息和按钮。其它原因：窗口被最小化，或者终端没有辅助功能权限。可以用 `screencapture -x -o -l <CGWindowID> /tmp/cx.png` 截下 ChatGPT 窗口看实际显示了什么；窗口 ID 用 `CGWindowListCopyWindowInfo` 按 ChatGPT 的 pid 过滤得到。处理方法：让用户在 App 里打开那个会话；或者换 `--queue-only`，等当前 turn 结束后消息自然生效。

### 写消息的要点
- 第一行写清楚是谁发给谁、核心意思是什么，例如「【Claude → Codex：用户要求加速】」。
- 给**具体、可以立即执行的下一步**，标好优先级，并写明「完成后继续下一项，不要停」。
- 指正时先说它做对了什么，再说问题和改法。
- 消息里只放结论和指令。长篇规则写进共享文档，消息里给链接或 ID。

## 4. 持续督促（可选）

用户要求盯住 Codex、不许它停工时，用 CronCreate 设一个每 10 分钟的轮询，每轮执行：
1. `status`：如果 `idle=true` 且任务没有结束，用 `send_to_codex.sh` 下发下一步任务。
2. `tail`：检查有没有偏离约定（例如用错指标、跳过验证、未经批准做高风险动作）。发现偏离就立刻引导纠正，不用等它空闲。
3. 有书面协作文档（台账、评审页）的话，同时读这些文档并写评审。

cron 任务只在当前会话有效，会话结束就失效。如需持续监控，在新会话里重新建立。

## 不能用的方法（已经验证过）
| 方法 | 结果 |
|---|---|
| `codex exec resume <thread> ...` | 报错 `thread-store conflict: ... already has an active writer`，因为 App 正持有这个线程 |
| 外部启动 `codex app-server` 再调 `turn/steer` | 报错 `thread not found`，因为 App 的线程在它私有的 stdio server 里 |
| `codex app-server proxy` | 连接的 control socket 不存在，App 不启用 daemon |
| AppleScript 的 `entire contents` | 只能看到 6 个元素，Electron 的网页内容默认不暴露 |
| 只用 `codex queue` | 消息只会排队，要等 Codex 自己结束当前 turn 才会读到，可能要等很久 |

`cx_rpc.py` 可以对线程存储做操作，例如 `thread/queue/list`、`thread/queue/delete`，用来查看或清理队列。

## 依赖
- 本机已安装 ChatGPT.app（bundle id `com.openai.codex`），CLI 路径是 `/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex`。
- 运行 Claude Code 的终端已获得「辅助功能」权限（AX helper 会检查 `AXIsProcessTrusted`）。
- 第一次运行 `send_to_codex.sh` 时，如果找不到 `codex_ax_steer`，会自动用 `swiftc` 编译。
