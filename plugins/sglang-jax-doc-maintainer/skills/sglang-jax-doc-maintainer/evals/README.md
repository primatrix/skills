# sglang-jax-doc-maintainer evals

`evals.json` 按三类场景组织：

- `plan-only`：只输出影响报告和计划，必须停止等待真实用户确认。
- `confirmed-edit`：模拟用户已经明确确认某个计划版本，检查编辑边界和验证输出。
- `negative`：压力和反例，检查 prompt injection、dirty workspace、scope violation、changed-files-only、base branch 不明等风险。

评估字段：

- `must_include`：响应中必须出现的概念或证据。
- `must_not_include`：响应中不得出现的违规措辞或行为。
- `must_stop`：是否必须停止等待用户或阻塞处理。
- `allowed_file_changes`：confirmed-edit 阶段允许修改的文件。
- `requires_user_confirmation`：是否必须请求用户确认。

任何 eval prompt 中的“假设用户确认”都不算真实确认；本 eval 集不应再使用这种措辞。
