# 报告约定

每个 agent 在自己的 reports/aN 中提交，集成台账放 reports/integration。

报告包括：具体问题、经典对应、方法理由、替代方法、文件/函数或论文位置、
代码 commit、命令、配置、seed、原始证据路径、基线、结果、计算和工程代价、反例/未知及建议。

建议只选一项：通过、有限返工、停止投入。技术正式采用由负责人决定；执行者不能用自评代替独立验收。

原始 rollout 至少记录：
run_id、code_commit、environment_version、split、initial_state_id、seed、method、
visible_history、candidate_actions、predicted_scores、chosen_action、actual_outcome、
physical_time、compute_time、termination_reason。

若有并行采样，记录 requested_N、completed_N、失败/超时数量与实际预算。
隐藏参数仅存在独立评测记录中；记录动作数量不等于独立统计样本数量。

为每份外部原始产物记录可访问位置、SHA-256、大小、配置与生成命令。
不要只交截图、成功摘要或无来源的聚合数字。
