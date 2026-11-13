# 接续台账 CONTINUATION

登记时间：2026-11-12 18:52（UTC+8），20:30 终版补记。登记人：Grok 总控。本文件是接续补充，原任务、agent、起点、截止时间都不变。

## 1. 时间
- 任务起点约 17:56，截止约 20:56；最后 30 分钟（20:26–20:56）冻结、集成、复核。按时完成。

## 2. 工作目录与版本
- 实际工程目录：Grok 计算机 `/workspace/wmprobe`，独立本地 git 仓库，分支 `master`，无 remote；与本仓库历史不同，不合并历史。登记时 HEAD `c2513f75a7f90586c8f2140899fbb6fc5ebe768c`，终版 `996d7ba`。
- 环境：Python 3.12 venv，numpy 2.5.4、scipy 1.18.1、pytest；无 torch；8 核 CPU、约 15GB 内存，无 GPU。
- Cosmos Policy 实际 checkout：`adapters/cosmos/src`，HEAD = `18a2accadf4e7a3531e56754102af5a24d2316da`（A1 实测，与已审读参考一致）；未下载权重、未运行官方模型。

## 3. 共享接口（沿用）
- `shared/interface.py` v1 约 18:05 冻结；v1.1（18:10）只做向后兼容新增 `DecisionRecord.candidate_counts`。坐标系 2D 桌面 ±0.5m，动作为推杆速度 m/s（|v|≤0.5），DT 0.1s，段 (H,2)/候选 (K,H,2)/端点 (K,2)，随机性均用显式 seed，隐藏参数只放在 `env.hidden`。
- 映射：`src/contracts/` ← `shared/interface.py`；fixture `shared/fixture_env.py` 只用于接线验证，不算实测证据。

## 4. 目录映射（原目录 → 本仓库目录）
| agent | 原目录（/workspace/wmprobe） | 本仓库目录 | 状态 |
|---|---|---|---|
| 总控 | shared/, runner/, configs/, ledger/ | src/contracts/, scripts/, configs/, reports/integration/ | 冻结 |
| A1 | adapters/cosmos（clone 不提交）, reports/a1 | adapters/cosmos, reports/a1 | 完成 / 有限返工 |
| A2 | probes/source, reports/a2 | probes/source, reports/a2 | 完成 / 有限返工 |
| A3 | reports/a3, runs/a3 | reports/a3 | 完成（含来源勘误） |
| A4 | src/env, reports/a4 | 同名 | 通过 |
| A5 | src/candidates, src/planning, reports/a5 | 同名 | 通过 |
| A6 | src/dynamics, src/adaptation, reports/a6 | 同名 | 冻结 / 有限返工 |
| A7 | src/scoring, reports/a7 | 同名 | 冻结 |
| A8 | src/probing, reports/a8 | 同名 | 冻结 / 有限返工 |
| A9 | src/direct_baselines, reports/a9 | 同名 | 冻结 |
| A10 | evaluation, tests/evaluation, reports/a10 | 同名 | 冻结 / 有限返工 |
| 原始证据 | runs/<agent>/, runs/controller/ | artifacts/（只放说明与 manifest） | 本地保存 |
模块代码仍在原目录，未搬迁；等负责人决定后选择性导入并保留来源 commit。

## 5. 成果（全部为 CPU 二维 toy 结果）
- E0 通过：复放 72/72 逐位一致；低/高摩擦最佳动作 12/12 不同；tap 辨识准确率 0.969（加噪 0.80）；预注册 G5 仅 1/12，已如实记录。
- E1：TimedRelease K=4 oracle 成功率 0.567、随机 0.142（A10 复现）；54 候选网格 oracle 1.0。
- 源码（A1/A2 静态审查 + numpy 移植）：四个疑点均成立。
- 密封 E2/E3（18:46–18:54，单次）：审计有效、0 问题、0 复放不一致；结果见 FINAL_REPORT.md。确定性复跑 5040/5040 一致；pytest 71 项通过（需 --ignore=adapters）。
- 官方 Cosmos smoke test 阻塞：需要 CUDA GPU；负责人 2026-11-13 决定等服务器维护后再定。

## 6. 证据位置
- 冻结 manifest `configs/frozen_manifest.json`；协议 `evaluation/PROTOCOL.md`（A-1…A-9）；决策 `ledger/LEDGER.md`；测试集访问日志 `reports/a10/test_access_log.jsonl`；逐 episode 结果 `runs/controller/sealed{,_stoch}/episodes.csv`（均在 /workspace/wmprobe）。

## 7. 来源纠错（依据 d5d09ad SOURCES.md）
- A1/A2 使用 NVlabs/cosmos-policy@18a2acc，正确。
- A3 已补勘误：REA = InfRec-Cardinal/REA（2610.42053，不是 morluto/rea），每个能力边界都做检查，不只在开头路由；可作 runner/A10 共享预算与只读评分的设计参考，不集成（Node.js）。OpenDM = dexmal/opendm，排除结论不变。Spotter = zqc3117/Spotter，cf_bench 的快照校验与失败态分支复放可供 A4/A7/A10 参考。Dual-Site-Adapter 只是网页仓库，不可安装。Best-of-N 未引用任何仓库。
- 访问失败：Spotter cf_bench/README.md 返回 404；api.github.com 两次返回 403（速率限制）。
- 未确认官方代码：Best-of-N/HeadroomBench、Few-Shot Residual 算法代码、Liang 2020，均直接读论文。
