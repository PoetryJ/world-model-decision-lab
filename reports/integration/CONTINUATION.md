# 接续台账 CONTINUATION

登记时间：2026-11-12 18:52（UTC+8）。登记人：Grok 总控。本文件是接续补充，原任务、agent、起点、截止时间都不变。

## 1. 时间
- 任务起点：约 17:56；截止：约 20:56；最后 30 分钟（20:26–20:56）用于冻结、集成与复核。
- 登记时剩余：约 2 小时 04 分。

## 2. 当前工作目录与版本
| 项 | 值 |
|---|---|
| 实际工程目录（Grok 计算机，Linux） | `/workspace/wmprobe` |
| Git | 独立的本地仓库，分支 `master`，无 remote（与本交付仓库历史不同，不合并历史） |
| 登记时 HEAD | `c2513f75a7f90586c8f2140899fbb6fc5ebe768c`（所有模块已冻结，含 runner 与 val pilot） |
| 未提交 | `runs/controller/sealed*/`（正在运行的密封测试产物）、`reports/a10/test_access_log.jsonl`（访问日志追加） |
| 交付仓库 checkout | `/workspace/world-model-decision-lab`，`main` @ `d5d09ad` |
| Python | `/workspace/wmprobe/.venv`，Python 3.12，numpy 2.5.4、scipy 1.18.1、pytest；无 torch |
| 硬件 | 8 核 CPU、约 15GB 内存，**无 GPU**；用户 PC MEISHIJIAN-PCKN 只有 Intel 核显 |
| Cosmos Policy 实际 checkout | `/workspace/wmprobe/adapters/cosmos/src`，`git rev-parse HEAD` = `18a2accadf4e7a3531e56754102af5a24d2316da`，与已审读参考 SHA 一致（A1 实测，只读，git status 干净）。未下载权重，未运行官方模型。 |

## 3. 共享接口（沿用，不另开一套）
- `/workspace/wmprobe/shared/interface.py`：v1 于约 18:05 冻结；v1.1 在 18:10 做了向后兼容的新增（可选字段 `DecisionRecord.candidate_counts`，依据 A2 疑点 d）。
- 坐标系、单位、动作、形状、随机数来源都写在文件头：世界系为 2D 桌面 ±0.5m；动作是推杆速度 m/s，|v|≤0.5；DT 为 0.1s；动作段形状 (H,2)，候选 (K,H,2)，端点 (K,2)；所有随机性都用显式 seed；隐藏参数只放在 `env.hidden`。
- 映射到本仓库：`src/contracts/` ← `shared/interface.py`（以及 fixture `shared/fixture_env.py`，**只用于接线验证，不算实测证据**）。

## 4. 目录映射（原目录 → 本仓库目录）
| agent | 原目录（/workspace/wmprobe） | 本仓库目录 | 状态 / 建议 |
|---|---|---|---|
| 总控 | shared/, runner/, configs/, ledger/ | src/contracts/, scripts/, configs/, reports/integration/ | 接口冻结；runner 完成 |
| A1 | adapters/cosmos（clone 不提交）, reports/a1 | adapters/cosmos, reports/a1 | 完成 / 有限返工 |
| A2 | probes/source, reports/a2, runs/a2 | probes/source, reports/a2 | 完成 / 有限返工 |
| A3 | reports/a3, runs/a3 | reports/a3 | 完成；18:52 按 SOURCES.md 补来源勘误（进行中） |
| A4 | src/env, reports/a4 | src/env, reports/a4 | 完成 / 通过 |
| A5 | src/candidates, src/planning, reports/a5, runs/a5 | 同名 | 完成 / 通过 |
| A6 | src/dynamics, src/adaptation, reports/a6, runs/a6 | 同名 | 18:24 冻结 / 有限返工 |
| A7 | src/scoring, reports/a7, runs/a7 | 同名 | 冻结 / 通过（不部署方差门控） |
| A8 | src/probing, reports/a8, runs/a8 | 同名 | 冻结 / 有限返工 |
| A9 | src/direct_baselines, reports/a9, runs/a9 | 同名 | 冻结 / 通过 |
| A10 | evaluation, tests/evaluation, reports/a10, runs/a10 | 同名 | 协议冻结 + 密封测试集 / 有限返工 |
| 原始证据 | runs/<agent>/, runs/controller/ | artifacts/（只放说明和 manifest；原始运行目录保留在本地） | — |

导入策略：在检查点按选择导入代码和报告，保留来源 commit；不搬迁、不重写、不覆盖原目录。

## 5. 已有成果（全部是 CPU 二维 toy 结果，另有标注的除外）
- **E0 通过**（A4 产出，总控复跑一致）：复放逐位一致 72/72；低/高摩擦下最佳动作 12/12 不同；`tap_fast` 能辨识摩擦（准确率 0.969，加噪声后 0.80）；负对照里存在稳健动作 12/12（事后标准 G5b；预注册的 G5 未通过，已如实记录）。
- **E1 完成**（A5 产出，A10 独立复现）：TimedRelease K=4 的 oracle 成功率 0.567，随机 0.142；A4 的 54 候选网格 oracle 为 1.0。
- **源码**（A1/A2，静态审查加 numpy 移植）：训练目标是 rectified-flow 速度预测，用 EDM σ 参数化；只预测端点；价值不等于可信度。四个疑点的结论：单样本 LCB 出 NaN 成立；majority_mean 存在尺度敏感；并行 seed 部分重复成立；worker 失败后候选数记录不全成立。
- **先行工作**（A3，论文报告的结果）：三个方向都有直接先行工作，只保留 D2 作为受控诊断。
- **val 结果**（A6–A9 与总控 pilot）：适应 vs 同样试探但不更新为 +0.083 [0.044, 0.128]（pilot）；VOI 和目标距离规则打平；学出来的直接历史基线弱，手写力学基线强。
- 官方 Cosmos smoke test 被**阻塞**：需要 ≥10GB 显存的 CUDA GPU、约 3.9GB 的检查点、Docker 和 LIBERO。

## 6. 运行中的实验
- 18:46 启动密封测试 E2/E3，只跑一次：`runner/SEALED_CMD.md`。数据集为 `evaluation/sealed/test_sealed.json`，sha256 `b6e9c903…6bd`，40 个正常态加 40 个负对照态，摩擦 0.1/0.3/0.6；共 21 个方法，分确定性和噪声 0.02×3 次重复两组。18:50 时两组运行和审计都已完成（审计 OK、密封访问 OK），正在计算 regret 和汇总。
- 产物位置：`/workspace/wmprobe/runs/controller/sealed{,_stoch}/`。

## 7. 证据位置与 manifest
- 冻结 manifest：`/workspace/wmprobe/configs/frozen_manifest.json`，若有哈希变化 runner 会拒绝运行。
- 协议与修订：`evaluation/PROTOCOL.md`（A-1…A-9）；总控决策：`ledger/LEDGER.md`。
- 测试集访问日志：`reports/a10/test_access_log.jsonl`；暴露记录：`reports/a10/test_exposure.jsonl`。原 `test`/`test_neg` 已降级为 dev-exposed（A-8）。

## 8. 来源纠错（依据 d5d09ad SOURCES.md）
- A1/A2：已使用 NVlabs/cosmos-policy @ 18a2acc，正确，无需纠正。
- A3：Best-of-N（2610.37196）、T2-z（T2-z-latent-optimization/T2-z）、WorldFoundry（OpenEnvision）一致。A3 先前写“无法确认 REA”，现已指向 InfRec-Cardinal/REA（2610.42053，不是 morluto/rea），并核对 OpenDM=dexmal/opendm、Spotter=zqc3117/Spotter、Dual-Site-Adapter 只是网页仓库。18:51 A3 已完成勘误（reports/a3/REPORT.md §Source corrections）：REA 已识别，不只在开头路由，作为 runner/A10 的共享预算与只读评分设计参考，不集成（Node.js）；OpenDM 排除结论不变；Spotter cf_bench 的快照校验与失败态分支复放可供 A4/A7/A10 参考；Dual-Site-Adapter 明确为不可安装；Best-of-N 未引用仓库。访问失败：Spotter cf_bench/README.md 返回 404；api.github.com 两次返回 403（速率限制）。
- 尚未确认官方代码的：Best-of-N/HeadroomBench、Few-Shot Residual 算法代码、Liang 2020。三者都直接读论文，不等代码。

## 9. 下一道验收门
1. 密封 E2/E3 汇总完成后，过 A10 审计，并独立复跑关键结果（确定性重跑逐 episode 一致）。
2. E3：比较世界模型路线与直接历史路线、适应与不适应、负对照中的试探率。
3. 20:26 冻结，之后导入本仓库并提交最终报告。

## 10. 终版（20:30）
- 最终报告：reports/integration/FINAL_REPORT.md，复制自 /workspace/wmprobe@996d7ba 中的 reports/controller/FINAL_REPORT.md。
- 确定性密封测试复跑 5040/5040 一致；pytest 71 项全部通过（运行时需加 --ignore=adapters）。
- 模块代码仍留在原目录 /workspace/wmprobe，未搬迁；等负责人决定后再选择性导入。

## 11. 模块代码选择性导入（2026-11-13，负责人批准）
来源：`/workspace/wmprobe` @ `996d7ba`（用 `git archive 996d7ba <路径>` 导出，**不取工作区**；工作区里 `reports/a10/test_access_log.jsonl`、`reports/a4/e0_raw.json` 的未提交改动未导入）。不合并历史。
路径与原仓库保持**完全相同的相对布局**，因此 `python -m runner.run_e2e3`、`from shared.interface import …`、`from src.… import …` 以及 `configs/frozen_manifest.json` 中按路径记录的 41 个哈希均可直接使用（已核对 41/41 一致；密封集 sha256 `b6e9c903…66bd` 一致）。
因此 `shared/` 没有搬到 `src/contracts/`，`runner/` 没有搬到 `scripts/`（搬迁会破坏 import 和 manifest 路径）；§4 的目录映射仅作为归属说明。

导入内容（共 112 个文件，另修改 `.gitignore`）：
- 源代码：`shared/`（冻结接口 v1.1 + fixture）、`src/{env,candidates,planning,dynamics,adaptation,scoring,probing,direct_baselines}/`、`runner/`、`probes/source/`、`evaluation/`（含 `PROTOCOL.md`）。
- 测试：`tests/`、`src/env/test_env.py`、`src/probing/test_probing.py`、`src/scoring/test_scoring.py`、`probes/source/test_a2_probes.py`、`reports/a5/test_module.py`。
- 冻结配置/清单：`configs/frozen_manifest.json`、`src/probing/frozen_config.json`，以及 manifest 校验所需的 `reports/a6/FROZEN.md`、`reports/a7/frozen_scorer_config.json`、`reports/a9/config_frozen.json`、`reports/a9/API.md`。
- 复现必需的小数据：`evaluation/frozen_sets.json`（178KB）、`evaluation/sealed/test_sealed.json`（30KB）；冻结权重 `runs/a6/models/ens_push_env_v3_s0.pkl`(+meta，198KB)、`runs/a9/frozen_strike54/models/{obs_only,history_direct}.npz`（13KB）。
- 可复用代码：`reports/a5/{closed_loop,diagnose,env_loader,oracle,run_all}.py`、`reports/controller/indep_check.py`。
- 台账/审计记录：`ledger/{LEDGER,RULES}.md`、`reports/a10/{test_access_log,test_exposure}.jsonl`。

唯一的代码改动：`reports/a5/` 中 4 个文件把硬编码的 `ROOT=/workspace/wmprobe` 改为 `Path(__file__).resolve().parents[2]`。不改的话，测试会把旧仓库的 `shared/` 插入 sys.path，导致 `tests/evaluation/test_eval.py::test_tracked_hidden_flags_env_rollout_inside_predict` 失败（帧路径不在本仓库根内）。这些文件不在 frozen manifest 内，不影响任何冻结哈希。

排除及原因：
- `__pycache__/`、`*.pyc`（原仓库误提交了 11 个）：属于生成物。
- `runs/` 下的原始运行产物（约 1.2GB：decisions.jsonl、训练行日志、各 agent 的实验输出）：体积大，属于证据而非代码。仍保留在原目录，`.gitignore` 新增 `/runs/**`，只放行上面 4 个冻结权重文件。
- 其它 `runs/a6/models/*`（fixture、v1/v2、s1/s2、latent）：不在冻结 manifest 内，复现不需要。
- `adapters/cosmos/src`（官方 clone @18a2acc）、`.venv/`、`.pytest_cache/`：外部源码和环境，按 third_party 规则不提交。
- `reports/a1–a10` 中的叙述报告、日志、图表、json 汇总，以及 `reports/controller/{FINAL_REPORT.md,indep_check.log}`：属于报告类，不是模块代码。结论已在本目录的 `FINAL_REPORT.md` 中，原件留在原目录。
- 控制器过程文件、checkpoint（`runs/controller/*`）：属于运行产物。

已知限制：`src/adaptation/analyze_a6.py` 仍然硬编码 `/workspace/wmprobe/runs/a6/e2/`（这是事后分析脚本，依赖未导入的 runs，故保持原样）。部分 docstring 里的运行说明仍写着 `cd /workspace/wmprobe`，在本仓库里应从仓库根运行。`reports/controller/indep_check.py` 需要本地的 `runs/controller/sealed*/episodes.csv`。

导入后核验（2026-11-13 16:3x，UTC+8，使用 `/workspace/wmprobe/.venv` 的 Python 3.12）：
- `python -m pytest -q`：71/71 通过（本仓库没有 `adapters/cosmos/src`，所以不需要 `--ignore=adapters`）。
- 在本仓库确定性重跑一次密封集：`python -m runner.run_e2e3 --split sealed --out runs/controller/repro_wmdl_det --workers 6`，manifest 校验通过，耗时 72.5 秒。与原 `runs/controller/sealed/episodes.csv` 逐 episode 比较 5040/5040，0 个不一致；`evaluation.audit runner` 结果为 OK，密封访问为 OK。成功率与 FINAL_REPORT 相同（例如 wm_noprobe 0.758、wm_goalrule_update 0.883、belief_goalrule_update 1.000）。数值结论没有变化。这次重跑的密封访问已追加到 `reports/a10/test_access_log.jsonl` 并一起提交；它只是确定性复现，没有调参。
