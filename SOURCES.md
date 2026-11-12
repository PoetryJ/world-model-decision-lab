# 准确来源与阅读入口

核验日期：2026-11-12。以下链接用于消歧和阅读，不等于依赖已安装、权重可下载或实验已复现。
机器可读版本见 [sources.json](sources.json)。来源状态是任务输入的一部分。

## 1. Cosmos Policy — 首要源码拆解对象

完整标题：**Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning**

- [论文](https://arxiv.org/abs/2601.16163) · [全文 v1](https://arxiv.org/html/2601.16163v1)
- [官方算法仓库：NVlabs/cosmos-policy](https://github.com/NVlabs/cosmos-policy)
- [官方项目页](https://research.nvidia.com/labs/cosmos-lab/cosmos-policy/)
- 已审读参考提交：`18a2accadf4e7a3531e56754102af5a24d2316da`。执行者记录自己的实际 checkout；不要把这个值填成未验证的运行版本。
- [训练数据与时间对齐](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/datasets/libero_dataset.py)
- [latent 打包与条件处理](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/models/policy_video2world_model.py)
- [动作、未来、价值、Q 与聚合函数](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/cosmos_utils.py)
- [LIBERO 执行与候选搜索](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/experiments/robot/libero/run_libero_eval.py)
- [规划数据配置](https://github.com/NVlabs/cosmos-policy/blob/18a2accadf4e7a3531e56754102af5a24d2316da/cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py#L373-L393)注明 648 条 ALOHA planning rollout 未公开。不能写“所有训练数据均公开”。

用途：A1/A2 拆解，A5/A7 建立已有方法基线。先看源码；是否运行官方模型取决于现有资源。

## 2. Best-of-N / HeadroomBench — 评测与选择器先行工作

完整标题：**How Much Can Best-of-N Help a World Action Model? Measuring Selectable Headroom and Realizing It in Closed Loop**

- [论文](https://arxiv.org/abs/2610.37196) · [全文 v1](https://arxiv.org/html/2610.37196v1)
- 状态：`code_pending`。论文表示 HeadroomBench 将公开发布，本轮未核实官方算法仓库。
- 不猜 GitHub 地址，不等待代码。先读方法、反事实分支、重复 continuation、选择/评估分离及 failure-gated selector。

用途：A3/A5/A10。单次分支成功的最大值可能混入后续随机性；同状态候选比较本身不是本项目创新。

## 3. T2-z — 环境 latent 与测试时适应

完整标题：**T2-z: Test-Time Optimization of Environment Latents for Robotic World Models**

- [论文](https://arxiv.org/abs/2610.22364) · [全文 v1](https://arxiv.org/html/2610.22364v1)
- [官方算法仓库：T2-z-latent-optimization/T2-z](https://github.com/T2-z-latent-optimization/T2-z)
- [官方项目页](https://t2-z-latent-optimization.github.io/)
- [官方数据入口](https://huggingface.co/datasets/huagailuowen/T2-z-datasets)
- 阅读入口：README、`t2z/train.py`、`t2z/infer.py`；实际路径随 checkout 核对。
- 测试时随机交互是重要对照。训练环境 latent 与世界模型需要配套训练，不能假设给 Cosmos 随手加向量就能工作。

用途：A3/A6/A8。本轮先拆机制，不默认下载整套大模型和数据。

## 4. Few-Shot Residual Adaptation — 轻量残差修正

完整标题：**Few-Shot Residual Adaptation of Pretrained Latent World Models**

- [论文](https://arxiv.org/abs/2611.13592) · [全文 v1](https://arxiv.org/html/2611.13592v1)
- [官方项目页](https://intuitive-robots.github.io/Dual-Site-Adapter/)
- [官方网页仓库：intuitive-robots/Dual-Site-Adapter](https://github.com/intuitive-robots/Dual-Site-Adapter)
- 状态：`website_repository_verified / algorithm_code_pending`。
- **上述仓库当前核实的是网页和展示素材，不是已确认的算法实现。**
- [网页源码](https://github.com/intuitive-robots/Dual-Site-Adapter/blob/master/index.html)的算法代码链接仍是注释中的待发布占位。
- 只读论文，提取冻结基座、内部/输出残差与少样本适应的思想；不等待安装这个网页仓库。

用途：A3/A6。与 Cosmos 的模型表示和接口兼容性需单独研究。

## 5. 经典任务导向探索 — 查重和强基线

完整标题：**Learning Active Task-Oriented Exploration Policies for Bridging the Sim-to-Real Gap**

- [论文](https://arxiv.org/abs/2006.01952) · [全文 v2](https://arxiv.org/html/2006.01952v2)
- [作者项目页](https://sites.google.com/view/task-oriented-exploration/)
- 状态：`algorithm_code_unconfirmed`。本轮未确认官方算法仓库；不表示绝对不存在。
- 阅读任务相关参数、试探策略、辨识与下游控制的关系。

用途：A3/A8。任务导向试探已有先行研究，不直接声称这个大思想原创。

## 6. WorldFoundry — 实验记录与评测组织

完整标题：**WorldFoundry: Towards Unified Infrastructure for World Intelligence**

- [论文](https://arxiv.org/abs/2611.10771)
- [官方基础设施仓库：OpenEnvision/WorldFoundry](https://github.com/OpenEnvision/WorldFoundry)
- [官方项目页](https://openenvision.org/WorldFoundry/)
- [架构文档](https://openenvision.github.io/WorldFoundry/docs/maintainers/architecture)
- [评测文档](https://openenvision.github.io/WorldFoundry/docs/evaluation)

用途：A3/总控/A10 借鉴模型适配器、manifest、产物记录与评测组织。它不是物理模拟器或机器人策略，全套安装不是本轮前置条件。

## 7. OpenDM / DM0.5 — Dexmal 具身策略参考

- [官方算法仓库：dexmal/opendm](https://github.com/dexmal/opendm)
- [官方 DM0.5 技术博客](https://www.dexmal.com/blog/dm0.5?lang=en-US)
- [核心架构](https://github.com/dexmal/opendm/blob/main/opendm/model/dm05/dm05_arch.py)
- [推理文档](https://github.com/dexmal/opendm/blob/main/docs/en/dm05_inference.md)
- [LIBERO 文档](https://github.com/dexmal/opendm/blob/main/docs/en/dm05_libero.md)
- 独立 arXiv 论文本轮未确认；不编造论文编号。**不要与 OpenDroneMap/ODM 混淆。**

用途：A1/A3 按需比较 VLM、动作专家与动作分块设计。本轮不要求切换策略基座。

## 8. Spotter — 监测、恢复与失败状态评测

完整标题：**Spotter: Let the Embodied Model Lead, and the VLM Reflect for It**

- [论文](https://arxiv.org/abs/2609.36808)
- [官方算法仓库：zqc3117/Spotter](https://github.com/zqc3117/Spotter)
- [官方项目页](https://zqc3117.github.io/Spotter/)
- [失败状态反事实评测：cf_bench](https://github.com/zqc3117/Spotter/tree/main/cf_bench)
- [监测与恢复：recovery_explore](https://github.com/zqc3117/Spotter/tree/main/recovery_explore)

用途：A3/A7/A10 按需阅读。仅当对应当前元件问题时借鉴，不启动整套恢复系统集成。

## 9. 机器人 REA — 路由、异步执行与恢复预算

完整标题：**REA: A Routed Embodied Agent Runtime for Asynchronous Brain-Cerebellum Control**

- [论文](https://arxiv.org/abs/2610.42053)
- [官方算法仓库：InfRec-Cardinal/REA](https://github.com/InfRec-Cardinal/REA)
- [官方项目页](https://mikeeeeeeeeeeeeee.github.io/REA/)
- [架构](https://github.com/InfRec-Cardinal/REA/blob/main/docs/architecture.md)
- [路由](https://github.com/InfRec-Cardinal/REA/blob/main/docs/routed-systems.md)
- [异步执行](https://github.com/InfRec-Cardinal/REA/blob/main/docs/async-systems.md)
- **这里不是 morluto/rea；后者是逆向工程工具。**

用途：A3 按需研究路由、检查边界与共享预算。区分各 REA 版本，不概括成“只在开始时路由”。

## 遇到链接或版本问题

1. 记录完整标题、论文编号、官方 owner/repo、访问结果及实际 commit。
2. 优先从论文、作者项目页和官方 README 的相互链接确认身份。
3. 暂时打不开记为访问阻塞；不凭名称换同名仓库。
4. 待发布/未确认算法代码的项目直接开展论文分析，额外找代码最多 5 分钟。
5. 对 main 分支链接，阅读时补记具体 commit；引用和实验绑定到实际版本。
