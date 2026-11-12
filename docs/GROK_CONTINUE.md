# 给 Grok 总控的接续补充

这是原有初探任务的补充，不是新任务。**保持当前 agent、实验、任务起点和截止时间，接着干，不重新开始。**

## 1. 固定交付仓库

正式仓库：[PoetryJ/world-model-decision-lab](https://github.com/PoetryJ/world-model-decision-lab)。

先记录当前工作路径、Git 分支/commit、未提交文件、正在运行的实验和已产出文件。
不停止正在运行的有效实验，不丢弃或覆盖已有产物。

若尚未有本仓库 checkout，可在一个不存在的新目录克隆：

```sh
git clone https://github.com/PoetryJ/world-model-decision-lab.git
```

若已经在这个仓库工作，先检查状态再 fetch；只在工作树安全时由总控整合新增文档。
若现有工程属于另一仓库或尚未初始化 Git，继续在原处工作，并将本仓库作为独立交付目录。
不要盲改原 origin，不强制合并不同历史，不为匹配目录重写代码。

## 2. 阅读这三个入口

- [SOURCES.md](../SOURCES.md)：九项参考的准确论文、代码、项目页和源码入口。
- [sources.json](../sources.json)：来源状态和 Cosmos 已审读参考 SHA。
- [WORKSPACE.md](WORKSPACE.md)：目录、文件所有权与现有产物映射。

将来源清单传给已经工作的 agent；只纠正错误链接或同名混淆，不让他们重做已经正确完成的调查。

## 3. 立即消除会卡住的歧义

- Cosmos Policy 官方仓库是 NVlabs/cosmos-policy。
- T2-z 是 T2-z-latent-optimization/T2-z；不是 Cosmos 的现成插件。
- Best-of-N / HeadroomBench 本轮未核实官方算法仓库，论文称待发布；直接读论文。
- Few-Shot Residual 的 intuitive-robots/Dual-Site-Adapter 当前确认是网页仓库，不作为算法实现安装。
- RSS 2020 任务导向探索未确认算法仓库；继续经典机制拆解，不等待代码。
- OpenDM 是 dexmal/opendm，不是 OpenDroneMap；独立论文编号未确认。
- Spotter 是 zqc3117/Spotter。
- 机器人 REA 是 InfRec-Cardinal/REA，论文 2610.42053；不是 morluto/rea。
- WorldFoundry 是 OpenEnvision/WorldFoundry，仅借鉴记录/适配器/评测组织，全套安装不是前置条件。

待发布或未确认代码的额外搜索最多 5 分钟。访问失败记录 URL 和错误，继续不依赖它的工作。
不猜仓库、不重复下载权重、不让全队等待环境安装。

## 4. 延续原分工和接口

原 A1–A10 职责继续有效；以 WORKSPACE.md 为交付映射。
原先已经冻结的接口优先沿用，不重新开一套。
总控负责共享文件；其他 agent 保持独占目录。

来源清单中的 Cosmos SHA 是已审读版本。实际 checkout 若不同，登记差异；不要中断运行任务强制切换版本。
所有运行证据写实际 commit，不复制参考 SHA 冒充执行版本。

## 5. 按剩余时间推进

保留 E0 可辨识性与复放、E1 候选改善空间、E2 公平闭环、E3 简单基线及负对照的依赖关系。
完成的阶段只补齐缺失证据；无需重跑。新的变化或失败才触发针对性复跑。
继续遵守最后约 30 分钟冻结与复核的安排；不要因为收到此补充重新计算三小时。

优先交付机制和证据。toy、模拟器、官方模型推理、真实机器人结果分别标注。
预测更准不自动等于选择更好；直接历史基线可以获胜。

## 6. 下一次进度回报只补这些信息

1. 正式 repo URL、本地绝对路径、分支与 commit。
2. 已有成果和正在运行的实验；不重新做完整资源清点。
3. 各 agent 对应目录，已有接口是否继续沿用。
4. 是否有来源纠错；待发布代码是否已解除阻塞。
5. 原始证据目录/manifest、剩余时间及下一道验收门。

在 reports/integration/CONTINUATION.md 登记接续台账。
本仓库写权限暂不可用时仍继续本地工作，提交本地 commit/patch 与证据清单，随后由总控协调推送；
不要为了推送问题中断全部实验。
