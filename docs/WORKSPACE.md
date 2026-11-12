# 目录结构与文件所有权

这些目录提供交付位置；已有代码的等价路径可由总控登记映射，不要求为了对齐目录重新实现或立即搬迁。
当前占位目录不表示模块已完成。原先已冻结的接口继续有效；变更由总控统一协调。

## 仓库结构

```text
world-model-decision-lab/
├── README.md
├── SOURCES.md
├── sources.json
├── docs/
│   ├── GROK_CONTINUE.md
│   ├── WORKSPACE.md
│   └── STATUS.md
├── adapters/cosmos/             # A1：官方模型适配
├── probes/source/               # A2：数据、张量、源码边界探针
├── src/
│   ├── contracts/               # 总控：共享接口
│   ├── env/                     # A4：环境与快照
│   ├── candidates/              # A5：动作候选
│   ├── planning/                # A5：采样规划
│   ├── dynamics/                # A6：预测模型
│   ├── adaptation/              # A6：历史适应
│   ├── scoring/                 # A7：价值与可信度
│   ├── probing/                 # A8：试探选择
│   └── direct_baselines/        # A9：直接历史基线
├── evaluation/                 # A10：独立评测
├── tests/evaluation/           # A10：评测不变量与反证
├── configs/                    # 总控：共享配置
├── scripts/                    # 总控：runner 与汇总入口
├── reports/
│   ├── README.md
│   ├── a1/ ... a10/            # 各 agent 独占
│   └── integration/            # 总控：台账、集成和映射
├── artifacts/README.md         # 原始证据说明；实际运行目录本地保存
├── third_party/README.md       # 外部源码本地保存，记录版本，不整体提交
└── .gitignore
```

## 分工

- 总控：共享接口、依赖、配置、runner、sources.json 及集成台账；确认各模块自己的测试路径。
- A1：adapters/cosmos、reports/a1；源码链路和必要的官方 smoke test。
- A2：probes/source、reports/a2；最小边界复现与数据/张量检查。
- A3：reports/a3；先行工作、适配成本、原创性反证；来源修改交总控汇总。
- A4：src/env、reports/a4。
- A5：src/candidates、src/planning、reports/a5。
- A6：src/dynamics、src/adaptation、reports/a6。
- A7：src/scoring、reports/a7。
- A8：src/probing、reports/a8。
- A9：src/direct_baselines、reports/a9。
- A10：evaluation、tests/evaluation、reports/a10；独立检查数据泄漏和预算。
- 每个模块的针对性测试放本模块自有子目录，避免其他 agent 修改 tests/evaluation。
- 不同 agent 可使用独立分支/worktree；采用 codex/ 前缀并由总控登记。不要把同一共享工作树当成互相隔离的分支。

## 接续已有产物

总控在 reports/integration/CONTINUATION.md 登记原目录、当前分支/commit、未提交文件、已完成工作、
当前运行中的实验、产物位置、与本仓库目录的映射及剩余时间。没有的字段写 unknown，不伪造。

先保留原始产物，再在适当检查点选择性导入。禁止覆盖未提交工作、强推、清理旧目录或重跑已完成阶段来“统一格式”。
不同 Git 历史不要直接执行强制合并；可以将本仓库作为独立交付目录，在检查点导入必要代码和报告，并保留来源记录。

## 共享接口需要明确的内容

沿用已冻结签名，补齐动作单位、坐标系、时间跨度、数组形状、随机数来源和版本。
环境 reset/observe/step/snapshot/restore，候选 propose，预测 predict，适应 update，
评分 score，试探 choose_probe，以及原始记录 schema。

模拟器隐藏参数只进入环境与独立评测证据；策略只能访问规定的观察和已经发生的交互。
接口 fixture 仅用于接线验证，不能计为算法实测证据。
