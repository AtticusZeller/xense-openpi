# 在线 RL 后训练：实现组织

本文说明 `src/openpi/rl/` 的代码组织、运行时数据流和人的操作流程。方法本身（规则、公式、数据结构）以研究工作区
`vla-post-train` 里 `docs/xense-openpi/<algo>-spec.md` 为准；本文只写这些规则落在哪些模块、模块之间怎么串起来。

## 1 分层与依赖方向

```text
src/openpi/rl/
  env/            机器人连接：服务器与机器人端之间的会话协议
  vla/            冻结 VLA：checkpoint 解析、预处理、身份，以及 VLA 的动作表示
  algos/<algo>/   算法私有的模型、loss、replay、采集逻辑、配置和 serving
configs/rl/<algo>/  各算法的 YAML
scripts/rl/<algo>/  各算法的入口
```

依赖只能从算法指向共享层：

- `algos/*` 可以依赖 `env/`、`vla/` 和 openpi 的其余部分；
- `env/`、`vla/` 不依赖 `algos/`；
- `src/openpi` 里 `rl/` 之外的模块不依赖 `openpi.rl`，只有 `scripts/`、`examples/` 直接调用它。

`src/openpi/rl/layering_test.py` 检查后两条，只查非测试模块：共享层的测试可以借用算法的参照实现，例如 action space
测试对照 RLT 的 TacXense codec。

一个组件被第二个使用方（另一个算法或评测）用到时再提进共享层；只有一个使用方时留在算法目录里。

## 2 模块职责

共享层：

| 模块 | 职责 |
|---|---|
| `env/protocol.py` | 服务器端的机器人连接 `RemoteEnv`：监听、握手时校验协议与维度、逐请求收发，断线抛 `EnvConnectionLostError` |
| `vla/frozen.py` | `FrozenVLA` 与 `resolve()`：按 TrainConfig 名和 checkpoint 加载 VLA，norm stats 只取 checkpoint 自带的 `assets/`；与 serving 一致的输入、输出 transform；VLA 身份（配置名、参数与 norm stats 指纹），供 resume 和 serving 校验 |
| `vla/action_space.py` | `ActionSpace`：VLA 的动作表示。TCP 写成相对当前 state 的 delta，夹爪保持绝对值，按 VLA 的 quantile stats 归一化并裁到 `[-1, 1]`，rot6d 正交化；`encode` / `decode` 在 JAX 里批量、可微 |

RLT（`algos/rlt/`）：

| 模块 | 职责 |
|---|---|
| `config.py` | `RLTConfig`（`token_training`、`model`、`rl` 三块）与 YAML 加载；`RLConfig` 划分 training contract 与 operational fields |
| `prefix_cache.py`、`token_model.py` | phase one：冻结 VLA 的 prefix hidden states 落盘缓存；RL token encoder-decoder |
| `features.py` | 在线特征：一次 VLA 前向同时得到参考 chunk 和 prefix hidden，再经 token encoder 得到 `z_rl` |
| `mlp_policy.py`、`td.py` | actor 与 twin-Q critic；chunked TD critic loss 和带 BC 项的 actor loss |
| `critical_trace.py`、`replay.py` | 关键阶段的逐步执行记录，打标签后切成 sliding C-step transition；replay ring buffer |
| `learner.py` | actor、critic、target、replay 与 UTD 节奏；checkpoint 与 resume |
| `collector.py` | 一轮在线采集：按 chunk 驱动机器人，决定由 VLA 还是 actor 驾驶，把打过标签的关键阶段变成 replay 行 |
| `diagnostics.py` | W&B 三个轴与 `events.jsonl`、transition dump、动作诊断 |
| `serving.py` | 把训练好的 actor 接到 `scripts/serve_policy.py policy:rlt` |
| `tacxense_reference.py` | 测试用：导入 TacXense 的 RLT 实现做数值对照 |

入口：`scripts/rl/rlt/precompute_prefix.py`、`train_token.py`（phase one），`train_rl.py`（phase two）。机器人端：
`examples/bi_flexiv_rizon4_rt/rlt_mode.py`（按键状态机与执行），`intervention.py`（Pico4 接管）。

## 3 `rlt-spec.md` 到模块

| spec | 实现位置 |
|---|---|
| §1 同步采集工作流：开窗时机、接管与释放、标签挂起、anchor | 机器人端 `rlt_mode.py`：按键、接管分段、标签随完整执行单元上报；`collector.py`：窗口内由谁驾驶、打标签后建 transition；`critical_trace.py`：sliding anchor 与终止窗口 |
| §2 MLP 输入输出表示 | `vla/action_space.py`：归一化、delta、裁剪；`mlp_policy.py`：固定方差的 Gaussian actor |
| §3 MLP 网络结构 | `mlp_policy.py` |
| §4 Replay transition：human intervention、actor / critic loss、metadata | `collector.py` 在打标签后生成行，`replay.py` 存储；`td.py` 把人工步的 BC 目标换成人的动作，并计算两个 loss |
| §5 Replay 规模与更新节奏 | `config.RLConfig` 的 `buffer_size`、`replay_stride`、`warm_up`、`utd`、`critic_actor_ratio`；`learner.py` 的 warm_up 门控、UTD 欠账和 actor 更新间隔；`diagnostics.py` 的 warm_up 估计 |
| phase one（RL token） | `prefix_cache.py`、`token_model.py`，入口 `precompute_prefix.py`、`train_token.py` |

## 4 运行时数据流

两个进程：机器人端（`examples/bi_flexiv_rizon4_rt/main.py --args.rlt`，lerobot-xense 环境）主动连接训练服务器
（`scripts/rl/rlt/train_rl.py`，监听 `rl.listen`）。消息是 websocket 上的 msgpack，服务器发请求，机器人回复：

```text
握手    机器人发 {protocol, state_dim, action_dim}，与服务器不一致即被拒绝
reset   机器人归位，等操作员按 A，回复首帧观测
chunk   服务器：对当前观测做一次 VLA 前向，得到参考 chunk 与 z_rl；
                窗口已开且 replay 已 warm_up 时由 actor 出 chunk，否则执行 VLA 参考（actor 的输出只记日志）
        机器人：逐步执行；接管时按 C 步切段在本地执行；回复各段执行的动作、人工标记、窗口状态、标签，
                以及窗口内每 replay_stride 步采的观测
        服务器：执行步追加到关键阶段记录；收到标签后补算窗口端点特征，切成 replay 行，暂存到本轮结束
        重复 chunk，直到操作员按 A 结束本轮
轮末    服务器：本轮的行一次性进 replay，按 UTD 欠账训练，记日志，按间隔存 checkpoint；机器人归位等待
```

断线时服务器丢弃当前轮，等机器人重连，本轮数据不进 replay。

产物在 `<token_training.checkpoint_base_dir>/<配置名>/<exp-name>/` 下：

- `token/<step>/`：phase one 的 token checkpoint；
- `rl/<round>/learner.pkl`：权重、优化器、计数器与 replay，供 resume；`rl/wandb_id.txt` 让 resume 接上同一个 W&B run；
- `transitions/`（`rl.dump_transitions` 打开时）：每轮进 replay 的 transition（`round_<n>.npz`）、每轮训练后的
  actor / critic 快照（`actor_critic_round<n>.pkl`）、每轮首个 chunk 的诊断（`initial_actions_round<n>.npz`），以及与
  W&B 同内容的 `events.jsonl`。

W&B：project 默认 `openpi-rlt`，run 名 `<配置名>/<exp-name>/rl`，三个轴分别是 `round/*`、`update/*`（每次梯度更新）
和 `chunk/*`（每个执行的 chunk，含 actor 输出与 VLA 参考的对比）。

## 5 人的流程

操作员的一轮（Pico4）：

1. 机器人归位后按 A 开始本轮，VLA 驾驶非关键阶段。
2. 到关键阶段按 B 开窗，从下一个执行单元起记录（细则见 spec §1）；warm_up 之后窗口内由 actor 驾驶。
3. 需要时握住运动键并移动手柄接管，位移或转角过阈值才算接管；松开即交还，服务器从当前观测重新推理。
4. 按 B 标成功、按 Y 标失败。当前执行单元跑满后标签才上报，终端显示这次进 replay 的 transition 数。
5. 按 X 作废本轮已标的数据并关窗；按 A 结束本轮，有挂起的标签时等它上报后再结束。
6. 服务器训练期间机器人归位等待，训练完回到第 1 步。

启动命令与按键表见 `examples/bi_flexiv_rizon4_rt/README.md`。

研发流程。spec、plan、实验记录在研究工作区 `vla-post-train` 的 `docs/xense-openpi/`：

```text
<algo>-spec.md   方法的唯一真相：规则、公式、数据结构
      ↓
plan.md          本次改动新增哪些模块、复用哪些，以及验收标准
      ↓
代码 + 本文档     在同一个 commit 里改
      ↓
训练 / 评测       真机运行，产物见第 4 节
      ↓
experiments.md   原始记录 → 结果分析 → 下一步（未确认）
      ↓
下一步经确认进 plan.md；方法本身变了就同时改 spec
```

## 6 接口与版本

- **机器人协议**：`openpi-rlt/1`。常量在 `env/protocol.py`，机器人端 `rlt_mode.py` 另存一份，因为机器人端不 import
  openpi。两边必须来自同一分支；协议语义变化时两处一起升版本。
- **配置**：RLT 配置在 `configs/rl/rlt/<名字>.yaml`，`_` 开头的文件只作说明，`_example.yaml` 列出全部字段。它引用的
  VLA TrainConfig 在 `configs/<名字>.yaml` 或 `configs/_examples/`。旧位置 `configs/rlt/` 里的个人配置要手动挪过来。
- **resume contract**：`RLConfig.OPERATIONAL_FIELDS` 之外的字段决定权重、优化器和 replay 的含义，resume 时不能改。
  checkpoint 还绑定 VLA 身份与 token checkpoint，不一致就拒绝加载。
- **serving**：`scripts/serve_policy.py policy:rlt --policy.config <名字> --policy.exp-name <run>`，`--policy.dir` 指定
  某一轮的 `rl/<round>` 或快照；请求里的 `rlt_switch` 在 actor 与 VLA 之间切换。

## 7 新算法接入清单

1. 在 `docs/xense-openpi/` 写 `<algo>-spec.md`。复现的算法以论文为基线，平台适配造成的差异单独列出。
2. `plan.md` 写清新增和复用的模块。只有新算法用到的组件放 `algos/<algo>/`；要把组件提进共享层时，在 plan 里写明
   是哪个组件、第二个使用方是谁。
3. 代码：`algos/<algo>/` 放模型、loss、replay 行格式、采集逻辑、配置 dataclass 和 serving 适配；
   `configs/rl/<algo>/_example.yaml`；`scripts/rl/<algo>/` 放入口。
4. 测试：用固定种子与参考实现做数值对照（参照 RLT 的 `tacxense_reference.py`），`layering_test.py` 保持通过。
5. 本文档：第 2 节加模块表，第 3 节加 spec 映射，第 4、5 节补上与 RLT 不同的数据流和操作流程。
