# 鸡蛋盒长程任务：按 subtask 切分 prompt

## 为什么要切

"打开鸡蛋盒 → 取蛋装盘"一个 episode 约 60–90 秒，里面有两段动作分布差别很大：开盒和夹蛋。夹蛋时间长、占了大部分帧，
如果整段只用一个 prompt 训练，模型学到的主要是夹蛋，开盒学得差。

这个分支的做法：

1. **逐帧切 prompt。** 人工在每个 episode 里找出开盒和夹蛋的分界秒数（`cuts.csv`），分界前的帧打 `Open the egg carton.`，
   分界后的帧打 `Take out the eggs one by one and place them in the egg tray.`。一个 episode 里允许出现多个 subtask。
2. **补难的阶段的数据。** 单独采集开盒数据（task 只有开盒 prompt），和切好的数据合并，再训练一个模型。
3. **真机按键切换。** 客户端按 `1` / `2` 切换 prompt，切换时清掉缓存的 action chunk，用新 prompt 重新推理。

训练和推理服务都不用改：LeRobot 按 `task_index` 逐帧取 `task`，`LeRobotBiEliteDataConfig` 把它作为 `prompt` 传给模型；
客户端在 observation 里带上 `prompt`，服务端直接使用。

## 文件

| 文件 | 作用 |
| --- | --- |
| `egg_prompts.py` | 两个 prompt 字符串，必须和客户端 `examples/bi_elite_cs66_rt/prompt_switch.py` 里的逐字一致 |
| `cuts.csv` | 每个 episode 的分界秒数，用删除坏 episode 之前的编号 |
| `drop_episodes.py` / `verify_drop.py` | 只改元数据地删除坏 episode，视频硬链接、不重编码；以及逐项核对 |
| `relabel.py` / `verify_relabel.py` | 校验分界并逐帧打 task；以及核对只改了 task 相关字段 |
| `merge.py` | 合并前检查（fps、相机、feature、tasks），必要时改写开盒数据的 task，再调官方 `merge_datasets` |
| `compare_frames.py` | 逐帧解码比较两个数据集对应 episode 的图像和 state/action |
| `check_prompts.py` | 按训练管线打印每帧最终送进模型的 prompt |

客户端相关文件在 `examples/bi_elite_cs66_rt/prompt_switch.py`；`main.py` 的 `--args.prompt-switch` 默认开启。
训练 config 是 `configs/_examples/pi05_base_bi_elite_egg_carton_relabeled_plus_open_1103_h100.yaml`。

## 注意

- 不要用 `lerobot-edit-dataset delete_episodes` 删 H.264 数据集里的 episode：它会把相关 mp4 用 libsvtav1 重编码成 AV1，
  之后官方 merge 按 stream copy 拼接视频会报 `non monotonically increasing dts`，或写出无法解码的视频。合并前用 ffprobe
  确认所有 mp4 都是同一编码。
- 合并、删除之后都要用 `compare_frames.py` 全量核对，用 `check_prompts.py` 检查分界帧前后的 prompt。
- decoupled 模式（`--args.action-hz`）不支持按键切换，启用时会直接报错。
