# Pick-tube RDP 真机部署定责与离线排障手册（2026-08-15）

## 1. 当前结论与真机安全门

【已确认】本手册中的四个工具只读取显式路径，不连接 WebSocket、Typhon HTTP 或相机，也不启动控制器；默认只向标准输出报告，只有显式给出 `--output` 才写新 JSON，且不得覆盖已有文件。

【已确认】桥接层的固定 50 ms 轮询延迟已经修复，但夹爪五步边界跳变、FRS 输入域偏移、模型左右不对称和 Typhon 非原子下发风险尚未解决。因此，在第 13 节全部通过前禁止再次上真机。

【已确认】不得对 20D 动作做全局放大。相同 FRS 初始帧的 32 个 seed 中右臂已经比左臂更活跃；全局缩放会同时放大右臂、旋转和夹爪，不能作为“左臂动作小”的安全修复。

【未知】最终根因是原始示教/转换、AT、LDP 条件预测，还是部署输入域偏移；必须按第 10 节逐阶段定责，不能依据一次真机现象直接归因。

## 2. 仓库、模型和日志清单

【已确认】服务端仓库为 `/home/typhon/RDP_vb3_robot_server/vb3_robot_server`，模型仓库为 `/home/typhon/RDP_vitamin`，FRS 保存观测为 `/home/typhon/FRS_Tact/outputs/frs_remote_observations/20260813_162558`。

【已确认】部署配置为 `/home/typhon/RDP_vitamin/configs/deploy_pick_tube_rdp.yaml`；其中 LDP、AT 和触觉编码器分别指向：

```text
data/weights/wjstx_rdp/ldp_20260813_214114/checkpoints/latest.ckpt
data/weights/wjstx_rdp/at_20260813_115524/checkpoints/latest.ckpt
data/encoder_ckpt_0809
```

【已确认】两次真机动作日志为：

```text
action_debug_logs/20260815_145643/action_debug.jsonl
action_debug_logs/20260815_160855/action_debug.jsonl
```

【已确认】训练机必须另行提供有序的 `pick_tube_01_04_rdp_zarr` 或原始 LeRobot episode 行；本机只有 checkpoint normalizer 和保存观测，没有足以重建“每个 episode 起始动作顺序”的训练行。

## 3. 观测与动作数据契约

【已确认】单帧 state 是 20D、单位为米和弧度，固定布局为：

```text
[0:6]   左臂相对 episode 起点 xyz + axis-angle
[6]     左夹爪实际宽度
[7:13]  右臂相对 episode 起点 xyz + axis-angle
[13]    右夹爪实际宽度
[14:20] 左臂相对右臂 xyz + axis-angle
```

【已确认】单步 action 是连续左右布局的 20D 相对动作：

```text
[0:3]   左臂 xyz
[3:9]   左臂旋转 6D（旋转矩阵前两列）
[9]     左夹爪实际宽度
[10:13] 右臂 xyz
[13:19] 右臂旋转 6D（旋转矩阵前两列）
[19]    右夹爪实际宽度
```

【已确认】相机/触觉顺序为 `camera0 -> policy camera1`、`camera1 -> policy camera2`，触觉为 `left_0, right_0, left_1, right_1`；四个 512D 触觉 embedding 展平为 2048D。

【已确认】夹爪观测标定为 `actual_width = 1.77 * commanded_width + 0.050`；执行时逆变换后把 command 裁剪到 `[0.01, 0.04]` m。不得把 action 中的实际宽度直接当成 Typhon command。

【已确认】checkpoint normalizer 的训练 action xyz 向量 RMS 为左臂 `2.380442 mm/step`、右臂 `2.540288 mm/step`，简写为 `2.38 mm` 和 `2.54 mm`；这两个值描述训练分布，不是允许的真机速度上限。

## 4. 部署数据流与时序

【已确认】数据流为：双三联相机帧裁剪/缩放 → 相机、触觉、20D state 打包 → msgpack WebSocket 发送 → Vitamin 触觉编码器/LDP/AT 推理 → 返回 `(1, 20)` → 相对位姿累计成双臂绝对目标 → Quest/EE 标定逆变换 → 共享内存 waypoint → 80 Hz 控制子进程 IK → Typhon HTTP。

【已确认】RDP 协商值为 `policy_type=rdp`、`data_type=vitac`、`action_horizon=1`、`steps_per_inference=1`；服务端观测分辨率为 `224x224`，control frequency 为 `30 Hz`，controller frequency 为 `80 Hz`，slow update interval 为五步，动作从接收时刻加 `50 ms` lead 调度。

【已确认】每个在线循环会调用三次 `get_obs()`：第一次构造发往策略的观测，第二次在动作转换前取得当前基准位姿，第三次在执行后生成 debug jump；初始化阶段另有两次 warmup `get_obs()`。这三次读取会增加循环开销，不能把 `30 Hz` 配置值当成实际闭环频率。

【已确认】控制子进程每个 80 Hz 周期依次为四个 target 调用 `set_joint_angle`；Typhon wrapper 的每次调用都会构造含四个 key 的完整 body 并独立 `POST /action`，因此一个控制周期是四次顺序 HTTP POST，而不是一次原子双臂提交。

【推断】四次非原子 POST 可能产生同周期内左右臂/夹爪不同步和中途失败后的部分更新；在获得 Typhon backend 的事务语义或改成单次原子 POST 前，它是控制层风险，不是已证实的本次左右不对称根因。

## 5. 两次真机运行对比

【已确认】`20260815_145643` 有 112 条连续记录、112/112 动作被调度，持续 `11.879516 s`，有效频率 `9.343815 Hz`（`9.34 Hz`），周期 p50/p95 为 `104.531/117.373 ms`。

【已确认】`20260815_160855` 有 249 条连续记录、249/249 动作被调度，持续 `12.548027 s`，有效频率 `19.764064 Hz`（`19.76 Hz`），周期 p50/p95 为 `52.264/65.947 ms`。

【已确认】第二次运行的 command lead p50/p95 为 `45.789/46.253 ms`，没有 late-action skip；桥接吞吐约为第一次的 `2.115x`，但仍没有达到配置的 `30 Hz`。

【已确认】延迟修复提交为 `8b361059accca3a828a6cc6e4ad322fe93142726`：独立 sender thread 用 condition 事件唤醒发送，接收端由 `recv(timeout=0.05)` 轮询改为阻塞 `recv()`；`3d925efbee8074d6a6a09033397f53e01a773a34` 又补充连接关闭、sender 错误传播和每连接独立 packer。

【已确认】第二次日志按五步计划边界统计，左夹爪 controller command jump：边界 mean `5.058 mm`、p95 `24.70 mm`、max `26.62 mm`，非边界 mean `0.0107 mm`；右夹爪边界 mean `4.062 mm`、p95 `13.73 mm`、max `16.72 mm`。这些是命令序列跳变，不等同于真机速度、实际夹爪位移或最终根因。

【已确认】旧 logger 直接相减 axis-angle，在左臂分支切换处报告约 `6.28 rad`；同帧 raw/controller 旋转增量只有约 `0.002--0.004 rad`。这是 SO(3) 表示的日志伪影，新 summarizer 使用 SO(3) geodesic，不应把 `2*pi` 当作真机旋转跳变。

## 6. FRS 172 帧离线回放结果

【已确认】FRS 目录包含 172 个保存步骤；所有 state 为 `(20,)`，左右夹爪几乎恒为 `0.1385 m`。checkpoint 训练 state 的夹爪范围为左 `[0.070727, 0.124618] m`、右 `[0.063521, 0.116377] m`，所以两个夹爪输入都高于训练最大值，属于已确认 OOD。

【已确认】相同 FRS 初始帧每个 seed 都重置 runtime 的 32-seed 离线实验中，`32/32` 为 `Rpos > Lpos`。左臂 first-step position norm mean/p05/p95 为 `0.131/0.0956/0.1656 mm`，右臂为 `0.432/0.2188/0.7525 mm`，mean R/L ratio 为 `3.45`。

【推断】32-seed 一致的不对称说明现象不是某一个随机 seed 的偶然采样；它仍不能区分 checkpoint 学习到的不对称与保存输入的域偏移。

【未知】完整 172 帧逐帧回放的最终汇总应以本 bundle 的 `saved_observation_replay.json` 为准；若本节参考值与新报告不一致，应保留两个 JSON、Git SHA 和 checkpoint SHA256 后重新定责，不能挑选更符合预期的一组结果。

## 7. 已解决问题

【已确认】WebSocket 出站观测/ack 不再受 50 ms 接收轮询门控；事件驱动 sender 已把实测有效频率从 `9.34 Hz` 提升到 `19.76 Hz`。

【已确认】sender shutdown 已处理阻塞 send、意外 sender 异常、重叠连接 packer 复用和 close-before-join；localhost 50 个完整 obs/action/ack 周期的 p95 contract 为 `<15 ms`。

【已确认】动作/观测左右切片没有交换：`robot0/camera0` 为左，`robot1/camera1` 为右，action 固定为 `[left 0:10, right 10:20]`。

【已确认】latent 时间索引没有发现 off-by-one：慢更新 latent 在五步内复用，AT 快解码取当前 tactile history 的最后输出；五步锯齿属于计划边界不连续候选，不是已证实的数组越界。

## 8. 已排除问题

【已确认】两次日志均为全部动作成功调度（112/112、249/249），因此这两段采样中的主要频率差异不是 late-action 丢弃造成。

【已确认】日志中的约 `2*pi` 左臂 rotation jump 是 axis-angle 直接相减伪影；不能用它证明机械臂真的旋转一周。

【已确认】全局缩放动作不是可接受修复：训练 normalizer 左右 xyz RMS 接近，而保存输入的 32-seed 输出已经右大于左；缩放既不定责，也会扩大已活跃右臂的风险。

【未知】没有有序训练行就不能排除示教起始侧、转换错位或 action/state lag；normalizer 统计只能描述边际分布，不能替代 episode 顺序审计。

## 9. 未解决问题与证据等级

【已确认】五步边界存在毫米到厘米级夹爪 command jump；其来源可能是 latent 重规划、AT tactile history 在边界重置、输入 OOD 或 checkpoint 输出，但当前证据不能单独选择其中一项。

【已确认】Typhon 曾以 HTTP 400 终止控制链：`POST /action` 抛出 `RuntimeError`，控制子进程记录共享 fatal error，父进程在 `get_obs`/执行健康检查时转成 `ControllerProcessError`，清理阶段再进入 shutdown/standby。必须保留首次 HTTP 400 的 request body、response 和第一段 stack trace；后续 shutdown 报错通常是次生现象。

【未知】HTTP 400 的 backend 校验原因、四次非原子 POST 中失败的是第几次、失败前是否已接受部分 target，现有日志不能回答。

【推断】FRS 双夹爪 `0.1385 m` 超出训练 state 上界，可能改变 LDP 条件分布；在用训练机数据和 stage comparator 量化前，不能写成最终根因。

【未知】AT 是否能高保真重建真实训练 action、LDP 是否在正确 AT latent 上预测错误、训练 episode 是否本来就右臂先动，必须由第 11 节生成的两个 JSON 回答。

## 10. 四阶段定责矩阵

| 阶段 | 离线输入/检查 | 判定规则 | 责任候选 |
| --- | --- | --- | --- |
| A 原始数据/转换 | 有序 Zarr 或 LeRobot episode；审计 20D action/state、起始 30/60 帧和 lag | 【已确认】ground truth 已错、左右切片或 episode 边界异常 | 数据采集或转换 |
| B AT 重建 | 同一真实 20 步 action 与对应 20 步 tactile | 【推断】A 正常但 AT encode/decode 误差超阈值 | AT 架构、AT checkpoint 或配对 tactile |
| C LDP→AT | 初始视觉/state 预测 latent，再由同一 AT 解码 | 【推断】B 正常但 LDP→AT 输出异常 | LDP checkpoint 或 observation conditioning |
| D 部署保存输入 | FRS 保存观测顺序回放并与训练样本比较 | 【推断】A--C 训练样本正常而 FRS 回放异常 | 输入域、normalization、标定或传感器映射 |

【未知】阈值必须结合训练 action RMS `2.38/2.54 mm`、AT 重建误差和任务容差共同制定；不能先凭真机观感设阈值再反推责任。

## 11. 训练机执行步骤

【已确认】先创建新输出目录并记录代码版本；以下命令均不连接硬件：

```bash
cd /absolute/path/to/vb3_robot_server
mkdir -p debug_outputs
git rev-parse HEAD
git -C /absolute/path/to/RDP_vitamin rev-parse HEAD

python tools/rdp_debug/summarize_action_log.py \
  /absolute/path/to/20260815_145643/action_debug.jsonl \
  /absolute/path/to/20260815_160855/action_debug.jsonl \
  --replan-interval 5 \
  --output debug_outputs/action_log_summary.json

python tools/rdp_debug/audit_training_dataset.py \
  /absolute/path/to/pick_tube_01_04_rdp_zarr \
  --format zarr --start-windows 30 60 --max-lag 10 \
  --output debug_outputs/training_dataset_audit.json

python tools/rdp_debug/replay_saved_observations.py \
  --vitamin-repo /absolute/path/to/RDP_vitamin \
  --config /absolute/path/to/configs/deploy_pick_tube_rdp.yaml \
  --observations /absolute/path/to/20260813_162558 \
  --device cuda:0 \
  --output debug_outputs/saved_observation_replay.json

python tools/rdp_debug/compare_policy_stages.py \
  --vitamin-repo /absolute/path/to/RDP_vitamin \
  --config /absolute/path/to/configs/deploy_pick_tube_rdp.yaml \
  --dataset /absolute/path/to/pick_tube_01_04_rdp_zarr \
  --episode 0 --start-frame 0 --horizon 20 \
  --output debug_outputs/policy_stages_ep000000_f000000.json
```

【已确认】在运行模型前记录本地 artifact 身份：

```bash
sha256sum \
  /absolute/path/to/ldp/checkpoints/latest.ckpt \
  /absolute/path/to/at/checkpoints/latest.ckpt \
  /absolute/path/to/encoder/files/*

cp /absolute/path/to/ldp/.hydra/config.yaml \
  debug_outputs/ldp_hydra_config.yaml
```

【已确认】若输出文件已经存在，先改用新的 bundle 目录；不要删除或覆盖旧报告来“重跑”。若训练数据只有 LeRobot 格式，把 `--format zarr` 改为 `--format lerobot` 并传 dataset root。

## 12. 如何回传结果

【已确认】最小回传包必须包含：

- 【已确认】`debug_outputs/training_dataset_audit.json`。
- 【已确认】`debug_outputs/policy_stages_ep000000_f000000.json`。
- 【已确认】建议同时返回 `action_log_summary.json` 和 `saved_observation_replay.json`。
- 【已确认】服务端与 Vitamin 的精确 Git SHA。
- 【已确认】LDP、AT、encoder artifact 的 SHA256。
- 【已确认】checkpoint 的 Hydra config，以及生成 Zarr 的完整 conversion command（含参数和输入数据版本）。
- 【已确认】任一脚本非零退出时，返回第一段失败 stack trace、完整命令、Python/CUDA 版本；不要只返回最后一行。

【已确认】默认禁止回传完整专有图片、视频或原始数据集；先回传 JSON、哈希、配置和最小错误文本。只有后续定责确实需要具体帧时，再按指定帧号最小化提供。

## 13. 再次上真机前的检查清单

- 【已确认】四个 CLI 的 `--help` 全部离线成功，contract 扫描未发现 bridge、HTTP、camera 或 controller 依赖。
- 【已确认】训练数据审计明确 episode 边界、左右首动侧、前 30/60 帧 RMS 和最佳 lag，且 action/state 均为有限 20D。
- 【已确认】AT 重建和 LDP→AT comparator 已生成 JSON，失败责任阶段已按第 10 节确定或明确标为未知。
- 【已确认】保存观测回放的 172 帧 action 全部有限，五步边界统计已复核；FRS gripper OOD 已校正或有书面风险处置。
- 【已确认】禁止使用全局动作缩放、临时平滑或绕过 safety limits 来掩盖不对称。
- 【已确认】Typhon HTTP 400 的首个 request/response 已解释；四次非原子 POST 风险已有单次原子提交、backend 保证或可验证的安全替代方案。
- 【已确认】先做无动力/隔离 dry-run，再做低风险单步；急停、工作空间、速度/增量/夹爪限位和操作者位置均复核。
- 【未知】任一项仍为未知或缺报告时，不得恢复无人值守连续真机运行。
