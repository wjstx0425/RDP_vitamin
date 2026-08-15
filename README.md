# VB3 Robot Server
> **Deployment assets:** This branch intentionally excludes robot-specific `quest_2_ee_*.npy` hand-eye calibration files and `real_world/robot_api/assets/`. Before hardware startup, copy the calibration and URDF/mesh assets from the existing `vb3_robot_server` installation on the robot computer. Do not reuse calibration files across different robots.

## 系统结构

```text
双目相机 + 双臂状态
        ↓
vb3_robot_server（真机侧）
        ⇅ WebSocket + Token
VB3 SmolVLA 客户端（模型推理）
        ↓
动作过滤 → 安全检查 → waypoint 时间插值 → 双臂控制器
```

默认部署约定：

- 服务端仓库：`/home/typhon/vb3_robot_server`
- VB3 客户端仓库：`/home/typhon/vb3`
- 模型目录：`/home/typhon/models/tactile-test-03-1w`
- 服务地址：`127.0.0.1:26421`
- 左、右相机：`/dev/video0`、`/dev/video2`
- 控制频率：30 Hz
- 动作块长度：50

## 1. 克隆仓库

```bash
git clone git@github.com:KaiyueChen-code/vb3_robot_server.git
cd /home/typhon/vb3_robot_server
```


## 2. 安装服务端环境

以下流程面向 Ubuntu 24.04 x86_64。客户端遵循 LeRobot 官方的
Python 3.12、`uv` 和 Linux CUDA 12.8 环境约定。先安装通用系统依赖和
`uv`：

```bash
sudo apt-get update
sudo apt-get install -y git curl ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

如果系统已有 `uv`，可以跳过安装步骤。不要用 `pip install tmux`、
不要在 Conda 环境中共用这两个项目的 `.venv`。

### 2.1 服务端（Python 3.11）

使用 `--managed-python` 确保 `uv` 不会误选本机已有的 Conda Python：

```bash
cd /home/typhon/vb3_robot_server
bash scripts/setup_environment.sh
```

该脚本会在找不到 `uv` 时安装独立版本，然后按照 `uv.lock` 创建 Python
3.11 环境并运行测试。安装完成后，启动脚本会直接使用本仓库的 `.venv/bin/python`，
运行时不需要激活 Conda、虚拟环境或再调用 `uv`。

### 2.2 VB3 客户端（Python 3.12 + CUDA）

先用 `nvidia-smi` 确认 NVIDIA 驱动。当前 LeRobot Linux `uv.lock` 使用
CUDA 12.8，NVIDIA 驱动至少需要 570.86；当前机器的 580.159.03 满足要求。

```bash
nvidia-smi
cd /home/typhon/vb3
bash scripts/setup_environment.sh
```

两个仓库都使用自己的 `.venv`。不需要先激活 Conda；需要进入交互式
shell 时再执行 `source .venv/bin/activate`。

## 3. 下载 Hugging Face 模型

模型仓库：`KaiyueChen/tactile_test_03_1w`

固定 revision：`4f01b035a13df0c1c5db3e1e4e0c3d2bc5e5b098`

优先使用 Hugging Face 官方端点下载。先清除 shell 中可能遗留的镜像变量：

```bash
mkdir -p /home/typhon/models
unset HF_ENDPOINT

/home/typhon/vb3/.venv/bin/hf download \
  KaiyueChen/tactile_test_03_1w \
  --revision 4f01b035a13df0c1c5db3e1e4e0c3d2bc5e5b098 \
  --local-dir /home/typhon/models/tactile-test-03-1w
```

SmolVLA 初始化还会读取基础 VLM 的配置、分词器和初始权重。准备离线部署时，
先将它完整缓存：

```bash
HF_HOME=/home/typhon/.cache/huggingface \
/home/typhon/vb3/.venv/bin/hf download \
  HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

确认模型文件已经下载并且权重不是空文件：

```bash
test -s /home/typhon/models/tactile-test-03-1w/model.safetensors
find /home/typhon/models/tactile-test-03-1w -maxdepth 2 -type f
```

模型权重不要复制到本仓库，也不要提交 `*.safetensors`、`*.ckpt`、`*.pth` 或 `*.pt`。

## 4. 配置访问 Token

直接启动时，如果没有 token 文件，Bash 脚本会在终端中询问 token，
并仅创建本次运行使用的临时文件。如果不想每次输入，可选地创建：

```text
/home/typhon/vb3_robot_server/token_list.txt
```

在一个终端生成 token，并写入服务端允许列表：

```bash
cd /home/typhon/vb3_robot_server
read -rsp "VB robot token: " VB_ROBOT_TOKEN
echo
export VB_ROBOT_TOKEN
printf '%s\n' "$VB_ROBOT_TOKEN" > token_list.txt
chmod 600 token_list.txt
stat -c '%a %n' token_list.txt
```

每行可填写一个允许连接的 token。该文件已被 Git 忽略，禁止把真实 token 写进 README、配置文件或提交历史。

启动 VB3 客户端前，在客户端终端设置相同的 token：

```bash
read -rsp "VB robot token: " VB_ROBOT_TOKEN
echo
export VB_ROBOT_TOKEN
```

## 5. 检查相机

先确认设备存在：

```bash
ls -l /dev/video0 /dev/video2
```

启动双相机预览：

```bash
cd /home/typhon/vb3_robot_server
.venv/bin/python deploy_scripts/preview_cameras.py --side both
```

在预览窗口按 `Q`，或在终端按 `Ctrl+C` 退出。相机参数统一定义在 `configs/camera_config.py`。

确认机械臂控制服务可达，地址由 `configs/typhon_am2.yaml` 定义：

```bash
curl --max-time 3 http://192.168.100.100:8081
```

如果连接超时，先检查本机有线网络、IP 网段和机械臂控制服务，不能进入真机步骤。

## 6. 硬件无关联调

首次部署建议先运行 dry-run。它不会初始化相机或机械臂。

服务端终端：

```bash
cd /home/typhon/vb3_robot_server
bash scripts/bimanual_smolvla.sh --dry-run
```

VB3 客户端终端：

```bash
cd /home/typhon/vb3
bash src/vb3/deploy/run_client.sh --max-iterations 2
```

dry-run 完成两次 ACK 后会自动发送 STOP 并退出。

## 7. 启动真机部署

### 7.1 启动服务端

所有默认参数由 `configs/server_config.py` 管理，真机启动只需：

```bash
cd /home/typhon/vb3_robot_server
bash scripts/bimanual_smolvla.sh
```

临时保存 OBS 时可以附加 `--save_obs true`。

服务端和客户端在同一台机器运行时必须保持 `127.0.0.1`。当前连接没有 TLS，不要直接把服务端监听地址改成局域网或公网地址；远程部署应使用受控的加密隧道和防火墙策略。

### 7.2 启动 VB3 客户端

确认 `/home/typhon/vb3/src/vb3/configs/remote_client.toml` 至少包含：

```toml
[model]
checkpoint = "/home/typhon/models/tactile-test-03-1w"
device = "cuda"

[connection]
address = "127.0.0.1"
port = 26421
add_port = true
retry_interval_s = 1.0
action_ack_timeout_s = 30.0
token_env = "VB_ROBOT_TOKEN"
require_token = true

[observation]
data_type = "vision"
language_prompt = "Pick up two tubes"
single_arm_mode = false
no_state_obs_mode = false

[control]
control_frequency = 30.0
controller_frequency = 80.0
action_horizon = 50
steps_per_inference = 40

[runtime]
auto_start = false
warmup_runs = 2
status_interval_s = 2.0
max_iterations = 0

[logging]
save_observations = false
output_dir = "outputs/vb3_tactile_test_observations"
save_every = 1
queue_size = 32
```

启动客户端：

```bash
cd /home/typhon/vb3

UV_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
.venv/bin/vb3-deploy \
  --device cuda \
  --config src/vb3/configs/remote_client.toml
```

客户端完成模型加载和 warmup 后，根据终端提示确认开始。真机运动期间必须有人观察，并保持急停可用。

## 8. OBS 和动作日志

使用 `--save_obs true` 时，处理后的 256×256 OBS 会保存到：

```text
eval_obs_data/eval_obs_YYYYMMDD_HHMMSS/
```

每一步目录包含左右相机图片、时间戳、末端位姿和夹爪状态。

动作诊断日志实时写入：

```text
action_debug_logs/YYYYMMDD_HHMMSS/action_debug.jsonl
```

正常停止服务端后，会尝试生成：

```text
action_debug_logs/YYYYMMDD_HHMMSS/plots/
ee_action_logs/
```

`ee_action_logs/` 中包含左右臂 `x`、`y`、`z`、`rx`、`ry`、`rz` 和 `g` 的实际值与目标值曲线。蓝线是实际 EE，红线是目标。

PNG 导出依赖 Plotly、Kaleido 和 Chrome。如果终端提示无法导出 PNG，可运行 `plotly_get_chrome` 安装所需浏览器组件。

上述 OBS、日志和图片均为本地运行产物，不应提交到 Git。

## 9. 动作调度和安全参数

常用服务端参数：

| 参数 | 含义 | 首次真机建议值 |
| --- | --- | ---: |
| `--max-pos-speed` | 末端最大线速度，m/s | `0.25` |
| `--max-rot-speed` | 末端最大角速度，rad/s | `0.16` |
| `--max_gripper_speed` | 夹爪命令最大速度，m/s | `0.05` |
| `--max_action_pos_delta` | 模型原始单步位置 delta 校验阈值，m | `0.03` |
| `--max_action_rot_delta` | 模型原始单步旋转 delta 校验阈值，rad | `0.35` |
| `--max-executed-actions` | 每个动作块最多调度的有效动作数 | `50`（实际还受客户端 `steps_per_inference` 限制） |

解码后的 waypoint 不再因为目标相对当前位姿的跳变而被运行时丢弃。控制器通过
waypoint 时间插值执行动作，并用 `--max-pos-speed`、`--max-rot-speed` 和
`--max_gripper_speed` 限制实际命令速度；不再施加额外的加速度、jerk 或响应时间滤波。
位置和旋转 delta 参数仍用于模型原始单步输出校验。

`steps_per_inference` 决定每次推理后最多调度多少步动作。当前
`remote_client.toml` 保留 50-step 模型输出，但每轮最多执行 `40` 步，在
30 Hz 下对应约 1.33 秒。这样可在下一轮推理期间保留动作余量，并缩短单次
开放环执行长度。修改该值前必须结合
`infer_latency_ms`、`loop out of time` 和 action debug 日志做短时间真机验证。

## 10. 常见问题

### `VB3 server environment not found`

```bash
cd /home/typhon/vb3_robot_server
uv sync --locked --managed-python --python 3.11
```

这个错误表示本仓库的 `.venv` 尚未创建，与 Conda 是否激活无关。

### Hugging Face 下载失败

先确认官方端点可达，并打开调试输出：

```bash
unset HF_ENDPOINT
HF_DEBUG=1 \
/home/typhon/vb3/.venv/bin/hf download \
  KaiyueChen/tactile_test_03_1w \
  --revision 4f01b035a13df0c1c5db3e1e4e0c3d2bc5e5b098 \
  --local-dir /home/typhon/models/tactile-test-03-1w
```

下载完成后使用离线环境变量启动，避免运行时再次访问 Hugging Face。
如果官方端点不可达再尝试 `HF_ENDPOINT=https://hf-mirror.com`；部分镜像与
新版 `huggingface_hub` 的 HEAD 元数据协议不兼容，会误报资源不存在。

### `--save_obs` 报 boolean 错误

正确写法：

```bash
--save_obs true
```

### 相机请求 30 FPS，但驱动返回其他帧率

服务端会继续使用驱动实际配置。该警告本身不会阻止启动，但实际帧率必须在
相机预览中确认；训练和当前控制配置均为 30 Hz。

### 持续出现 `loop out of time`

表示模型推理、图像处理和通信的总周期超过当前动作覆盖时间。检查客户端是否使用 CUDA，并结合 `action_debug.jsonl` 分析实际批周期和过期动作数量。

### 原始模型 action delta 超过限制

表示模型输出的单步相对位置或旋转超过
`--max_action_pos_delta` / `--max_action_rot_delta`，整个 action chunk 会在解码前被拒绝。
解码后的目标与当前位姿之间即使存在较大差异，也会交给控制器按位置和旋转速度上限插值执行。

### 没有生成 EE/action PNG

图像只在服务端正常停止时生成。确认终端出现 `Plotting ee vs target`；如果提示 Kaleido 或 Chrome 错误，安装 Chrome 组件后重新运行。

## 11. 真机安全要求

- 首次测试前清空机械臂工作空间，确认相机对应左右手。
- 保持机械急停可触达，并安排人员持续观察。
- 先运行 dry-run，再进行短时间、低速、空载测试。
- 不要未经验证放宽模型原始单步 delta 的 `0.03 m` 和 `0.35 rad` 校验阈值。
- 模型、提示词、相机标定或手眼标定发生变化后，必须重新进行低速测试。
- 出现持续过期动作、大幅跳变、坐标异常或控制器错误时立即停止。

## 12. 不应提交到 Git 的内容

- `token_list.txt` 和 `.env*`
- `.venv/`、Python 缓存和测试缓存
- 模型权重、checkpoint 和 Hugging Face 缓存
- `eval_obs_data/`
- `action_debug_logs/`
- `ee_action_logs/`
- `recordings/`
- 相机坏帧和本地诊断输出

提交前检查：

```bash
git status --short --ignored
git diff --check
```

任何真实 token、模型权重、现场 OBS 或动作日志都不应进入 Git 历史。
