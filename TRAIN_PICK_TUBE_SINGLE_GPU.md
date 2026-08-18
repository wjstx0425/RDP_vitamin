# Pick-tube PCA30 单卡训练

这套入口面向单张 NVIDIA RTX PRO 6000，训练六个数据集合并后的
PCA30 数据。AT 和 LDP 使用当前项目配置：20 维动作、30 维触觉、
`n_latent_dims=16`、`conv_latent_dims=32`、`rnn_latent_dims=64`、
`n_embed=32`。

## 1. 安装环境

```bash
cd /path/to/reactive_diffusion_policy-main
bash scripts/install_pick_tube_training_env.sh
source .venv/bin/activate
```

默认安装 Python 3.12、PyTorch 2.10 CUDA 13.0 和
`requirements-rdp-training.txt`。可通过环境变量调整安装位置或镜像：

```bash
VENV_DIR=/path/to/.venv \
PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
bash scripts/install_pick_tube_training_env.sh
```

脚本安装的是 `hf` CLI；需要访问私有 Hugging Face 数据时使用：

```bash
.venv/bin/hf auth login
```

## 2. 准备六数据集 PCA30 Zarr

已有 `data/pick_tube_01_06_pca30_rdp_zarr/replay_buffer.zarr` 时跳过转换。
现有 `tactile_pca_2x15.npz` 已由六个数据集共同拟合。

```bash
DATASET_PATH="$PWD/data/pick_tube_01_06_pca30_rdp_zarr" \
TACTILE_PCA_PATH="$PWD/data/PCA_Transform_PickTube/tactile_pca_2x15.npz" \
bash scripts/setup_pick_tube_data.sh convert
```

## 3. 单卡完整训练

```bash
GPU_ID=0 \
RUN_ID=pca30_latent32_full6_v1 \
AT_EPOCHS=20 \
LDP_EPOCHS=10 \
AT_BATCH=64 \
LDP_BATCH=64 \
NUM_WORKERS=8 \
AT_CHECKPOINT_EVERY=1 \
LDP_CHECKPOINT_EVERY=1 \
MIXED_PRECISION=bf16 \
bash scripts/train_pick_tube_single_gpu.sh all
```

默认输出为：

```text
data/outputs/pick_tube_01_06/
├── at_pca30_latent32_full6_v1/
│   └── checkpoints/latest.ckpt
└── ldp_pca30_latent32_full6_v1/
    ├── checkpoints/latest.ckpt
    └── normalizer.pkl
```

## 4. 断点续训

使用相同的 `RUN_ID` 再执行同一命令即可。`AT_EPOCHS` 和 `LDP_EPOCHS`
表示目标总 epoch 数，不是额外增加的 epoch 数。例如已有 LDP epoch 7，
`LDP_EPOCHS=10` 会继续训练到总计 10 个 epoch。

只训练 LDP 时显式指定 AT：

```bash
GPU_ID=0 \
RUN_ID=pca30_latent32_full6_v1 \
AT_CKPT=/absolute/path/to/at/checkpoints/latest.ckpt \
bash scripts/train_pick_tube_single_gpu.sh ldp
```

只打印命令、不启动训练：

```bash
DRY_RUN=1 RUN_ID=pca30_latent32_full6_v1 \
bash scripts/train_pick_tube_single_gpu.sh all
```

