# Ultra-Scale Demos：手写分布式训练并行，每一步都和单卡对答案

**中文** | [English summary](#english-summary)

一组配合 [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)（Hugging Face 出品的大模型分布式训练指南）学习的**最小可运行 demo**（非官方）。

- 用一两百行 PyTorch 手写 DP、ZeRO、TP/SP、PP、CP、EP 和二维并行，不依赖 Megatron / DeepSpeed。
- 每个脚本都把并行结果和**单卡 baseline 逐元素对比**，打印 `[PASS]` / `[FAIL]`：先确认"算得对"，再看"省了什么、花了什么"。
- 只依赖 PyTorch。没有 GPU 也能在笔记本 CPU 上用 gloo 后端跑通全部逻辑；有 GPU 时自动用 NCCL。
- 附带 Slurm 脚本，支持单节点多卡、两节点四卡，以及 MIG 切片。

👉 **第一次看？从 [上手教程](docs/tutorial.md) 开始**：每个 demo 讲清目标、运行命令、怎么读输出（附真实输出）和动手练习。

## Demo 一览

| 脚本 | 内容 | 进程数 |
| --- | --- | --- |
| `00_collectives.py` | 通信原语语义 + all-reduce 带宽测试 | ≥2 |
| `01_memory.py` | 显存账本、激活重计算、梯度累积 | 1 |
| `02_dp.py` | 手写 DP：naive / overlap / bucket，对比 DDP | ≥2 |
| `03_zero.py` | 手写 ZeRO-1/2/3，`--fsdp` 对比 FSDP2 | ≥2 |
| `04_tp_sp.py` | Megatron 风格 TP 和 TP+SP 的 Transformer block | ≥2 |
| `05_pp.py` | 流水线并行 AFAB / 1F1B 调度 + ASCII 时间线 | ≥2 |
| `06_cp.py` | 上下文并行：Ring Attention、all-gather KV、zig-zag | ≥2 |
| `07_ep.py` | 专家并行：MoE all-to-all dispatch/combine | ≥2 |
| `08_2d_dp_tp.py` | DeviceMesh 二维 DP×TP，`--native` 用 DTensor TP + FSDP2 | 4（或 2，tp=2） |

依赖：只有 PyTorch，**建议 torch ≥ 2.6**（FSDP2 的 `fully_shard` 需要；其余 demo 2.2+ 即可）。

## 快速开始

### 1. 笔记本 / CPU：先跑通逻辑

```bash
pip install "torch>=2.6"
bash run_all_local.sh                 # 全部跑一遍
./run_local.sh 05_pp.py 4 --micro 8   # 单独跑一个：脚本 进程数 参数...
```
没有 GPU 时自动用 gloo 后端；GPU 机器上可以 `FORCE_CPU=1` 强制走 CPU。

### 2. 一台多卡 GPU 机器

```bash
torchrun --standalone --nproc_per_node 2 02_dp.py --dim 2048 --layers 8 --bsz 64 --bench
torchrun --standalone --nproc_per_node 4 08_2d_dp_tp.py --native      # dp=2 × tp=2
```

### 3. Slurm 集群

```bash
sbatch slurm/run_1node.sbatch                          # 单节点 2 卡，跑全部 demo
sbatch slurm/run_1node.sbatch 03_zero.py --fsdp        # 只跑一个
sbatch --gpus=a100:2 slurm/run_1node.sbatch            # 指定卡型（写法以你的集群为准）
VENV=$HOME/venvs/dist sbatch slurm/run_1node.sbatch    # 指定虚拟环境（或提交前先 conda activate）
sbatch slurm/run_2node.sbatch                          # 两节点 × 2 卡 = 4 卡，DP×TP
squeue -u $USER; tail -f logs/*.out
```

**只有 MIG 切片也能跑。** 很多共享集群把一张大卡切成多个 MIG 实例，每人能拿到的整卡有限。`run_1node.sbatch` 会自动识别 MIG：给每个 rank 只暴露自己的切片（`slurm/mig_rank.sh`），并设置 `NCCL_MULTI_RANK_GPU_ENABLE=1`（两块切片可能在同一张物理卡上，NCCL 默认会拒绝）。可以先用 `sbatch --gpus=<MIG 型号>:2 slurm/check_mig_nccl.sbatch` 检查 NCCL 是否可用。注意：切片之间测出的带宽和速度不代表真实的多卡互联，适合验证正确性，不适合做性能对比。

## 已验证的环境

| 环境 | 结果 |
| --- | --- |
| Linux，2 块 H100 MIG 切片（单节点），PyTorch 2.15 + NCCL 2.30 | 9 个 demo、32 项校验全部 `[PASS]` |

欢迎在 issue 里补充其他环境（CPU、整卡多卡、多节点）的运行结果。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| `Duplicate GPU detected` | 两个 rank 用了同一张卡：检查 `--nproc_per_node` 不超过实际 GPU 数 |
| `Multiple Ranks are using the same GPU/Partition` | 两块 MIG 切片在同一张物理卡上：设置 `NCCL_MULTI_RANK_GPU_ENABLE=1`（`run_1node.sbatch` 已自动处理） |
| MIG 下 `torch.cuda.device_count()` 比申请的切片数少 | CUDA 不一定把多块切片都暴露给一个进程：用 `slurm/mig_rank.sh` 给每个 rank 单独指定切片 |
| 多节点卡在初始化 | `export NCCL_DEBUG=INFO` 看日志；设置 `NCCL_SOCKET_IFNAME=<网卡名>`；或 `NCCL_IB_DISABLE=1` |
| `[FAIL]` 误差 1e-3 量级 | GPU 开了 TF32；`common.setup()` 已关闭，如果自己改了代码注意保留 |
| `ImportError: fully_shard` | torch < 2.6，升级或去掉 `--fsdp` / `--native` |
| Slurm 作业被拒 | 没写 `--gpus`，或者 GPU 型号写错（`sinfo -o "%N %G"` 查看集群的 GPU 名字） |
| 装包或编译时报 /tmp 空间不足 | 一些集群 /tmp 很小：`export TMPDIR=$HOME/tmp` |

## 目录

```
00_collectives.py … 08_2d_dp_tp.py   各个 demo
common.py                            进程组初始化、结果校验、计时、小模型
run_local.sh / run_all_local.sh      本地 CPU 运行
slurm/                               Slurm 作业脚本（单节点、两节点、MIG）
docs/tutorial.md                     上手教程
```

## English summary

Minimal, runnable PyTorch demos to accompany [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (unofficial). Each script hand-implements one parallelism strategy — DP (naive / overlap / bucketed), ZeRO-1/2/3 and FSDP2, Megatron-style TP and TP+SP, pipeline parallelism (AFAB / 1F1B with ASCII timelines), context parallelism (Ring Attention, all-gather KV, zig-zag), expert parallelism (MoE all-to-all) and 2D DP×TP with DeviceMesh — and checks the result element-wise against a single-device baseline, printing `[PASS]` / `[FAIL]`. PyTorch is the only dependency; everything runs on CPU with gloo, and on GPUs with NCCL. Slurm scripts for 1-node, 2-node and MIG-slice setups are included. Comments and the [tutorial](docs/tutorial.md) are in Chinese.

## 致谢与许可

- 内容组织参考 [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)（Hugging Face nanotron 团队）；本仓库为独立实现，与其作者无关。
- 代码以 [MIT License](LICENSE) 发布。
