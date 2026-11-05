#!/bin/bash
# ---------------------------------------------------------------------------
# 两个节点 × 每节点 2 卡 = 4 卡，跑 DP×TP 二维并行（Slurm）
#   sbatch slurm/run_2node.sbatch                       # 默认跑 08_2d_dp_tp.py
#   sbatch slurm/run_2node.sbatch 00_collectives.py     # 看跨节点 all_reduce 带宽掉多少
#   VENV=$HOME/venvs/dist sbatch slurm/run_2node.sbatch # 指定虚拟环境
# TP 组 = 同一节点内的 2 张卡（rank 0,1 | 2,3），DP 组跨节点（rank 0,2 | 1,3）
# 需要整卡（不支持 MIG 切片跨节点混用）。
# ---------------------------------------------------------------------------
#SBATCH --job-name=ultrascale-2n
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2           # 按你的集群改：型号写法如 --gpus-per-node=a100:2，必要时加 --partition / --constraint
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ -n "${VENV:-}" ]]; then set +u; source "$VENV/bin/activate"; set -u; fi

export OMP_NUM_THREADS=4
export TORCH_CPP_LOG_LEVEL=ERROR
export NCCL_DEBUG=WARN
# 跨节点卡住时依次尝试：
#   export NCCL_SOCKET_IFNAME=<用 `ip -br a` 看到的网卡名，如 eno1>
#   export NCCL_IB_DISABLE=1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
echo "nodes: $SLURM_JOB_NODELIST   master: $MASTER_ADDR:$MASTER_PORT"

SCRIPT=${1:-08_2d_dp_tp.py}
shift || true
ARGS=("$@")
if [[ "$SCRIPT" == "08_2d_dp_tp.py" && ${#ARGS[@]} -eq 0 ]]; then ARGS=(--tp 2 --native); fi

# srun 在每个节点起一个 torchrun，torchrun 再在本节点起 2 个进程
srun --kill-on-bad-exit=1 torchrun \
  --nnodes "$SLURM_NNODES" --nproc_per_node 2 \
  --rdzv_id "$SLURM_JOB_ID" --rdzv_backend c10d --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
  "$SCRIPT" "${ARGS[@]}"
