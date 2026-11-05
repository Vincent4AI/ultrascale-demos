#!/bin/bash
# ---------------------------------------------------------------------------
# 检查同一节点上的 2 块 MIG 切片之间能不能跑 NCCL
#   sbatch --gpus=<MIG 型号>:2 slurm/check_mig_nccl.sbatch
# 依次尝试三种方式，日志里看每种的"结果"行：
#   A：原样 torchrun（每个进程按 LOCAL_RANK 选卡）
#   B：每个 rank 只暴露自己的 MIG 实例（slurm/mig_rank.sh）
#   C：B + NCCL_MULTI_RANK_GPU_ENABLE=1（两块切片在同一张物理卡上时需要）
# ---------------------------------------------------------------------------
#SBATCH --job-name=mig-nccl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ -n "${VENV:-}" ]]; then source "$VENV/bin/activate"; fi
export TORCH_CPP_LOG_LEVEL=ERROR OMP_NUM_THREADS=4

echo "== host: $(hostname)   job: $SLURM_JOB_ID"
nvidia-smi -L
python -c "import torch; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version(), 'visible cuda devices:', torch.cuda.device_count())"

echo; echo "########## A：原样 torchrun ##########"
NCCL_DEBUG=WARN timeout 120 torchrun --standalone --nproc_per_node 2 00_collectives.py --max_mb 256 \
  && echo "== A 结果：成功" || echo "== A 结果：失败（退出码 $?）"

MIG_UUIDS=$(nvidia-smi -L | grep -oE "MIG-[0-9a-f-]+" | tr '\n' ' ' || true)
if [[ $(wc -w <<< "$MIG_UUIDS") -lt 2 ]]; then echo "== 只看到 $(wc -w <<< "$MIG_UUIDS") 个 MIG 实例，跳过 B/C"; exit 0; fi
export MIG_UUIDS
FILTER="NCCL INFO (cudaDriver|Bootstrap|NET/Plugin|comm 0x.*Init COMPLETE|Setting affinity|NVLS|P2P Chunksize|threadThresholds|[0-9]+ coll channels|CC Off|TUNER|ENV)"

echo; echo "########## B：每个 rank 一个 MIG 实例 ##########"
NCCL_DEBUG=INFO timeout 180 torchrun --standalone --nproc_per_node 2 --no-python slurm/mig_rank.sh \
  00_collectives.py --max_mb 256 2>&1 | grep -v -E "^W[0-9]{4}" | grep -E -v "$FILTER" \
  && echo "== B 结果：成功" || echo "== B 结果：失败"

echo; echo "########## C：B + NCCL_MULTI_RANK_GPU_ENABLE=1 ##########"
NCCL_MULTI_RANK_GPU_ENABLE=1 NCCL_DEBUG=INFO timeout 180 torchrun --standalone --nproc_per_node 2 --no-python slurm/mig_rank.sh \
  00_collectives.py --max_mb 256 2>&1 | grep -v -E "^W[0-9]{4}" | grep -E -v "$FILTER" \
  && echo "== C 结果：成功" || echo "== C 结果：失败"
