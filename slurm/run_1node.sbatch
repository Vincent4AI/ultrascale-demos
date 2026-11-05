#!/bin/bash
# ---------------------------------------------------------------------------
# 单节点多卡跑 demo（Slurm）
#   跑全部：          sbatch slurm/run_1node.sbatch
#   跑某一个：        sbatch slurm/run_1node.sbatch 03_zero.py --fsdp
#   指定卡型/约束：   sbatch --gpus=a100:2 --constraint=<feature> slurm/run_1node.sbatch
#   指定虚拟环境：    VENV=$HOME/venvs/dist sbatch slurm/run_1node.sbatch
#                     （或者提交前先 conda activate，sbatch 默认继承当前环境变量）
# 支持 MIG 切片：脚本会自动识别，给每个 rank 只暴露自己的切片（slurm/mig_rank.sh），
#   并设置 NCCL_MULTI_RANK_GPU_ENABLE=1（两块切片可能在同一张物理卡上，NCCL 默认拒绝）。
#   注意：切片之间测出的带宽 / 速度不代表真实多卡互联。
# ---------------------------------------------------------------------------
#SBATCH --job-name=ultrascale
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2                    # 按你的集群改：型号写法如 --gpus=a100:2，必要时加 --partition / --constraint
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# ---- 环境 ----
if [[ -n "${VENV:-}" ]]; then set +u; source "$VENV/bin/activate"; set -u; fi
# 有些集群 /tmp 很小，可以用 TMPDIR 指到自己的目录：export TMPDIR=$HOME/tmp

export OMP_NUM_THREADS=4
export TORCH_CPP_LOG_LEVEL=ERROR
export NCCL_DEBUG=WARN            # 出问题时改成 INFO

echo "== host: $(hostname)   job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
nvidia-smi topo -m                # 看两张卡之间是 NV#（NVLink）还是 PIX/SYS（PCIe）
MIG_UUIDS=$(nvidia-smi -L | grep -oE "MIG-[0-9a-f-]+" | tr '\n' ' ' || true)
if [[ -n "$MIG_UUIDS" ]]; then
  NGPU=$(wc -w <<< "$MIG_UUIDS")
  export MIG_UUIDS NCCL_MULTI_RANK_GPU_ENABLE=1
  echo "== MIG 模式：$NGPU 个切片"
else
  NGPU=$(nvidia-smi -L | wc -l)
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

FAILED=()
run() {  # run <script> [args...]；单个 demo 失败不影响后面的 demo
  echo; echo "########## $* ##########"
  local rc=0
  if [[ "$1" == "01_memory.py" ]]; then
    if [[ -n "$MIG_UUIDS" ]]; then
      CUDA_VISIBLE_DEVICES=${MIG_UUIDS%% *} python "$@" || rc=$?
    else
      python "$@" || rc=$?
    fi
  elif [[ -n "$MIG_UUIDS" ]]; then
    torchrun --standalone --nproc_per_node "$NGPU" --no-python slurm/mig_rank.sh "$@" || rc=$?
  else
    torchrun --standalone --nproc_per_node "$NGPU" "$@" || rc=$?
  fi
  [[ $rc -eq 0 ]] || { echo "!! $* 失败（退出码 $rc）"; FAILED+=("$*"); }
}

if [[ $# -gt 0 ]]; then
  run "$@"
else
  run 00_collectives.py
  run 01_memory.py --dim 2048 --layers 8 --bsz 512
  run 02_dp.py --dim 2048 --layers 8 --bsz 64 --bench
  run 03_zero.py --fsdp
  run 04_tp_sp.py
  run 05_pp.py
  run 05_pp.py --sim_ms 0 --dim 4096 --mbs 64 --micro 8
  run 06_cp.py --seq 64
  run 07_ep.py
  run 08_2d_dp_tp.py --tp "$NGPU" --native   # 单节点 2 卡时 dp=1；4 卡请用 run_2node.sbatch
fi

echo; echo "########## 汇总 ##########"
if [[ ${#FAILED[@]} -eq 0 ]]; then echo "全部 demo 运行完成"; else printf '失败：%s\n' "${FAILED[@]}"; exit 1; fi
