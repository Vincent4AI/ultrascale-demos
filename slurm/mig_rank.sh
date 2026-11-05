#!/bin/bash
# MIG 切片用的 torchrun 启动包装：每个 rank 只看到自己那一个 MIG 实例。
#   MIG_UUIDS="MIG-xxx MIG-yyy" torchrun --no-python ... slurm/mig_rank.sh 02_dp.py [参数...]
# 原因：MIG 下 CUDA 不一定把多个切片都暴露给一个进程，按 LOCAL_RANK 选卡会出错；
#       给每个 rank 单独设置 CUDA_VISIBLE_DEVICES 后，进程里的卡号统一是 0。
set -euo pipefail
read -r -a U <<< "${MIG_UUIDS:?需要设置 MIG_UUIDS（空格分隔的 MIG UUID）}"
export CUDA_VISIBLE_DEVICES=${U[$LOCAL_RANK]} LOCAL_RANK=0
exec python "$@"
