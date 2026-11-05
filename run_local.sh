#!/bin/bash
# 本地 CPU 冒烟测试：bash run_local.sh 02_dp.py 2
# 过滤掉 torchrun 的启动警告
NP=${2:-2}
export TORCH_CPP_LOG_LEVEL=ERROR OMP_NUM_THREADS=1
torchrun --nproc_per_node "$NP" "$1" "${@:3}" 2>&1 | grep -v -E "^W[0-9]{4}|\*{10}|OMP_NUM_THREADS|socket.cpp"
