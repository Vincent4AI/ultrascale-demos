#!/bin/bash
# 在 Mac / 没有 GPU 的机器上，用 CPU + gloo 把所有 demo 跑一遍（约 1 分钟）
# 用法：bash run_all_local.sh
set -e
cd "$(dirname "$0")"
R=./run_local.sh
python 01_memory.py
$R 00_collectives.py 2
$R 02_dp.py 2
$R 03_zero.py 2 --fsdp
$R 04_tp_sp.py 2
$R 05_pp.py 4 --micro 8
$R 06_cp.py 2
$R 07_ep.py 2
$R 08_2d_dp_tp.py 4 --native
