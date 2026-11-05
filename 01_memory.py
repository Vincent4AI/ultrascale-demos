"""第0章 单卡显存账本：为什么一张卡放不下，以及两把"单卡就能用"的刀。

运行：  python 01_memory.py                (单进程即可；GPU 上会额外打印真实显存)
        python 01_memory.py --dim 1024 --layers 8 --bsz 64

看什么：
  1) 静态显存：参数 + 梯度 + Adam 状态，和 batch 无关，按 Ψ（参数量）线性增长
  2) 激活显存：和 batch × seq 线性增长 —— 用 saved_tensors_hooks 精确统计"为 backward 存了多少"
  3) 激活重计算（activation checkpointing）：只存每个 block 的输入，backward 时重算
  4) 梯度累积：把大 batch 拆成 micro-batch，梯度完全一样，激活峰值按 micro-batch 算
"""
import argparse
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from common import (setup, cleanup, log0, header, check, make_model, get_batch, num_params, fmt_bytes, is_cuda,
                    SavedTensorCounter)


def forward(model, x, recompute: bool):
    for blk in model.blocks:
        # use_reentrant=False 是 PyTorch 推荐的新实现
        x = checkpoint(blk, x, use_reentrant=False) if recompute else blk(x)
    return x


def static_memory(model):
    header("1. 静态显存（与 batch 无关）")
    psi = num_params(model)
    log0(f"  本 demo 模型参数量 Ψ = {psi:,}")
    log0(f"  fp32 训练：参数 4Ψ + 梯度 4Ψ + Adam(m,v) 8Ψ = 16Ψ = {fmt_bytes(16 * psi)}")
    log0("  bf16 混合精度：bf16 参数 2Ψ + bf16 梯度 2Ψ + fp32 主权重 4Ψ + Adam 8Ψ = 16Ψ"
         "（若梯度用 fp32 累积再 +4Ψ）")
    log0("  —— 混合精度并不省'静态'显存，省的是激活和算力。")
    log0(f"\n  {'模型':>8s} {'16Ψ (只算权重+优化器)':>24s}")
    for name, n in [("1B", 1e9), ("8B", 8e9), ("70B", 70e9), ("405B", 405e9)]:
        log0(f"  {name:>8s} {fmt_bytes(16 * n):>24s}")
    log0("  => 8B 模型光静态部分就要 ~120GB，单张 80GB 卡放不下：这就是 ZeRO 的动机")


def activation_memory(model, dim, bsz, dev):
    header("2. 激活显存：随 batch 线性增长；3. 重计算后的激活")
    log0(f"  {'batch':>6s} {'不重计算':>14s} {'重计算':>14s} {'节省':>8s}")
    for b in [bsz // 4, bsz // 2, bsz]:
        x, y = get_batch(0, b, dim, dev)
        res = []
        for rc in [False, True]:
            with SavedTensorCounter(model) as c:
                out = forward(model, x, rc)
                loss = F.mse_loss(out, y)
            loss.backward()
            model.zero_grad(set_to_none=True)
            res.append(c.bytes)
        log0(f"  {b:>6d} {fmt_bytes(res[0]):>14s} {fmt_bytes(res[1]):>14s} {1 - res[1] / res[0]:>7.0%}")
    log0("  每个 block 不重计算时要存：LN 输入/统计量、fc1 输入、GELU 输入、fc2 输入 …")
    log0("  重计算只存 block 输入，backward 时多跑一次 forward（约多 1/3 的计算）")


def real_gpu_memory(model, dim, bsz, dev):
    if not is_cuda(dev):
        log0("\n  (CPU 模式：跳过真实显存测量；在 GPU 节点上会打印 torch.cuda.memory_allocated)")
        return
    header("GPU 真实显存（torch.cuda.memory_allocated）")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, y = get_batch(0, bsz, dim, dev)
    torch.cuda.reset_peak_memory_stats()
    mem = lambda: torch.cuda.memory_allocated()
    m0 = mem()
    out = forward(model, x, False)
    loss = F.mse_loss(out, y)
    m1 = mem()
    loss.backward()
    m2 = mem()
    opt.step()
    m3 = mem()
    log0(f"  只有参数               : {fmt_bytes(m0)}")
    log0(f"  forward 后 (+激活)     : {fmt_bytes(m1)}")
    log0(f"  backward 后 (+梯度-激活): {fmt_bytes(m2)}")
    log0(f"  第一次 step 后 (+Adam) : {fmt_bytes(m3)}")
    log0(f"  峰值                   : {fmt_bytes(torch.cuda.max_memory_allocated())}")
    opt.zero_grad(set_to_none=True)


def grad_accumulation(model, dim, bsz, dev):
    header("4. 梯度累积：4 个 micro-batch 累积 == 一个大 batch")
    x, y = get_batch(0, bsz, dim, dev)
    # 大 batch 一次算完
    F.mse_loss(model(x), y).backward()
    ref = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    # 拆成 4 份累积；注意 loss 要除以累积步数，否则梯度是 4 倍
    n_acc = 4
    for xm, ym in zip(x.chunk(n_acc), y.chunk(n_acc)):
        (F.mse_loss(model(xm), ym) / n_acc).backward()
    got = [p.grad for p in model.parameters()]
    check("grad accumulation == big batch", got, ref)
    model.zero_grad(set_to_none=True)
    log0("  global_batch = micro_batch × grad_acc_steps × dp_size —— 后面每章都会用到这个公式")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--bsz", type=int, default=256)
    args = ap.parse_args()
    rank, world, dev = setup()
    model = make_model(args.dim, args.layers, dev)
    static_memory(model)
    activation_memory(model, args.dim, args.bsz, dev)
    real_gpu_memory(model, args.dim, args.bsz, dev)
    grad_accumulation(model, args.dim, args.bsz, dev)
    cleanup()
