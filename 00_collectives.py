"""第1章 通信原语：分布式训练的全部"积木"。

运行：  torchrun --nproc_per_node 2 00_collectives.py      （也可以 4）
GPU 上会额外跑一个 all-reduce 带宽测试，可以和 nvidia-smi topo -m 的拓扑对照着看。

要记住的结论：
  * all_reduce  = reduce_scatter + all_gather （ring 实现里正好就是这两步）
  * 每张卡在 ring all-reduce 里发送的数据量 ≈ 2(N-1)/N × 消息大小，几乎与卡数无关
  * DP 用 all_reduce；ZeRO 用 reduce_scatter / all_gather；TP 用 all_reduce（SP 下换成 RS/AG）；
    PP 用 send/recv；EP 用 all_to_all；CP 用 send/recv（ring）或 all_gather
"""
import argparse
import torch
import torch.distributed as dist
from common import setup, cleanup, log0, log_all, header, check, is_cuda, fmt_bytes


def show(name, t):
    return f"  rank{dist.get_rank()} {name:<12s} {t.tolist()}"


def semantics(rank, world, dev):
    header("1. 每个原语做了什么（看每个 rank 前后的张量）")

    # broadcast：rank0 的数据复制到所有 rank（DP 开训前同步初始权重就用它）
    x = torch.tensor([float(rank)] * 2, device=dev)
    dist.broadcast(x, src=0)
    log0("broadcast(src=0)：所有 rank 拿到 rank0 的值")
    log_all(show("after", x))

    # all_reduce：所有 rank 求和，结果每个 rank 都有一份（DP 同步梯度）
    x = torch.tensor([rank + 1.0, 10.0 * (rank + 1)], device=dev)
    log0("all_reduce(SUM)：逐元素求和，每个 rank 都拿到完整结果")
    log_all(show("before", x))
    dist.all_reduce(x)
    log_all(show("after", x))

    # reduce_scatter：先求和，再把结果切成 N 份，第 i 份给 rank i（ZeRO-2/3 的梯度）
    x = torch.arange(world * 2, dtype=torch.float32, device=dev) + 100 * rank
    out = torch.empty(2, device=dev)
    log0("reduce_scatter：求和后切片，rank i 只拿第 i 片")
    log_all(show("before", x))
    dist.reduce_scatter_tensor(out, x)
    log_all(show("after", out))

    # all_gather：每个 rank 贡献一片，拼起来每个 rank 都有完整的（ZeRO-3 取回参数）
    x = torch.full((2,), float(rank), device=dev)
    out = torch.empty(world * 2, device=dev)
    dist.all_gather_into_tensor(out, x)
    log0("all_gather：拼接所有 rank 的分片")
    log_all(show("after", out))

    # all_to_all：rank i 的第 j 片发给 rank j（MoE 专家并行分发 token）
    x = torch.tensor([10.0 * rank + j for j in range(world)], device=dev)
    out = torch.empty(world, device=dev)
    log0("all_to_all：rank i 的第 j 片 -> rank j 的第 i 个位置（相当于分布式转置）")
    log_all(show("before", x))
    dist.all_to_all_single(out, x)
    log_all(show("after", out))

    # send/recv：点对点（PP 传激活和梯度；Ring Attention 传 KV）
    nxt, prv = (rank + 1) % world, (rank - 1) % world
    x = torch.tensor([float(rank)], device=dev)
    buf = torch.empty(1, device=dev)
    ops = [dist.P2POp(dist.isend, x, nxt), dist.P2POp(dist.irecv, buf, prv)]
    for req in dist.batch_isend_irecv(ops):
        req.wait()
    log0("send/recv 组成一个环：每个 rank 发给下一个、从上一个收")
    log_all(f"  rank{rank} 收到来自 rank{prv} 的 {buf.item():.0f}")


def identity(rank, world, dev):
    header("2. 验证 all_reduce == reduce_scatter + all_gather")
    torch.manual_seed(rank)
    n = world * 1000
    x = torch.randn(n, device=dev)

    ref = x.clone()
    dist.all_reduce(ref)

    shard = torch.empty(n // world, device=dev)
    dist.reduce_scatter_tensor(shard, x)          # 第一步：每人得到 1/N 的"和"
    full = torch.empty(n, device=dev)
    dist.all_gather_into_tensor(full, shard)      # 第二步：把 1/N 拼回完整
    check("all_reduce vs RS+AG", full, ref)
    log0("  => 这就是 ZeRO 能'免费'切分优化器状态的原因：DP 本来就要做 RS+AG，"
         "\n     只是 ZeRO 在两步之间只更新自己那 1/N 的参数")


def bandwidth(rank, world, dev, max_mb):
    header("3. all_reduce 带宽测试（busbw 才能跨卡数比较）")
    log0(f"  {'size':>10s} {'time/iter':>11s} {'algbw':>11s} {'busbw':>11s}")
    size = 1 << 20  # 1 MB
    while size <= max_mb << 20:
        x = torch.ones(size // 4, device=dev)
        for _ in range(3):
            dist.all_reduce(x)
        iters = 10
        if is_cuda(dev):
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(iters):
                dist.all_reduce(x)
            e.record()
            torch.cuda.synchronize()
            t = s.elapsed_time(e) / 1000 / iters
        else:
            import time
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(iters):
                dist.all_reduce(x)
            t = (time.perf_counter() - t0) / iters
        algbw = size / t
        busbw = algbw * 2 * (world - 1) / world  # nccl-tests 的定义
        log0(f"  {fmt_bytes(size):>10s} {t * 1e3:9.3f}ms {fmt_bytes(algbw) + '/s':>11s} "
             f"{fmt_bytes(busbw) + '/s':>11s}")
        size *= 4
    log0("  小消息受延迟主导，大消息才接近链路带宽 => 这就是 DP 要做 gradient bucketing 的原因")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_mb", type=int, default=None, help="带宽测试最大消息 (MB)")
    args = ap.parse_args()
    rank, world, dev = setup()
    semantics(rank, world, dev)
    identity(rank, world, dev)
    bandwidth(rank, world, dev, args.max_mb or (1024 if is_cuda(dev) else 16))
    cleanup()
