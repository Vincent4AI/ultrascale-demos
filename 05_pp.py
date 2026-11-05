"""第6章 流水线并行：手写 AFAB 和 1F1B 两种调度，打印每个 rank 的时间线。

运行：  torchrun --nproc_per_node 2 05_pp.py
        torchrun --nproc_per_node 4 05_pp.py --micro 8
        torchrun --nproc_per_node 2 05_pp.py --sim_ms 0 --dim 2048   # GPU 上用真实计算看时间线

模型的 L 层平均分给 p 个 stage（p = 进程数）。每个 micro-batch：
  forward : stage i 从 i-1 recv 激活 → 计算 → send 给 i+1
  backward: stage i 从 i+1 recv 激活的梯度 → 反传 → send 输入的梯度给 i-1
两种调度：
  AFAB（all-forward-all-backward，GPipe）：先跑完所有 m 个 forward，再跑所有 backward
          => 每个 stage 要同时存 m 个 micro-batch 的激活
  1F1B  ：先做 p-rank-1 个 warmup forward，然后一个 forward 一个 backward 交替
          => 同时在飞的 micro-batch 最多 p 个，bubble 和 AFAB 一样：(p-1)/m（相对理想时间）

--sim_ms：用 sleep 模拟每个 forward 的耗时（backward = 2 倍），让 CPU 上也能画出干净的时间线。
"""
import argparse
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from common import setup, cleanup, log0, header, check, make_model, get_batch, is_cuda


# ----------------------------------------------------------------------------
# 点对点通信：一次 batch_isend_irecv 可以同时"发给 A + 从 B 收"，避免死锁
# ----------------------------------------------------------------------------
def p2p(send=None, dst=None, recv_shape=None, src=None, dev=None):
    ops, buf = [], None
    if send is not None:
        ops.append(dist.P2POp(dist.isend, send.contiguous(), dst))
    if src is not None:
        buf = torch.empty(recv_shape, device=dev)
        ops.append(dist.P2POp(dist.irecv, buf, src))
    if ops:
        for req in dist.batch_isend_irecv(ops):
            req.wait()
        # NCCL 的 wait() 只让 GPU 流等待、CPU 不阻塞；这里同步一下，
        # 让"等数据"的时间算作空闲（和 CPU/gloo 上的行为一致），否则时间线上看不到 bubble
        if dev is not None and is_cuda(dev):
            torch.cuda.synchronize()
    return buf


class PipelineStage:
    def __init__(self, ref_model, rank, world, mb_shape, dev, sim_ms):
        L = len(ref_model.blocks)
        assert L % world == 0, "层数需能被 stage 数整除"
        per = L // world
        self.module = nn.Sequential(*copy.deepcopy(ref_model.blocks[rank * per:(rank + 1) * per]))
        self.rank, self.world = rank, world
        self.first, self.last = rank == 0, rank == world - 1
        self.prev = None if self.first else rank - 1
        self.next = None if self.last else rank + 1
        self.shape, self.dev, self.sim = mb_shape, dev, sim_ms / 1000
        self.events, self.t0 = [], 0.0
        self.max_inflight = 0

    # ---- 通信的四种组合（first/last stage 上自动变成 no-op）----
    def recv_forward(self):
        return p2p(recv_shape=self.shape, src=self.prev, dev=self.dev)

    def send_forward(self, out):
        if not self.last:
            p2p(send=out.detach(), dst=self.next)

    def recv_backward(self):
        return p2p(recv_shape=self.shape, src=self.next, dev=self.dev)

    def send_backward(self, g):
        if not self.first:
            p2p(send=g, dst=self.prev)

    def send_forward_recv_backward(self, out):
        if self.last:
            return None
        return p2p(send=out.detach(), dst=self.next, recv_shape=self.shape, src=self.next, dev=self.dev)

    def send_backward_recv_forward(self, g):
        if self.first:
            return None
        return p2p(send=g, dst=self.prev, recv_shape=self.shape, src=self.prev, dev=self.dev)

    # ---- 计算 ----
    def _mark(self, kind, i, t_start):
        if is_cuda(self.dev):
            torch.cuda.synchronize()
        self.events.append((kind, i, t_start - self.t0, time.perf_counter() - self.t0))

    def forward(self, i, inp, xs, ys, m):
        t = time.perf_counter()
        if self.first:
            inp = xs[i]
        else:
            inp.requires_grad_()
        out = self.module(inp)
        if self.last:
            out = F.mse_loss(out, ys[i]) / m          # 梯度累积：loss 除以 micro-batch 数
        if self.sim:
            time.sleep(self.sim)
        self._mark("F", i, t)
        return inp, out

    def backward(self, i, inp, out, grad_out):
        t = time.perf_counter()
        if self.last:
            out.backward()
        else:
            torch.autograd.backward(out, grad_out)
        if self.sim:
            time.sleep(2 * self.sim)
        self._mark("B", i, t)
        return None if self.first else inp.grad

    # ---- 调度 1：AFAB ----
    def run_afab(self, xs, ys, m):
        store = []
        for i in range(m):
            inp, out = self.forward(i, self.recv_forward(), xs, ys, m)
            self.send_forward(out)
            store.append((i, inp, out))
            self.max_inflight = max(self.max_inflight, len(store))
        while store:
            i, inp, out = store.pop(0)
            self.send_backward(self.backward(i, inp, out, self.recv_backward()))

    # ---- 调度 2：1F1B（与 Megatron / picotron 相同的结构）----
    def run_1f1b(self, xs, ys, m):
        warmup = min(self.world - self.rank - 1, m)
        remaining = m - warmup
        store = []
        for i in range(warmup):                       # 热身：只做 forward
            inp, out = self.forward(i, self.recv_forward(), xs, ys, m)
            self.send_forward(out)
            store.append((i, inp, out))
            self.max_inflight = max(self.max_inflight, len(store))
        inp = self.recv_forward() if remaining > 0 else None
        for j in range(remaining):                    # 稳态：1 个 forward + 1 个 backward
            i = warmup + j
            inp, out = self.forward(i, inp, xs, ys, m)
            store.append((i, inp, out))
            self.max_inflight = max(self.max_inflight, len(store))
            grad_out = self.send_forward_recv_backward(out)
            bi, b_inp, b_out = store.pop(0)
            g = self.backward(bi, b_inp, b_out, grad_out)
            if j == remaining - 1:
                self.send_backward(g)
            else:
                inp = self.send_backward_recv_forward(g)
        while store:                                  # 收尾：只做 backward
            bi, b_inp, b_out = store.pop(0)
            self.send_backward(self.backward(bi, b_inp, b_out, self.recv_backward()))


def draw_timeline(all_events, quantum):
    """把每个 rank 的 (F/B, mb, start, end) 画成 ASCII 甘特图。'.' = 空闲（bubble）。"""
    end = max(e[3] for evs in all_events for e in evs)
    ncol = int(end / quantum + 0.999)
    lines = []
    for r, evs in enumerate(all_events):
        row = [". "] * ncol
        for kind, i, s, e in evs:
            a, b = int(s / quantum + 0.5), max(int(e / quantum + 0.5), int(s / quantum + 0.5) + 1)
            for c in range(a, min(b, ncol)):
                row[c] = f"{kind}{i % 10}"
        busy = sum(e - s for _, _, s, e in evs)
        lines.append(f"  stage{r}: {''.join(row)}  空闲 {1 - busy / end:5.1%}")
    return "\n".join(lines)


def run(schedule, args, rank, world, dev):
    ref = make_model(args.dim, args.layers, dev)
    m = args.micro
    x, y = get_batch(0, m * args.mbs, args.dim, dev)
    xs, ys = list(x.chunk(m)), list(y.chunk(m))

    # baseline：单卡完整模型，m 个 micro-batch 梯度累积
    ref_loss = sum(F.mse_loss(ref(xi), yi) / m for xi, yi in zip(xs, ys))
    ref_loss.backward()
    per = args.layers // world
    ref_grads = [p.grad for blk in ref.blocks[rank * per:(rank + 1) * per] for p in blk.parameters()]

    st = PipelineStage(ref, rank, world, (args.mbs, args.dim), dev, args.sim_ms)
    for it in range(2):            # 第 1 遍热身（首次 backward/通信有一次性开销），第 2 遍计时
        st.module.zero_grad(set_to_none=True)
        st.events, st.max_inflight = [], 0
        dist.barrier()
        st.t0 = time.perf_counter()
        getattr(st, f"run_{schedule}")(xs, ys, m)
        dist.barrier()

    check(f"{schedule.upper()}: stage grads == single-GPU", [p.grad for p in st.module.parameters()], ref_grads)
    gathered = [None] * world
    dist.all_gather_object(gathered, (st.events, st.max_inflight))
    if rank == 0:
        log0(draw_timeline([g[0] for g in gathered], args.sim_ms / 1000 if args.sim_ms else
                           max(e[3] for g in gathered for e in g[0]) / 60))
        log0("  执行顺序 stage0: " + " ".join(f"{k}{i}" for k, i, *_ in gathered[0][0]))
        log0(f"  stage{world - 1}: " + " ".join(f"{k}{i}" for k, i, *_ in gathered[-1][0]))
        log0("  每个 stage 同时存着的 micro-batch 激活数（峰值）: "
             + ", ".join(f"stage{r}={g[1]}" for r, g in enumerate(gathered)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--micro", type=int, default=6, help="micro-batch 数 m")
    ap.add_argument("--mbs", type=int, default=4, help="每个 micro-batch 的样本数")
    ap.add_argument("--sim_ms", type=float, default=30.0)
    args = ap.parse_args()
    rank, world, dev = setup()
    p, m = world, args.micro
    log0(f"p={p} 个 stage，m={m} 个 micro-batch")
    log0(f"理论 bubble：(p-1)/m = {(p - 1) / m:.2f}（相对理想时间），"
         f"占总时间 (p-1)/(m+p-1) = {(p - 1) / (m + p - 1):.1%}")
    for sched in ["afab", "1f1b"]:
        header(f"调度：{sched.upper()}   (F3 = 第 3 个 micro-batch 的 forward，B = backward，. = 空闲)")
        run(sched, args, rank, world, dev)
    log0("\n结论：两种调度 bubble 一样大，但 1F1B 的激活峰值从 m 降到 p。"
         "\n想再减 bubble：增大 m（但 global batch 变大）、interleaved 1F1B（每卡多个虚拟 stage），"
         "\n或 zero-bubble / DualPipe（把 backward 拆成 B_input 和 B_weight 来填空）。")
    cleanup()
