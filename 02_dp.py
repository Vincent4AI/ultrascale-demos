"""第2章 数据并行：手写三种 DP，再和 PyTorch DDP 对比。

运行：  torchrun --nproc_per_node 2 02_dp.py
        torchrun --nproc_per_node 2 02_dp.py --dim 1024 --layers 8 --bsz 64 --bench   # GPU 上看速度

三个版本（与 Playbook 的三步优化一一对应）：
  v1 naive    : backward 全部结束后，逐个参数 all_reduce            —— 计算和通信完全串行
  v2 overlap  : 每个参数梯度一算好（post-accumulate-grad hook）就异步 all_reduce
                —— 通信和剩余的 backward 重叠
  v3 bucket   : 把梯度攒成 ~bucket_mb 的桶再发 —— 减少小消息的启动延迟（DDP 默认 25MB）
另外演示：梯度累积时只在最后一个 micro-batch 同步（DDP 的 no_sync）。

正确性：每个 rank 拿全局 batch 的 1/N，同步后的梯度必须等于单卡用整个全局 batch 算出的梯度。
"""
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from common import setup, cleanup, log0, header, check, make_model, get_batch, timer, num_params


# ----------------------------------------------------------------------------
# v1：naive
# ----------------------------------------------------------------------------
class NaiveDP:
    def __init__(self, model):
        self.model = model
        self.world = dist.get_world_size()

    def sync_grads(self):
        for p in self.model.parameters():
            dist.all_reduce(p.grad)          # SUM
            p.grad.div_(self.world)           # -> 平均


# ----------------------------------------------------------------------------
# v2：每个参数的梯度一就绪就发异步 all_reduce
# ----------------------------------------------------------------------------
class OverlapDP:
    def __init__(self, model):
        self.model = model
        self.world = dist.get_world_size()
        self.handles = []
        self.require_sync = True  # 梯度累积的前几个 micro-batch 设为 False
        for p in model.parameters():
            p.register_post_accumulate_grad_hook(self._hook)

    def _hook(self, p):
        if self.require_sync:
            self.handles.append(dist.all_reduce(p.grad, async_op=True))

    def sync_grads(self):
        for h in self.handles:
            h.wait()
        self.handles.clear()
        for p in self.model.parameters():
            p.grad.div_(self.world)


# ----------------------------------------------------------------------------
# v3：bucket —— 参数按"梯度就绪顺序"（大致是逆序）分桶，桶满了才发
# ----------------------------------------------------------------------------
class BucketDP:
    def __init__(self, model, bucket_mb: float = 1.0):
        self.model = model
        self.world = dist.get_world_size()
        self.require_sync = True
        params = [p for p in model.parameters() if p.requires_grad]
        cap = int(bucket_mb * 2**20 / params[0].element_size())
        self.buckets, cur, size = [], [], 0
        for p in reversed(params):  # backward 时靠后的层先出梯度
            cur.append(p)
            size += p.numel()
            if size >= cap:
                self.buckets.append(cur)
                cur, size = [], 0
        if cur:
            self.buckets.append(cur)
        self.p2b = {p: i for i, b in enumerate(self.buckets) for p in b}
        self.bufs = [torch.zeros(sum(p.numel() for p in b), device=b[0].device, dtype=b[0].dtype)
                     for b in self.buckets]
        self.ready = [0] * len(self.buckets)
        self.handles = {}
        for p in params:
            p.register_post_accumulate_grad_hook(self._hook)

    def _hook(self, p):
        if not self.require_sync:
            return
        i = self.p2b[p]
        self.ready[i] += 1
        if self.ready[i] == len(self.buckets[i]):  # 桶里所有梯度都到齐
            buf, off = self.bufs[i], 0
            for q in self.buckets[i]:              # 拷贝进连续 buffer（真实 DDP 让 grad 直接是 buffer 的 view）
                buf[off:off + q.numel()].copy_(q.grad.view(-1))
                off += q.numel()
            self.handles[i] = dist.all_reduce(buf, async_op=True)

    def sync_grads(self):
        for i, h in self.handles.items():
            h.wait()
            buf, off = self.bufs[i], 0
            for q in self.buckets[i]:
                q.grad.copy_(buf[off:off + q.numel()].view_as(q.grad)).div_(self.world)
                off += q.numel()
        self.handles.clear()
        self.ready = [0] * len(self.buckets)


# ----------------------------------------------------------------------------
def local_shard(x, rank, world):
    return x.chunk(world)[rank]


def reference_grads(args, dev):
    """单卡 baseline：同一模型，用完整的全局 batch。"""
    model = make_model(args.dim, args.layers, dev)
    x, y = get_batch(0, args.bsz, args.dim, dev)
    F.mse_loss(model(x), y).backward()
    return [p.grad.clone() for p in model.parameters()]


def run_variant(name, wrapper_cls, args, rank, world, dev, ref, **kw):
    model = make_model(args.dim, args.layers, dev)
    dp = wrapper_cls(model, **kw)
    x, y = get_batch(0, args.bsz, args.dim, dev)
    xl, yl = local_shard(x, rank, world), local_shard(y, rank, world)
    F.mse_loss(model(xl), yl).backward()
    dp.sync_grads()
    check(f"{name}: grads == single-GPU big batch", [p.grad for p in model.parameters()], ref)


def run_ddp(args, rank, world, dev, ref):
    model = make_model(args.dim, args.layers, dev)
    ddp = DDP(model, device_ids=[dev.index] if dev.type == "cuda" else None,
              bucket_cap_mb=args.bucket_mb)
    x, y = get_batch(0, args.bsz, args.dim, dev)
    F.mse_loss(ddp(local_shard(x, rank, world)), local_shard(y, rank, world)).backward()
    check("torch DDP: grads == single-GPU big batch", [p.grad for p in model.parameters()], ref)


def grad_acc_demo(args, rank, world, dev):
    header("梯度累积 + DP：只在最后一个 micro-batch 通信")
    n_acc = 4
    # baseline：单卡、全局 batch
    ref_model = make_model(args.dim, args.layers, dev)
    x, y = get_batch(0, args.bsz, args.dim, dev)
    F.mse_loss(ref_model(x), y).backward()
    ref = [p.grad.clone() for p in ref_model.parameters()]

    model = make_model(args.dim, args.layers, dev)
    dp = BucketDP(model, args.bucket_mb)
    xl, yl = local_shard(x, rank, world), local_shard(y, rank, world)
    n_comm = 0
    for i, (xm, ym) in enumerate(zip(xl.chunk(n_acc), yl.chunk(n_acc))):
        dp.require_sync = (i == n_acc - 1)       # 等价于 DDP 的 `with model.no_sync():`
        (F.mse_loss(model(xm), ym) / n_acc).backward()
        n_comm += len(dp.handles)
    dp.sync_grads()
    check(f"grad_acc={n_acc} x dp={world}: grads == big batch", [p.grad for p in model.parameters()], ref)
    log0(f"  {n_acc} 个 micro-batch 一共只发了 {n_comm} 次 all_reduce（= 桶数 {len(dp.buckets)}）")


def bench(args, rank, world, dev):
    header(f"速度对比（{args.iters} 次迭代，Ψ = {num_params(make_model(args.dim, args.layers, 'cpu')):,}）")
    x, y = get_batch(0, args.bsz, args.dim, dev)
    xl, yl = local_shard(x, rank, world), local_shard(y, rank, world)
    results = {}
    for name, cls, kw in [("v1 naive", NaiveDP, {}), ("v2 overlap", OverlapDP, {}),
                          ("v3 bucket", BucketDP, {"bucket_mb": args.bucket_mb})]:
        model = make_model(args.dim, args.layers, dev)
        dp = cls(model, **kw)
        for it in range(args.iters + 3):
            model.zero_grad(set_to_none=False)
            key = name if it >= 3 else "warmup"  # 前 3 次热身不计时
            with timer(dev, results, key):
                F.mse_loss(model(xl), yl).backward()
                dp.sync_grads()
    model = DDP(make_model(args.dim, args.layers, dev),
                device_ids=[dev.index] if dev.type == "cuda" else None, bucket_cap_mb=args.bucket_mb)
    for it in range(args.iters + 3):
        with timer(dev, results, "torch DDP" if it >= 3 else "warmup"):
            F.mse_loss(model(xl), yl).backward()
    for k in ["v1 naive", "v2 overlap", "v3 bucket", "torch DDP"]:
        log0(f"  {k:<12s} {results[k] / args.iters * 1e3:8.2f} ms/iter")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--bsz", type=int, default=32, help="全局 batch，需能被 world_size*4 整除")
    ap.add_argument("--bucket_mb", type=float, default=0.25)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    rank, world, dev = setup()

    header(f"正确性：{world} 个 rank 各算 {args.bsz // world} 个样本 vs 单卡算 {args.bsz} 个")
    ref = reference_grads(args, dev)
    run_variant("v1 naive", NaiveDP, args, rank, world, dev, ref)
    run_variant("v2 overlap", OverlapDP, args, rank, world, dev, ref)
    run_variant("v3 bucket", BucketDP, args, rank, world, dev, ref, bucket_mb=args.bucket_mb)
    run_ddp(args, rank, world, dev, ref)
    grad_acc_demo(args, rank, world, dev)
    if args.bench:
        bench(args, rank, world, dev)
    cleanup()
