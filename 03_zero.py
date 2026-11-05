"""第3章 ZeRO-1 / 2 / 3：手写实现，和"普通 DP + Adam"逐步对比参数。

运行：  torchrun --nproc_per_node 2 03_zero.py
        torchrun --nproc_per_node 4 03_zero.py --fsdp      # 额外跑 PyTorch FSDP2（ZeRO-3 的官方实现）

核心思想（N = DP 卡数，Ψ = 参数量）：
  ZeRO-1：优化器状态切 N 份。梯度照常 all_reduce，每卡只更新自己那 1/N 参数，再 all_gather 参数
  ZeRO-2：梯度也切。all_reduce 换成 reduce_scatter，每卡只留自己那 1/N 的梯度
  ZeRO-3：参数也切。forward/backward 用到哪层就 all_gather 哪层，用完就丢；梯度 reduce_scatter
通信量（以 Ψ 为单位，每卡发送量近似）：DP = ZeRO-1 = ZeRO-2 = 2Ψ，ZeRO-3 = 3Ψ
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from common import setup, cleanup, log0, log_all, header, check, make_model, get_batch, fmt_bytes

COMM = {"elems": 0}  # 统计每个 rank "参与集合通信"的元素数（以完整 tensor 大小计）


def all_gather_flat(shard):
    out = torch.empty(shard.numel() * dist.get_world_size(), device=shard.device, dtype=shard.dtype)
    dist.all_gather_into_tensor(out, shard)
    COMM["elems"] += out.numel()
    return out


def reduce_scatter_flat(full):
    world = dist.get_world_size()
    out = torch.empty(full.numel() // world, device=full.device, dtype=full.dtype)
    dist.reduce_scatter_tensor(out, full)
    COMM["elems"] += full.numel()
    return out


def all_reduce_flat(full):
    dist.all_reduce(full)
    COMM["elems"] += 2 * full.numel()  # all_reduce = RS + AG


def pad_to(n, world):
    return (n + world - 1) // world * world


# ----------------------------------------------------------------------------
# Baseline：普通 DP 的数学等价物 = 单卡、全局 batch、torch.optim.Adam
# ----------------------------------------------------------------------------
def baseline(args, dev):
    model = make_model(args.dim, args.layers, dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for step in range(args.steps):
        x, y = get_batch(step, args.bsz, args.dim, dev)
        opt.zero_grad()
        F.mse_loss(model(x), y).backward()
        opt.step()
    return [p.detach().clone() for p in model.parameters()]


# ----------------------------------------------------------------------------
# ZeRO-1 / ZeRO-2：模型完整复制；把所有参数拍平成一个 flat 向量，每个 rank 负责其中 1/N
# ----------------------------------------------------------------------------
class Zero12:
    def __init__(self, model, stage, lr):
        self.model, self.stage = model, stage
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.params = list(model.parameters())
        self.numel = sum(p.numel() for p in self.params)
        self.padded = pad_to(self.numel, self.world)
        self.shard_size = self.padded // self.world
        lo = self.rank * self.shard_size
        flat = self.flatten([p.data for p in self.params])
        # 每个 rank 只持有自己分片的"主权重"和 Adam 状态 —— 这就是 ZeRO-1 省下的显存
        self.master = nn.Parameter(flat[lo:lo + self.shard_size].clone())
        self.opt = torch.optim.Adam([self.master], lr=lr)

    def flatten(self, tensors):
        flat = torch.cat([t.reshape(-1) for t in tensors])
        return F.pad(flat, (0, self.padded - self.numel))

    def step(self):
        g = self.flatten([p.grad for p in self.params])
        if self.stage == 1:
            all_reduce_flat(g)                      # 完整梯度每卡一份
            g_shard = g.view(self.world, -1)[self.rank]
        else:
            g_shard = reduce_scatter_flat(g)        # 每卡只拿到自己分片的梯度和
            # 真实 ZeRO-2 会在 backward 过程中按 bucket 做 RS 并立刻释放非本分片的梯度
        self.master.grad = g_shard / self.world
        self.opt.step()                            # 只更新 1/N 的参数，Adam 状态也只有 1/N
        full = all_gather_flat(self.master.data)    # 把更新后的参数拼回来
        off = 0
        for p in self.params:
            p.data.copy_(full[off:off + p.numel()].view_as(p))
            off += p.numel()
        for p in self.params:
            p.grad = None


# ----------------------------------------------------------------------------
# ZeRO-3：每个 Linear 只存 1/N 的参数分片
# 自定义 autograd.Function：forward 时 all_gather 取回完整权重，算完立刻丢弃；
# backward 时再 all_gather 一次（第二次通信！），算完梯度 reduce_scatter 回各自分片。
# ----------------------------------------------------------------------------
class Zero3LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shard, layer):
        w, b = layer.gather()                       # 通信 1：all_gather（forward）
        ctx.save_for_backward(x, shard)             # 注意：不保存完整 w，只存分片
        ctx.layer = layer
        return F.linear(x, w, b)                    # 函数返回后 w, b 被释放

    @staticmethod
    def backward(ctx, gy):
        x, shard = ctx.saved_tensors
        layer = ctx.layer
        w, _ = layer.gather()                       # 通信 2：all_gather（backward 再取一次）
        gx = gy @ w
        gw = gy.reshape(-1, gy.shape[-1]).t() @ x.reshape(-1, x.shape[-1])
        gb = gy.reshape(-1, gy.shape[-1]).sum(0)
        flat_g = F.pad(torch.cat([gw.reshape(-1), gb]), (0, layer.padded - layer.numel))
        g_shard = reduce_scatter_flat(flat_g)       # 通信 3：reduce_scatter 梯度
        return gx, g_shard / dist.get_world_size(), None


class Zero3Linear(nn.Module):
    def __init__(self, ref: nn.Linear):
        super().__init__()
        rank, world = dist.get_rank(), dist.get_world_size()
        self.out_f, self.in_f = ref.weight.shape
        self.numel = ref.weight.numel() + ref.bias.numel()
        self.padded = pad_to(self.numel, world)
        flat = F.pad(torch.cat([ref.weight.data.reshape(-1), ref.bias.data]), (0, self.padded - self.numel))
        s = self.padded // world
        self.shard = nn.Parameter(flat[rank * s:(rank + 1) * s].clone())  # 只存 1/N

    def gather(self):
        full = all_gather_flat(self.shard.data)[:self.numel]
        nw = self.out_f * self.in_f
        return full[:nw].view(self.out_f, self.in_f), full[nw:]

    def forward(self, x):
        return Zero3LinearFn.apply(x, self.shard, self)


def to_zero3(model):
    """把模型里的 Linear 换成 Zero3Linear。LayerNorm 参数很小，这里保持复制 + all_reduce 梯度。"""
    for blk in model.blocks:
        blk.fc1, blk.fc2 = Zero3Linear(blk.fc1), Zero3Linear(blk.fc2)
    return model


def zero3_step(model, opt, world):
    for blk in model.blocks:                         # 复制的小参数（LN）走普通 DP
        for p in blk.ln.parameters():
            all_reduce_flat(p.grad)
            p.grad /= world
    opt.step()
    opt.zero_grad(set_to_none=True)


def zero3_full_params(model):
    """把分片参数拼回完整形状，按 baseline 的参数顺序返回，用于对比。"""
    out = []
    for blk in model.blocks:
        out += [blk.ln.weight.data, blk.ln.bias.data]
        for lin in (blk.fc1, blk.fc2):
            w, b = lin.gather()
            out += [w, b]
    return out


# ----------------------------------------------------------------------------
def run_stage(stage, args, rank, world, dev, ref_params):
    model = make_model(args.dim, args.layers, dev)
    COMM["elems"] = 0
    if stage in (1, 2):
        z = Zero12(model, stage, args.lr)
    else:
        model = to_zero3(model)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for step in range(args.steps):
        x, y = get_batch(step, args.bsz, args.dim, dev)
        xl, yl = x.chunk(world)[rank], y.chunk(world)[rank]
        F.mse_loss(model(xl), yl).backward()
        if stage in (1, 2):
            z.step()
        else:
            zero3_step(model, opt, world)
    comm_per_step = COMM["elems"] / args.steps
    got = zero3_full_params(model) if stage == 3 else [p.data for p in model.parameters()]
    check(f"ZeRO-{stage}: params after {args.steps} Adam steps == baseline", got, ref_params, atol=1e-5)
    return comm_per_step


def memory_table(args, world):
    psi = sum(p.numel() for p in make_model(args.dim, args.layers, "cpu").parameters())
    header(f"每卡显存（fp32 训练，Ψ = {psi:,}，N = {world}；括号内是 bf16 混合精度的系数）")
    k = 12  # 混合精度下 fp32 主权重 + m + v = 12 字节/参数
    rows = [("DP", 2 + 2 + k, 4 + 4 + 8),
            ("ZeRO-1", 2 + 2 + k / world, 4 + 4 + 8 / world),
            ("ZeRO-2", 2 + (2 + k) / world, 4 + (4 + 8) / world),
            ("ZeRO-3", (2 + 2 + k) / world, (4 + 4 + 8) / world)]
    for name, mixed, fp32 in rows:
        log0(f"  {name:<7s} {fmt_bytes(fp32 * psi):>11s}   ({mixed:5.2f} Ψ bytes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fsdp", action="store_true", help="额外演示 PyTorch FSDP2 (fully_shard)")
    args = ap.parse_args()
    rank, world, dev = setup()

    header(f"正确性：ZeRO-1/2/3 训练 {args.steps} 步 vs 单卡 Adam")
    ref = baseline(args, dev)
    comm = {s: run_stage(s, args, rank, world, dev, ref) for s in (1, 2, 3)}

    psi = sum(p.numel() for p in make_model(args.dim, args.layers, "cpu").parameters())
    header("每步通信量（以完整 tensor 大小计，单位 Ψ）")
    for s in (1, 2, 3):
        log0(f"  ZeRO-{s}: {comm[s] / psi:.2f} Ψ")
    log0("  ZeRO-1: all_reduce(2Ψ) + all_gather 参数(Ψ) —— 实际实现里 all_reduce 拆成 RS+AG，"
         "\n          参数 AG 正好顶替梯度 AG，所以真实是 2Ψ；这里为了好读多发了一次")
    log0("  ZeRO-2: reduce_scatter(Ψ) + all_gather(Ψ) = 2Ψ，和 DP 一样")
    log0("  ZeRO-3: forward AG(Ψ) + backward AG(Ψ) + RS(Ψ) = 3Ψ，多 50%，换来参数也切分")
    memory_table(args, world)

    if args.fsdp:
        header("PyTorch FSDP2 (fully_shard) —— 生产环境的 ZeRO-3")
        from torch.distributed.fsdp import fully_shard
        model = make_model(args.dim, args.layers, dev)
        for blk in model.blocks:
            fully_shard(blk)          # 每个 block 一个通信单元：用到时 all_gather，用完 reshard
        fully_shard(model)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for step in range(args.steps):
            x, y = get_batch(step, args.bsz, args.dim, dev)
            F.mse_loss(model(x.chunk(world)[rank]), y.chunk(world)[rank]).backward()
            opt.step()
            opt.zero_grad()
        local = sum(p.to_local().numel() for p in model.parameters())
        log_all(f"  rank{rank}: 本地只存 {local:,} / {psi:,} 个参数（DTensor 分片）")
        full = [p.full_tensor() for p in model.parameters()]
        check("FSDP2: params == baseline", full, ref, atol=1e-5)
    cleanup()
