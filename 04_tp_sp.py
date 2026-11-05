"""第4章 张量并行 (TP) + 序列并行 (SP)：手写 Megatron 风格的 Transformer block。

运行：  torchrun --nproc_per_node 2 04_tp_sp.py
        torchrun --nproc_per_node 4 04_tp_sp.py --heads 8

切法（tp = TP 组大小）：
  Attention：Wq/Wk/Wv 按"头"列切（每卡 heads/tp 个头），Wo 行切
  MLP      ：fc1 列切（4h/tp），fc2 行切
  每个子层只在开头和结尾各有一次通信：
      纯 TP：f = forward 恒等 / backward all_reduce；   g = forward all_reduce / backward 恒等
      TP+SP：f'= forward all_gather(seq) / backward reduce_scatter(seq)
              g'= forward reduce_scatter(seq) / backward all_gather(seq)
  SP 的好处：LayerNorm/残差/dropout 的激活也按序列切开了（纯 TP 里它们在每卡上是完整重复的）。
  SP 的代价：LayerNorm 和行切 Linear 的 bias 的梯度只来自本卡那段序列，要在 TP 组内 all_reduce。

正确性：前向输出、输入梯度、每个参数分片的梯度都和单卡完整 block 对比。
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from common import setup, cleanup, log0, log_all, header, check, SavedTensorCounter, fmt_bytes


# ----------------------------------------------------------------------------
# 通信原语（带 autograd），group 参数让 08_2d_dp_tp.py 可以复用
# ----------------------------------------------------------------------------
def _all_gather_dim(x, dim, group):
    world = dist.get_world_size(group)
    parts = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(parts, x.contiguous(), group=group)
    return torch.cat(parts, dim=dim)


def _reduce_scatter_dim(x, dim, group):
    world = dist.get_world_size(group)
    chunks = [c.contiguous() for c in x.chunk(world, dim=dim)]
    out = torch.empty(chunks[0].numel(), device=x.device, dtype=x.dtype)
    dist.reduce_scatter_tensor(out, torch.cat([c.view(-1) for c in chunks]), group=group)
    return out.view_as(chunks[0])


class CopyToTP(torch.autograd.Function):          # f
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        g = g.contiguous().clone()
        dist.all_reduce(g, group=ctx.group)
        return g, None


class ReduceFromTP(torch.autograd.Function):      # g
    @staticmethod
    def forward(ctx, x, group):
        x = x.contiguous().clone()
        dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, g):
        return g, None


class GatherSeq(torch.autograd.Function):         # f'（SP）
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _all_gather_dim(x, 1, group)

    @staticmethod
    def backward(ctx, g):
        return _reduce_scatter_dim(g, 1, ctx.group), None


class ScatterSeq(torch.autograd.Function):        # g'（SP）
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _reduce_scatter_dim(x, 1, group)

    @staticmethod
    def backward(ctx, g):
        return _all_gather_dim(g, 1, ctx.group), None


# ----------------------------------------------------------------------------
# 单卡参考实现
# ----------------------------------------------------------------------------
def attention(q, k, v, heads):
    B, T, _ = q.shape
    split = lambda t: t.view(B, T, heads, -1).transpose(1, 2)        # [B, heads, T, hd]
    o = F.scaled_dot_product_attention(split(q), split(k), split(v), is_causal=True)
    return o.transpose(1, 2).reshape(B, T, -1)


class RefBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(dim, dim) for _ in range(4))
        self.fc1, self.fc2 = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.wo(attention(self.wq(h), self.wk(h), self.wv(h), self.heads))
        h = self.ln2(x)
        return x + self.fc2(F.gelu(self.fc1(h)))


# ----------------------------------------------------------------------------
# TP / TP+SP 实现：权重从参考模型切出来，保证初值一致
# ----------------------------------------------------------------------------
def col_shard(lin: nn.Linear, r, tp):     # 按输出维切：weight [out/tp, in]，bias [out/tp]
    new = nn.Linear(lin.in_features, lin.out_features // tp)
    new.weight.data.copy_(lin.weight.data.chunk(tp, 0)[r])
    new.bias.data.copy_(lin.bias.data.chunk(tp, 0)[r])
    return new


def row_shard(lin: nn.Linear, r, tp):     # 按输入维切：weight [out, in/tp]，bias 完整（reduce 后再加）
    new = nn.Linear(lin.in_features // tp, lin.out_features, bias=False)
    new.weight.data.copy_(lin.weight.data.chunk(tp, 1)[r])
    return new


class TPBlock(nn.Module):
    def __init__(self, ref: RefBlock, group=None, sequence_parallel=False):
        super().__init__()
        self.group, self.sp = group, sequence_parallel
        tp, r = dist.get_world_size(group), dist.get_rank(group)
        assert ref.heads % tp == 0, "头数必须能被 tp 整除"
        self.local_heads = ref.heads // tp
        self.ln1, self.ln2 = nn.LayerNorm(ref.ln1.normalized_shape), nn.LayerNorm(ref.ln2.normalized_shape)
        self.ln1.load_state_dict(ref.ln1.state_dict())
        self.ln2.load_state_dict(ref.ln2.state_dict())
        self.wq, self.wk, self.wv = (col_shard(l, r, tp) for l in (ref.wq, ref.wk, ref.wv))
        self.wo = row_shard(ref.wo, r, tp)
        self.fc1, self.fc2 = col_shard(ref.fc1, r, tp), row_shard(ref.fc2, r, tp)
        self.bo = nn.Parameter(ref.wo.bias.data.clone())    # 行切 Linear 的 bias 在 reduce 之后加
        self.b2 = nn.Parameter(ref.fc2.bias.data.clone())

    def enter(self, h):   # 进入列切 Linear 之前
        return GatherSeq.apply(h, self.group) if self.sp else CopyToTP.apply(h, self.group)

    def exit(self, y):    # 离开行切 Linear 之后
        return ScatterSeq.apply(y, self.group) if self.sp else ReduceFromTP.apply(y, self.group)

    def forward(self, x):
        # x: 纯 TP 时是完整序列 [B, T, H]；SP 时是本卡那段序列 [B, T/tp, H]
        h = self.enter(self.ln1(x))
        a = attention(self.wq(h), self.wk(h), self.wv(h), self.local_heads)   # 本卡的头，完整序列
        x = x + self.exit(self.wo(a)) + self.bo
        h = self.enter(self.ln2(x))
        x = x + self.exit(self.fc2(F.gelu(self.fc1(h)))) + self.b2
        return x

    def sync_sp_grads(self):
        """SP 下这些"复制"参数只看到了本卡那段序列，梯度要在 TP 组内求和。"""
        if self.sp:
            for p in [*self.ln1.parameters(), *self.ln2.parameters(), self.bo, self.b2]:
                dist.all_reduce(p.grad, group=self.group)


def ref_grads_for(ref: RefBlock, r, tp):
    """把参考模型的梯度按 TPBlock 的参数顺序切好，用于对比。"""
    g = lambda l: l.weight.grad
    out = [ref.ln1.weight.grad, ref.ln1.bias.grad, ref.ln2.weight.grad, ref.ln2.bias.grad]
    for l in (ref.wq, ref.wk, ref.wv):
        out += [g(l).chunk(tp, 0)[r], l.bias.grad.chunk(tp, 0)[r]]
    out += [g(ref.wo).chunk(tp, 1)[r]]
    out += [g(ref.fc1).chunk(tp, 0)[r], ref.fc1.bias.grad.chunk(tp, 0)[r]]
    out += [g(ref.fc2).chunk(tp, 1)[r]]
    out += [ref.wo.bias.grad, ref.fc2.bias.grad]
    return out


def tp_grads(blk: TPBlock):
    out = [blk.ln1.weight.grad, blk.ln1.bias.grad, blk.ln2.weight.grad, blk.ln2.bias.grad]
    for l in (blk.wq, blk.wk, blk.wv):
        out += [l.weight.grad, l.bias.grad]
    out += [blk.wo.weight.grad, blk.fc1.weight.grad, blk.fc1.bias.grad, blk.fc2.weight.grad]
    out += [blk.bo.grad, blk.b2.grad]
    return out


def run(args, rank, tp, dev, sp):
    torch.manual_seed(42)                                  # 所有 rank 同一个参考模型和输入
    ref = RefBlock(args.dim, args.heads).to(dev)
    x = torch.randn(args.bsz, args.seq, args.dim, device=dev)
    w = torch.randn(args.bsz, args.seq, args.dim, device=dev)   # loss = sum(out * w)
    xr = x.clone().requires_grad_()
    with SavedTensorCounter(ref) as c_ref:
        out_ref = ref(xr)
    (out_ref * w).sum().backward()

    blk = TPBlock(ref, None, sp).to(dev)
    if sp:   # 输入、输出、loss 都只用本卡那段序列
        xl = x.chunk(tp, 1)[rank].clone().requires_grad_()
        wl = w.chunk(tp, 1)[rank]
    else:
        xl, wl = x.clone().requires_grad_(), w
    with SavedTensorCounter(blk) as c_tp:
        out = blk(xl)
    (out * wl).sum().backward()
    blk.sync_sp_grads()

    name = "TP+SP" if sp else "TP"
    if sp:
        check(f"{name}: forward output (本卡序列段)", out, out_ref.chunk(tp, 1)[rank])
        check(f"{name}: input grad", xl.grad, xr.grad.chunk(tp, 1)[rank])
    else:
        check(f"{name}: forward output", out, out_ref)
        check(f"{name}: input grad", xl.grad, xr.grad)
    check(f"{name}: all param-shard grads", tp_grads(blk), ref_grads_for(ref, rank, tp), atol=1e-4)
    return c_ref.bytes, c_tp.bytes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seq", type=int, default=16, help="需能被 tp 整除")
    ap.add_argument("--bsz", type=int, default=2)
    args = ap.parse_args()
    rank, tp, dev = setup()

    header(f"张量并行：tp={tp}，每卡 {args.heads // tp} 个头、fc1 输出 {4 * args.dim // tp} 维")
    a_ref, a_tp = run(args, rank, tp, dev, sp=False)
    header("张量并行 + 序列并行")
    _, a_sp = run(args, rank, tp, dev, sp=True)

    header("每卡为 backward 保存的激活（一个 block）")
    log0(f"  单卡完整 block : {fmt_bytes(a_ref)}")
    log0(f"  TP            : {fmt_bytes(a_tp)}   <- attention/MLP 内部切了，但 LN、残差、block 输入仍是完整的")
    log0(f"  TP + SP       : {fmt_bytes(a_sp)}   <- LN 等也按序列切了")
    log0("\n  通信：每个 block forward 2 次 all_reduce（TP）或 2×(all_gather + reduce_scatter)（SP），"
         "\n  backward 同样多。通信量相同，但都在关键路径上无法和计算重叠 => TP 一般只放在节点内（NVLink）")
    cleanup()
