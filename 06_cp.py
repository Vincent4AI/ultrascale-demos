"""第5章 上下文并行 (CP)：把超长序列切到多张卡上做 attention。

运行：  torchrun --nproc_per_node 2 06_cp.py
        torchrun --nproc_per_node 4 06_cp.py --seq 64

两种实现：
  A) Ring Attention（只演示 forward，讲清原理）：
     每卡持有自己那段 Q、K、V。K/V 沿环转 N-1 次，每收到一块就用 online softmax
     （和 FlashAttention 同一个技巧）把结果累加进来。通信可以和计算重叠。
  B) All-gather KV（Llama 3 的做法，可训练）：
     每卡 all_gather 完整的 K/V，只算本卡的 Q。backward 时 K/V 梯度 reduce_scatter 回去。
     GQA 下 K/V 很小，所以这种"笨"办法在实际中很好用。
两种切分：
  contiguous：rank i 拿第 i 段  -> causal mask 下 rank 0 几乎不用算，最后一个 rank 算最多（负载不均）
  zigzag    ：把序列切 2N 块，rank i 拿第 i 块和第 2N-1-i 块 -> 每卡计算量相同
"""
import argparse
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
from common import setup, cleanup, log0, log_all, header, check


def positions(T, rank, world, mode):
    """本 rank 负责的 token 位置下标。"""
    if mode == "contiguous":
        return torch.arange(T).chunk(world)[rank]
    blocks = torch.arange(T).chunk(2 * world)
    return torch.cat([blocks[rank], blocks[2 * world - 1 - rank]])


def full_attention(q, k, v):
    """参考实现：[B, H, T, D]，causal。"""
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


# ----------------------------------------------------------------------------
# A) Ring Attention forward
# ----------------------------------------------------------------------------
def ring_attention_forward(q, k, v, q_pos, k_pos):
    rank, world = dist.get_rank(), dist.get_world_size()
    scale = 1 / math.sqrt(q.shape[-1])
    NEG = -1e30                                     # 用很大的负数代替 -inf，避免 inf-inf=nan
    m = torch.full(q.shape[:-1], NEG, device=q.device)          # 每行当前的最大 logit
    l = torch.zeros(q.shape[:-1], device=q.device)              # 每行当前的 softmax 分母
    acc = torch.zeros_like(q)                                    # 未归一化的输出
    work = 0
    for step in range(world):
        # 先发起下一轮的 K/V 交换（真实实现里这一步和下面的计算重叠）
        if step < world - 1:
            nk, nv, npos = torch.empty_like(k), torch.empty_like(v), torch.empty_like(k_pos)
            ops = []
            for send, recv in [(k, nk), (v, nv), (k_pos, npos)]:
                ops += [dist.P2POp(dist.isend, send.contiguous(), (rank + 1) % world),
                        dist.P2POp(dist.irecv, recv, (rank - 1) % world)]
            reqs = dist.batch_isend_irecv(ops)
        mask = q_pos[:, None] >= k_pos[None, :]                   # causal：只能看见位置 <= 自己的 key
        if mask.any():                                           # 整块被 mask 掉就跳过（这就是负载不均的来源）
            work += int(mask.sum())
            s = (q @ k.transpose(-1, -2)) * scale
            s = s.masked_fill(~mask, NEG)
            m_new = torch.maximum(m, s.amax(-1))
            alpha = torch.exp(m - m_new)                         # 旧结果的缩放系数
            p = torch.exp(s - m_new[..., None])
            l = l * alpha + p.sum(-1)
            acc = acc * alpha[..., None] + p @ v
            m = m_new
        if step < world - 1:
            for r in reqs:
                r.wait()
            k, v, k_pos = nk, nv, npos
    return acc / l[..., None], work


# ----------------------------------------------------------------------------
# B) All-gather KV CP（带 autograd，可训练）
# ----------------------------------------------------------------------------
class AllGatherSeq(torch.autograd.Function):
    """forward: 沿序列维 all_gather；backward: reduce_scatter（每卡的 Q 都对全部 K/V 有梯度贡献）。"""

    @staticmethod
    def forward(ctx, x):
        world = dist.get_world_size()
        parts = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(parts, x.contiguous())
        return torch.cat(parts, dim=2)

    @staticmethod
    def backward(ctx, g):
        world = dist.get_world_size()
        chunks = [c.contiguous().view(-1) for c in g.chunk(world, dim=2)]
        out = torch.empty_like(chunks[0])
        dist.reduce_scatter_tensor(out, torch.cat(chunks))
        return out.view(g.chunk(world, dim=2)[0].shape)


def allgather_cp_attention(q, k, v, q_pos, k_pos_local):
    k_all, v_all = AllGatherSeq.apply(k), AllGatherSeq.apply(v)
    world = dist.get_world_size()
    pos = [torch.empty_like(k_pos_local) for _ in range(world)]
    dist.all_gather(pos, k_pos_local)
    k_pos = torch.cat(pos)
    mask = q_pos[:, None] >= k_pos[None, :]
    return F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask)


# ----------------------------------------------------------------------------
def main(args):
    rank, world, dev = setup()
    B, H, T, D = args.bsz, args.heads, args.seq, args.head_dim
    assert T % (2 * world) == 0
    torch.manual_seed(0)                                         # 所有 rank 同样的完整 Q/K/V
    q, k, v = (torch.randn(B, H, T, D, device=dev) for _ in range(3))
    gout = torch.randn(B, H, T, D, device=dev)                  # loss = sum(out * gout)
    qr, kr, vr = (t.clone().requires_grad_() for t in (q, k, v))
    ref = full_attention(qr, kr, vr)
    (ref * gout).sum().backward()

    for mode in ["contiguous", "zigzag"]:
        header(f"切分方式：{mode}")
        pos = positions(T, rank, world, mode).to(dev)
        pick = lambda t: t[:, :, pos]
        log_all(f"  rank{rank} 负责位置 {pos.tolist()}")

        out, work = ring_attention_forward(pick(q), pick(k), pick(v), pos, pos)
        check(f"Ring Attention forward ({mode})", out, pick(ref))
        works = [None] * world
        dist.all_gather_object(works, work)
        log0(f"  每卡实际计算的 (q,k) 对数: {works}  -> 最大/平均 = {max(works) / (sum(works) / world):.2f}")

        ql, kl, vl = (pick(t).clone().requires_grad_() for t in (q, k, v))
        out = allgather_cp_attention(ql, kl, vl, pos, pos)
        (out * pick(gout)).sum().backward()
        check(f"All-gather CP forward ({mode})", out, pick(ref))
        check(f"All-gather CP dQ/dK/dV ({mode})", [ql.grad, kl.grad, vl.grad],
              [pick(qr.grad), pick(kr.grad), pick(vr.grad)])

    log0("\n  注意：真实模型里 Q/K/V 来自 Linear，CP 组内每卡只看到部分 token，"
         "\n  所以权重梯度还要在 CP 组内 all_reduce（和 DP 一起做即可）。")
    cleanup()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--head_dim", type=int, default=16)
    main(ap.parse_args())
