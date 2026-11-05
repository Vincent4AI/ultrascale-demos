"""第7章 专家并行 (EP)：MoE 层的 token 用 all_to_all 分发到专家所在的卡。

运行：  torchrun --nproc_per_node 2 07_ep.py
        torchrun --nproc_per_node 4 07_ep.py --experts 8 --topk 2

流程（每卡有自己的一批 token，E 个专家平均放在 N 张卡上）：
  1. router 打分，每个 token 选 top-k 个专家
  2. 按目标专家排序，统计发往每张卡的 token 数，先 all_to_all 交换"数量"
  3. all_to_all 交换 token 本身（dispatch）
  4. 本卡专家计算
  5. all_to_all 把结果送回原卡（combine），按 gate 权重加权求和
正确性：输出和梯度都和"每卡都有全部专家"的稠密实现对比。
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from common import setup, cleanup, log0, log_all, header, check


class AllToAll(torch.autograd.Function):
    """带 autograd 的 all_to_all：backward 就是把 split 反过来再做一次 all_to_all。"""

    @staticmethod
    def forward(ctx, x, in_splits, out_splits):
        ctx.in_splits, ctx.out_splits = in_splits, out_splits
        out = x.new_empty(sum(out_splits), x.shape[1])
        dist.all_to_all_single(out, x.contiguous(), out_splits, in_splits)
        return out

    @staticmethod
    def backward(ctx, g):
        gx = g.new_empty(sum(ctx.in_splits), g.shape[1])
        dist.all_to_all_single(gx, g.contiguous(), ctx.in_splits, ctx.out_splits)
        return gx, None, None


class Expert(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MoE(nn.Module):
    """ep_size=1 时就是普通单卡 MoE（作为参考实现）。"""

    def __init__(self, dim, hidden, n_experts, topk, ep_rank=0, ep_size=1, seed=0):
        super().__init__()
        torch.manual_seed(seed)                       # 所有卡上 router 和全部专家初值一致
        self.router = nn.Linear(dim, n_experts, bias=False)
        all_experts = [Expert(dim, hidden) for _ in range(n_experts)]
        self.per_rank = n_experts // ep_size
        lo = ep_rank * self.per_rank
        self.experts = nn.ModuleList(all_experts[lo:lo + self.per_rank])  # 只保留本卡的专家
        self.E, self.k, self.ep_size = n_experts, topk, ep_size
        self.last_load = None

    def forward(self, x):                             # x: [N, dim]
        N, dim = x.shape
        probs = F.softmax(self.router(x), dim=-1)
        gate, eid = probs.topk(self.k, dim=-1)        # [N, k]
        gate = gate / gate.sum(-1, keepdim=True)
        flat_e = eid.reshape(-1)                      # 每个 (token, slot) 一行
        order = flat_e.argsort(stable=True)           # 按专家编号排序 => 发往同一张卡的 token 连续
        tokens = x.repeat_interleave(self.k, dim=0)[order]
        cnt = torch.bincount(flat_e, minlength=self.E)               # 本卡发往每个专家的 token 数

        if self.ep_size > 1:
            recv_cnt = torch.empty_like(cnt)
            dist.all_to_all_single(recv_cnt, cnt)     # 第一步：先告诉对方要收多少
            in_splits = cnt.view(self.ep_size, -1).sum(1).tolist()
            out_splits = recv_cnt.view(self.ep_size, -1).sum(1).tolist()
            recv = AllToAll.apply(tokens, in_splits, out_splits)       # dispatch
            # recv 的布局：[来自 rank0 的(专家0..专家e)...][来自 rank1 的...]，按本地专家重新分组
            per_src = recv_cnt.view(self.ep_size, self.per_rank)       # [src, local_expert]
            local_cnt = per_src.sum(0)
            self.last_load = local_cnt.tolist()
            src_id = torch.arange(self.ep_size * self.per_rank, device=x.device) % self.per_rank
            expert_of_row = torch.repeat_interleave(src_id, per_src.reshape(-1))
            regroup = expert_of_row.argsort(stable=True)
            h = recv[regroup]
            outs = [e(c) for e, c in zip(self.experts, h.split(local_cnt.tolist()))]
            y = torch.empty_like(recv)
            y[regroup] = torch.cat(outs)
            y = AllToAll.apply(y, out_splits, in_splits)               # combine：原路送回
        else:
            self.last_load = cnt.tolist()
            y = torch.cat([e(c) for e, c in zip(self.experts, tokens.split(cnt.tolist()))])

        out = torch.empty_like(y)
        out[order] = y                                 # 还原成 (token, slot) 顺序
        out = (out.view(N, self.k, dim) * gate[..., None]).sum(1)
        return out


def main(args):
    rank, world, dev = setup()
    assert args.experts % world == 0
    # 每卡一批不同的 token（EP 通常和 DP 共用这一维）
    gen = lambda r: torch.Generator().manual_seed(100 + r)
    xs = [torch.randn(args.tokens, args.dim, generator=gen(r)).to(dev) for r in range(world)]
    ws = [torch.randn(args.tokens, args.dim, generator=gen(1000 + r)).to(dev) for r in range(world)]

    header(f"EP={world}：{args.experts} 个专家，每卡 {args.experts // world} 个，top-{args.topk}")
    # 参考：单卡拥有全部专家，处理所有卡的 token
    ref = MoE(args.dim, args.hidden, args.experts, args.topk).to(dev)
    ref_out = [ref(x) for x in xs]
    sum((o * w).sum() for o, w in zip(ref_out, ws)).backward()

    moe = MoE(args.dim, args.hidden, args.experts, args.topk, rank, world).to(dev)
    x = xs[rank].clone().requires_grad_()
    out = moe(x)
    (out * ws[rank]).sum().backward()
    log_all(f"  rank{rank}: 本卡专家 {list(range(rank * moe.per_rank, (rank + 1) * moe.per_rank))}"
            f" 收到的 token 数 {moe.last_load}")

    check("EP forward output", out, ref_out[rank])
    # 专家参数只在拥有它的卡上，梯度已经通过 all_to_all 的 backward 汇总了所有卡的 token
    lo = rank * moe.per_rank
    ref_expert_grads = [p.grad for e in ref.experts[lo:lo + moe.per_rank] for p in e.parameters()]
    check("EP expert grads (本卡专家)", [p.grad for e in moe.experts for p in e.parameters()],
          ref_expert_grads)
    # router 是复制的（和 DP 一样），梯度要 all_reduce
    dist.all_reduce(moe.router.weight.grad)
    check("EP router grad (all_reduce 后)", moe.router.weight.grad, ref.router.weight.grad)
    log0("\n  负载不均衡是 MoE 的头号问题：某个专家收到的 token 太多就会成为瓶颈。"
         "\n  实际系统用 auxiliary loss / 无辅助损失的 bias 调整（DeepSeek-V3）来平衡，"
         "\n  并设置 capacity factor 丢弃溢出的 token。")
    cleanup()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=16, help="每卡 token 数")
    main(ap.parse_args())
