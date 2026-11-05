"""第8章 组合并行：DP × TP 二维网格（4 卡起）。

运行：  torchrun --nproc_per_node 4 08_2d_dp_tp.py              # dp=2, tp=2
        torchrun --nproc_per_node 4 08_2d_dp_tp.py --sp         # TP 换成 TP+SP
        torchrun --nproc_per_node 4 08_2d_dp_tp.py --native     # 再用 PyTorch 原生 API（DTensor TP + FSDP2）做一遍
        两节点各 2 卡：见 slurm/run_2node.sbatch

要点：
  * DeviceMesh 把 world 排成 [dp, tp] 网格；同一行是一个 TP 组，同一列是一个 DP 组
  * 放置原则：通信最频繁、无法重叠的维度放在最内层（相邻 rank = 同一节点 = NVLink）。
    TP 每层都要通信 -> 最内层；DP 每步才通信一次且能和 backward 重叠 -> 外层（可跨节点）
  * 同一个 TP 组内的卡吃同一份数据；不同 DP 组吃不同数据
  * 梯度同步：先在 TP 内部完成（04 里的 f/g 算子已经做了），再在 DP 组里 all_reduce
"""
import argparse
import importlib
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from common import setup, cleanup, log0, log_all, header, check

tp_mod = importlib.import_module("04_tp_sp")   # 复用第 4 章的 RefBlock / TPBlock
RefBlock, TPBlock, ref_grads_for, tp_grads = tp_mod.RefBlock, tp_mod.TPBlock, tp_mod.ref_grads_for, tp_mod.tp_grads


def main(args):
    rank, world, dev = setup()
    tp = args.tp
    assert world % tp == 0, "world_size 必须能被 tp 整除"
    dp = world // tp
    mesh = init_device_mesh(dev.type, (dp, tp), mesh_dim_names=("dp", "tp"))
    tp_group, dp_group = mesh["tp"].get_group(), mesh["dp"].get_group()
    dp_rank, tp_rank = mesh.get_local_rank("dp"), mesh.get_local_rank("tp")

    header(f"2D 网格：dp={dp} × tp={tp}")
    log0(f"  mesh =\n{mesh.mesh.tolist()}  （行 = TP 组，列 = DP 组）")
    log_all(f"  global rank {rank}: dp_rank={dp_rank}, tp_rank={tp_rank}, "
            f"TP 组={dist.get_process_group_ranks(tp_group)}, DP 组={dist.get_process_group_ranks(dp_group)}")

    # 参考：单卡、完整模型、全局 batch
    torch.manual_seed(42)
    refs = nn.ModuleList([RefBlock(args.dim, args.heads) for _ in range(args.layers)]).to(dev)
    gb = args.bsz * dp
    x = torch.randn(gb, args.seq, args.dim, device=dev)
    w = torch.randn(gb, args.seq, args.dim, device=dev)
    h = x
    for b in refs:
        h = b(h)
    (h * w).sum().backward()

    # 2D 并行版本
    blocks = nn.ModuleList([TPBlock(r, tp_group, args.sp) for r in refs]).to(dev)
    xl, wl = x.chunk(dp)[dp_rank], w.chunk(dp)[dp_rank]          # DP：按 batch 切
    if args.sp:                                                    # SP：再按序列切
        xl, wl = xl.chunk(tp, 1)[tp_rank], wl.chunk(tp, 1)[tp_rank]
    h = xl
    for b in blocks:
        h = b(h)
    (h * wl).sum().backward()
    for b in blocks:
        b.sync_sp_grads()                                          # SP 的 LN/bias 梯度：TP 组内求和
    for p in blocks.parameters():                                  # DP：在 DP 组内求和
        dist.all_reduce(p.grad, group=dp_group)

    name = "DP×(TP+SP)" if args.sp else "DP×TP"
    got = [g for b in blocks for g in tp_grads(b)]
    ref = [g for r in refs for g in ref_grads_for(r, tp_rank, tp)]
    check(f"{name}: all grads == single-GPU global batch", got, ref, atol=1e-4)

    if args.native:
        native(args, mesh, dev, dp_rank)
    cleanup()


def native(args, mesh, dev, dp_rank):
    """PyTorch 原生写法：DTensor 张量并行 + FSDP2，生产里（torchtitan）就是这么组合的。"""
    header("PyTorch 原生 API：parallelize_module (TP) + fully_shard (FSDP2 over dp)")
    from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
    from torch.distributed.fsdp import fully_shard
    import warnings, logging
    warnings.filterwarnings("ignore")                       # 屏蔽 FSDP2/DTensor 的提示性警告
    logging.getLogger("torch.distributed.tensor").setLevel(logging.ERROR)

    def mlp():
        torch.manual_seed(7)
        return nn.Sequential(nn.Linear(args.dim, 4 * args.dim), nn.GELU(), nn.Linear(4 * args.dim, args.dim)).to(dev)

    ref = mlp()
    x = torch.randn(args.bsz * mesh["dp"].size(), args.dim, generator=torch.Generator().manual_seed(3)).to(dev)
    ref(x).pow(2).sum().backward()

    model = mlp()
    parallelize_module(model, mesh["tp"], {"0": ColwiseParallel(), "2": RowwiseParallel()})
    fully_shard(model, mesh=mesh["dp"])
    model(x.chunk(mesh["dp"].size())[dp_rank]).pow(2).sum().backward()
    # FSDP2 会对梯度做平均（除以 dp），所以和参考比较时乘回来
    got = [p.grad.full_tensor() * mesh["dp"].size() for p in model.parameters()]
    check("native TP+FSDP2: grads == single-GPU", got, [p.grad for p in ref.parameters()], atol=1e-4)
    log0(f"  每个参数现在是一个 DTensor，fc1.weight.placements = {model[0].weight.placements}")
    log0("  = 在 tp 维上按行切（ColwiseParallel），再在 dp 维上把本地分片继续切（FSDP2）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--bsz", type=int, default=2, help="每个 DP 组的 batch")
    ap.add_argument("--sp", action="store_true")
    ap.add_argument("--native", action="store_true")
    main(ap.parse_args())
