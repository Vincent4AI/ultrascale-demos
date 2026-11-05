"""公共工具：分布式初始化、打印、正确性检查、小模型、确定性数据。

所有 demo 都用 torchrun 启动：
    torchrun --nproc_per_node 2 02_dp.py
有 GPU 时自动用 NCCL，没有就用 gloo（Mac / CPU 节点也能跑通逻辑）。
设置环境变量 FORCE_CPU=1 可以在 GPU 机器上强制走 CPU + gloo。
"""
import os
import time
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ----------------------------------------------------------------------------
# 初始化
# ----------------------------------------------------------------------------
def setup(seed: int = 0):
    """初始化进程组，返回 (rank, world_size, device)。"""
    if "RANK" not in os.environ:  # 允许 `python xx.py` 单进程直接跑
        os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0",
                          MASTER_ADDR="127.0.0.1", MASTER_PORT="29511")
    use_cuda = torch.cuda.is_available() and os.environ.get("FORCE_CPU") != "1"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        # 关掉 TF32，否则 GPU 上和 baseline 对比会有 1e-3 级别误差
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        device = torch.device("cpu")
    dist.init_process_group("nccl" if use_cuda else "gloo")
    torch.manual_seed(seed)
    return dist.get_rank(), dist.get_world_size(), device


def cleanup():
    dist.barrier()
    dist.destroy_process_group()


def is_cuda(device) -> bool:
    return torch.device(device).type == "cuda"


# ----------------------------------------------------------------------------
# 打印
# ----------------------------------------------------------------------------
def log0(*args):
    """只在 rank0 打印。"""
    if dist.get_rank() == 0:
        print(*args, flush=True)


def log_all(msg: str, group=None):
    """按 rank 顺序依次打印（每个 rank 一行），便于看每张卡上发生了什么。"""
    world = dist.get_world_size(group)
    me = dist.get_rank(group)
    for r in range(world):
        if r == me:
            print(msg, flush=True)
        dist.barrier(group)
        time.sleep(0.01)  # 让 stdout 有机会刷出来，避免交错


def header(title: str):
    log0("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# ----------------------------------------------------------------------------
# 正确性检查：所有 rank 的最大误差取 max，rank0 打印 PASS/FAIL
# ----------------------------------------------------------------------------
def check(name: str, got, ref, atol: float = 1e-5, rtol: float = 1e-4) -> bool:
    if isinstance(got, (list, tuple)):
        pairs = list(zip(got, ref))
    else:
        pairs = [(got, ref)]
    err = 0.0
    ok = True
    for g, r in pairs:
        g = g.detach().float().cpu()
        r = r.detach().float().cpu()
        if g.shape != r.shape:
            raise ValueError(f"{name}: shape {tuple(g.shape)} vs {tuple(r.shape)}")
        err = max(err, (g - r).abs().max().item() if g.numel() else 0.0)
        ok = ok and torch.allclose(g, r, atol=atol, rtol=rtol)
    # NCCL 只能通信 GPU 张量，所以用 NCCL 时把这个小张量放到当前卡上
    dev = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else torch.device("cpu")
    t = torch.tensor([err, 0.0 if ok else 1.0], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)  # gloo/nccl 都支持 MAX
    passed = t[1].item() == 0
    log0(f"  [{'PASS' if passed else 'FAIL'}] {name:<48s} max|diff|={t[0].item():.2e}")
    return passed


# ----------------------------------------------------------------------------
# 计时：GPU 上要 synchronize，否则测到的只是 kernel launch 时间
# ----------------------------------------------------------------------------
@contextlib.contextmanager
def timer(device, out: dict, key: str):
    if is_cuda(device):
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    yield
    if is_cuda(device):
        torch.cuda.synchronize(device)
    out[key] = out.get(key, 0.0) + time.perf_counter() - t0


# ----------------------------------------------------------------------------
# 一个小模型：残差 MLP 块堆叠（足够展示 DP / ZeRO / PP，不依赖任何外部库）
# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, mult * dim)
        self.fc2 = nn.Linear(mult * dim, dim)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.ln(x))))


class MLPStack(nn.Module):
    def __init__(self, dim: int = 64, n_layers: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim) for _ in range(n_layers)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def make_model(dim=64, n_layers=4, device="cpu", seed=1234):
    """所有 rank 用同一个 seed 建模型 => 初始权重完全一致（DP 的前提）。"""
    g = torch.random.fork_rng()
    with g:
        torch.manual_seed(seed)
        m = MLPStack(dim, n_layers)
    return m.to(device)


def get_batch(step: int, bsz: int, dim: int, device, seed: int = 0):
    """确定性数据：同一个 step 在所有 rank 上生成同一份 *全局* batch。
    各 rank 再按需切自己那一份，这样单卡 baseline 和并行版本看到的数据完全一样。"""
    gen = torch.Generator().manual_seed(seed * 100003 + step)
    x = torch.randn(bsz, dim, generator=gen)
    y = torch.randn(bsz, dim, generator=gen)
    return x.to(device), y.to(device)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def fmt_bytes(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:7.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


# ----------------------------------------------------------------------------
# 统计激活：autograd 为 backward 保存了多少字节
# ----------------------------------------------------------------------------
class SavedTensorCounter:
    """统计 autograd 为 backward 保存的张量字节数（按 storage 去重，排除参数本身）。"""

    def __init__(self, model):
        self.param_ptrs = {p.untyped_storage().data_ptr() for p in model.parameters()}
        self.seen, self.bytes = set(), 0

    def pack(self, t):
        ptr = t.untyped_storage().data_ptr()
        if ptr not in self.param_ptrs and ptr not in self.seen:
            self.seen.add(ptr)
            self.bytes += t.untyped_storage().nbytes()
        return t

    def __enter__(self):
        self.ctx = torch.autograd.graph.saved_tensors_hooks(self.pack, lambda t: t)
        self.ctx.__enter__()
        return self

    def __exit__(self, *a):
        self.ctx.__exit__(*a)
