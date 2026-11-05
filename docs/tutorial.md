# 上手教程：9 个 demo 读懂大模型分布式训练

建议配合 [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) 阅读：先读对应章节，再跑 demo、对照输出。每个 demo 约 10–20 分钟。

> 本教程里的输出节选来自一次真实运行：单节点、2 块 H100 MIG 切片（实际位于同一张物理卡上），PyTorch 2.15。
> **正确性（`[PASS]`）、显存、通信量、负载这类数字可以直接参考；带宽、速度会因为两块切片共用一张卡而失真**，不要拿来比较方案的快慢。
> 你在 CPU 或其他 GPU 上跑，数值会不同，但 `[PASS]` 应该都一样。

## 0. 准备：三分钟了解怎么跑

**安装**：只需要 PyTorch（建议 ≥ 2.6）。

```bash
pip install "torch>=2.6"
```

**启动多进程**：每个 demo 都是"一份代码、N 个进程同时跑"。`torchrun` 负责启动 N 个进程，并给每个进程设置环境变量：

| 变量 | 含义 |
| --- | --- |
| `WORLD_SIZE` | 进程总数（= "卡数"） |
| `RANK` | 本进程的全局编号，0 … N−1 |
| `LOCAL_RANK` | 本机内的编号，用来选第几张卡 |

```bash
torchrun --standalone --nproc_per_node 2 02_dp.py   # 有 GPU：每个进程一张卡，NCCL 后端
./run_local.sh 02_dp.py 2                           # 没有 GPU：2 个 CPU 进程，gloo 后端
```

**怎么读结果**：每个 demo 都会打印若干行 `[PASS]` / `[FAIL]`，含义是"并行版本的结果"和"单卡完整计算的结果"逐元素比较、误差在容忍范围内。`max|diff|` 是最大误差，1e-6 量级是正常的浮点误差。

`common.py` 是所有 demo 共用的工具：`setup()` 初始化进程组、`check()` 跨所有 rank 做比较、`make_model()` 构造一个小 MLP / Transformer。读任何一个 demo 之前，先花两分钟看一下它。

---

## 1. `00_collectives.py`：通信原语

**目标**：分布式训练里所有并行方式，底层都是几种"集合通信"的组合。先把它们的语义看明白。

```bash
torchrun --standalone --nproc_per_node 2 00_collectives.py
```

**输出节选**：

```
all_reduce(SUM)：逐元素求和，每个 rank 都拿到完整结果
  rank0 before       [1.0, 10.0]
  rank1 before       [2.0, 20.0]
  rank0 after        [3.0, 30.0]
  rank1 after        [3.0, 30.0]
reduce_scatter：求和后切片，rank i 只拿第 i 片
  rank0 before       [0.0, 1.0, 2.0, 3.0]
  rank1 before       [100.0, 101.0, 102.0, 103.0]
  rank0 after        [100.0, 102.0]
  rank1 after        [104.0, 106.0]
  ...
  [PASS] all_reduce vs RS+AG                              max|diff|=0.00e+00
        size   time/iter       algbw       busbw
     1.00 MB     0.068ms   14.37 GB/s   14.37 GB/s
    16.00 MB     0.093ms  168.75 GB/s  168.75 GB/s
     1.00 GB     3.000ms  333.36 GB/s  333.36 GB/s
```

**要点**：

- **all_reduce = reduce_scatter + all_gather**。这是后面 ZeRO 的基础：DP 本来就要做这两步，ZeRO 只是在两步之间"只更新自己那 1/N"。
- 各并行方式用到的原语：DP → all_reduce；ZeRO → reduce_scatter / all_gather；TP → all_reduce（SP 下换成 RS/AG）；PP → send/recv；EP → all_to_all；CP → send/recv（环）或 all_gather。
- 带宽表：**小消息受延迟主导，大消息才接近链路带宽**。这就是为什么 DP 要把梯度攒成桶再发（第 3 节）。
- `busbw`（总线带宽）把不同卡数的结果折算到可比的口径，比较拓扑时看它。

**动手试试**：CPU 上用 4 个进程跑（`./run_local.sh 00_collectives.py 4`），看 reduce_scatter 每个 rank 拿到几片。

---

## 2. `01_memory.py`：单卡显存账本

**目标**：搞清楚训练时显存花在哪，为什么一张卡放不下大模型，以及两种单卡就能用的省显存办法。

```bash
python 01_memory.py --dim 2048 --layers 8 --bsz 512
```

**输出节选**：

```
  本 demo 模型参数量 Ψ = 268,550,144
  fp32 训练：参数 4Ψ + 梯度 4Ψ + Adam(m,v) 8Ψ = 16Ψ =    4.00 GB
        8B                119.21 GB
   batch           不重计算            重计算       节省
     512      328.03 MB       40.00 MB     88%
  只有参数               :    1.01 GB
  forward 后 (+激活)     :    1.32 GB
  backward 后 (+梯度-激活):    2.02 GB
  第一次 step 后 (+Adam) :    4.02 GB
  峰值                   :    5.02 GB
  [PASS] grad accumulation == big batch                   max|diff|=2.61e-08
```

**要点**：

- **静态显存 = 16Ψ 字节**（参数 + 梯度 + Adam 两个状态），和 batch 无关。实测 4.02 GB 正好对上理论 4.00 GB。按这个算，8B 模型光静态部分就要约 120 GB——这是 ZeRO 的动机。
- 混合精度并不省"静态"显存（还要存 fp32 主权重），省的是激活和算力。
- **激活显存随 batch 线性增长**；**激活重计算**只存每个 block 的输入，backward 时重算一遍 forward，用约 1/3 的额外计算换 88% 的激活显存。
- **梯度累积**：把大 batch 拆成几个 micro-batch 依次算、梯度相加，结果和一次算大 batch 完全一样。记住 `global_batch = micro_batch × 累积步数 × DP 卡数`，后面每章都用。

**动手试试**：把 `--bsz` 换成 128、256，看激活显存是否线性变化。

---

## 3. `02_dp.py`：数据并行（DP）

**目标**：每张卡放一份完整模型、吃不同的数据，backward 后把梯度求平均。手写三个逐步优化的版本，再和 PyTorch DDP 对比。

```bash
torchrun --standalone --nproc_per_node 2 02_dp.py --dim 2048 --layers 8 --bsz 64 --bench
```

**输出节选**：

```
  [PASS] v1 naive: grads == single-GPU big batch          max|diff|=4.95e-10
  [PASS] v2 overlap: grads == single-GPU big batch        max|diff|=4.95e-10
  [PASS] v3 bucket: grads == single-GPU big batch         max|diff|=4.95e-10
  [PASS] torch DDP: grads == single-GPU big batch         max|diff|=4.95e-10
  [PASS] grad_acc=4 x dp=2: grads == big batch            max|diff|=7.42e-10
  4 个 micro-batch 一共只发了 17 次 all_reduce（= 桶数 17）
```

**要点**：

- **v1 naive**：backward 全部结束后逐个参数 all_reduce，计算和通信完全串行。
- **v2 overlap**：某个参数的梯度一算好就立刻异步发出去，和剩下的 backward 重叠。
- **v3 bucket**：把梯度攒成桶再发，减少小消息的启动开销（DDP 默认 25 MB 一桶）。
- **梯度累积 + DP**：只在最后一个 micro-batch 通信（DDP 的 `no_sync`），4 个 micro-batch 只发了 17 次（= 桶数），而不是 4 倍。
- `--bench` 会打印每个版本的 ms/iter。在真正的多卡上，overlap / bucket 应该更快；本教程的运行环境是两块切片共用一张卡，计算和通信抢同一份资源，所以速度对比没有意义。

**动手试试**：改 `--bucket_mb`（默认 0.25），看桶数怎么变。

---

## 4. `03_zero.py`：ZeRO-1 / 2 / 3 和 FSDP2

**目标**：DP 里每张卡都存一份完整的优化器状态、梯度和参数，很浪费。ZeRO 把它们依次切成 N 份。

```bash
torchrun --standalone --nproc_per_node 2 03_zero.py --fsdp
```

**输出节选**：

```
  [PASS] ZeRO-1: params after 5 Adam steps == baseline    max|diff|=2.38e-07
  [PASS] ZeRO-2: params after 5 Adam steps == baseline    max|diff|=2.38e-07
  [PASS] ZeRO-3: params after 5 Adam steps == baseline    max|diff|=2.38e-07
  ZeRO-2: reduce_scatter(Ψ) + all_gather(Ψ) = 2Ψ，和 DP 一样
  ZeRO-3: forward AG(Ψ) + backward AG(Ψ) + RS(Ψ) = 3Ψ，多 50%，换来参数也切分
每卡显存（fp32 训练，Ψ = 99,648，N = 2；括号内是 bf16 混合精度的系数）
  DP          1.52 MB   (16.00 Ψ bytes)
  ZeRO-1      1.14 MB   (10.00 Ψ bytes)
  ZeRO-2    973.12 KB   ( 9.00 Ψ bytes)
  ZeRO-3    778.50 KB   ( 8.00 Ψ bytes)
  rank0: 本地只存 49,824 / 99,648 个参数（DTensor 分片）
  [PASS] FSDP2: params == baseline                        max|diff|=2.38e-07
```

**要点**：

| | 切什么 | 每步通信量 |
| --- | --- | --- |
| ZeRO-1 | 优化器状态 | 2Ψ（和 DP 一样） |
| ZeRO-2 | + 梯度 | 2Ψ（和 DP 一样） |
| ZeRO-3 | + 参数（用到哪层临时 all_gather 哪层） | 3Ψ |

- ZeRO-1/2 的显存是"白省"的：通信量和普通 DP 相同。ZeRO-3 多 50% 通信，换来参数也能切分。
- 输出里 ZeRO-1 显示 3Ψ，是 demo 为了好读多发了一次 all_gather；脚本里有说明，真实实现是 2Ψ。
- FSDP2（`fully_shard`）是 PyTorch 里 ZeRO-3 的官方实现，2 卡时每个 rank 正好存一半参数。

**动手试试**：用 4 个进程跑，看每卡显存的系数怎么变（卡越多，ZeRO 省得越多）。

---

## 5. `04_tp_sp.py`：张量并行（TP）和序列并行（SP）

**目标**：模型一层就放不下时，把每一层的矩阵切到多张卡上（Megatron 风格），再用 SP 把 LayerNorm 这些"切不了"的部分按序列切开。

```bash
torchrun --standalone --nproc_per_node 2 04_tp_sp.py
```

**输出节选**：

```
  [PASS] TP: forward output                               max|diff|=4.77e-07
  [PASS] TP: all param-shard grads                        max|diff|=1.91e-06
  [PASS] TP+SP: forward output (本卡序列段)                    max|diff|=4.77e-07
  单卡完整 block :  129.52 KB
  TP            :   81.02 KB   <- attention/MLP 内部切了，但 LN、残差、block 输入仍是完整的
  TP + SP       :   72.77 KB   <- LN 等也按序列切了
```

**要点**：

- Attention 按"头"切（每卡 heads/tp 个头），MLP 的 fc1 按列切、fc2 按行切。这样每个子层只在开头和结尾各通信一次。
- 纯 TP 每个 block 的 forward 要 2 次 all_reduce；SP 把它换成 all_gather + reduce_scatter，通信量相同，但激活显存更少（129.5 → 81.0 → 72.8 KB）。
- TP 的通信在计算的关键路径上、无法和计算重叠，所以**一般只在节点内（NVLink）做 TP**。

**动手试试**：`torchrun --nproc_per_node 4 04_tp_sp.py --heads 8`。

---

## 6. `05_pp.py`：流水线并行（PP）

**目标**：把模型按层切成 p 段（stage），每张卡负责一段，数据像流水线一样依次经过。对比两种调度，并用 ASCII 时间线看"气泡"（bubble，即卡在空等的时间）。

```bash
torchrun --standalone --nproc_per_node 2 05_pp.py
```

**输出节选**（`F3` = 第 3 个 micro-batch 的 forward，`B` = backward，`.` = 空闲）：

```
p=2 个 stage，m=6 个 micro-batch
理论 bubble：(p-1)/m = 0.17（相对理想时间），占总时间 (p-1)/(m+p-1) = 14.3%

调度：AFAB
  [PASS] AFAB: stage grads == single-GPU                  max|diff|=3.73e-09
  stage0: F0F1F2F3F4F5. . . B0B0B1B1B1B2B2B3B3B4B4B5B5  空闲 15.3%
  stage1: . F0F1F2F3F4F5B0B0B1B1B2B2B2B3B3B4B4B5B5. .   空闲 15.1%
  每个 stage 同时存着的 micro-batch 激活数（峰值）: stage0=6, stage1=6

调度：1F1B
  [PASS] 1F1B: stage grads == single-GPU                  max|diff|=3.73e-09
  stage0: F0F1. . B0B0F2B1B1F3B2B2F4. B3B3F5B4B4. B5B5  空闲 14.9%
  stage1: . F0B0B0F1B1B1F2B2B2F3B3B3. F4B4B4F5B5B5. .   空闲 14.7%
  每个 stage 同时存着的 micro-batch 激活数（峰值）: stage0=2, stage1=1
```

**怎么读时间线**：

- stage1 开头的 `.`：它要等 stage0 算完 F0、把激活送过来才能开始。
- AFAB 里 stage0 中间的 `. . .`：所有 forward 跑完后，它要等 stage1 把第一个梯度送回来。
- 实测空闲 ~15%，和理论值 14.3% 基本一致。

**要点**：

- **AFAB**（GPipe）：先跑完所有 forward 再跑所有 backward，每个 stage 要同时存 m 份激活。
- **1F1B**：一个 forward、一个 backward 交替进行。**bubble 和 AFAB 一样大**，但激活峰值从 m 份降到约 p 份（这里 6 → 2）。
- 想继续减小 bubble：增大 m（但 global batch 变大）、interleaved 1F1B、zero-bubble / DualPipe。
- `--sim_ms` 默认用 sleep 模拟计算时间（backward = 2 倍 forward），让 CPU 上也能画出干净的时间线；`--sim_ms 0` 用真实计算。

**动手试试**：`--micro 12` 看 bubble 变小；CPU 上 `./run_local.sh 05_pp.py 4 --micro 8` 看 4 个 stage 的时间线。

---

## 7. `06_cp.py`：上下文并行（CP）

**目标**：序列很长时，把一条序列切到多张卡上做 attention。

```bash
torchrun --standalone --nproc_per_node 2 06_cp.py --seq 64
```

**输出节选**：

```
切分方式：contiguous
  [PASS] Ring Attention forward (contiguous)              max|diff|=6.56e-07
  每卡实际计算的 (q,k) 对数: [528, 1552]  -> 最大/平均 = 1.49
  [PASS] All-gather CP dQ/dK/dV (contiguous)              max|diff|=4.77e-07
切分方式：zigzag
  rank0 负责位置 [0, …, 15, 48, …, 63]
  rank1 负责位置 [16, …, 47]
  [PASS] Ring Attention forward (zigzag)                  max|diff|=6.56e-07
  每卡实际计算的 (q,k) 对数: [1040, 1040]  -> 最大/平均 = 1.00
```

**要点**：

- **Ring Attention**：K/V 沿环传 N−1 次，每收到一块就用 online softmax（和 FlashAttention 同一个技巧）累加，通信可以和计算重叠。
- **All-gather KV**（Llama 3 的做法）：每卡收集完整的 K/V、只算本卡的 Q。GQA 下 K/V 很小，这种"笨"办法很实用。
- **负载均衡**：因果遮罩下，按顺序切（contiguous）时后半段的卡要算的多得多（528 对 1552）；zig-zag 让每张卡各拿一头一尾，计算量完全相等。

**动手试试**：用 4 个进程、`--seq 64`，看 contiguous 的不均衡是不是更严重。

---

## 8. `07_ep.py`：专家并行（EP）

**目标**：MoE 模型里有很多"专家"（小 MLP），每个 token 只用其中 top-k 个。把专家分到不同卡上，用 all_to_all 把 token 送到专家所在的卡、算完再送回来。

```bash
torchrun --standalone --nproc_per_node 2 07_ep.py
```

**输出节选**：

```
EP=2：4 个专家，每卡 2 个，top-2
  rank0: 本卡专家 [0, 1] 收到的 token 数 [13, 20]
  rank1: 本卡专家 [2, 3] 收到的 token 数 [19, 12]
  [PASS] EP forward output                                max|diff|=0.00e+00
  [PASS] EP expert grads (本卡专家)                           max|diff|=9.54e-07
  [PASS] EP router grad (all_reduce 后)                    max|diff|=0.00e+00
```

**要点**：

- 流程：router 打分选 top-k → 先 all_to_all 交换"每张卡要收多少 token" → all_to_all 发送 token（dispatch）→ 本卡专家计算 → all_to_all 送回（combine）→ 按 gate 权重加权。
- 各专家收到的 token 数不均（12～20 个）：**负载不均衡是 MoE 的头号问题**，实际系统用辅助损失、bias 调整（DeepSeek-V3）和 capacity factor 来控制。

**动手试试**：`torchrun --nproc_per_node 4 07_ep.py --experts 8 --topk 2`。

---

## 9. `08_2d_dp_tp.py`：二维并行 DP × TP

**目标**：把前面的方法组合起来。用 DeviceMesh 把所有卡排成 `[dp, tp]` 网格：同一行是一个 TP 组，同一列是一个 DP 组。

```bash
torchrun --standalone --nproc_per_node 4 08_2d_dp_tp.py --native   # dp=2 × tp=2，需要 4 卡（或 CPU 上 4 个进程）
torchrun --standalone --nproc_per_node 2 08_2d_dp_tp.py --tp 2 --native   # 只有 2 卡时：dp=1 × tp=2
```

**输出节选**（2 卡，dp=1 × tp=2）：

```
  global rank 0: dp_rank=0, tp_rank=0, TP 组=[0, 1], DP 组=[0]
  global rank 1: dp_rank=0, tp_rank=1, TP 组=[0, 1], DP 组=[1]
  [PASS] DP×TP: all grads == single-GPU global batch      max|diff|=2.38e-06
  [PASS] native TP+FSDP2: grads == single-GPU             max|diff|=5.36e-07
  每个参数现在是一个 DTensor，fc1.weight.placements = (_StridedShard(dim=0, sf=2), Shard(dim=0))
```

**要点**：

- **放置原则**：通信最频繁、无法重叠的维度放最内层（相邻 rank = 同一节点 = NVLink）。TP 每层都通信 → 最内层；DP 每步才通信一次、还能和 backward 重叠 → 外层，可以跨节点。
- `--native` 用 PyTorch 原生 API 再做一遍：`parallelize_module` 做 TP，`fully_shard`（FSDP2）在 DP 维上继续切。
- 2 卡只能 dp=1，DP 那一维实际没用上；用 4 卡（或 `slurm/run_2node.sbatch` 两节点各 2 卡）才能看到真正的二维并行。

**动手试试**：4 进程加 `--sp`，把 TP 换成 TP+SP。

---

## 下一步

- 在真正的多卡（最好有 NVLink）上重跑 `00` 和 `02 --bench`，对比 naive / overlap / bucket 的速度差异。
- 用 `slurm/run_2node.sbatch 00_collectives.py` 看跨节点后 all_reduce 带宽掉了多少，体会为什么 TP 不跨节点。
- 读每个 demo 的源码：每个一两百行，开头的文档字符串写了实现思路。
