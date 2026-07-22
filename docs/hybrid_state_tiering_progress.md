# State-as-a-Memory-Tier 实验进度报告

> 本文是 `docs/hybrid_state_tiering_experiment_plan.md` 的活文档。每完成一部分,就把该实验的
> **配置**、**过程**、**结果** 追加进来。每个阶段内最新验证结果置顶;顶部状态看板用于一眼总览。

**论文:** *State as a Memory Tier: Free Prefix Reuse and Graceful KV Eviction for
Hybrid Linear-Attention LLMs.*
**分支:** `yyq/gdn-head-aware-prefix-cache`。
**硬件:** 2× H20-3e(sm90),CUDA 12.9,TP=2。
**主模型:** Qwen3-Next-80B-A3B-Instruct(NVFP4)—— 混合架构 GDN(36 个线性层)+ 12 个全注意力层。

---

## 状态看板

| 阶段 | 内容 | 关卡 | 状态 | 结论 |
|------|------|------|------|------|
| A2 | 因果消融(线性态 vs 全注意力 KV) | **GATE-A** | ✅ 完成(2026-07-17) | **通过** —— 全注意力是长程召回的因果来源;线性态可降级 |
| A1 | 逐层贡献探针 | — | ✅ 完成(2026-07-17) | 检索由**最后 3 个全注意力层(39/43/47)冗余承担**;前 9 个全注意力层与检索无关 |
| B1 | 方法 A/B/C/D 容量/时延扫描 | — | ✅ 完成(2026-07-17) | **免重建(no-recon)全面占优** —— 最低 TTFT + 最高吞吐 + 1.33~1.60x 容量 |
| B2 | 缓存下的正确性(needle/GSM8K) | — | ✅ 完成(2026-07-17) | no-recon == dense:needle 1.000、GSM8K 与 dense 同噪声带内 |
| B3 | 架构边界(全注意力占比 → 精度差) | — | ✅ 完成(2026-07-17) | no-recon 召回随全注意力预算**优雅退化**(12层1.0→晚带3层0.77→单层0.39→0),无断崖 |
| C0 | fold vs drop 离线可行性 | **GATE-C0** | ✅ 通过(2026-07-17) | **通过** —— LSE 合并数值正确(1e-6);小预算 fold 全面胜 drop(分散召回 cos 0.998 vs 崩塌) |
| C1 | 门控 fold(真实 GDN gate) | — | ✅ 完成(2026-07-17) | **真实门控救回深层 needle** —— 高记忆 GDN 层(L24)每步保留 r_med=0.998,深度 1720 处累积保留 2.98e-4 = **固定 0.99 的 9580×**;按真实保留折叠把深 needle 的 cos 从 0.171(0.99)拉到 0.357(逼近理想 1.0 的 0.530) |
| C2 | 融合 kernel + serving | **GATE-C2a** | 🟨 kernel 通过 / serving 负边界(2026-07-17) | **C2-a kernel 通过**(fold-decode GPU == C0 参考 rel<5e-3;微基准 fold 全面胜 dense,1.83~174x)。**C2-b 忠实 serving 是负结果** —— 全注意力层用**固定衰减**折叠中段,深 needle 召回 **1.000→0.000**;印证 C1"固定衰减清零深层",而全注意力层**没有 GDN 那样的逐 token 门控可救回** |
| D | 统一系统 + 端到端(层选择性 fold) | **GATE-D** | 🟨 有界正结果(2026-07-18) | **A1 因果预测器迁移成立** —— 只折**前 9 个检索无关**全注意力层(保留 39/43/47)把 C2-b 的 needle 召回从 **0.000 救回 0.922**(≈dense,残留 ~8% gap);对照 `fold_unsafe`(只折 39/43/47)= **0.000** 崩塌 —— 双向解离。perf(in256/out512<预算)未触发 fold → 与 baseline 同噪声带,容量/时延收益仍由 C2-a 微基准**投影** |
| G | 因果驱逐优先级(value vs LRU mamba 检查点驱逐) | **GATE-G** | ✅ 通过(2026-07-22) | **value 策略实测胜出(时延/吞吐)** —— A2 证 mamba 检查点是**可降级层**(驱逐召回中性:miss 精确重算前缀),故可把 radix 驱逐 victim 从**纯 recency(LRU)**换成**按复用频率(value)**;checkpoint pool 饱和 + Zipf 前缀倾斜下,两个独立工作点均 **hit-rate +10~12pt、TTFT -15~17%、吞吐 +17%**;env 未设时与 LRU 字节一致 |
| H | 纯 GDN 正对照(打破准确率天花板) | **GATE-H1** | 🟥 诚实 NULL(2026-07-22) | **H1 未通过 —— 片上致盲无法造出纯 GDN 区。** 致盲全部 12 个全注意力层的 prefix KV 后,`gdn_only` 召回在**所有深度(含紧邻 d=0.95)均为 0.000**,`hybrid` 恒为 1.000。**关键 nuance:** 失败模式是**模型坍缩**(输出退化为重复:`State State State`/`( ( (`/`1111111111`),而非干净的"GDN 取不到 needle" —— 说明全注意力 prefix KV 对**基本连贯性**(非仅检索)也是承载的,模型被联合训练成依赖全注意力做上下文整合。⇒ 片上致盲是**被污染的**纯 GDN 模拟,度量无法区分"GDN 缺 needle"与"模型坏了"。按计划在 H1 处**以诚实 null 停止**,不进 H2/H3;正对照需**真纯 GDN 模型**(fallback b)或改探针(短程/聚合任务) |
| K | KDA 逐通道 head-aware(实现 + 离线门) | **GATE-K1** | ✅ 通过(2026-07-22) | **逐通道 head-aware 机制就位、离线正确。** KDA 衰减逐 K 通道(90.6% head 混合,逐 head 不迁移)→ keep/drop 单元 = 全局 `(head, d_k 列)` = 一个 `d_v` 向量。按 `dt_bias` 宽度自动分派(GDN 逐 head 字节一致);GATE-K1 合成 + **真实 Kimi-Linear-48B 权重**双过:global 列位精确 / local 清零 / mask 正确 / 真机 **35.31x** 容量(99.8% 列 local@w512)/ GDN 非回归。**serving A/B(GATE-K2)未跑** —— 阻塞于 KimiLinear 缺 `extra_buffer` 白名单 + KDA extra_buffer 0.63 bug;可落地路径 = Idea4 value 驱逐 + Idea1 no-recon(`no_buffer` 全提示复用) |

图例:✅ 完成 · 🟨 进行中 · ⬜ 待做。

---

## Phase A2 —— 因果消融(GATE-A) ✅

**问题(RQ0):** 全注意力是否是长程召回的**因果**来源,从而线性递归态是一个可降级的记忆层?
**假设:** 清零线性态几乎不伤 needle 召回;清零全注意力的 prefix KV 会使召回崩塌。

### 配置

- **模型 / serving:** Qwen3-Next-80B-A3B-Instruct-NVFP4,TP=2,`mem-fraction-static=0.7`,
  `--mamba-radix-cache-strategy extra_buffer`,`--attention-backend triton`
  (cu12.9 的 flash-attn 缺 FA3 `only_qv` 入参,PR #28394),`--disable-overlap-schedule`,
  **`--disable-cuda-graph`**(必需,原因见"过程"),PORT 31007,`CUDA_VISIBLE_DEVICES=0,1`。
- **探针:** `test/manual/needle_longrange.py` 直接检索 —— 一篇 3000 词文档,在深度 0.5 处埋
  一个唯一的 5 位金库码 needle,先灌一次再问纯数字问题。16 组 × 4 问 = **每 arm 64 次试验**。
  贪心解码(温度 0)。
- **消融注入(仅实验用,env 门控,未设时全程 no-op):**
  `python/sglang/srt/debug/state_ablation.py`。
  - `ABLATE_LINEAR_STATE = off|zero|noise` —— 在**每次** kernel 读前就地破坏 GDN `ssm_state`
    (持续 = GDN 不携带任何跨步记忆)。挂钩在 `linear/gdn_backend.py` 的 forward_decode + forward_extend。
  - `ABLATE_FULL_KV = off|zero|noise` —— 对 12 个全注意力层的 prefix KV 做 snapshot → 破坏 →
    **恢复**(围绕一次 forward,非持久;事后 KV 池逐字节不变 → 在连续批处理 / 重复提问下安全)。
    挂钩在 `hybrid_linear_attn_backend.py` 的 forward_decode + forward_extend(仅全注意力层)。
  - 主 arm 用 `zero`;`ABLATE_NOISE_SCALE`/`ABLATE_SEED` 控制 `noise` 模式。
- **驱动脚本:** `test/manual/run_state_ablation_2x2.sh <baseline|ablate_linear|ablate_full_kv|ablate_both|report>`。
  每个 arm 各自起一份 server(env 在 import 时冻结),跑探针,追加到同一份 JSON
  `state_ablation_logs/state_ablation_2x2.json`;`report` 以 baseline 为参照做 diff。

### 2×2 的四个 arm

| Arm | `ABLATE_LINEAR_STATE` | `ABLATE_FULL_KV` | 含义 |
|-----|-----------------------|------------------|------|
| baseline | off | off | 两个 tier 都完好(上界) |
| ablate_linear | zero | off | 只破坏 GDN 递归态 |
| ablate_full_kv | off | zero | 只对全注意力遮住 prefix |
| ablate_both | zero | zero | 两个都破坏(sanity) |

### 过程

- 四个 arm 顺序跑(共用 PORT + 两张卡 ⇒ 不能并行),全部落到同一 JSON,最后 `report`。
- **跑的过程中修了两个 bug,都只在 full-KV 路径,baseline 不受影响:**
  1. `ForwardBatch` 没有 `req_to_token_pool` 属性 → 改为从 hybrid backend 把
     `self.req_to_token_pool` 传进 `corrupt_full_kv`;loc 收集函数显式接收 `req_to_token` 表。
  2. **CUDA graph 不兼容(根因):** 该消融是 Python 的 snapshot/破坏/恢复,必须在*每一步* decode
     上执行,但 CUDA graph 只在 capture 时跑 Python(replay 不跑),且 capture 还禁止 loc 收集里的
     device→host 拷贝。→ 对**所有** arm 禁用 decode graph(召回是粗粒度、与 graph 无关的 0/1 指标,
     arm 间仍可比)。baseline 也用 graph-disabled 重跑了一遍,保证配置一致。

### 结果

```
=== 长上下文直接检索(以 baseline 为参照),每 arm 64 次 ===
mode              retr_acc  cache_hit_frac  ans==baseline
baseline            1.000       0.98           64/64
ablate_full_kv      0.000       0.98            0/64
ablate_linear       1.000       0.98           64/64
ablate_both         0.000       0.98            0/64
```

### 判定 —— **GATE-A 通过**

- 遮住全注意力的 prefix KV,召回 **1.000 → 0.000** 崩塌:全注意力是长程召回的**因果**来源。
- 清零 GDN `ssm_state`,召回**保持 1.000**,且 **64/64** 输出与 baseline 逐字一致:线性递归态是
  一个**可降级、定长的记忆层**。
- 四个 arm 的 `cache_hit_frac` 都是 **0.98** ⇒ needle 在每个 arm 里被同样地缓存,唯一变量就是
  "哪个 tier 能读到 prefix"。因此效果是**因果,而非相关**。`ablate_both` 崩塌验证了 sanity 一支。
- 这一张 2×2 同时支撑 **Idea 1**(no-recon 前缀缓存)和 **Idea 2**(fold-to-state 驱逐),并从因果上
  证实了此前 no-recon == dense 的发现。

**产物:** `state_ablation_logs/state_ablation_2x2.json`(以及各 arm 的 `server_*.log`、`needle_*.log`)。

---

## Phase A1 —— 逐层贡献探针 ✅

**问题:** 长程 needle 检索具体由**哪些层**承担?("谁在做检索"的图。)

### 设计选择 —— 因果定位,而非注意力质量

计划原文是"测每层在 needle 位置的注意力/贡献质量"。但注意力质量对**线性层无明确定义**(GDN 没有显式注意力矩阵),且要捕获内部张量。这里改用**逐层因果定位**:复用 A2 的全注意力 KV 消融,只不过用 `ABLATE_FULL_KV_LAYERS` 把破坏**限制在一个深度带**,看遮住该带后召回掉多少。这与全文"因果"叙事一致,也比相关性的注意力质量更强。

### 配置

- 模型 / serving / 探针:与 Phase A2 完全一致(NVFP4,TP=2,mem-frac 0.7,extra_buffer,triton,
  disable-overlap,**disable-cuda-graph**,needle 3000 词 / 深度 0.5 / 16×4=64)。
- **层过滤:** `state_ablation.py` 新增 `ABLATE_FULL_KV_LAYERS`(逗号分隔的层 id 白名单;
  `all`/空 = 全部,退化为 A2 行为)。`corrupt_full_kv` 仅对白名单内的层生效。
- **全注意力层索引:** Qwen3-Next-80B 共 48 层,`full_attention_interval=4` ⇒ 全注意力层在
  `{3,7,11, 15,19,23, 27,31,35, 39,43,47}`(12 个)。按深度分 4 带(每带 3 层)。
- **驱动脚本:** `test/manual/run_state_ablation_a1.sh <none|early|mid_early|mid_late|late|all12|late_39|late_43|late_47|report>`,
  写入 `state_ablation_logs/state_ablation_a1.json`,`report` 以 `none` 为参照。

### 过程

1. **深度带扫描:** none(参考)+ 4 带(early=3,7,11 / mid_early=15,19,23 / mid_late=27,31,35 /
   late=39,43,47)+ all12(交叉校验,应 == A2 的 0.0)。
2. **细粒度定位:** 对崩塌的 late 带,再把 3 层各自单独遮(late_39 / late_43 / late_47),
   判断是整带冗余还是某一层关键。

### 结果

```
=== 深度带扫描(以 none 为参照),每 arm 64 次 ===
mode         retr_acc  cache_hit_frac
none           1.000       0.98      <- 参考
early          1.000       0.98         3,7,11
mid_early      1.000       0.98         15,19,23
mid_late       1.000       0.98         27,31,35
late           0.000       0.98         39,43,47   <-- 崩塌
all12          0.000       0.98         全部12层(交叉校验,与 A2 一致)

=== late 带单层定位 ===
late_39        1.000       0.98         仅遮第39层
late_43        0.969       0.98         仅遮第43层
late_47        0.984       0.98         仅遮第47层
```

### 判定

- **检索集中在最后 3 个全注意力层(39/43/47)。** 遮住任意早/中深度带(共 9 个全注意力层),
  召回**纹丝不动 1.0**;唯独遮住 late 带崩到 0.0,且与遮全部 12 层等效。
- **late 带内部是冗余的:** 单独遮 39/43/47 中任一层,召回仅 0.97~1.0(另两层能恢复);
  三层全遮才彻底崩。即"任一晚层大致足够,但整带是必需的"。
- **对 Idea 2 的直接启示(RQ4 预测器):** 早/中全注意力层(3..35)对检索无贡献 ⇒ 是**可激进 fold**
  的安全对象;晚层(39/43/47)必须保留精确 KV。这把"哪层可降级"从 A2 的模型级细化到了层级。
- 与"检索发生在较晚层"的普遍现象一致:遮早层可被后续完好的层重新检索找回,遮晚层则在出 logit
  前无处补救。四带 + 单层所有 arm 的 `cache_hit_frac` 均为 0.98 ⇒ 因果,非相关。

**产物:** `state_ablation_logs/state_ablation_a1.json`(+ 各 arm `server_*.log`、`needle_*.log`)。

## Phase B —— Idea 1:免状态重建的前缀缓存 🟨

**问题(RQ1/RQ2):** 混合模型上,线性态的前缀重建能否**跳过**而不掉精度?一旦跳过,还能拿到多少
容量与 TTFT 收益?这套结论在**多大的全注意力预算**下才成立(边界在哪)?

### 方法 ↔ 计划的 A/B/C/D 对应

| 方法 | 计划名 | 本仓实现(`bench_gdn_prefix_hitrate.sh` MODE) | 语义 |
|------|--------|----------------------------------------------|------|
| A | no-cache | (对照,全量 prefill) | 每请求重算,速度下界 |
| B | recon / re-prefill | `headaware_a`(`NORECON=0`,含 seam 变体) | 命中后回放前缀重建 GDN 态 |
| C | checkpoint(精确 bf16) | `dense_bf16`(`WMAX=0` 全头快照) | 存精确定长快照,命中零重算(容量 1.0x) |
| D | **no-recon** | `headaware_a`(`NORECON=1`) | 丢弃 local 头 + **不重建**,靠全注意力兜底 |

> 说明:本阶段直接复用 Route-A 头感知前缀缓存的既有 GPU 数据(2026-07-08/09,真实 Qwen3-Next-80B
> NVFP4,TP=2)。Route-A 的"存端丢 local 头"给出 1.33~1.60x 容量;**是否在载入端重建**(`NORECON`)
> 正是 B(重建)与 D(免重建)之分。C2 seam 是 B 与 D 之间的折中(只缩重建窗口)。

### B1 —— 容量 / 时延扫描 ✅

**配置:** Qwen3-Next-80B-A3B-NVFP4,TP=2,mem-frac 0.7,extra_buffer radix,triton attn,
`DISABLE_OVERLAP=1`(Route-A seam overlap 不安全)。共享前缀工作负载
`bench_serving --dataset-name generated-shared-prefix`:sys=2048,64 组 × 16 = 1024 请求,
q=128 / out=256,zipf(α=1.1),`--max-concurrency 64`。容量口径由服务器日志的 checkpoint 池
`bytes/slot vs dense` 读出。

```
=== 方法扫描(sys=2048,g64×16=1024,conc=64,单位 ms / tok·s⁻¹) ===
方法               TTFT_med  TTFT_p99  TPOT_med  out_thr   req_thr  容量        GSM8K  needle
C  dense_bf16(ckpt)  2007.5    5388.6    19.65    2290.2    8.95    1.00x       0.950  1.000
B  recon(full W)     7141.3    9758.1    25.17    1196.5    4.67    1.33-1.60x  0.960  1.000
B  recon seam=64     2138.8    6299.3    19.51    2203.6    8.61    1.33-1.60x  0.950   —
B  recon seam=32     1991.1    6185.2    18.91    2244.8    8.77    1.33-1.60x  0.950   —
B  recon seam=16     1969.8    6200.9    18.62    2248.5    8.78    1.33-1.60x  0.960  1.000
B  recon seam=8      1967.0    6204.0    19.40    2250.7    8.79    1.33-1.60x  0.950   —
D  no-recon          1656.9    5354.7    18.68    2380.8    9.30    1.33-1.60x  0.940  1.000
```

容量来自服务器日志(TP0/TP1 两卡 bytes/slot):dense 37.7MB/slot = 1.00x;headaware_a
28.3MB(TP0)/23.6MB(TP1)/slot ⇒ **1.33x~1.60x** 容量(丢 local 头),1200 槽 vs 476 槽。

### B1 判定

- **免重建(D)全面占优。** 在**相同的 1.33~1.60x 容量**下,no-recon 的 TTFT(1657ms)比精确
  checkpoint(C,2007ms)还低 **0.82x**,比全窗重建(B,7141ms)低 **0.23x**;吞吐最高
  (out 2381 / req 9.30),TPOT 最低。**重建是纯开销**:B 的全窗重建把 TTFT 拉到 3.6x、吞吐砍半,
  却没有任何精度回报(见 B2)。
- **seam(C2 折中)能把重建 TTFT 从 7141 拉回 ~1970ms**,但仍被 no-recon 压制(更高 TTFT、更低吞吐),
  且不带来 no-recon 之外的精度。⇒ 在**混合模型**上没有保留重建的理由。
- 一句话:Idea 1 = "在混合模型上把线性态前缀重建整段删掉",拿到 **1.33~1.60x 容量 + 更低 TTFT +
  更高吞吐 + 精度中性**。

### B2 —— 缓存下的正确性 ✅

**配置:** 两路正确性判据,均在上面各 arm 的同一 server 上跑。
- **GSM8K**(8-shot 前缀是跨题共享前缀 ⇒ 第 2 题起走命中/重建路径):见 B1 表 `GSM8K` 列。
- **长程 needle 输出一致性**(`needle_longrange.py`,3000 词 / 深度 0.5 / 16×4=64,以 dense 为参照):

```
mode                 retr_acc  cache_hit_frac
dense_bf16(C)          1.000       0.98
headaware_a(B recon)   1.000       0.98
headaware_a seam=16    1.000       0.98
headaware_a no-recon   1.000       0.98
```

### B2 判定

- **no-recon == dense。** needle 检索 **1.000**,与重建、精确 checkpoint 逐一持平;GSM8K
  0.940 落在 dense 0.950 的噪声带内(±0.005~0.01 口径,8-shot×100 题)。**跳过线性态重建不掉精度。**
- 这与 Phase A 因果结论一致:Qwen3-Next 的 12 个全注意力层携带长程记忆,GDN 线性态可降级 ⇒
  免重建之所以无损,是**架构性**的,不是调参凑出来的。needle 的 `cache_hit_frac` 各 arm 都 0.98,
  说明比较是在"同样缓存命中"下做的。

### B3 —— 架构边界(RQ2)✅

**设计:** 用 A1/A2 的 KV 消融机制,在**同一个** Qwen3-Next 上**逐步遮住更多全注意力层的精确前缀
KV**,模拟"有效全注意力预算"下降。x 轴 `keep-k` = 仍保留精确 KV 的全注意力层数(保留**晚层**,
因 A1 证明检索集中在晚层);当预算降到某点,免重建的线性态无从兜底 ⇒ 召回崩塌。这就是 RQ2 的
**模型内**边界曲线代理(全注意力占比 → no-recon 精度差)。

- **驱动:** `test/manual/run_state_ablation_b3.sh <keep12|keep9|keep6|keep3|keep2|keep1|keep0|report>`
  → `state_ablation_logs/state_ablation_b3.json`,`report` 以 keep12 为参照。keep12(=A1 none)、
  keep9(=A1 early)由 A1 覆盖(均 1.0);新增 keep3/keep2/keep1/keep0 追崩塌拐点。

### 结果

```
=== 全注意力预算边界(以 keep12 为参照),每 arm 64 次 ===
keep-k  保留的精确全注意力层        retr_acc
keep12  全部 12 层                   1.000    <- 满预算
keep3   仅晚带 39,43,47             0.766    <- 晚带独扛 ~77%
keep2   43,47                       0.750
keep1   仅 47                       0.391    <- 单层 ~39%
keep0   无                          0.000    <- 全崩
```

### B3 判定

- **no-recon 召回随全注意力预算优雅退化,而非悬崖式。** 满预算(12 层)= 1.000;砍到只剩晚带
  3 层(占全注意力预算 25%、占全 48 层的 6%)仍保 **0.766**;单层保 0.391;归零才彻底崩。曲线单调、
  连续,除非预算清零否则无断崖。
- **细化 A1:** A1 单独遮任一早/中带(遮 3 层)召回纹丝不动 1.0,是**冗余**;但 B3 一次遮掉 9 个
  非晚层,召回从 1.0 掉到 0.766 —— 冗余是**有限**的。即:检索晚层集中 + 早/中层提供**可耗尽的冗余
  支撑**,两者一致。
- **对 Idea 1 的边界结论(RQ2):** 富混合模型(Qwen3-Next 有 12 个全注意力层,预算充裕)⇒ no-recon
  安全,这正是 B1/B2 观测到的 no-recon==dense。曲线预测:当**有效全注意力层数趋近 0**(退化为纯 GDN)
  时,免重建必然崩塌,须回退到重建 —— 与 Route-A 记忆"recon 只在纯 GDN 上才该保留"的既有结论吻合。
  阈值:近乎无损的 no-recon 需要**完整(冗余)的全注意力层补集**;少数功能性全注意力层只能撑起部分召回。
- 所有 arm 的 `hit_frac` 恒为 0.98 ⇒ 因果、非相关(与 A1/A2 同口径)。

**产物:** `state_ablation_logs/state_ablation_b3.json`(+ 各 arm `server_*.log`、`needle_*.log`);
驱动 `test/manual/run_state_ablation_b3.sh`。

### Phase B 总结

Idea 1 三问全部落地(真实 Qwen3-Next-80B TP=2):**(B1)** 免重建在相同 1.33~1.60x 容量下 TTFT 最低
(0.82x vs 精确 checkpoint、0.23x vs 全窗重建)、吞吐最高;**(B2)** 精度无损(needle 1.000、GSM8K
噪声带内);**(B3)** 该结论随全注意力预算优雅退化,边界在"全注意力层趋零"处。⇒ **在混合模型上,
线性态前缀重建可整段删除**,Idea 1 成立且可独立成文。

## Phase C —— Idea 2:fold-to-state 驱逐 🟨

**问题(RQ3):** 在固定的**精确 KV 预算**下,把被驱逐的中段 token **fold(折叠)进一个线性态**、
解码时读回并与精确窗口做 LSE 合并,是否比直接 **drop(丢弃)** 保留更多召回?

**结构形态**(受门控衰减的路径依赖所迫,每个全注意力层):
`[sink 精确] + [中段 → fold 进线性态] + [近窗 精确]`,精确 KV 预算 B = |sink| + |window|。

### Phase C0 —— 离线可行性(GATE-C0)✅ **通过**

**问题:** 这套 fold + LSE 合并**在数值上正确吗**?在小预算下**fold 是否真的赢过 drop**?
若 fold 不赢,则**不建 kernel**(C2)——要么合并写错(debug),要么此模型上论点不成立(记为负边界)。

**方法(零新 kernel,纯离线 CPU):** `test/manual/fold_vs_drop_c0.py`。
- **合并(C-risk,承重):** 精确窗口用在线 softmax 统计量 `(numerator, denom, max)`;fold 的中段在
  驱逐时(**query 未知**)存 q 无关线性态 `S = Σ_i g_i·φ(k_i)·v_iᵀ`、`z = Σ_i g_i·φ(k_i)`,解码时
  读回 `N_mid=φ(q)@S, D_mid=φ(q)@z`,再与精确部分在同一 exp 尺度下相加归一。
- **特征映射 φ**(让线性 fold 与精确 softmax 同尺度):`taylor2`(基于 exp 二阶泰勒,
  `⟨φ(q),φ(k)⟩=1+s·q·k+(s·q·k)²/2`,恒正)与 `elu1`(线性注意力标准,诚实下界)。
- **单元测试(GATE 前置):** 把一段序列切成两个**精确**分区做合并,须逐一等于整段 full softmax。
- **needle 合成:** 中段埋一个 query 对齐的 needle(top-1 logit,独特 value 方向),扫精确预算 B;
  当窗口 < needle 深度时 needle 落在中段 → drop 丢光 / fold 折叠。度量输出对 full-attn 的 cos 与相对
  L2 误差,以及 needle 方向信号的恢复量。

### 结果

```
[单元测试] 两精确分区合并 vs full softmax:最坏相对误差 = 9.64e-07 -> PASS
           (GATE 前置:LSE 合并代数数值正确)

=== needle 扫描(φ=taylor2, decay=1.0, 锐利 needle logit_gap=6)===
full-attn:needle softmax 权重=0.165  needle 信号(o·û)=0.481
预算B  needle在中段  cos_drop  cos_fold   rel_drop  rel_fold
 128       是         -0.054    0.456      2.292     0.890
 256       是          0.008    0.456      1.554     0.890
 512       是          0.219    0.456      1.128     0.890
1024       是          0.276    0.456      1.021     0.890
1152       否(入窗)   0.968    1.000      0.667     0.000
2048       否          1.000    1.000      0.000     0.000

=== needle 扫描(φ=taylor2, decay=1.0, 柔和/分散召回 logit_gap=3)===
预算B  needle在中段  cos_drop  cos_fold   rel_drop  rel_fold
 128       是          0.335    0.998      3.537     0.064
 512       是          0.533    0.998      1.647     0.064
1024       是          0.702    0.998      0.987     0.064   <- fold≈精确,drop 崩

=== 固定衰减的陷阱(φ=taylor2, decay=0.99, 锐利 needle)===
预算B=128(needle 深 ~900 tok):cos_fold 掉到 0.156(< decay=1.0 的 0.456)
   0.99^900 ≈ 1e-4 → 深层 needle 被衰减清零 → 固定激进衰减有害
```

### 判定 —— **GATE-C0 通过**

- **LSE 合并数值正确**(单元测试 1e-6)⇒ C-risk 消解,可放心进 serving。
- **小预算下 fold 全面赢过 drop**,且分两个体制:
  1. **锐利单 needle(线性折叠的最坏情形):** fold 恢复**方向**优于 drop(cos 0.456 vs ≤0.28),
     幅度部分恢复 —— 即"优雅退化"。
  2. **柔和/分散召回(LongBench 式典型):** fold **≈ 精确**(cos 0.998、rel 0.06),drop 崩塌
     (cos 0.34~0.70、rel>1)—— **大幅胜出**。
- **固定衰减 <1 会清零深层内容**(decay 0.99 把 900 tok 深的 needle 抹掉)⇒ 最佳固定衰减 ≈ 1.0(纯
  累加);真正的解法是 **C1 用模型自己的门控 g**(内容自适应、对要保留的信息 ≈1),而非固定激进衰减。
- **结论:** fold-to-state 在混合模型全注意力层上,以"图质量/优雅退化 > drop"成立 ⇒ **建 kernel 值得**,
  继续 C1→C2。核心图(recall vs 精确 KV 预算,fold 曲线压在 drop 之上)已由本离线实验给出方向性证据。
- **口径说明:** C0 是**离线合成**可行性(计划即如此界定:"offline feasibility, zero new kernel"),
  已覆盖对抗(锐利 needle)与有利(分散召回)两端;真实 Q/K/V 捕获可作为后续加强,但门已被确定性地判过。

**产物:** `test/manual/fold_vs_drop_c0.py`(单元测试 + needle 扫描 + φ/decay 变体)。

### Phase C1 —— 门控 fold(真实 GDN g)✅ **通过**

#### 动机

C0 暴露的核心矛盾:**固定衰减 <1 会清零深层内容**(0.99^900 ≈ 1e-4,把 900 tok 深的 needle 抹掉),
但衰减 =1(纯累加)又缺乏内容选择性。计划里的解法是用模型**自己学到的、内容自适应**的门控 g 来折叠。
Qwen3-Next GDN 的遗忘门恰好就是这样的信号:

```
g = -exp(A_log) * softplus(a + dt_bias)   (逐 token、逐 head,g <= 0)
每步保留 r = exp(g) ∈ (0,1]
```

对显著 token,模型把 a 压得很负 → softplus→0 → g→0 → r≈1(保留);填充 token → g<0 → 衰减。
C1 就是把这个**真实门控**从一次真实 needle prefill 中抓出来,离线验证它是否让深层显著内容存活
(保留 >> 固定衰减),并用抓到的 g 驱动 C0 的 fold。

#### 配置

- **模型/服务:** Qwen3-Next-80B-A3B-Instruct(NVFP4),TP=2,triton 后端,`--disable-overlap-schedule
  --disable-cuda-graph`,mem-frac 0.7,extra_buffer,PORT 31007。
- **探针:** 复用 `needle_longrange.build_group`,group=0,doc_words=3000,depth=0.5 → needle
  "…vault authorization number for the Meridian-00 account is **10271**…" 埋在文档正中。
- **一次真实 prefill:** `doc + "\n\nAcknowledged."`(prompt_tokens=**3452**),max_new_tokens=1,
  触发整篇文档的整体 prefill。
- **捕获:** 仅由裸环境变量 `CAPTURE_TIER_DIR` 开启的 EXPERIMENT-ONLY 钩子
  (`srt/debug/tier_capture.py`),在 `gdn_backend.forward_extend` 里 `fused_gdn_gating` 之后
  转储真实逐 token g(+k/v/positions/input_ids)。抓 GDN 层 {0,10,24,36,46},每层一次
  (首个 ≥2000 token 的 prefill)。未设该变量时全程 no-op,不上任何发布/默认路径。
- **离线分析:** `analyze_gdn_gate_c1.py` —— 用真实 tokenizer 在 input_ids 里定位 needle 的
  5 位码 token(token-match,精确命中 pos=1731,depth=1720),度量每步保留 r_t 的分布、从每个
  token 到序列末尾的累积保留 R(i)=exp(Σ_{j>i} ḡ_j),并复用 C0 的 fold 数学做"C0 bridge"。

#### 结果

```
=== 每层真实门控(needle_pos=1731,到末尾深度 depth=1720)===
 层    r_med    r_needle_win   R_needle(深1720)   0.99^1720     REAL/固定
  0   0.000000    0.000000       0.000e+00         3.108e-08       0.0x   <- 快速遗忘(局部混合)
 10   0.549437    0.449533       0.000e+00         3.108e-08       0.0x   <- 中等衰减
 24   0.997966    0.973762       2.978e-04         3.108e-08    9580.3x   <- 长记忆层(28% token r>0.999)
 36   0.982802    0.944611       8.381e-16         3.108e-08       0.0x
 46   0.807460    0.794248       0.000e+00         3.108e-08       0.0x

=== C0 BRIDGE:深 needle(N=2048,pos=1148,age 899,budget=256,φ=taylor2)===
full-attn:needle softmax 权重=0.165  full-sig=0.494
        衰减          sig_fold  cos_fold   sig_drop  cos_drop
   固定 0.99           -0.019    0.171     -0.026    0.126
   真实中位 0.99797     0.024    0.357     -0.026    0.126    <- 用高记忆层的真实保留折叠
   真实 needle 0.97376  -0.026    0.149     -0.026    0.126
     理想 1.0           0.037    0.530     -0.026    0.126
```

#### 判定 —— **C1 通过(真实门控解掉 C0 的深层衰减陷阱)**

- **真实门控强烈按层异质,且存在"长记忆" GDN 层。** L24 每步保留中位 **0.998**、28% 的 token
  r>0.999 → 深度 1720 处累积保留 **2.98e-4**,是固定 0.99 曲线(3.1e-8)的 **≈9580×**。也就是说
  **模型自己的门控把深层内容保活比那个抹掉 C0 needle 的任意常数高约 4 个数量级**。
- **另一些 GDN 层是快速遗忘的局部混合器**(L0 r≈0、L10 r_med=0.55)。这与"记忆是分布式的"
  一致 —— 并非每个线性层都需承载长程(全注意力层 + 少数高保留 GDN 层负责),恰好呼应论文主旨:
  **线性态是可降级的记忆层**。
- **C0 bridge 决定性:** 用高记忆层的真实保留(0.998)折叠,把深 needle 的 cos 从固定 0.99 的
  **0.171 提到 0.357**(逼近理想累加 1.0 的 0.530),而 drop 恒为 0.126 —— **在固定 0.99 清零
  needle 的同一深度上,按真实门控折叠救回了 needle**。
- **诚实标注:** ① needle token 相对全局中位是**略微 less open**(L24:0.974 vs 0.998)—— 门控没有
  把 needle 折得比周围填充更开,但其绝对保留 r>0.97 仍很高,needle **没被遗忘**;"保活"成立,
  "相对最开"不成立。② C0 bridge 用单一标量保留(层特征中位)作真实逐 token g 的代理;完全逐 token
  的 fold 需把捕获的 g 直接喂进 kernel,留待 C2。③ 全注意力层的 q 未捕获(fold 数学已在 C0 验证),
  本阶段被测量是**门控本身**。

**产物:** `srt/debug/tier_capture.py`(env-gated 捕获,EXPERIMENT-ONLY)、
`gdn_backend.py` 内的捕获钩子(env 未设即 no-op)、`test/manual/run_gdn_gate_capture.sh`(驱动)、
`test/manual/capture_gdn_gate_c1.py`(单次 needle prefill 客户端)、
`test/manual/analyze_gdn_gate_c1.py`(离线分析)、`gdn_gate_capture/c1_analysis.json`(结果)。

### Phase C2 —— 融合 kernel + serving 🟨(kernel 通过 / serving 负边界)

分两级:**C2-a** 把 C0/C1 验证的 fold 数学做成真实 GPU decode 原语并对全注意力 ground-truth
单测(计划强制的 gate);**C2-b** 才在真 80B 上做忠实 serving A/B。C2-b 不通过 C2-a 不启动。

#### C2-a —— 融合 kernel + 正确性 + 微基准 ✅ **通过**(GATE-C2a)

fold-decode 原语 = `merge( window_softmax(q, K[sink+recent]), state_readout(q, S, z) )`,
其中 S=Σ gᵢφ(kᵢ)vᵢᵀ、z=Σ gᵢφ(kᵢ)(df=d,φ=elu1),LSE 合并复用 `merge_state`(自然对数帧)。

- **正确性(`test/manual/test_fold_decode_gt.py`,5/5 OK):** ① window flash-decode == 对
  [sink+window] 的整段 softmax(rel<1e-4);② fold-decode(GPU triton)== C0 python
  `eval_budget[...]["fold"]`(df=d/elu1,decay∈{1.0,0.999},B∈{256,512,1024},rel<5e-3);
  ③ needle:`cos(fold,full)>cos(drop,full)`(fp32+bf16)且 `sig_fold>sig_drop`(fp32),
  needle 落在被折叠的中段;④ state_readout kernel==torch 参考;fused==torch 参考 rel<2e-3。
- **微基准(`test/manual/bench_fold_decode.py`,bf16,H=16 d=dv=128):** fold-decode 时延
  ~O(B + d·dv) 对 N 恒定,dense decode O(N)。**fold 全面胜**(最小 1.83x@N=4096/B=2048):
  batch=1 N=65536/B=256 → 81x;batch=64 → 150x;batch=256 N=65536 dense KV OOM(fold 靠容量胜)。
  所有 B≤2048 的 crossover N 都 <4096。
- **踩坑:** ① 微基准 batch=64/N=65536 触发 triton int32 偏移溢出(8.6e9>2³¹)→ 指针算术全部
  转 int64。② needle 测试 bf16 一度 fold<drop:`make_needle` 在两次 dtype 迭代间消耗 RNG 不同 →
  入口 reseed 保证同一 canonical needle,并用更稳的 cos 判据。

#### C2-b —— 忠实 serving fold-eviction(真 Qwen3-Next-80B TP=2)⚠️ **负边界**

**决策(用户选定):忠实输出 + 投影容量。** decode 时把每个全注意力层输出替换为
`merge(window[sink+recent], state_readout(fold(middle)))`,KV 池**只读、不放页**(env 未设即字节
一致);O(B) decode 时延 + max-batch/容量赢法由 C2-a 微基准 + 精确 KV 足迹(B vs N)**投影**,不靠放页测。
产物:`srt/debug/fold_evict.py`(env `FOLD_KV_BUDGET` 门控,unset→no-op)+
`hybrid_linear_attn_backend.py` decode 钩子 + `test/manual/run_fold_evict_e2e.sh`。

**needle A/B(doc_words=3000,depth=0.5,groups=16×4,n=64,B=1024,sink=4,decay=1.0,φ=elu1,
12 个全注意力层全折,--disable-cuda-graph/-overlap):**

```
mode        retr_acc  hit_frac  ans==dense
baseline      1.000     0.98       1.000
fold          0.000     0.98       0.000
```

- **钩子在全部 12 个全注意力层(L3/7/…/47)干净触发,无 NaN/崩溃** —— 模型正常出词但答错,是
  **忠实执行的负结果**,非 bug。
- **判定 —— C2-b 负边界:** 用**固定/等权衰减**折叠全注意力层中段,深 needle 召回 **1.000→0.000**,
  与计划"recall(fold)≈dense"的预期相反。这**正好印证 C1**:固定衰减把深层内容清零(等权把 needle
  信号淹没在 ~3000 个中段 token 里),而 C1 里救回 needle 的是**模型自己的逐 token GDN 门控** ——
  **全注意力层根本没有这样的门控**(门控在 GDN 线性层上),所以在全注意力层折叠没有救回机制。呼应
  Phase A2:全注意力是**因果来源**、折叠它就丢 needle;GDN 线性态才是**可降级层**(本阶段正确地不碰它)。
- **诚实标注:** ① 本次只测了固定 decay(1.0);把捕获的真实 GDN g 喂进全注意力 fold **在语义上不成立**
  (那是 GDN 层的门控,不是全注意力层的),所以"用真实门控救回"这条路对全注意力层不适用。② 因召回已
  归零、配置不可用,**未跑 live perf 扫描**(TTFT/TPOT 投影中性,且用户需紧急切分支);O(B) 时延/容量
  收益仍以 C2-a 微基准为准。③ kernel 侧(C2-a)是干净的正向结果,可复用于纯 GDN(非混合)模型的线性态折叠。

**产物:** `test/manual/{fold_decode_kernel,test_fold_decode_gt,bench_fold_decode}.py`(C2-a)、
`srt/debug/fold_evict.py` + `hybrid_linear_attn_backend.py` decode 钩子 + `test/manual/run_fold_evict_e2e.sh`(C2-b,均 EXPERIMENT-ONLY,env 未设即字节一致)、`e2e_fold_evict_logs/`(结果)。

## Phase D —— 统一系统 + 端到端(层选择性 fold)🟨 有界正结果(GATE-D)

**问题(RQ4/D3):** C2-b 证明**全折 12 个全注意力层**把深 needle 召回清零(1.000→0.000),因为固定衰减把
深层内容淹没、而全注意力层没有 GDN 那样的逐 token 门控可救回。但 **A1** 已因果证明:12 个全注意力层里,
检索**只由最后 3 层 {39,43,47}** 承担,前 9 层 {3,7,11,15,19,23,27,31,35} 与检索无关。**Phase-D 命题:**
A1 标定的"可降级层"恰好就是 fold 安全的层 —— 只折前 9 层、保留后 3 层精确,应能把召回救回 ≈dense,
同时仍折掉 9/12 的全注意力 KV。这把 A1 的**因果预测器**从消融机制搬到真实 fold 机制上验证。

### 配置

- 模型 / serving / 探针:与 C2-b 完全一致(Qwen3-Next-80B-A3B-NVFP4,TP=2,mem-frac 0.7,extra_buffer,
  triton,`--disable-overlap-schedule --disable-cuda-graph`,PORT 31017;needle doc_words=3000、depth=0.5、
  16×4=64;fold B=1024、sink=4、decay=1.0、φ=elu1)。**唯一变量 = `FOLD_KV_LAYERS`**。
- **零新代码逻辑:** `srt/debug/fold_evict.py` 早已支持 `FOLD_KV_LAYERS`(`_read_layer_set` / `should_fold_layer`);
  `needle_longrange.py` 以 `--mode` 作任意 arm 标签;`run_fold_evict_e2e.sh` 早已透传 `FOLD_KV_LAYERS`。
  本阶段**仅在驱动脚本里加三个 arm**(`fold_safe` / `fold_unsafe` / `perf_fold_safe`),各自默认层集但保持
  env 可覆盖;其余(启动 flag、needle 旋钮、report)不变。EXPERIMENT-ONLY,env 未设即字节一致。
- **三个 arm:**
  - `fold_safe` —— `FOLD_KV_LAYERS=3,7,11,15,19,23,27,31,35`(折 9 个检索无关层,保留 39/43/47 精确)。
  - `fold_unsafe`(对照)—— `FOLD_KV_LAYERS=39,43,47`(只折 3 个检索层 → 预测崩塌;把 A1 的晚带从消融搬到 fold)。
  - `perf_fold_safe` —— 同 fold_safe 层集,跑 bench_serving decode 扫描。

### 过程

1. `fold_safe` arm 起独立 server,needle 探针。`[fold_evict]` 钩子在 **{3,7,11,15,19,23,27,31,35}** 干净触发,
   **不碰 {39,43,47}**(server 日志核对)。
2. `fold_unsafe` arm 起独立 server,needle 探针。钩子在 **{39,43,47}** 触发。
3. `report` 汇 4 行表(以 baseline 为参照)。
4. D2 perf:`perf_baseline` + `perf_fold_safe`,random in=256 / out=512,conc {1,16,64}。

### 结果

```
=== Stage D1:长上下文直接检索(以 baseline 为参照),每 arm 64 次 ===
mode                       retr_acc  hit_frac  ans==dense   FOLD_KV_LAYERS
baseline                     1.000     0.98       1.000      (fold OFF,dense 上界)
fold(全 12 层,来自 C2-b)   0.000     0.98       0.000      3,7,…,47
fold_safe                    0.922     0.98       0.922      3,7,11,15,19,23,27,31,35   <- 救回
fold_unsafe                  0.000     0.98       0.000      39,43,47                   <- 崩塌(对照)
```

```
=== Stage D2:decode 扫描 A/B(random in=256/out=512,单位 ms / tok·s⁻¹)===
conc  arm         TTFT_med  TPOT_med   out_thr  total_thr
 1    baseline     106.13     67.39     14.80     20.15
 1    fold_safe    105.98     68.09     14.65     19.95
16    baseline     162.70     72.58    195.70    297.71
16    fold_safe    164.60     74.18    193.09    293.74
64    baseline     161.36     90.45    654.06    998.90
64    fold_safe    162.30     89.70    654.53    999.62
```

### 判定 —— **GATE-D 有界通过(A1 因果预测器迁移成立)**

- **双向解离,决定性。** 只折 A1 标定的 **9 个检索无关**全注意力层 → 召回从 C2-b 全折的 **0.000 救回
  0.922**(≈dense);反过来**只折 3 个检索层** {39,43,47} → 召回 **0.000** 崩塌。两个 arm 的 `hit_frac`
  均 0.98(同缓存命中)⇒ 因果、非相关。这把 A1 的因果预测器从**消融**(A1:遮住晚带→0.0、遮早/中带→1.0)
  一比一搬到了**真实 fold 机制**上:**能安全 fold 的层 == A1 说检索无关的层**。C2-b 的负结果由此转成
  Idea 2 在混合模型上的**有界正结果**。
- **诚实边界:** ① `fold_safe` = 0.922 < dense 1.000,残留 **~8% gap**,略低于 ~0.95 目标 —— 固定/等权衰减
  fold 即使在"安全"层上也引入了微小扰动,是**有界正结果、非完全无损**(不同于 Idea 1 no-recon 的 ==dense)。
  ② `fold_unsafe` 精确复刻 A1 晚带的因果地位:检索由后 3 层承载,fold 它们等价于消融它们。
- **D2 perf = 忠实/投影,非实测容量。** perf 负载 in=256/out=512 ⇒ 最大 seqlen 768 **< 预算 B=1024**,fold
  的早退分支(`if not any(n > _BUDGET): return out`)使**没有一步真正 fold**(server 日志零 `[fold_evict]`),
  故 `fold_safe` 三档并发均与 baseline 落在 **~1-2% 噪声带内**(TTFT 106/165/162 vs 106/163/161,TPOT 68/74/90
  vs 67/73/90,吞吐同量级)。**结论:env-gated 钩子在欠预算快路径上零开销、字节一致;真正的 O(B) decode 时延
  与 max-batch/容量收益仍由 C2-a 微基准**投影**(fold ~O(B+d·dv) 对 N 恒定 vs dense O(N),1.83~174x),
  与 C2-b"KV 池只读、容量投影"的决策一致。诚实范围:fold 覆盖 9/12 全注意力层,后 3 层 + 36 个 GDN 层保持
  精确/原生 ⇒ **部分容量,非全量**。

### 统一系统框架(D1 of the plan)

混合状态分层系统 = **Idea 1**(免重建前缀缓存,作用于 GDN**可降级层**,Phase B 已在真 80B 验证:1.33~1.60x
容量 / 0.92x TTFT / 精度中性)**+ Idea 2**(选择性 fold,作用于 A1**检索无关**的全注意力子集,recall 0.922)。
两者由**同一个 A1 因果预测器**统辖:no-recon 安全,因为 GDN 线性态可降级;fold 安全,因为前 9 个全注意力层
检索无关;而**后 3 个全注意力层 {39,43,47} 是两者共享的"必须保精确"核心**。

### 范围外(诚实)

- Kimi-Linear / KDA fold(逐 K 门控与 fold 相斥,ReplaySSM 0.7~0.9x 回归)。
- LongBench / GSM8K 全精度矩阵与多预算扫描(计划的宽 D2)—— 给定 C2-b,needle 召回 + 一次 perf 扫描已是决定性
  信号;需要全矩阵再展开。
- 带外放页 / 真实容量实测(C2-b 决策 = 投影)。

**产物:** `test/manual/run_fold_evict_e2e.sh`(+3 arm,唯一代码改动)、
`e2e_fold_evict_logs/`(`fold_evict_needle.json` 四行表 + `perf_*_conc*.log` + `server_*.log`)。

---

## Phase G —— Idea 4:因果驱逐优先级(value vs LRU)✅ 通过(GATE-G)

### 动机 —— 从"可降级层"到"可重排驱逐"

Phase A2 证明:**mamba(GDN)线性态是可降级层**(清零线性态,混合模型召回仍 1.0)。把这个结论从"计算"迁移
到"缓存管理":radix 里缓存的 **mamba checkpoint(前缀共享的 GDN 状态 S)也是可降级/可重建层** —— 驱逐一个
checkpoint 是**召回中性**的(下次命中 miss ⇒ 精确重算该前缀,答案不变)。既然驱逐谁都不影响正确性,就有自由度把
victim 选择从**纯 recency(LRU)**换成**按价值(复用频率)**,在偏斜(Zipf)前缀流行度下保留最热前缀 ⇒
换取更高的 cached-prefix 命中率,**不换任何精度**。

这是与 Idea 1/2 正交的第三条线:Idea 1/2 动的是**读取/折叠**,Phase G 动的是**驱逐策略**;都被同一个
"线性态可降级"的 A2 因果结论授权。

### 实现(实验专用、env 门控、未设时字节一致)

- 新 env `SGLANG_MAMBA_EVICT_POLICY`(`environ.py`,`EnvStr("lru")`):`"lru"` = 原纯 recency 路径(**字节一致**);
  `"value"` = 按价值驱逐。
- `mamba_radix_cache.py`:
  - `TreeNode.hit_count`(原本未用)在 `match_prefix` 命中且节点持有 mamba_value 时 +1 —— 复用频率计数。
  - `evict_mamba` 在 `policy=="value"` 时分派到新 `_evict_mamba_value`:对当前**未锁定**的 mamba 节点建
    `(hit_count, last_access_time, id)` 小根堆,**先驱逐价值最低**者(复用最少,recency 破平);复用既有的
    内部节点 tombstone / `_evict_leaf_node` 机制;对已被联动释放的陈旧堆项用 `in_list()`/锁重检跳过。
    `"lru"` 路径原样保留。
- 驱动 `bench_gdn_prefix_hitrate.sh` 增 `MAMBA_EVICT_POLICY` 透传(所有 arm 统一,公平 A/B)。

### GATE-G1 —— 离线门控(纯 python,无 GPU)✅ 通过

`test/manual/test_mamba_evict_policy.py`,直接跑真 `TreeNode`+`LRUList`+`evict_mamba`(仅桩掉 GPU 物理释放):
①LRU 按纯 recency 驱逐最老(即使最热);②value 按最低频率驱逐;③**recency 与 frequency 反相关**(热前缀早
缓存)时 value 保留 155 复用质量 vs LRU 仅 55;④锁定 checkpoint 永不被 value 驱逐;⑤value 正确 tombstone
低价值内部节点。全部 PASS。

### GATE-G2 —— 真 80B 端到端 A/B(命中率/TTFT)✅ 通过

真 Qwen3-Next-80B-A3B-NVFP4,TP=2,headaware_a(route A,extra_buffer,NORECON=1,triton,`--disable-overlap-schedule`)。
关键是让 **mamba checkpoint pool 饱和且不饥饿**:pool 槽位须 > 并发(锁定集),又 < 活跃不同前缀数(才会驱逐)。
诚实踩坑记录:pool=16 + conc32 ⇒ `Can not alloc int8 mamba checkpoint slot` 断言(16 槽全被在飞请求锁定,
驱逐无可释放)⇒ **pool 必须 > 并发锁定集**。最终工作点:conc=8、pool=40~48、256/200 个不同 Zipf(α=1.2~1.3)
前缀、sys=1024、`generated-shared-prefix`。两个独立工作点(evict_ab3/ab4),输入 token 总量逐 arm 完全一致:

| 工作点 | 策略 | token 命中率 | TTFT 均值 | TTFT p90 | E2E 均值 | 吞吐(tok/s) |
|--------|------|-------------|-----------|----------|----------|-------------|
| A(ng256/ckpt48/α1.3) | LRU   | 0.5035 | 430.4 | 599.9 | 538.2 | 18120 |
| A                     | value | **0.6080** | **358.3** | **474.0** | **458.0** | **21286** |
| B(ng200/ckpt40/α1.2) | LRU   | 0.4315 | 436.3 | 594.9 | 548.8 | 17742 |
| B                     | value | **0.5485** | **367.5** | **501.4** | **468.2** | **20790** |

两点一致:**命中率 +10~12pt(相对 +21~27%)、TTFT 均值 −15~17%、p90 −16~21%、吞吐 +17%**。机制正如预测:
Zipf 倾斜下 value 保留热前缀 checkpoint ⇒ 更多 `#cached-token` ⇒ 更少重算前缀 ⇒ 更低 TTFT / 更高吞吐。

### 判定 —— **GATE-G 通过(实测时延/吞吐正结果)**

- **这是 Idea 1/2 之外少见的 *实测*(非投影)正结果**:因为动的是纯软件驱逐策略,不涉及 kernel/容量投影,可直接
  端到端量化。
- **诚实边界:** ① 收益是**命中率/TTFT/吞吐**,**不是精度** —— 混合模型上驱逐召回中性(这正是 A2 授权的前提);
  精度可判别的版本仍缺**纯 GDN 正对照**(与 Idea 1 同一 gap)。② 收益只在 **pool 饱和 + 前缀流行度偏斜**时出现;
  pool 不饱和(如早期 pool=160 峰值仅用 ~22 槽)则 `evict_mamba` 基本不触发,两策略等价。③ 工作点须满足
  `pool > 并发锁定集` 否则饥饿崩溃(已记录)。

**产物:** `environ.py`(+1 env)、`mamba_radix_cache.py`(hit_count 计数 + `_evict_mamba_value`)、
`test/manual/test_mamba_evict_policy.py`(GATE-G1)、`bench_gdn_prefix_hitrate.sh`(+env 透传);
bench 日志 `gdn_prefix_hitrate_logs/evict_ab{3,4}/`(未入库)。

---

## Phase H —— 纯 GDN 正对照:打破准确率天花板 ⬜(代码就位待跑)

### 动机(为什么)

到目前为止,所有 state-tier 结果(Idea 1 no-recon、Idea 2 fold、Idea 4 value 驱逐 / Phase G)
在混合 Qwen3-Next 上**只能被展示为准确率中性**:dense == no-recon == fold-safe == value-evict ==
**1.000** needle 召回。这就是**准确率天花板问题** —— 度量被钉在 1.0,因为 12 个全注意力层承载长程检索,
GDN 态是一个真正**可降级**的层,碰它永远不动精度。我们反复展示了因果预测器的**负臂**("GDN 态可降级
→ 策略中性"),却从未展示**正臂**("当态确实承载因果信息时,尊重它的策略会**提升**精度")。这一 gap 在
memory 里被标记了 3 次:*"精度可判别的版本仍缺纯 GDN 正对照"*。

**Phase H 关闭它。** 在**同一个模型**上造出一个 GDN 态是所检索事实**唯一**载体的区域 —— 用已提交的
`state_ablation.corrupt_full_kv` 挂钩(`ABLATE_FULL_KV=zero`,挂在
`hybrid_linear_attn_backend.py:821,866`)**致盲全注意力层**对 prefix 的读取。在这个纯 GDN 区里天花板消失,于是:
(1) 把事实移出 GDN 的递归视野会**降低**精度 = GDN 承载的正对照;(2) 一个重建 GDN 态的 recon 策略会
**找回** no-recon 丢掉的精度 = **用户要的"提升精度的策略"**。复用已提交的 needle 探针 + recon 旋钮,
无需下载新模型。

### 决定性设计选择 —— 扫 needle 的**距 query 距离**,而非文档长度

关键风险是 Qwen3-Next 的 GDN 在**任何**深度都做不到精确检索(线性注意力弱于精确召回,模型学会把检索路由到
全注意力)。我们通过扫 needle **距 query 的距离**来降低这个风险:`needle_longrange.py --depth` 把 needle
插在文档的 `depth` 比例处,问题出现在**整篇文档之后**。于是 `depth→1.0` = needle 紧邻 query(在 GDN 递归
窗内,几乎必然保留),`depth→0.0` = needle 很远(已从 GDN 态衰减掉)。全注意力致盲后,GDN-only 召回应画出
一条 **近高→远低** 的曲线,而混合 baseline 在所有深度都 ~1.0。这个对比把**近上下文记忆因果定位到 GDN、
远上下文定位到全注意力** —— 一个近乎必然存在的正对照(GDN 总保留紧邻的 token),而不是赌一个魔法文档长度。

### 复用(不重写)

- `python/sglang/srt/debug/state_ablation.py` —— `ABLATE_FULL_KV=zero|noise`(逐 forward snapshot→
  破坏→恢复,非持久)+ `ABLATE_FULL_KV_LAYERS`(层白名单);`corrupt_full_kv` 已挂在
  `hybrid_linear_attn_backend.py:821,866`。`ABLATE_LINEAR_STATE=zero` + `corrupt_linear_state_`(GDN
  ssm_state 清零)此前**函数存在但未接入 kernel 路径** —— Phase H2 补上(见下)。
- `test/manual/needle_longrange.py` —— 每组唯一 5 位码的直接检索探针(`--depth`/`--doc-words`/`--groups`/
  `--k-per-group`,温度 0),输出 `retrieval_acc` + `cache_hit_frac`。
- `test/manual/run_state_ablation_2x2.sh` —— arms `baseline|ablate_linear|ablate_full_kv|ablate_both`,
  每 arm 一份 server,`report` 打印 2×2 —— **就是 H2 驱动,现成可用**(一旦线性挂钩接上)。
- `test/manual/bench_gdn_prefix_hitrate.sh` —— 底层启动驱动(headaware_a route-A 旗标,PORT 31007,
  TP2);继承 shell 环境 → 导出 `ABLATE_*` 即可组合。H3 需要它逐 forward 跑致盲挂钩 → **新增
  `DISABLE_CUDA_GRAPH=1` env 旋钮**(与 `DISABLE_OVERLAP` 平行,未设时字节一致)。

### H2 一行接线(仅实验用,`ABLATE_LINEAR_STATE=off` 时字节一致)

在 GDN kernel 读 `ssm_states` 的入口(`linear/gdn_backend.py` 的 `forward_decode` 与 `forward_extend`,
`ssm_states` + `mamba_cache_indices` 都在手处)加一行 env 门控调用:

```python
if state_ablation.linear_state_enabled():
    state_ablation.corrupt_linear_state_(ssm_states, cache_indices)
```

镜像已提交的 `corrupt_full_kv` 接线风格;未设 env 时是 no-op。`decode` 路径插在 `cache_indices` 解析后、
conv/SSM kernel 读态前;`extend` 路径插在 `ssm_states = mamba_cache_params.temporal` 后、把它当 chunk
初始态读入前。

### Stage H1 —— 关键探针:GDN-only 检索到底能不能行?(GATE-H1)

**无新后端代码。** 新驱动 `test/manual/run_gdn_only_depth_sweep.sh`:
- 中等文档(`NEEDLE_DOC_WORDS≈800`,小到 GDN 视野有望覆盖),`NEEDLE_GROUPS≈12`,`NEEDLE_K≈4`,
  `DISABLE_OVERLAP=1`,`--disable-cuda-graph`(致盲挂钩逐 forward 必须跑)。
- 两次 server 启动(dense 权重,head-aware **关**,纯消融):
  - arm `hybrid`   : `ABLATE_FULL_KV=off`  → 期望每个深度 acc ~1.0(全注意力承载)。
  - arm `gdn_only` : `ABLATE_FULL_KV=zero` → GDN-only 曲线。
- 每 arm 循环 `NEEDLE_DEPTH ∈ {0.95,0.85,0.7,0.5,0.3,0.1}`,每深度一次探针(tag `${arm}_d${depth}`),
  追加到一份 JSON;内嵌 python reporter 打印两 arm 的 acc-vs-depth 表并**自动建议操作深度 d\***。

**GATE-H1(成败关键):**
- **成功:** `gdn_only` acc **近 query 高、随距离衰减**(如 depth≥0.85 时 ≳0.5,depth≤0.3 时 →~0),
  同时 `hybrid` 保持 ~1.0 → GDN 承载区存在;取 `gdn_only` 落在判别中带(~0.4–0.8)的**操作深度 d\***,
  进入 H2/H3。
- **诚实回退:** 若 `gdn_only` 在**每个**深度都 ~0(连紧邻也是),则 Qwen3-Next 的 GDN 做不到精确检索 →
  记录为发现并**停止**(不伪造正结果);回退选项 =(a)换成 GDN 能承载的任务(短程召回/聚合,超出本计划
  默认范围),或(b)取一个真正的纯 GDN 模型。

### Stage H2 —— 显式因果 knockout 2×2(GATE-H2,加固)

确认近上下文记忆**具体就是** GDN 态(而非"存在别处")。用 H1 的操作点跑现成
`run_state_ablation_2x2.sh`(`NEEDLE_DEPTH=d*`,`NEEDLE_DOC_WORDS≈800`),arms
`baseline|ablate_full_kv|ablate_both|ablate_linear`。

**GATE-H2(双解离):** `baseline`~1.0;`ablate_full_kv`≈`gdn_only(d*)`(中带,>0);`ablate_both`→~0
(在致盲全注意力之上再清 GDN 态 = 摧毁最后的载体 = **正对照**:此处 GDN 态承载);`ablate_linear`≈1.0
(全注意力完好 → 中性 = 已知负臂)。这就是完成的正+负因果表。

### Stage H3 —— 提升精度的策略(GATE-H3,交付物)

在 GDN-only 区证明 **recon 策略找回 no-recon 丢掉的精度**。新驱动
`test/manual/run_gdn_only_recon_sweep.sh` = recon 扫描的薄覆盖层:
- 所有 arm 导出 `ABLATE_FULL_KV=zero`(GDN-only 覆盖)于 `NEEDLE_DEPTH=d*`,doc≈800,headaware_a route-A
  开(GDN 检查点在 prime→question 缓存边界被丢弃/重建)。
- arms:`norecon`(`NORECON=1`,GDN 态保持丢弃)vs `recon_full`(route-A full-W_max recon,重建 GDN 态);
  可选 `seam=16`(有损部分 recon,`RUN_SEAM=1`)。
- **先做组合冒烟:** norecon arm 先跑,确认 `ABLATE_FULL_KV=zero` + route-A recon 能跑完且缓存命中
  (探针打印 `cache_hit_frac`>0)。full_kv 的 snapshot/restore 逐 forward、与 GDN 检查点路径正交。

**GATE-H3(成功):** 在 d\*,`recon_full` acc **> `norecon` acc** 且差距明显(recon 向 hybrid 的 1.0 回收,
norecon 停在丢弃态地板)。这是 **提升精度的策略** —— recon 的首个非中性正结果,正是此前因缺纯 GDN 区而
无法检验的"recon 在纯 GDN 上有价值"。**诚实回退:** 若纯 GDN 区里 `recon_full`≈`norecon`,则 route-A recon
没有忠实重建 needle 承载态 —— 带数字记录 null(仍是真发现:recon 即使离开天花板也中性),不夸大。

### 诚实边界

- 片上全注意力致盲是**模拟**纯 GDN 区,不是原生纯 GDN 模型 —— 明说。真纯 GDN 检查点是更强外部对照,但在
  本机 cu12.9 天花板(base `753aa89a83`)上有下载/兼容风险 → 仅 H1 失败时回退。
- H1 真正可证伪 —— 若 GDN 任何深度都检索不到,计划在 H1 处以诚实 null **停止**,不伪造正结果。
- 一切仅实验用、env 门控,`ABLATE_*` / `NORECON` 未设时字节一致。不用 cuda graph(数据相关的消融),
  `--disable-overlap-schedule` 保留。这里的度量是**准确率**(retrieval_acc),非时延。

### 状态 / 产物

- **代码就位(2026-07-22):** H2 挂钩(`linear/gdn_backend.py` decode+extend 各一行 env 门控)、H1 驱动
  `test/manual/run_gdn_only_depth_sweep.sh`、H3 驱动 `test/manual/run_gdn_only_recon_sweep.sh`、
  `bench_gdn_prefix_hitrate.sh` 新增 `DISABLE_CUDA_GRAPH` 旋钮。
- **结果落地:** `state_ablation_logs/gdn_only_depth_sweep.json`、`.../gdn_only_recon_sweep.json`、
  `.../state_ablation_2x2.json`(未入库);数字回填本文 + memory `phase-h-pure-gdn-positive-control.md`。

### GATE-H1 结果 —— 🟥 诚实 NULL(2026-07-22,真 80B TP2)

配置:Qwen3-Next-80B-A3B-NVFP4,TP2,extra_buffer,triton,`--disable-overlap-schedule`,
`--disable-cuda-graph`,doc 800 词、12 组 × 4 问 = 48 试验/深度、温度 0。两 arm 各起一份 server。

| depth | 距 query | hybrid acc | gdn_only acc | hit_frac |
|-------|----------|-----------|--------------|----------|
| 0.95 | near | 1.000 | **0.000** | 1.00 |
| 0.85 | near | 1.000 | **0.000** | 1.00 |
| 0.70 | near | 1.000 | **0.000** | 1.00 |
| 0.50 | mid  | 1.000 | **0.000** | 1.00 |
| 0.30 | far  | 1.000 | **0.000** | 1.00 |
| 0.10 | far  | 1.000 | **0.000** | 1.00 |

**判定:GATE-H1 未通过。** `gdn_only` 在**所有深度(含紧邻 d=0.95,needle 距 query 仅 ~80 token)均为 0**,
没有 near→far 衰减曲线可言,取不到判别中带的 d\*。缓存 100% 命中(探针路径正常);`hybrid` 恒 1.000
(消融确实生效:致盲把 1.0 打到 0.0,与 A2 一致)。

**关键 nuance(比纯 null 更有信息量):** 逐条看 `gdn_only` 输出,失败模式是**模型坍缩**而非干净的
"GDN 缺 needle" —— 输出退化为重复串(`State State State…`、`( ( (…`、`1111111111`、`What What…`),
而同一探针/缓存路径下 `hybrid` 输出干净的正确码(`10271`)。这说明**致盲全部 12 个全注意力层的 prefix KV,
移除的远不止 needle,而是模型做上下文整合、维持基本连贯所依赖的全部长程注意力** —— Qwen3-Next 被联合训练成
依赖全注意力做通用上下文处理,不只是检索。

**结论:** 片上全注意力致盲**不是**一个干净的纯 GDN 模拟 —— 度量无法区分"GDN 态里没有 needle"与
"模型整体坏掉"这两种解释,正对照被污染。按计划在 H1 处**以诚实 null 停止**,**不**进入 H2/H3
(二者以 GATE-H1 成功为前提)。这本身是一个真发现:**在混合模型上,全注意力 prefix KV 对基本连贯性
(非仅检索)是因果承载的**,因此不能靠清零全注意力来"关掉"它。要拿到精度可判别的正对照,需要:
(a)**真正的纯 GDN 模型**(如 GatedDeltaNet 检查点;本机 cu12.9 base `753aa89a83` 有下载/兼容风险),或
(b)改用 GDN 本就能承载的任务(短程召回 / 聚合类,而非精确 5 位码检索)—— 两者都超出本计划默认范围,
待与用户确认后再定。

**产物:** `state_ablation_logs/gdn_only_depth_sweep.json` + 各 arm `server_gdnonly_*.log`、
`needle_gdnonly_*.log`(未入库)。代码(H2 挂钩 + H1/H3 驱动 + `DISABLE_CUDA_GRAPH` 旋钮)保持
**未提交**(GATE-H1 未通过 → 按提交规范暂不入库)。

---

## 探索 —— KDA 逐通道衰减画像:local/global 是双峰吗?(离线,零风险)

### 动机 —— 把 head-aware 从 GDN 迁到 KDA 的前提验证

GDN head-aware prefix cache 依赖一个**干净的逐 head local/global 划分**(GDN 衰减 = 逐 head 标量
`g_h = -exp(A_log_h)·softplus(a+dt_bias_h)`,tau_h 逐 head 可分)。**KDA 的衰减是逐 K 通道**(elementwise):
`g[t,h,k] = -exp(A_log[h])·softplus(a[t,h,k] + dt_bias[h,k])`,其中 `A_log`=[H] 逐 head 标量、
`dt_bias`=[H·K] 逐 (head,通道)、`a`=`f_b_proj(f_a_proj(x))`=[T,H,K] 输入相关。所以要回答用户的
问题"**KDA 不好区分 local/global,有什么办法**",先要弄清:(head,通道) 的衰减是**双峰分离**(存在硬通道
阈值)还是**连续**的?

Kimi-Linear-48B-A3B-Instruct:27 层,20 KDA + 7 全注意力 {4,8,12,16,20,24,27};H=32,K=head_dim=128
⇒ 20·32·128 = 81920 个通道。度量:保留率 `alpha=exp(g)`、记忆视野 `tau=ln(eps)/g`(token);
Sarle 双峰系数 `BC=(skew²+1)/kurtosis`,`BC>0.555` 判为双峰。

### 工具(新增,未入库,纯离线 CPU/单卡)

`test/manual/profile_kda_decay.py` —— GDN `profile_gdn_decay.py` 的 KDA 逐通道版。两种模式:
- `weights`:从 safetensors 读 `A_log`[H] + `dt_bias`[H·K],在**常数 a 括号**(a=0 / a=1 / a=2)下算
  静态骨架分布。a=0 曾被当作"最少遗忘上界"。
- `empirical`:HF transformers 真跑 48B,钩住全部 20 个 KDA 层的 `f_b_proj` 抓真实逐 token `a`,逐通道对
  token 求均值 → 真实输入下的 (head,通道) 衰减。

**跑通 48B 的 6 个兼容补丁**(cu12.9 + transformers 5.6.0 + 本地 fla 源
`/home/…/flash-linear-attention`,全部在 `profile_kda_decay.py` 内):①`OutputRecorder` no-op shim;
②载后强制所有 config `_attn_implementation="eager"`(modeling 硬塞 flash_attention_2 会在 `s_aux=None.to()`
崩;全注意力是 MLA,eager 原生支持);③`fused_kda_gate` 签名适配器(modeling 用旧 API `(g,A_log,head_dim,
g_bias=dt_bias)`,fla 源是 `(g,A_log,dt_bias=,lower_bound=,…)`)+ 把 g `[…,H·K]→[…,H,K]` reshape(长序列走
`chunk_kda` 断言 4-D g);④`device_map={"":0}` 单卡(48B bf16 ~94GB 进一张 143GB H20,免跨卡 gemm);
⑤`use_cache=False`(免 Kimi 混合缓存 `get_mask_sizes` int-vs-tensor 崩);⑥`PYTHONPATH` 指向本地 fla 源
(pip wheel 是无 `fla.ops` 的命名空间桩)。

### 结果 —— 3 个诚实发现(以 12 条多域长 prompt × 558 tok = 6696 tokens 的**可信** empirical 跑为准)

> 说明:先跑 5 条短 prompt(≤37 tok,走非 chunk 容忍 3-D g 的路径,轻微偏差);随后**修正 g reshape**
> 后用 12 条 diverse 长 prompt 重跑(正确 chunk-mode 门控),分位数坐实。3 个发现两跑一致,量级在长跑更长。

1. **逐通道衰减是 UNIMODAL/连续,不是双峰。** `BC(log10 tau)=0.422`(<0.555;单个宽驼峰,峰在
   log10 tau≈1.4 即 tau≈25,右侧长尾)。→ 对"是不是双峰"的回答是**否 —— 连续,无硬通道阈值**。
   (`BC(alpha)`/`BC(g)` 读数 >0.555 只因 alpha 被界在 [0,1] 两端堆积;记忆视野的正确度量 log10 tau 是单峰。)
2. **KDA 记忆整体偏 SHORT,但真实输入下尾巴比静态骨架重。** tau 分位:p25=11、p50=45、p75=260、p95=2010、
   max≈1.16M。local 覆盖率:55.8%@w64、66.6%@128、74.8%@256、82.2%@512、88.9%@1024、95.1%@2048、
   97.9%@4096。**"a=0 = tau 上界"的假设是错的** —— 真实 `a` 会取**负值** ⇒ 更少遗忘 ⇒ 尾巴比任何
   常数 a≥0 骨架都长(极端尾到 ~1M token)。
3. **逐 head 可分性在真实输入下坍塌 —— 关键修正。** 静态权重骨架曾显示逐 head local-fraction 呈 U 形
   (57~70% 近纯 head);**empirical 在判别窗 w=45(整体 50% local)下只有 9.4% head 是纯的、90.6% 是
   MIXED**,逐 head local-fraction 直方图近**扁平**(非 U 形)。原因:静态骨架 head 内只变 `dt_bias`(逐 head
   `A_log` 标量主导 → head 聚成两簇);真实输入相关 `a` 给 head 内加了大的逐通道离散度 ⇒ 几乎每个 head 都
   同时握**快 + 慢**两类通道。**所以 KDA 的 local/global 是真·逐通道,而非逐 head —— GDN 式逐 head 分层不迁移。**

### 结论 / 策略(回答"KDA 不好区分 local/global,有什么办法")

- **不要**按硬通道阈值切 local/global(衰减连续,BC=0.31~0.42),**也不要**逐 head keep/drop(真实输入下
  90.6% head 混合)。
- 正确设计 = **逐通道 SOFT 窗口,窗口 = 每个通道自己的 tau**(多数很小:p50=22~45、p95=314~2010)。一个
  w≈512–2048 的窗口覆盖 82~95% 通道,只有 ~5% 长尾(tau>2048)需要更长保留或精确保存。比权重骨架的
  故事更干净:一条连续的逐通道视野,而非二元分层。

### 诚实边界 / 后续

- empirical 仅 12 条 prompt(6696 tok),定性发现(通道单峰、head 混合)稳健,分位数可再加长/加多样 prompt
  坐实。工具 + 发现均**未入库**(探索性)。
- **重要迁移前提:唯一可用的真·逐通道 KDA 模型 Kimi-Linear 本身是混合架构(7 全注意力层)** —— 与
  Qwen3-Next 同样存在**准确率天花板**:混合模型上全注意力承载长程检索,重建 KDA 态在精度上多半**中性**
  (no-recon == recon == dense)。因此逐通道 soft 窗口 recon 的精度收益需要**纯 KDA 正对照**才能显形
  (与 Idea 1 / Phase H 同一 gap;Phase H 片上致盲模拟已因模型坍缩失败)。可直接落地且诚实的 KDA 实验是:
  (i) **Idea 4 value 驱逐**(`SGLANG_MAMBA_EVICT_POLICY=value`,衰减无关、模型无关,预期复现 Phase G 的
  命中率/TTFT/吞吐实测正结果),(ii) **Idea 1 no-recon**(丢弃 KDA 态 + 前缀复用,衰减无关,预期与 GDN
  同样容量正、精度中性)。逐通道 recon 是大改动且在混合 Kimi 上大概率撞天花板,列为后续可选。

---

## Phase K —— KDA 逐通道 head-aware:实现与离线门(GATE-K1)✅ 通过(2026-07-22)

### 动机 —— 把 head-aware 前缀缓存从 GDN(逐 head)迁到 KDA(逐通道)

GDN 的衰减是**逐 head 标量**,keep/drop 单元 = 一个 head 的整块态,head-aware 缓存天然逐 head。KDA 的衰减是
**逐 K 通道 elementwise**(见上"KDA 逐通道衰减画像":`BC(log10 tau)=0.42` 单峰、90.6% head 混合),
逐 head keep/drop **不迁移**。所以 KDA 的 head-aware 必须**逐通道**:keep/drop 单元 = 一个全局
`(head, d_k 列)` 对 = 一个 `d_v` 向量。KDA 态 `S_h` 形状 `[d_v, d_k]`,逐通道衰减作用在 `d_k` **列**上;
`KimiLinearStateShape.temporal = (num_heads, head_dim, head_dim)`(末轴 = `d_k`,`d_v==d_k==128`)。

### 实现(实验专用、按张量形状分派、GDN 字节一致)

全部改动集中在 `mamba_checkpoint_pool.py`(+`model_runner.py` 3 行注释),**靠 dt_bias 的宽度自动分派**,
未走 KDA 时 GDN 路径完全不变:

- `HeadAwarePlan`(msgspec.Struct)新增 5 个**默认字段**:`per_channel=False` / `w_chan` / `global_hk` /
  `GU_max`。GDN 构造时全部取默认 → 与旧路径字节一致。
- `build_plan` 分派:放宽形状断言到 `A_log.dim()==2 且 dt_bias.dim()==2 且行数相等`;取 `L,HV=A_log.shape`,
  若 `dt_bias.shape[1] != HV` ⇒ KDA(`K = dt_bias.shape[1]//HV`,断言 `K==d_k` 且 `route=="A"`)
  → `build_plan_per_channel`;否则(`==HV`)走原 GDN 逐 head 路径。**KDA 的 `A_log` 仍是逐 head `[HV]`,
  只有 `dt_bias` 是逐通道 `[HV*d_k]`**(`kimi_linear.py`:`projection_size=head_dim*num_heads`,head-major
  展平),`model_runner.py` 的 `.flatten()` 直接产出宽 dt_bias,无需 KDA 专用采集。
- `build_plan_per_channel`:逐层用 `gdn_gate(a_off[HV,K], A_log[:,None], dt3)` 算逐列 `tau`,
  `local = isfinite(tau) & (tau <= w_max)`;`w_chan[L,HV,K]` = 逐列窗口;`global_hk[L,GU_max,2]` 枚举
  `~local` 的 `(h,k)` 对(填充 -1);GDN 字段留占位。
- `HeadAwareCheckpointStore`:per-channel 分支分配 `state_buf_pc[L, num_slots, GU_max, d_v]`(**只存 global 列的
  `d_v` 向量**,这就是容量来源),`_u_valid = global_hk[...,0] >= 0`。
  - `_store_per_channel`:`picked = states[l][:, h_idx, :, k_idx]`(高级索引轴 1,3 → `[GU_max, N, d_v]`)存入。
  - `_load_per_channel`:逐层把有效 global 列 scatter 回 `out[l][:, h_idx, :, k_idx]`,返回
    `(out, w_chan>0)` 作 local 掩码 `[L,HV,d_k]`。
  - `copy_local_rows_from_scratch` per-channel 分支:从 re-prefill scratch 态按 `mask[l].nonzero()` 的
    `(head,列)` 逐列 masked copy-back 到 active 态(Route-A seam),GDN 逐 head 路径不动。

### GATE-K1 —— 离线正确性(纯 CPU/GPU-free)✅ 通过

`test/manual/test_kda_head_aware_prefix.py`(新增,零 GPU),合成 + 真实 Kimi 权重两路都过:

| 检查 | 合成 `L=3 HV=8` | 真实 Kimi-Linear-48B `20 KDA 层 HV=32` |
| --- | --- | --- |
| (1) 分类 local iff `tau<=w_max` | 1941/3072 local(exact) | 81779/81920 local、141 global(exact) |
| (2) round-trip:global 列位精确 / local 清零 / mask==`w_chan>0` | max\|dS\|=0、local=0、mask✓ | 同,全 0 / mask✓ |
| (3) 容量 bytes/slot vs dense | 0.59M vs 1.57M = **2.67x** | 1.19M vs 41.9M = **35.31x** |
| (4) GDN 非回归(dt_bias 塌成 `[L,HV]` → `per_channel=False`) | 仍精确 round-trip | 同 |
| (5) 逐通道 masked copy-back(真实 `copy_local_rows_from_scratch`) | OK | OK |

真实权重 **35.31x** 容量对应 **99.8% 列 local**(w=512)—— 与"KDA 记忆偏 SHORT"的 empirical 画像一致
(静态权重在 `a=-a_margin` 下 tau 更短,几乎全 local)。逐层 re-prefill 态重装配相对误差 3.9e-4(信息性;
Route-A 重建精度在真机上量)。**GDN 路径全程字节一致**(route A 3.37x、route B 1.21x 未变)。

### 诚实边界 / 后续(GATE-K2 待定)

- **离线机制已完备,serving A/B(GATE-K2)未跑且有真实模型级阻塞。** head-aware 池要复用共享前缀的 mamba 态,
  依赖 `extra_buffer` 缓存策略(`no_buffer` 只在整序列叶子快照 → 前缀复用 ~0)。**KimiLinear 不在
  `_support_mamba_cache_extra_buffer` 白名单**,且 KDA `extra_buffer` 有已知 0.63 精度 bug
  (`kda_backend.py forward_extend` 只返回 `core_attn_out`,无中间 SSM `h` track/restore)。
- 因此逐通道 head-aware 的**容量/命中率收益要落到 serving,需先解 extra_buffer**;而在混合 Kimi 上其
  **精度预期仍中性**(天花板)。可直接落地且诚实的 KDA serving 实验仍是 **Idea 4 value 驱逐**(衰减/模型无关)
  与 **Idea 1 no-recon**(`no_buffer` 全提示复用可绕开 extra_buffer bug,`no_buffer` KDA GSM8K=0.895 正常)。
- 代码 **env/形状门控**,未走 KDA(GDN)时字节一致;GATE-K1 通过后可入库,GATE-K2 待模型阻塞决策。
