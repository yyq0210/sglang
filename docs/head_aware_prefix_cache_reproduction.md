# 复现指南：head-aware / value-eviction 前缀缓存实验

> 面向对象：**零上下文的接手 agent / 工程师**。照本文可以在本机（2×H20，cu12.9）从头把
> `docs/head_aware_prefix_cache_paper.md` 里的实验全部重跑出来。论文讲「为什么 / 结论」，
> 本文讲「怎么跑 / 哪里会踩坑」。**先读完 §0～§2 再动手**，那几节全是血泪教训。

---

## 0. 一分钟全景

这套工作在 SGLang 上验证两个「感知记忆结构」的前缀缓存优化，跑在两个真实混合线性注意力大模型上：

| 缩写 | 模型 | 线性注意力类型 | 分类粒度 | 本机目录 |
| --- | --- | --- | --- | --- |
| **KDA** | Kimi-Linear-48B-A3B-Instruct | Kimi Delta Attention（20 KDA + 7 全注意力层） | 逐 (head, d_k 列) | `./Kimi-Linear-48B-A3B-Instruct` |
| **GDN** | Qwen3-Next-80B-A3B-Instruct-NVFP4 | Gated Delta Net（36 GDN + 12 全注意力层） | 逐 head | `./Qwen3-Next-80B-A3B-Instruct-NVFP4` |

- **Idea 1**：只精确保存「长记忆」通道，短记忆通道丢掉 → checkpoint 变小 → 容量倍增。精度中性。
- **Idea 2**：缓存满了淘汰「最不常复用」的（value）而非「最久没用」的（lru）→ 命中率↑、TTFT↓。召回中性。
- **Idea 2+**：cost-aware GDSF（频率 × 重建成本），value 是等长特例。离线机制成立，在线 ≈ value。

**这些收益全是 容量 / 命中率 / 延迟 / 吞吐，不是精度**。在混合模型上精度天然中性（天花板），
所以精度实验只能证明「无损」，不能证明「有提升」（要提升得有纯线性模型正对照，我们没有）。

分支：`yyq/gdn-head-aware-prefix-cache`。改动全是 env / 形状门控，不开开关时字节完全一致。

---

## 1. 环境天花板与强制 workaround（不做就跑不起来）

本机是 **CUDA 12.9 + torch cu129 + sgl-kernel 0.4.2.post2**，比上游 main 老。三条硬约束：

1. **基线 commit 必须是 `753aa89a83`**（2026-06-17）或其后本分支。更新的 main 需要 cu13-only 的
   sgl-kernel 0.4.4 + `flash_attn_with_kvcache(only_qv=...)`，在 cu12.9 上直接崩。
2. **放宽 sgl-kernel 版本检查**：每条命令都要 `export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`。
3. **FA3 strip hack（本地未提交改动，务必保留、不要提交）**：`python/sglang/jit_kernel/flash_attention_v3.py`
   里 `_call_fa3_kernel` 会剥掉 0.4.2.post2 内核不认识的 falsy kwarg（`only_qv`/`sinks`）后重试。
   Kimi/Qwen 的全注意力层会传这些参数，不打这个补丁会崩。

**代理**（下载 HF 权重 / GSM8K 数据集时需要，本地 server 流量要直连）：
```bash
export no_proxy='127.0.0.1,localhost' NO_PROXY='127.0.0.1,localhost'
export HTTP_PROXY=http://10.229.18.27:8412 HTTPS_PROXY=http://10.229.18.27:8412
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY"
```
默认环境 `no_proxy='*'` 会禁掉所有代理，务必按上面覆盖。

---

## 2. 六个必须知道的坑（每个都真实踩过，会直接毁掉结果）

1. **dense 对照 arm 的 OOM**：Idea 1 的 dense 参照是 `SGLANG_FORCE_HEAD_AWARE_WMAX=0`（全通道 global =
   不压缩），每个 checkpoint slot 约 **20 MB（未压缩）**。`--head-aware-mamba-ckpt-size 1200` 会要 ~23 GB，
   在 mem-fraction 0.85 下 OOM。**精度 A/B 只需缓存当前共享前缀，用 `CKPT_SIZE=200`（≈4 GB）即可**。
   （head-aware arm 因为压缩了，同样 slot 数省得多，不会 OOM。）

2. **推理模型的 token budget 截断 = 假低分（最隐蔽的坑）**：Kimi/Qwen 是 thinking 模型，AIME 一题
   thinking 链常要 2 万+ tokens。若 `max_new_tokens=8192`，**~70% 样本还在思考就被砍断**，pred_answer
   从思考中途瞎抓一个数字 → 精度腰斩（实测 dense AIME2025 0.28@8k vs 参考 ~0.6）。**必须用大 budget**。

3. **采样参数必须对齐参考**，否则绝对值对不上（下表是官方参考配置，务必照抄）：

   | 模型 | max_new_tokens | temperature | top_p | top_k | presence_penalty | repetition_penalty |
   | --- | --- | --- | --- | --- | --- | --- |
   | **KDA (Kimi)** | 262144 | 1.0 | 1.0 | -1 | 0 | 1 |
   | **GDN (Qwen)** | 65536 | 0.7 | 0.8 | 20 | 1.5 | 1 |
   | 上下文窗口 | Qwen 262144 | | | | | |

   `max_new_tokens` 是**上限不是每条都生成这么多**——多数样本推完自然 EOS 停下，只有极少不收敛的走长尾。

4. **驱逐池的大小要卡在中间**（Idea 2/2+）：
   - 池 > 并发数：否则在跑的请求把 slot 全锁死，报 `Can not alloc int8 mamba checkpoint slot`。
   - 池 < 不同前缀数：否则永不淘汰，策略空转（value/lru 无差异）。
   - 实测可用配方：**并发 8 / 池 40~48 / 不同前缀 200~256 / Zipf 倾斜**。

5. **命中率不能读 gauge**：`sglang:cache_hit_rate` 是瞬时/过期值，idle 后读到 0.0。正确做法是把服务日志里
   **所有 `Prefill batch` 行的 `#cached-token` / `#new-token` 累加**，`hit = Σcached / (Σcached+Σnew)`。
   而日志里的 `mamba usage: X` 是**活跃 bf16 池**占用，不是 checkpoint 池，别混。

6. **NORECON 是混合模型上的正确操作点**：`SGLANG_ENABLE_HEAD_AWARE_REPREFILL=0`。混合模型长记忆由全注意力
   兜底，丢掉的 local 通道不用重建（精度照样 1.000）。**recon-ON 会把每次「免费命中」变成「付费逐通道重建」，
   反而拖慢 TTFT**（K2c 实测过这个负结果）。做 Idea 2 的 TTFT A/B 一定要 NORECON=1。

---

## 3. 关键开关（env + CLI）

| 开关 | 作用 |
| --- | --- |
| `--mamba-radix-cache-strategy extra_buffer` | 让 radix 缓存共享前缀的线性状态（Idea 1/2 前提；`no_buffer` 几乎无前缀复用） |
| `--enable-head-aware-mamba-checkpoint --head-aware-route A` | 开 Idea 1 压缩（Route A：命中可选重建 local） |
| `--head-aware-mamba-ckpt-size N` | checkpoint 池 slot 数 |
| `SGLANG_FORCE_HEAD_AWARE_WMAX=W` | 窗口阈值 W_max。**`0` = 全 global = dense 对照**；`16/128/512/4096` = 越大丢越多、容量越高 |
| `SGLANG_ENABLE_HEAD_AWARE_REPREFILL=1/0` | Route-A 重建 开/关（`0` = NORECON，混合模型正确点） |
| `SGLANG_MAMBA_EVICT_POLICY=lru/value/gdsf` | Idea 2 驱逐策略（不设 = lru = 字节一致） |
| `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` | 放宽版本检查（本机必需，见 §1） |

**两个模型的后端 flag 不同（关键差异，别搞混）**：
- **KDA (Kimi)**：`--linear-attn-backend triton --chunked-prefill-size 2048`，`--mem-fraction-static 0.85`。
  （extra_buffer 需要 Triton `chunk_kda` 暴露逐 chunk 的 `h`；cu12.9 FA3 缺 `only_qv`，全注意力走 strip hack。）
- **GDN (Qwen3-Next-NVFP4)**：`--attention-backend triton`，`--mem-fraction-static 0.7`。
  （cu12.9 FA3 缺 `only_qv`，把所有注意力路由到 triton。）

两者都要：`--tp 2 --trust-remote-code --disable-overlap-schedule --enable-metrics`
（`--disable-overlap-schedule` 因为 Route-A 重建 seam 与 overlap 调度冲突）。

---

## 4. 实验按门（gate）顺序跑，从便宜到贵，失败即停

### GATE-K2a：KDA extra_buffer 正确性（成败硬门，先跑）

back-port PR #31474 让 KDA 接进 GDN 同款 mamba-track 路径（commit `dceaaafd88`）。先证明 extra_buffer
没把答案搞坏：
```bash
bash test/manual/bench_kda_extra_buffer_gsm8k_k2a.sh extra_buffer   # 期望 GSM8K ≈ 0.892
bash test/manual/bench_kda_extra_buffer_gsm8k_k2a.sh no_buffer      # 合并基线 ≈ 0.900
```
**判据**：extra_buffer 必须从修复前的 0.63 恢复到 ≈0.89 并与 no_buffer 一致。不一致就说明 track/restore
端口有问题，**停下先修**，别往下跑任何 head-aware / eviction。

### GATE-K2b：Idea 1 容量 + needle 精度中性

```bash
bash test/manual/bench_kda_headaware_k2b.sh          # KDA route-A 池 + needle 探针
```
**判据**：head-aware 池在真实权重上建成不崩，日志报出容量倍数（`x capacity` 行），needle NORECON == dense
（1.000，混合天花板）。GDN 侧用 `bench_gdn_prefix_hitrate.sh`。

### Idea 1 精度 A/B（真实推理数据集，检验「是否掉点」）—— 本次新增

needle/RULER 在混合模型上是天花板（dense 就是 1.000），证明不了 headroom 上的掉点。所以补了在
**AIME/GPQA 等有 headroom 的真实推理任务**上的 dense vs idea1(W_max=16, NORECON) A/B。

- 数据集：`dataset/datasets.zip` 解压到 `dataset/datasets_extracted/`（AIME2025 parquet、AIME2026 jsonl、
  hmmt、imo、gpqa_diamond parquet、MMLU-Pro parquet；gpqa 是 boxed 答案不是 4 选 1，已按 boxed-math 判分）。
- 驱动脚本 `dataset/bench_idea1_accuracy.py`（改编自 `dataset/bench_unified.py`，判分用
  `benchmark/reasoning_benchmark/{answer_extraction,eval_utils}.py` 的 `extract_math_answer`+`math_equal`）。
- 启动器 `test/manual/run_idea1_accuracy_ab.sh <kda|gdn>`：一次一模型，`dense`（WMAX=0）与
  `idea1`（WMAX=16 + NORECON）两 arm 顺序跑，末尾打印 dense/idea1/delta/SE 表。

**正确跑法（务必带上大 budget + 参考采样 + 小 CKPT_SIZE，见 §2 坑 1/2/3）**：
```bash
# KDA（Kimi 参考采样：temp=1 top_p=1，top_k=-1/presence=0 是默认；budget 262144；池 200 防 OOM）
CKPT_SIZE=200 MAXNEW=262144 TEMP=1.0 TOPP=1.0 TASKS="aime2026 aime2025" \
  bash test/manual/run_idea1_accuracy_ab.sh kda

# GDN（Qwen 参考采样 temp=0.7/top_p=0.8/top_k=20/presence=1.5 —— 注意 top_k/presence
# 当前启动脚本没透传，跑 GDN 前需在 run_idea1_accuracy_ab.sh 的 python 调用里补 --top-k 20 --presence-penalty 1.5）
CKPT_SIZE=200 MAXNEW=65536 TEMP=0.7 TOPP=0.8 TASKS="aime2026 aime2025 gpqa_diamond" \
  bash test/manual/run_idea1_accuracy_ab.sh gdn
```
**判据**：dense baseline 先要恢复到接近参考量级（AIME2026 ≈ 0.6），说明测量对齐了；然后看
`delta = acc(idea1) − acc(dense)`：**在 ±SE 内 ≈ 0 = 精度中性（不掉点）**。

**自检截断**（若 dense 仍偏低，先查是不是又被截断）：
```python
import json,statistics as st
rows=[json.loads(l) for l in open("idea1_accuracy_logs/out_kda_dense_aime2025.jsonl")]
lens=[r["output_length"] for r in rows]
near=sum(x>=max(lens)-50 for x in lens)   # 撞上限的比例；高 = budget 还不够
print("mean",st.mean(lens),"max",max(lens),"trunc",near,"/",len(lens))
```

### GATE-K2c：Idea 2 value vs lru（必须 NORECON）

```bash
NORECON=1 bash test/manual/bench_kda_evict_k2c.sh value
NORECON=1 bash test/manual/bench_kda_evict_k2c.sh lru
# 5-seed 误差棒：
bash test/manual/bench_kda_evict_multiseed.sh value && bash test/manual/bench_kda_evict_multiseed.sh lru
python test/manual/agg_kda_evict_multiseed.py
```
**判据**：NORECON 下 value 每个维度赢（命中率 +16pt 上下，TTFT −12~15%，吞吐 +9%），输入 token 逐 seed
完全一致（召回中性）。**若误设 recon-ON，命中率仍升但 TTFT 反而变差**（K2c 记录过的负结果）。

### Idea 2+：cost-aware GDSF（异质长度 + ρ-sweep）

```bash
for P in lru value gdsf; do bash test/manual/bench_kda_evict_het.sh $P; done       # rr=0.25 异质长度
bash test/manual/bench_kda_evict_rho_sweep.sh gdsf   # --gsp-len-pop-corr ρ 长度↔热度相关性
python test/manual/agg_kda_evict_rho_sweep.py
```
**判据（含诚实负结果）**：value/gdsf 都显著碾压 lru；但 gdsf ≈ value（全 ρ∈[-0.75,1] 谱不显著）——
因为独立/甚至满相关时二者淘汰决策仍高度重合。离线正确性门 `test/manual/test_mamba_evict_policy.py`（11 项）证机制成立。

### W_max 敏感性 sweep（§3.2.1）

```bash
NORECON=1 bash test/manual/bench_kda_wmax_sweep.sh          # 每档一次 server 启动
python test/manual/agg_kda_wmax_sweep.py                     # norecon 表
python test/manual/agg_kda_wmax_sweep.py --recon             # recon 表
```
**判据（NORECON）**：cap_x 随 W_max 单调上升；hit_rate/TTFT ~flat；in_tokens 恒定（召回中性）。
（W_max 在 `mamba_checkpoint_pool.py` 建池时消费，故每档必须重启 server。）
观测 τ 分布 → local 比例：`test/manual/analyze_tau_threshold.py`（GDN），`gdn_gate_capture/kda_empirical_big.log`（KDA）。

---

## 5. 通用运维

- **启动等待**：脚本都是 health-poll `http://127.0.0.1:PORT/health_generate`，最多 300×10s。server 死了
  会 tail 日志。48B/80B TP2 冷启动约 2~4 分钟。
- **GPU 释放**：换 arm / 换实验前确认 GPU 空：
  `pkill -9 -f sglang.launch_server; nvidia-smi --query-gpu=memory.free --format=csv,noheader`。
- **产物落盘位置**（都不提交 git）：`idea1_accuracy_logs/`、`gdn_prefix_hitrate_logs/`、`e2e_kda_ngram_logs/`。
- **端口**：K2a=31000，其余 head-aware 系列默认 31007。
- **提交纪律**：端口代码改动可提交（不开开关字节一致）；但 **FA3 strip hack、模型目录、bench 日志、
  Phase-H 的 `corrupt_linear_state_` 接线不要提交**。作者用 `yuyanqi`，无 Claude trailer。

---

## 6. 一句话记住每个门的期望

| 门 | 期望 | 失败含义 |
| --- | --- | --- |
| K2a | extra_buffer GSM8K 0.892 == no_buffer 0.900 | track/restore 端口错，先修 |
| K2b | 池建成、报容量倍数、needle NORECON == dense | 池 shape / 分类端口错 |
| Idea1 精度 A/B | dense 恢复到参考量级；delta ≈ 0（±SE） | 多半是 budget 截断或采样没对齐（§2） |
| K2c | NORECON 下 value 全面赢、召回中性 | 误开 recon（TTFT 反转）或池没饱和 |
| GDSF | value/gdsf 碾压 lru；gdsf ≈ value | —（这是如实的负结果） |
| W_max sweep | cap↑、hit/TTFT flat、in_tokens 恒定 | 没每档重启 server |
