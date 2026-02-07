---
title: "年末年始に BitNet を実装して実用性を確かめた"
emoji: "🔬"
type: "tech"
topics: ["bitnet", "triton", "pytorch", "gpu", "llm"]
published: false
---

## はじめに

年末年始に時間ができたので、以前話題になった BitNet について改めて調べてみた。

BitNet は Microsoft が 2024年に提案した 1.58-bit 量子化手法で、重みを {-1, 0, +1} の3値に制限することで大幅なメモリ削減を実現する。当時は「スマホでも大規模LLMが動く」という期待もあったが、2026年現在、実用化の話はあまり聞かない。

調べてみると、学習時の誤差逆伝播のオーバーヘッドや、GPU での高速化が難しいといった課題があるようだった。とはいえ、エッジデバイスで大規模モデルを動かすという魅力は捨てがたい。実際のところどうなのか、自分で実装して確かめてみることにした。

本記事では、BitNet を Triton カーネルとして GPU 上に実装し、様々な実験を通じて得られた知見をまとめる。

**実験で作成したリポジトリ:**
- [bitnet-triton](https://github.com/yukikotani231/bitnet-triton) - Triton カーネル実装
- [bitnet-mnist](https://github.com/yukikotani231/bitnet-mnist) - MNIST での検証

## BitNet とは

### 基本概念

BitNet b1.58 は、ニューラルネットワークの重みを3値 {-1, 0, +1} に量子化する手法だ[^bitnet-paper]。

```python
# 通常の Linear
y = x @ W  # W は FP32/FP16

# BitNet Linear
W_ternary = quantize(W)  # W ∈ {-1, 0, +1}
y = x @ W_ternary * scale
```

[^bitnet-paper]: [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)

### なぜ画期的だったのか

従来の量子化手法（INT8, INT4）と比べて、BitNet の何が革新的だったのか。

**1. 極端なメモリ圧縮**

| 精度 | ビット数 | 圧縮率 |
|------|---------|-------|
| FP32 | 32 bit | 1x |
| FP16 | 16 bit | 2x |
| INT8 | 8 bit | 4x |
| INT4 | 4 bit | 8x |
| **BitNet (2-bit)** | 2 bit | **16x** |

3値を2ビットでエンコードし、16個の重みを1つの int32 にパッキングすることで、FP32 比で 16倍のメモリ圧縮を実現する。

**2. 乗算の排除**

ternary weights {-1, 0, +1} の場合、行列演算が加算と減算だけで計算できる：

```
y = Σ(x where w=+1) - Σ(x where w=-1)
```

これは CPU では SIMD 命令で非常に効率的に実行できるため、理論上は大幅な高速化が期待できた。

### 実際の使用例

2025年、Microsoft は [BitNet b1.58 2B4T モデル](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)をリリースし、2B パラメータで完全精度モデルと競争力のあるパフォーマンスを実証した[^bitnet-2b4t]。

また、[BitNet.cpp](https://github.com/microsoft/BitNet) という CPU 向けの推論フレームワークも公開されており、ARM CPU で 1.37x〜5.07x、x86 CPU で 2.37x〜6.17x の高速化を実現している[^bitnet-cpp]。

[^bitnet-2b4t]: [BitNet b1.58 2B4T Technical Report](https://arxiv.org/pdf/2504.12285)
[^bitnet-cpp]: [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://arxiv.org/pdf/2502.11880)

## BitNet が広く実用化されなかった理由

では、なぜこれほど魅力的に見える BitNet が広く実用化されていないのか。調べてみると、いくつかの課題が見えてきた。

### 学習時のオーバーヘッド

BitNet は学習時に特殊な工夫が必要になる。重みは {-1, 0, +1} に量子化されるが、勾配とオプティマイザの状態は高精度（FP32）で保持する必要がある[^quantization-training]。

- **Straight-Through Estimator (STE)** を使って非微分可能な量子化関数を近似
- 勾配を低精度で蓄積するとゼロ勾配や高誤差が発生
- 結果として、学習時のメモリ削減効果は限定的

量子化学習 (QAT) の主な欠点は、ニューラルネットワークモデルの再学習にかかる計算コストで、特に低ビット精度の量子化では、精度を回復するために数百エポックの学習が必要になることもある[^qat-cost]。

[^quantization-training]: [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/pdf/2310.11453)
[^qat-cost]: [A White Paper on Neural Network Quantization](https://arxiv.org/pdf/2106.08295)

### GPU での高速化が困難

BitNet の最大の課題は、GPU での高速化が期待ほどできないことだ。

**Tensor Core との相性問題**

現代の GPU は FP16/FP32 の行列積に最適化された Tensor Core を搭載している。しかし：

- Tensor Core は浮動小数点演算に特化しており、BitNet の整数演算には不向き[^tensor-core-perf]
- LUT (Lookup Table) ベースの手法は GPU では効率が悪い[^bitnet-gpu-perf]
- 実際の Tensor Core 利用率は理論値の 9-31% 程度に留まる[^tensor-core-util]

**CPU との性能差**

A100 GPU では BitNet b1.58 2B モデルで 250 tokens/s を達成したが、CPU では 110 tokens/s、82 tokens/s、88 tokens/s と、GPU の 2.3x〜3x の性能差に留まっている[^cpu-gpu-perf]。GPU は CPU の 17x〜20x のメモリ帯域幅を持つことを考えると、期待されたほどの性能差がない。

[^tensor-core-perf]: [Benchmarking GPU Tensor Cores on General Matrix Multiplication Kernels through CUTLASS](https://www.mdpi.com/2076-3417/13/24/13022)
[^bitnet-gpu-perf]: [Advances to low-bit quantization enable LLMs on edge devices](https://www.microsoft.com/en-us/research/blog/advances-to-low-bit-quantization-enable-llms-on-edge-devices/)
[^tensor-core-util]: [The Power of 8: Getting the most out of Tensor Cores](https://medium.com/@michael.diggin/the-power-of-8-getting-the-most-out-of-tensor-cores-c7704ae0c5c1)
[^cpu-gpu-perf]: [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://arxiv.org/pdf/2502.11880)

### INT4 との競合

INT4 量子化（AWQ, GPTQ など）は：

- GPU での実装が成熟している
- Tensor Core を効率的に利用できる
- 精度の劣化が BitNet より少ない

結果として、GPU ベースの推論では INT4 が選ばれることが多い。

## それでも実験してみた理由

とはいえ、諦めるにはまだ早い。調査を進めるうちに、いくつか気になる点があった。

1. **ほとんどの評価が CPU ベース**：GPU での最適化はまだ改善の余地があるのでは？
2. **Triton での実装例が少ない**：Triton カーネルで工夫すれば何とかならないか？
3. **個人的な興味**：スマホで大規模 LLM を動かせたら面白いのでは？

特に3つ目が大きい。Claude Opus のような高性能モデルをスマホで動かせたら、プライバシーも保てるし、オフラインでも使える。課金も不要だ。

そんなわけで、年末年始の時間を使って実際に Triton カーネルを実装し、実用性を検証してみることにした。

## Triton カーネルの実装

### 2-bit パッキング

```python
def pack_weights(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 重みを 2-bit にパック"""
    # スケール計算
    scale = weight.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)

    # 量子化: {-1, 0, +1}
    w_scaled = weight / scale
    w_ternary = torch.clamp(torch.round(w_scaled), -1, 1).to(torch.int8)

    # {0, 1, 2} にマッピング
    w_mapped = (w_ternary + 1).to(torch.uint8)

    # 16個の2-bit値を1つの int32 にパック
    packed = torch.zeros(N, K // 16, dtype=torch.int32)
    for i in range(16):
        packed |= w_reshaped[:, :, i].to(torch.int32) << (i * 2)

    return packed, scale
```

### Triton MatMul カーネル

```python
@triton.jit
def _bitnet_matmul_kernel(
    x_ptr, packed_ptr, scale_ptr, output_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # ブロックインデックス
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)

    # アキュムレータ
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        # 入力をロード
        x = tl.load(x_ptr + ...)

        # パックされた重みをアンパック
        pack_idx = offs_k // 16
        bit_idx = offs_k % 16
        packed = tl.load(packed_ptr + ...)
        w_bits = (packed >> (bit_idx * 2)) & 0b11
        w = w_bits.to(tl.float32) - 1.0  # {0,1,2} -> {-1,0,+1}

        # Tensor Core で行列積
        acc += tl.dot(x, tl.trans(w), allow_tf32=True)

    # スケール適用
    output = acc * scales
    tl.store(output_ptr + ..., output)
```

## 実験1: BitNet DiT (Diffusion Transformer)

### 概要

Triton カーネルの実装が完成したので、まずは MNIST 画像生成タスクで動作確認をすることにした。Diffusion Transformer (DiT) を BitNet 化して学習させてみる。

「どうせ全部 BitLinear に置き換えればいいんでしょ」と軽い気持ちで実装したら、見事に失敗した。

### 最初の失敗: Loss が 1.0 で停滞

```python
# 失敗した実装
class BitNetDiT(nn.Module):
    def __init__(self, ...):
        # 全てを BitLinear に
        self.time_embed = BitLinear(dim, dim)  # ← 問題
        self.adaLN = BitLinear(dim, dim * 4)   # ← 問題
```

**原因**: 時間埋め込みと AdaLN（条件付け層）を BitLinear にすると、連続的な条件情報が量子化で破壊される。

### 修正版

```python
# 成功した実装
class BitNetDiT(nn.Module):
    def __init__(self, ...):
        # 時間埋め込みは通常の Linear（条件情報を保持）
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(dim),
            nn.Linear(dim, dim),  # ← FP32
            nn.GELU(),
            nn.Linear(dim, dim),  # ← FP32
        )

        # Attention と MLP は BitLinear OK
        self.qkv = BitLinear(dim, dim * 3)
        self.mlp = BitLinearMLP(dim)
```

### 結果

| モデル | Loss (50 epoch) | 画像品質 |
|--------|-----------------|---------|
| 失敗版 | ~1.0 (停滞) | ノイズ |
| 修正版 | 0.045 | 認識可能な数字 |

**教訓**: 条件付け層（時間埋め込み、クラス埋め込み等）は量子化してはいけない。

## 実験2: LUT ベースカーネル (BitNet.cpp 方式)

### 背景

実験1で DiT が動くことは確認できた。次は速度だ。

BitNet.cpp の README を読んでいると、T-MAC という手法で CPU で大幅な高速化を実現していることが分かった。LUT (Lookup Table) を使って乗算を完全に排除するというアイデアだ。「これを GPU に移植したら速くなるのでは？」と思い、実装してみることにした。

### T-MAC (BitNet.cpp) の仕組み

```
ternary weights {-1, 0, +1} の場合:
  y = Σ(x where w=+1) - Σ(x where w=-1)

乗算が不要！加算・減算のみで計算可能

CPU での実装:
  4つの重み (8bit) をグループ化
  → 256エントリの LUT を構築
  → SIMD shuffle (vpshufb) で高速ルックアップ
```

### GPU での実装と結果

```python
# LUT スタイルのカーネル
@triton.jit
def _bitnet_lut_kernel(...):
    # 条件分岐で加算/減算
    c = tl.where(w == 2, x, tl.where(w == 0, -x, 0.0))
    acc += c
```

**ベンチマーク結果:**

| Config | Current (tl.dot) | LUT | Ternary |
|--------|-----------------|-----|---------|
| (1, 4096, 4096) | 0.48 ms | 0.48 ms | 0.70 ms |
| (32, 4096, 4096) | 0.48 ms | 0.48 ms | 0.73 ms |

期待していたが、結果は散々だった。LUT 方式は `tl.dot` を使う現在の実装と同じ速度で、特に改善は見られなかった。それどころか、条件分岐を使う naive な ternary 実装は 1.5倍も遅くなった。

**結論**: GPU では LUT 方式のメリットがない。

### なぜ GPU では効果がないのか

| 要素 | CPU | GPU |
|------|-----|-----|
| 行列演算 | SIMD FMA | **Tensor Core** |
| LUT 配置 | L1 キャッシュ | Shared Memory |
| 高速命令 | vpshufb (shuffle) | なし |

GPU の Tensor Core は FP16/FP32 の行列積に最適化されており、LUT ルックアップより圧倒的に高速だ。

## 実験3: 実用性ベンチマーク

LUT 方式での高速化は失敗したが、メモリ削減効果は確実にあるはずだ。実際にどれくらいメモリが削減できて、スループットがどう変わるのか、定量的に測定してみることにした。

### メモリ使用量

| モデル | FP16 | BitNet | 圧縮率 |
|--------|------|--------|-------|
| GPT-2 | 0.11 GB | 0.01 GB | 16x |
| LLaMA-7B | 13.00 GB | 0.81 GB | 16x |

### スループット (hidden=4096)

| Batch Size | FP16 (tok/s) | BitNet (tok/s) | 比率 |
|------------|--------------|----------------|------|
| 1 | 11,087 | 2,136 | 0.19x |
| 32 | 345,879 | 68,919 | 0.20x |
| 512 | 1,997,117 | 237,756 | 0.12x |

### 最大バッチサイズ

```
FP16:   2,048 サンプル
BitNet: 8,192 サンプル (4x 多い！)
```

## BitNet の使いどころ

ここまでの実験結果を踏まえて、BitNet がどういう場面で有効なのかを整理してみる。

### ✓ 効果的なケース

| シナリオ | 理由 |
|---------|------|
| **メモリ不足** | 16x 圧縮で大モデルが動く |
| **大量バッチ推論** | 同メモリで 4x 多くのサンプル |
| **エッジデバイス** | 小さい GPU で大きいモデル |
| **コスト削減** | 小さい GPU インスタンス |

### ✗ 不向きなケース

| シナリオ | 理由 |
|---------|------|
| **メモリに余裕あり** | FP16 の方が速い |
| **低レイテンシ必須** | 単一サンプルは 0.2x |
| **最高品質が必要** | 量子化による精度低下 |

## GPU での高速化: BitNet 以外の選択肢

実験を通じて、BitNet は「メモリ削減」が目的で「速度」は犠牲になることが分かった。では、速度を重視する場合は何を使えばいいのか。参考までに、GPU での高速化手法をまとめておく。

| 手法 | 圧縮 | 速度 | 用途 |
|------|------|------|------|
| FP16 + Flash Attention | 2x | 2-4x | 汎用 |
| INT8 + TensorRT | 4x | 2-4x | 推論最適化 |
| INT4 (AWQ/GPTQ) | 8x | 1.5-2x | メモリ節約 |
| vLLM | - | 2-24x | LLM サービング |
| Speculative Decoding | - | 2-3x | 生成高速化 |

## まとめ

### 得られた知見

実験を通じて、以下のことが分かった。

1. **BitNet の本質はメモリ圧縮**
   - GPU では速度向上しない（Tensor Core が FP16/FP32 に最適化されている）
   - CPU では T-MAC/LUT 方式で速度も向上する

2. **条件付け層は量子化してはいけない**
   - 時間埋め込み、クラス埋め込みなどは FP32 を維持する必要がある
   - Attention、MLP の重み行列は BitLinear で問題ない

3. **LUT 方式は CPU 専用**
   - GPU の Tensor Core は LUT ルックアップより圧倒的に高速
   - CPU の SIMD shuffle 命令は LUT に最適化されている

4. **実用的な価値はあるが限定的**
   - メモリ 16x 圧縮で大モデルが小さい GPU で動く
   - バッチサイズを 4x に増やせるためスループットは向上
   - ただし単一サンプルの速度は 0.2x に低下

### 結論：素直に Anthropic に課金することにした

実験の結果、BitNet は確かにメモリを大幅に削減できるが、GPU では速度面で大きく劣ることが分かった。

```
LLaMA-7B での比較:
  FP16:   13GB メモリ必要、高速
  BitNet: 0.81GB で動作可能、ただし遅い
```

つまり、BitNet は「速度と引き換えにメモリを節約する」技術だ。メモリが制約のボトルネックなら強力な選択肢になるが、スマホで Claude Opus レベルのモデルを快適に動かすという当初の夢には程遠い。

**現実的な選択肢**

- メモリに余裕があるなら → FP16 + Flash Attention
- 推論最適化なら → INT8/INT4 + TensorRT
- LLM サービングなら → vLLM
- スマホで快適に使いたいなら → **素直に API を使う**

スマホで Opus を動かす夢は一旦諦めて、素直に Anthropic に課金することにした。Claude の API なら速度も速く、精度も高く、自分で GPU を管理する手間もない。技術的興味で実装するのは楽しかったが、実用性を考えると API 利用が現実的な選択だった。

とはいえ、この実験で BitNet の仕組みや GPU カーネルの最適化について深く学べたのは収穫だった。いつか本当にエッジデバイスで大規模モデルが快適に動く日が来るかもしれない。その時はまた挑戦してみたい。

## 参考資料

- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)
- [The Era of 1-bit LLMs (BitNet b1.58)](https://arxiv.org/abs/2402.17764)
- [BitNet.cpp](https://github.com/microsoft/BitNet)
- [T-MAC: Table Lookup for Ternary Matrix Multiplication](https://arxiv.org/abs/2407.00088)
