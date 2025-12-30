---
title: "BitNet を GPU で実装して分かったこと - Triton カーネル開発から実用性検証まで"
emoji: "🔬"
type: "tech"
topics: ["bitnet", "triton", "pytorch", "gpu", "llm"]
published: false
---

## はじめに

BitNet は Microsoft が提案した 1.58-bit 量子化手法で、重みを {-1, 0, +1} の3値に制限することで、大幅なメモリ削減と高速化を実現します。本記事では、BitNet を GPU 上で Triton カーネルとして実装し、様々な実験を通じて得られた知見を共有します。

**実験で作成したリポジトリ:**
- [bitnet-triton](https://github.com/yukikotani/bitnet-triton) - Triton カーネル実装
- [bitnet-mnist](https://github.com/yukikotani/bitnet-mnist) - MNIST での検証

## BitNet とは

### 基本概念

BitNet b1.58 は、ニューラルネットワークの重みを3値 {-1, 0, +1} に量子化する手法です。

```python
# 通常の Linear
y = x @ W  # W は FP32/FP16

# BitNet Linear
W_ternary = quantize(W)  # W ∈ {-1, 0, +1}
y = x @ W_ternary * scale
```

### メモリ効率

| 精度 | ビット数 | 圧縮率 |
|------|---------|-------|
| FP32 | 32 bit | 1x |
| FP16 | 16 bit | 2x |
| INT8 | 8 bit | 4x |
| INT4 | 4 bit | 8x |
| **BitNet (2-bit)** | 2 bit | **16x** |

3値を2ビットでエンコードし、16個の重みを1つの int32 にパッキングすることで、FP32 比で 16倍のメモリ圧縮を実現します。

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

MNIST 画像生成タスクで BitNet の有効性を検証しました。

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

BitNet.cpp は CPU 上で LUT（Lookup Table）を使い、乗算を完全に排除して高速化しています。この方式を GPU に移植できるか検証しました。

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

**結論**: GPU では LUT 方式のメリットがない。

### なぜ GPU では効果がないのか

| 要素 | CPU | GPU |
|------|-----|-----|
| 行列演算 | SIMD FMA | **Tensor Core** |
| LUT 配置 | L1 キャッシュ | Shared Memory |
| 高速命令 | vpshufb (shuffle) | なし |

GPU の Tensor Core は FP16/FP32 の行列積に最適化されており、LUT ルックアップより圧倒的に高速です。

## 実験3: 実用性ベンチマーク

### メモリ使用量

| モデル | FP16 | BitNet | 圧縮率 |
|--------|------|--------|-------|
| GPT-2 | 0.11 GB | 0.01 GB | 16x |
| LLaMA-7B | 8.00 GB | 0.50 GB | 16x |

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

BitNet は「メモリ」が目的で「速度」は犠牲になります。速度重視なら:

| 手法 | 圧縮 | 速度 | 用途 |
|------|------|------|------|
| FP16 + Flash Attention | 2x | 2-4x | 汎用 |
| INT8 + TensorRT | 4x | 2-4x | 推論最適化 |
| INT4 (AWQ/GPTQ) | 8x | 1.5-2x | メモリ節約 |
| vLLM | - | 2-24x | LLM サービング |
| Speculative Decoding | - | 2-3x | 生成高速化 |

## まとめ

### 得られた知見

1. **BitNet の本質はメモリ圧縮**
   - GPU では速度向上しない（Tensor Core が強すぎる）
   - CPU では T-MAC/LUT 方式で速度も向上

2. **条件付け層は量子化しない**
   - 時間埋め込み、クラス埋め込みは FP32 を維持
   - Attention、MLP は BitLinear OK

3. **LUT 方式は CPU 専用**
   - GPU の Tensor Core > LUT ルックアップ
   - CPU の SIMD shuffle は LUT に最適

4. **実用的な価値**
   - メモリ 16x 圧縮 → 大モデルが動く
   - バッチサイズ 4x → スループット向上
   - 速度は 0.2x → トレードオフ

### 結論

BitNet は「速度と引き換えにメモリを節約する」技術です。

```
LLaMA-7B:
  FP16: 26GB → 大きい GPU 必要
  BitNet: 1.6GB → 4GB GPU で動作可能！
```

メモリが制約のボトルネックなら、BitNet は「動かせない」を「動かせる」に変える強力な選択肢です。

## 参考資料

- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)
- [The Era of 1-bit LLMs (BitNet b1.58)](https://arxiv.org/abs/2402.17764)
- [BitNet.cpp](https://github.com/microsoft/BitNet)
- [T-MAC: Table Lookup for Ternary Matrix Multiplication](https://arxiv.org/abs/2407.00088)
