---
title: "年末年始に BitNet を実装して実用性を確かめた"
emoji: "🔬"
type: "tech"
topics: ["bitnet", "triton", "pytorch", "gpu", "llm"]
published: false
---

## はじめに

年末年始に時間ができたので、以前話題になった BitNet について改めて調べてみた。

BitNet は Microsoft が 2024年に提案した 1.58-bit 量子化手法で、重みを {-1, 0, +1} の3値だけで表現する。当時は「スマホでも大規模LLMが動くのでは」という期待もあったが、2026年現在、実用化の話はあまり聞かない。調べてみると学習時のオーバーヘッドや GPU との相性問題があるようだったが、エッジデバイスで大規模モデルを動かすという話は魅力的だし、実際のところどうなのか気になったので自分で Triton カーネルを実装して確かめてみることにした。なお、実装には Claude Code に大いに頼った。

実験で作成したリポジトリはこちら：
- [bitnet-triton](https://github.com/yukikotani231/bitnet-triton) - Triton カーネル実装
- [bitnet-mnist](https://github.com/yukikotani231/bitnet-mnist) - MNIST での検証

## BitNet の概要

BitNet b1.58[^bitnet-paper] は、ニューラルネットワークの重みを {-1, 0, +1} の3値に量子化する。通常の Linear 層が `y = x @ W`（W は FP32/FP16）なのに対し、BitNet では量子化した ternary weights とスケール係数を使って `y = x @ W_ternary * scale` のように計算する。

[^bitnet-paper]: [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)

これが画期的だったのは2つの理由がある。まず、3値を2ビットでエンコードして16個の重みを1つの int32 にパッキングできるので、FP32 比で 16倍、FP16 比でも 8倍のメモリ圧縮になる。そしてもう一つ、重みが {-1, 0, +1} しかないので行列演算が `y = Σ(x where w=+1) - Σ(x where w=-1)` のように加算と減算だけで計算でき、乗算が不要になる。CPU では SIMD 命令でこの演算を効率的に実行できるし、GPU でも Tensor Core を介さない軽量な演算に置き換えられる可能性があった（結論から言うと GPU では期待通りにはいかなかったのだが）。

実際に 2025年には Microsoft が [BitNet b1.58 2B4T モデル](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)をリリースしており、2B パラメータで完全精度モデルと遜色ない性能を出している[^bitnet-2b4t]。CPU 向けの推論フレームワーク [BitNet.cpp](https://github.com/microsoft/BitNet) も公開されていて、ARM CPU で 1.37x〜5.07x、x86 CPU で 2.37x〜6.17x の高速化が報告されている[^bitnet-cpp]。

[^bitnet-2b4t]: [BitNet b1.58 2B4T Technical Report](https://arxiv.org/pdf/2504.12285)
[^bitnet-cpp]: [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://arxiv.org/pdf/2502.11880)

## BitNet が広く実用化されていない理由

ここまで聞くと良いことずくめに見えるが、実際にはあまり普及していない。調べてみるといくつか課題があった。

### 学習時のオーバーヘッド

BitNet は推論時の重みこそ {-1, 0, +1} だが、学習時には勾配やオプティマイザの状態を FP32 で保持する必要がある[^quantization-training]。量子化関数は微分できないので Straight-Through Estimator (STE) で近似するのだが、勾配を低精度で蓄積するとゼロ勾配や高誤差が発生するため、結局学習時のメモリ削減効果は限定的になる。また、低ビット量子化は精度を回復するために数百エポックの再学習が必要になることもあり[^qat-cost]、学習コストの面でも楽ではない。

[^quantization-training]: [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/pdf/2310.11453)
[^qat-cost]: [A White Paper on Neural Network Quantization](https://arxiv.org/pdf/2106.08295)

### GPU での高速化が難しい

もっと根本的な問題として、GPU での高速化が期待ほどできないというのがある。現代の GPU は FP16/FP32 の行列積に最適化された Tensor Core を搭載しているが、BitNet の整数演算はこれとは相性が悪く[^tensor-core-perf]、LUT ベースの手法も GPU では効率的に動かない[^bitnet-gpu-perf]。

BitNet.cpp の論文[^cpu-gpu-perf]を見ると状況がよく分かる。A100 GPU で BitNet b1.58 2B モデルを動かすと 250 tokens/s だが、CPU でも 82〜110 tokens/s 出ている。性能差がわずか 2.3x〜3x しかない。普通の FP16 推論なら GPU は CPU の数十倍速いことを考えると、BitNet では GPU を使う旨味がかなり薄い。

[^tensor-core-perf]: [Benchmarking GPU Tensor Cores on General Matrix Multiplication Kernels through CUTLASS](https://www.mdpi.com/2076-3417/13/24/13022)
[^bitnet-gpu-perf]: [Advances to low-bit quantization enable LLMs on edge devices](https://www.microsoft.com/en-us/research/blog/advances-to-low-bit-quantization-enable-llms-on-edge-devices/)
[^cpu-gpu-perf]: [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://arxiv.org/pdf/2502.11880)

### INT4 量子化で十分なことが多い

さらに言えば、GPU で推論するなら AWQ や GPTQ といった INT4 量子化の方が実装も成熟しているし、Tensor Core をちゃんと活かせるし、精度の劣化も BitNet より少ない。GPU ベースの推論では INT4 で事足りる場面が多く、BitNet の出番がないというのが現状だ。

## それでも実験してみた理由

とはいえ、Triton での GPU 実装例はまだ少ないし、カーネルの書き方次第で結果が変わる可能性もある。それに、Claude Opus クラスのモデルをスマホで動かせたら API 課金もいらないしオフラインでも使えるし最高じゃないかという夢がどうしても捨てきれなかったので、年末年始の時間を使って実際に手を動かしてみることにした。

## Triton カーネルの実装

まず BitNet の行列演算を GPU で動かすための Triton カーネルを書く。2-bit にパッキングした重みをカーネル内でアンパックしながら行列積を計算するのがポイントになる。

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

Triton カーネルが動くようになったので、まずは実際のモデルで学習できるか確認したい。MNIST の画像生成タスクで Diffusion Transformer (DiT) を BitNet 化して学習させてみた。

最初は「全部の Linear を BitLinear に置き換えればいいでしょ」と思ってそうしたのだが、Loss が 1.0 付近で停滞して全く学習が進まなかった。

```python
# 失敗した実装：全てを BitLinear に
self.time_embed = BitLinear(dim, dim)  # ← これが問題
self.adaLN = BitLinear(dim, dim * 4)   # ← これも問題
```

原因は、時間埋め込みと AdaLN（条件付け層）まで BitLinear にしてしまったこと。これらの層は連続的な条件情報を扱うので、{-1, 0, +1} に量子化すると情報が壊れてしまう。時間埋め込みを通常の FP32 Linear に戻して、Attention と MLP だけ BitLinear にしたら 50 epoch で Loss 0.045 まで下がり、ちゃんと数字が認識できる画像が生成されるようになった。

```python
# 成功した実装：条件付け層は FP32 のまま
self.time_embed = nn.Sequential(
    SinusoidalPositionEmbedding(dim),
    nn.Linear(dim, dim),  # FP32
    nn.GELU(),
    nn.Linear(dim, dim),  # FP32
)
self.qkv = BitLinear(dim, dim * 3)  # ここは BitLinear OK
self.mlp = BitLinearMLP(dim)        # ここも OK
```

教訓としては、条件付け層（時間埋め込みやクラス埋め込みなど）は量子化してはいけないということ。逆に、Attention や MLP の重み行列は BitLinear で問題なく動く。

## 実験2: LUT ベースカーネル (BitNet.cpp 方式)

学習が動くことは確認できたので、次は推論速度を何とかしたい。

BitNet.cpp の README を読んでいると、T-MAC[^tmac] という手法で CPU 上で大幅な高速化を実現していることが分かった。4つの ternary weights をグループ化して 256 エントリの LUT (Lookup Table) を構築し、CPU の SIMD shuffle 命令（`vpshufb`）で高速にルックアップすることで乗算を完全に排除している。これを GPU に移植したら速くなるのでは？と思い試してみた。

[^tmac]: [T-MAC: Table Lookup for Ternary Matrix Multiplication](https://arxiv.org/abs/2407.00088)

GPU 上では Triton の `tl.where` による条件分岐で加算/減算を切り替える形で実装した。

```python
@triton.jit
def _bitnet_lut_kernel(...):
    c = tl.where(w == 2, x, tl.where(w == 0, -x, 0.0))
    acc += c
```

結果は散々で、LUT 方式は `tl.dot` を使う元の実装と全く同じ速度だった。条件分岐を使う naive な ternary 実装に至っては 1.5倍遅い。

| Config | tl.dot | LUT | Ternary |
|--------|--------|-----|---------|
| (1, 4096, 4096) | 0.48 ms | 0.48 ms | 0.70 ms |
| (32, 4096, 4096) | 0.48 ms | 0.48 ms | 0.73 ms |

理由は単純で、CPU では SIMD FMA → LUT + vpshufb という置き換えが高速化に繋がるが、GPU では行列演算を Tensor Core がハードウェアレベルで処理するので、LUT に置き換える意味がない。GPU に `vpshufb` 相当の命令もなく、LUT を Shared Memory に置いても L1 キャッシュ上の CPU には敵わない。

## 実験3: 実用性ベンチマーク

速度面では厳しいことが分かってきたが、メモリ削減効果は確実にあるはずだ。実際にどのくらいのインパクトがあるのか定量的に測ってみた。

メモリ使用量は理論通り 16 倍の圧縮が確認できた（GPT-2 で 0.11GB → 0.01GB、LLaMA-7B で 13GB → 0.81GB）。一方でスループットは hidden=4096 で測定したところ、Batch Size 1 で FP16 の 0.19x、Batch Size 32 でも 0.20x と、やはりかなり遅い。Batch Size が大きくなるほど差が広がり、512 では 0.12x まで落ちる。

| Batch Size | FP16 (tok/s) | BitNet (tok/s) | 比率 |
|------------|--------------|----------------|------|
| 1 | 11,087 | 2,136 | 0.19x |
| 32 | 345,879 | 68,919 | 0.20x |
| 512 | 1,997,117 | 237,756 | 0.12x |

ただし面白いのは最大バッチサイズで、メモリが小さい分だけ BitNet は FP16 の 4 倍のサンプルを同時に処理できる（FP16: 2,048 vs BitNet: 8,192）。速度が遅くてもバッチを大きく取れるので、スループット全体で見ればメモリが足りない環境では有利になるケースもありそうだ。

## まとめ

実験を通じて分かったのは、BitNet の本質はメモリ圧縮であって速度向上ではないということだ。GPU の Tensor Core は FP16/FP32 に最適化されているので、BitNet にしたところで速くはならない。LUT 方式も CPU の SIMD shuffle 命令があってこそ活きるもので、GPU に持ってきても意味がなかった。

モデルの設計としては、条件付け層（時間埋め込みやクラス埋め込み）は FP32 のままにして、Attention や MLP だけ BitLinear にするのが正解だった。全部 BitLinear にすると学習が壊れるというのは、やってみないと分からないことだったと思う。

メモリ 16x 圧縮の効果自体は本物で、LLaMA-7B が 13GB → 0.81GB になるのは確かにインパクトがある。ただし速度は FP16 の 0.2x 程度に落ちるので、メモリがボトルネックでない限り選ぶ理由がない。GPU で速度も求めるなら FP16 + Flash Attention、メモリを節約したいなら INT4 (AWQ/GPTQ)、LLM のサービングなら vLLM あたりが現実的な選択肢になる。

結局、スマホで Opus を動かすという当初の夢には程遠く、素直に Anthropic に課金することにした。BitNet の仕組みや GPU カーネルの最適化について理解が深まったのは収穫だったが、実用性を考えると API を使うのが一番賢い。エッジデバイスで大規模モデルが快適に動く日が来たら、また挑戦してみたい。

## 参考資料

- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)
- [The Era of 1-bit LLMs (BitNet b1.58)](https://arxiv.org/abs/2402.17764)
- [BitNet.cpp](https://github.com/microsoft/BitNet)
