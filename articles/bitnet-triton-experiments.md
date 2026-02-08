---
title: "年末年始にBitNetを実装して実用性を確かめた"
emoji: "🔬"
type: "tech"
topics: ["bitnet", "triton", "pytorch", "gpu", "llm"]
published: true
---

## はじめに

年末年始に時間ができたので、以前話題になったBitNetについて改めて調べてみました。

BitNetはMicrosoftが2024年に提案した1.58-bit量子化手法で、重みを{-1, 0, +1}の3値だけで表現します。当時は「スマホでも大規模LLMが動くのでは」という期待もありましたが、2026年現在、実用化の話はあまり聞きません。調べてみると学習時のオーバーヘッドやGPUとの相性問題があるようでしたが、エッジデバイスで大規模モデルを動かすという話は魅力的ですし、実際どうなのか気になったので自分でTritonカーネルを実装して確かめてみることにしました。なお、実装にはClaude Codeに大いに頼りました。

実験で作成したリポジトリはこちら：
- [bitnet-triton](https://github.com/yukikotani231/bitnet-triton) - Tritonカーネル実装
- [bitnet-mnist](https://github.com/yukikotani231/bitnet-mnist) - MNISTでの検証

## BitNetの概要

BitNet b1.58[^bitnet-paper]は、ニューラルネットワークの重みを{-1, 0, +1}の3値に量子化します。通常のLinear層が`y = x @ W`（WはFP32/FP16）なのに対し、BitNetでは量子化したternary weightsとスケール係数を使って`y = x @ W_ternary * scale`のように計算します。

「1.58-bit」という名前は、情報理論から来ています。3つの値を区別するのに必要な情報量は log₂(3) ≈ 1.585 bit です。実装上は2bitでエンコードしますが（00, 01, 10で3値を表現）、情報理論的には1パラメータあたり1.58bitの情報量しか持っていません。

[^bitnet-paper]: [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)

これが画期的だったのは2つの理由があります。まず、3値を2ビットでエンコードして16個の重みを1つのint32にパッキングできるので、FP32比で16倍、FP16比でも8倍のメモリ圧縮になります。そしてもう一つ、重みが{-1, 0, +1}しかないので行列演算が`y = Σ(x where w=+1) - Σ(x where w=-1)`のように加算と減算だけで計算でき、乗算が不要になります。CPUではSIMD命令でこの演算を効率的に実行できますし、GPUでもTensor Coreを介さない軽量な演算に置き換えられる可能性がありました（結論から言うとGPUでは期待通りにはいきませんでしたが）。

実際に2025年にはMicrosoftが[BitNet b1.58 2B4Tモデル](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)をリリースしており、2Bパラメータで完全精度モデルと遜色ない性能を出しています[^bitnet-2b4t]。CPU向けの推論フレームワーク[BitNet.cpp](https://github.com/microsoft/BitNet)も公開されていて、ARM CPUで1.37x〜5.07x、x86 CPUで2.37x〜6.17xの高速化が報告されています[^bitnet-cpp]。

[^bitnet-2b4t]: [BitNet b1.58 2B4T Technical Report](https://arxiv.org/pdf/2504.12285)
[^bitnet-cpp]: [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://arxiv.org/pdf/2502.11880)

## BitNetが広く実用化されていない理由

良いことずくめに見えますが、実際にはあまり普及していません。調べてみるといくつか課題がありました。

### 学習時のオーバーヘッド

BitNetは推論時の重みこそ{-1, 0, +1}ですが、学習時には勾配やオプティマイザの状態をFP32で保持する必要があります[^quantization-training]。量子化関数は微分できないのでStraight-Through Estimator(STE)で近似しますが、勾配を低精度で蓄積するとゼロ勾配や高誤差が発生するため、結局学習時のメモリ削減効果は限定的になります。低ビット量子化は精度を回復するために数百エポックの再学習が必要になることもあり[^qat-cost]、学習コストの面でも楽ではありません。

[^quantization-training]: [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/pdf/2310.11453)
[^qat-cost]: [A White Paper on Neural Network Quantization](https://arxiv.org/pdf/2106.08295)

### GPUでの高速化が難しい

もっと根本的な問題として、GPUでの高速化が期待ほどできないという点があります。現代のGPUはFP16/FP32の行列積に最適化されたTensor Coreを搭載していますが、BitNetの整数演算はこれとは相性が悪く[^tensor-core-perf]、LUTベースの手法もGPUでは効率的に動きません[^bitnet-gpu-perf]。

BitNet.cppの論文[^cpu-gpu-perf]を見ると状況がよく分かります。A100 GPUでBitNet b1.58 2Bモデルを動かすと250 tokens/sですが、CPUでも82〜110 tokens/s出ています。性能差がわずか2.3x〜3xしかありません。普通のFP16推論ならGPUはCPUの数十倍速いことを考えると、BitNetではGPUを使う旨味がかなり薄いです。

[^tensor-core-perf]: [Benchmarking GPU Tensor Cores on General Matrix Multiplication Kernels through CUTLASS](https://www.mdpi.com/2076-3417/13/24/13022)
[^bitnet-gpu-perf]: [Advances to low-bit quantization enable LLMs on edge devices](https://www.microsoft.com/en-us/research/blog/advances-to-low-bit-quantization-enable-llms-on-edge-devices/)
[^cpu-gpu-perf]: [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://arxiv.org/pdf/2502.11880)

### INT4量子化で十分なことが多い

さらに言えば、GPUで推論するならAWQやGPTQといったINT4量子化の方が実装も成熟していますし、Tensor Coreをちゃんと活かせますし、精度の劣化もBitNetより少ないです。GPUベースの推論ではINT4で事足りる場面が多く、BitNetの出番がないというのが現状です。

## それでも実験してみた理由

とはいえ、TritonでのGPU実装例はまだ少ないですし、カーネルの書き方次第で結果が変わる可能性もあります。それに、Claude Opusクラスのモデルをスマホでオフライン実行できたら最高だという夢がどうしても捨てきれなかったので、年末年始の時間を使って実際に手を動かしてみることにしました。

## Tritonカーネルの実装

まずBitNetの行列演算をGPUで動かすためのTritonカーネルを書きます。2-bitにパッキングした重みをカーネル内でアンパックしながら行列積を計算するのがポイントになります。

### 2-bitパッキング

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

### Triton MatMulカーネル

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

Tritonカーネルが動くようになったので、まずは実際のモデルで学習できるか確認します。MNISTの画像生成タスクでDiffusion Transformer(DiT)をBitNet化して学習させてみました。

最初は「全部のLinearをBitLinearに置き換えればいいでしょ」と思ってそうしましたが、Lossが1.0付近で停滞して全く学習が進みませんでした。

```python
# 失敗した実装：全てを BitLinear に
self.time_embed = BitLinear(dim, dim)  # ← これが問題
self.adaLN = BitLinear(dim, dim * 4)   # ← これも問題
```

原因は、時間埋め込みとAdaLN（条件付け層）までBitLinearにしてしまったことです。これらの層は連続的な条件情報を扱うので、{-1, 0, +1}に量子化すると情報が壊れてしまいます。時間埋め込みを通常のFP32 Linearに戻して、AttentionとMLPだけBitLinearにしたら50 epochでLoss 0.045まで下がり、ちゃんと数字が認識できる画像が生成されるようになりました。

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

教訓としては、条件付け層（時間埋め込みやクラス埋め込みなど）は量子化してはいけないということです。逆に、AttentionやMLPの重み行列はBitLinearで問題なく動きます。

## 実験2: LUTベースカーネル(BitNet.cpp方式)

学習が動くことは確認できたので、次は推論速度を何とかしたいところです。

BitNet.cppのREADMEを読んでいると、T-MAC[^tmac]という手法でCPU上で大幅な高速化を実現していることが分かりました。4つのternary weightsをグループ化して256エントリのLUT(Lookup Table)を構築し、CPUのSIMD shuffle命令（`vpshufb`）で高速にルックアップすることで乗算を完全に排除しています。これをGPUに移植したら速くなるのでは？と思い試してみました。

[^tmac]: [T-MAC: Table Lookup for Ternary Matrix Multiplication](https://arxiv.org/abs/2407.00088)

GPU上ではTritonの`tl.where`による条件分岐で加算/減算を切り替える形で実装しました。

```python
@triton.jit
def _bitnet_lut_kernel(...):
    c = tl.where(w == 2, x, tl.where(w == 0, -x, 0.0))
    acc += c
```

結果は散々で、LUT方式は`tl.dot`を使う元の実装と全く同じ速度でした。条件分岐を使うnaiveなternary実装に至っては1.5倍遅いです。

| Config | tl.dot | LUT | Ternary |
|--------|--------|-----|---------|
| (1, 4096, 4096) | 0.48 ms | 0.48 ms | 0.70 ms |
| (32, 4096, 4096) | 0.48 ms | 0.48 ms | 0.73 ms |

理由は単純で、CPUではSIMD FMA → LUT + vpshufbという置き換えが高速化に繋がりますが、GPUでは行列演算をTensor Coreがハードウェアレベルで処理するので、LUTに置き換える意味がありません。GPUに`vpshufb`相当の命令もなく、LUTをShared Memoryに置いてもL1キャッシュ上のCPUには敵いません。

## 実験3: 実用性ベンチマーク

速度面では厳しいことが分かってきましたが、メモリ削減効果は確実にあるはずです。実際にどのくらいのインパクトがあるのか定量的に測ってみました。

メモリ使用量は理論通り16倍の圧縮が確認できました（GPT-2で0.11GB → 0.01GB、LLaMA-7Bで13GB → 0.81GB）。一方でスループットはhidden=4096で測定したところ、Batch Size 1でFP16の0.19x、Batch Size 32でも0.20xと、やはりかなり遅いです。Batch Sizeが大きくなるほど差が広がり、512では0.12xまで落ちます。

| Batch Size | FP16 (tok/s) | BitNet (tok/s) | 比率 |
|------------|--------------|----------------|------|
| 1 | 11,087 | 2,136 | 0.19x |
| 32 | 345,879 | 68,919 | 0.20x |
| 512 | 1,997,117 | 237,756 | 0.12x |

ただし面白いのは最大バッチサイズで、メモリが小さい分だけBitNetはFP16の4倍のサンプルを同時に処理できます（FP16: 2,048 vs BitNet: 8,192）。速度が遅くてもバッチを大きく取れるので、スループット全体で見ればメモリが足りない環境では有利になるケースもありそうです。

## まとめ

実験を通じて分かったのは、BitNetの本質はメモリ圧縮であって速度向上ではないということです。GPUのTensor CoreはFP16/FP32に最適化されているので、BitNetにしたところで速くはなりません。LUT方式もCPUのSIMD shuffle命令があってこそ活きるもので、GPUに持ってきても意味がありませんでした。

モデルの設計としては、条件付け層（時間埋め込みやクラス埋め込み）はFP32のままにして、AttentionやMLPだけBitLinearにするのが正解でした。全部BitLinearにすると学習が壊れるというのは、やってみないと分からないことだったと思います。

メモリ16x圧縮の効果自体は本物で、LLaMA-7Bが13GB → 0.81GBになるのは確かにインパクトがあります。ただし速度はFP16の0.2x程度に落ちるので、メモリがボトルネックでない限り選ぶ理由がありません。GPUで速度も求めるならFP16 + Flash Attention、メモリを節約したいならINT4(AWQ/GPTQ)、LLMのサービングならvLLMあたりが現実的な選択肢になります。

結局、スマホでOpusを動かすという当初の夢には程遠いので素直にAnthropicに課金し続けることにします。

## 参考資料

- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)
- [The Era of 1-bit LLMs (BitNet b1.58)](https://arxiv.org/abs/2402.17764)
- [BitNet.cpp](https://github.com/microsoft/BitNet)
