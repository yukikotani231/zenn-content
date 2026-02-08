---
title: "GPT-3で知識が止まっている人のための、LLMアーキテクチャ進化論（2020-2026）"
emoji: "🧠"
type: "tech"
topics: ["llm", "transformer", "deeplearning", "ai", "architecture"]
published: false
---

## はじめに

「大規模言語モデル（LLM）って、要するに次に来る単語を予測するだけでしょ？」

確かにその基本設定は2020年のGPT-3から2026年の現在まで変わっていません。しかし、その「予測」の**内側**は劇的に進化しました。

2020年のGPT-3は1750億パラメータの**巨大な一枚岩**でした。推論時には全パラメータが動き、膨大な計算リソースを消費しました。一方、2026年の最新モデル（GPT-5.2、Claude 4.6、Gemini 3など）は、同等以上の性能を持ちながらも：

- **疎な活性化（Sparsity）**：全パラメータではなく必要な部分だけを動かす
- **効率的な記憶（Efficient Memory）**：長文処理時のメモリ消費を劇的に削減
- **推論時計算（Test-Time Compute）**：答えを出す前に「裏で考える」仕組み

といった革新的なアーキテクチャを採用しています。

この記事では、**GPT-3時代で知識が止まっている方**に向けて、LLMアーキテクチャがこの6年間でどう進化したのかを、技術的な詳細を交えながら整理します。

:::message
この記事は主に**アーキテクチャの変遷**に焦点を当てています。学習データの質向上、RLHF/DPOなどのアライメント手法、マルチモーダル化などの話題は別の観点として扱います。
:::

## 1. 「巨大な一枚岩」の終焉：MoE（Mixture of Experts）

### 1.1 全パラメータ活性化の限界

GPT-3（2020年）は1750億パラメータを持ち、推論時には**すべてのパラメータ**が計算に関与しました。これは以下の問題を引き起こしました：

- **推論コストの増大**：1トークン生成に膨大なFLOPs（浮動小数点演算）が必要
- **メモリ帯域幅のボトルネック**：GPUメモリとの帯域幅が律速になる
- **スケーリングの限界**：パラメータを増やすほどコストが線形以上に増加

### 1.2 MoEの基本原理

**Mixture of Experts（MoE）**は、この問題を「専門家の分業」で解決します[^moe-paper]。

```
入力トークン → Router（ルーター） → Expert選択 → 計算 → 出力
                    ↓
              Top-k Experts のみ活性化
```

具体的には：

1. **Router（ゲーティング関数）**：入力トークンを見て「どの専門家に任せるか」を決定
   $$
   G(x) = \text{Softmax}(x \cdot W_g)
   $$

2. **Top-k選択**：スコアが高いk個の専門家（Expert）のみを活性化

3. **専門家の計算**：選ばれた専門家だけがFFN（Feed-Forward Network）を実行
   $$
   y = \sum_{i \in \text{Top-k}} G(x)_i \cdot \text{Expert}_i(x)
   $$

この仕組みにより、**総パラメータ数は巨大だが、推論時に動くのは一部だけ**という「疎な活性化（Sparse Activation）」が実現されます。

[^moe-paper]: [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)

### 1.3 最新の工夫：DeepSeekMoE

2024年にDeepSeekが提案したDeepSeekMoE[^deepseek-moe]は、MoEの効率をさらに高めました：

- **細粒度の専門家（Fine-grained Experts）**：従来の8〜16個ではなく、64〜256個の小さな専門家に分割
- **共有専門家（Shared Experts）**：すべてのトークンに共通する知識を扱う専門家を別途用意
- **負荷分散の工夫**：専門家間の計算量バランスを保つための補助損失関数

例えば、DeepSeek-V3では671Bの総パラメータのうち、1トークンあたり**37Bだけ**が活性化されます（活性化率5.5%）[^deepseek-v3]。

[^deepseek-moe]: [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066)
[^deepseek-v3]: [DeepSeek-V3 Technical Report](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf)

:::details MoEのトレードオフ
**メリット：**
- 推論時の計算コスト削減（活性化パラメータが少ない）
- スケーラビリティ（専門家を増やすだけで性能向上）

**デメリット：**
- モデルサイズが巨大（全専門家をメモリに載せる必要）
- 学習時の負荷分散が難しい（特定の専門家に負荷が集中しがち）
- 推論時の遅延増加の可能性（動的なルーティングのオーバーヘッド）
:::

## 2. 「記憶」のパラダイムシフト：Attention機構の軽量化

### 2.1 Multi-Head Attentionのメモリ問題

Transformerの中核であるAttention機構は、以下の計算を行います：

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

ここで、**K（Key）とV（Value）のキャッシュ**が推論時のメモリを圧迫します。長文処理（コンテキスト長が大きい）では、このKVキャッシュのサイズが爆発的に増加します。

例えば、32個のAttentionヘッドを持つモデルで128kトークンを処理する場合：
- 各ヘッドのKVキャッシュ：`128k tokens × 128 dim × 2 (K,V) × 2 bytes (FP16) = 64 MB`
- 全ヘッド合計：`64 MB × 32 heads = 2 GB`
- 複数レイヤーでさらに増加

### 2.2 GQA（Grouped-Query Attention）

GQA[^gqa-paper]は、複数のQueryヘッドで**1つのKey/Valueペアを共有**する仕組みです。

```
従来のMHA（Multi-Head Attention）：
Q1, K1, V1
Q2, K2, V2
...
Q32, K32, V32

GQA（例：8グループ）：
Q1-Q4 → K1, V1（共有）
Q5-Q8 → K2, V2（共有）
...
Q29-Q32 → K8, V8（共有）
```

これにより、KVキャッシュのサイズを**ヘッド数分の1**に削減できます（上記の例では32→8で4分の1）。

GPT-4やLlama 3.1などの最新モデルはGQAを採用しています。

[^gqa-paper]: [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)

### 2.3 MLA（Multi-head Latent Attention）

DeepSeek-V3が導入したMLA[^mla-paper]は、さらに進んだ圧縮手法です：

$$
K = W_K^{down} W_K^{up} \cdot x, \quad V = W_V^{down} W_V^{up} \cdot x
$$

**低ランク行列分解（Low-Rank Factorization）**により、KとVを一度低次元空間（潜在空間）に圧縮してからキャッシュします。これにより、GQAよりもさらに小さいKVキャッシュで済みます。

DeepSeek-V3では、従来のMLAよりも**約93%のKVキャッシュ削減**を達成しています（512kトークンで380GBのVRAMを節約）[^deepseek-v3]。

[^mla-paper]: [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)

### 2.4 RoPE（Rotary Positional Embedding）

従来のTransformerは**絶対位置埋め込み**（各トークンの位置を固定ベクトルで表現）を使っていました。しかし、これには以下の問題がありました：

- 学習時の最大長を超える推論ができない
- 長文での位置情報の表現力が低い

**RoPE（Rotary Positional Embedding）**[^rope-paper]は、位置情報を**回転行列**として埋め込みます：

$$
f(x_m, m) = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_m^{(1)} \\ x_m^{(2)} \end{pmatrix}
$$

ここで、$m$はトークン位置、$\theta$は回転角度です。この仕組みにより：

- **相対位置の表現**：トークン間の距離が直接計算できる
- **外挿性（Extrapolation）**：学習時より長い文章でも位置情報が破綻しない

RoPEは、Llama、GPT-NeoX、DeepSeekなど多くのモデルで標準採用されています。さらに、YaRN[^yarn-paper]やABF[^abf-paper]などの拡張により、**1M（100万）トークン超**の超長文処理が可能になりました。

[^rope-paper]: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
[^yarn-paper]: [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)
[^abf-paper]: [Attention Bridge: Instruction-Tuned Attention for Long Context](https://arxiv.org/abs/2502.06526)

:::message
**なぜ回転なのか？**
回転行列は内積を保存します。つまり、`rot(x) · rot(y) = x · y`が成り立つため、Attention計算（内積ベース）と相性が良いのです。また、回転角度の差が相対位置を表すという幾何学的な美しさもあります。
:::

## 3. 「反射」から「熟考」へ：推論時計算（Test-Time Compute）

### 3.1 System 1とSystem 2

心理学者ダニエル・カーネマンの「ファスト&スロー」に登場する概念：

- **System 1（直感）**：即座に答えを出す。速いが浅い。
- **System 2（論理）**：じっくり考えて答えを出す。遅いが深い。

従来のLLMは「System 1型」でした。入力を受け取ったら即座にトークンを生成します。しかし、数学の証明や複雑な論理推論では「じっくり考える」時間が必要です。

### 3.2 OpenAI o1の「思考プロセス」

OpenAI o1（2024年9月リリース）[^openai-o1]は、**推論時に内部で試行錯誤する**仕組みを導入しました：

```
ユーザー入力
  ↓
[内部思考プロセス]
  ├─ 仮説1を立てる → 検証 → 失敗
  ├─ 仮説2を立てる → 検証 → 部分的成功
  └─ 仮説3を立てる → 検証 → 成功
  ↓
最終的な回答を出力
```

この「内部思考」は**Chain of Thought（CoT）**の発展形で、ユーザーには見せずに裏で複数の推論パスを探索します。

[^openai-o1]: [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)

### 3.3 推論時計算のスケーリング法則

従来のスケーリング法則：
$$
\text{性能} \propto \text{パラメータ数}^{\alpha} \times \text{データ量}^{\beta}
$$

新しいスケーリング法則：
$$
\text{性能} \propto \text{推論時計算量}^{\gamma}
$$

つまり、**答えを出すまでにどれだけ計算リソースを使うか**が性能を左右します。OpenAIのGreg Brockman氏は「推論時計算を10倍にすると、100倍のデータで学習したのと同等の性能向上が得られる」と述べています[^test-time-scaling]。

[^test-time-scaling]: [Reasoning and System 2 Thinking](https://www.youtube.com/watch?v=fGqFVOr5pGE)

:::message alert
**注意：o1の詳細は非公開**
OpenAIはo1の内部実装を公開していません。ここで述べた「思考プロセス」は、公開された挙動やサンプル出力から推測されるメカニズムです。
:::

### 3.4 DeepSeek-R1とオープンソースの挑戦

2025年1月、DeepSeekは**DeepSeek-R1**[^deepseek-r1]をリリースしました。これはo1スタイルの推論時計算をオープンソース化したもので：

- **純粋な強化学習（Pure RL）**でCoTを学習（教師データなし）
- **Aha Moment（ひらめきの瞬間）**：RLの過程で自発的に推論パターンを獲得
- **数学・コード・推論タスク**でGPT-4oやClaude 3.7 Sonnetを上回る性能

DeepSeek-R1の登場により、推論時計算の仕組みが学術的にも検証可能になりました。

[^deepseek-r1]: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)

## 4. 高速化の裏側：先読みと圧縮

### 4.1 Speculative Decoding（投機的実行）

LLMの生成は**逐次的（シーケンシャル）**です。1トークンずつしか生成できず、並列化が困難でした。

**Speculative Decoding**[^speculative-decoding]は、CPUの投機的実行（Speculative Execution）にヒントを得た手法です：

1. **小さな高速なモデル（Draft Model）**が複数トークンを先読み生成
2. **大きなモデル（Target Model）**がそれを一括検証
3. 正しければ採用、間違っていればその時点から大きなモデルで再生成

```
Draft Model（速い）: "The cat sat on the"
                              ↓（一括検証）
Target Model（遅い）: "The cat sat on the mat"
                      ✓   ✓   ✓   ✓  ✓   ×（mat）
                      → "mat"を採用して次へ
```

この手法により、大きなモデル単体よりも**2〜3倍高速**に生成できます。

[^speculative-decoding]: [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)

### 4.2 MTP（Multi-Token Prediction）

従来の言語モデルは「次の1トークン」だけを予測していました。**MTP（Multi-Token Prediction）**[^mtp-paper]は、**複数先のトークンを同時に予測**します：

```
入力: "The cat sat on"

従来: 次の1トークンだけ予測
  → "the" のみ

MTP: 次の5トークンを同時予測
  → "the", "mat", "and", "looked", "around"
```

これにより：

- **学習の効率化**：1サンプルから複数の予測タスクを学習
- **長期依存関係の学習**：数トークン先を見据えた文脈理解
- **推論の高速化**：Speculative Decodingとの相性が良い

Meta（Facebook）の研究では、7Bモデルで**最大3倍の学習効率向上**が報告されています[^mtp-paper]。

[^mtp-paper]: [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)

## 5. まとめ：2026年のエンジニアが押さえておくべき視点

### 5.1 パラメータ数だけでは性能を測れない時代

「このモデルは何兆パラメータ？」という問いは、もはや本質的ではありません。

重要なのは：

| 指標 | 説明 | 例 |
|------|------|-----|
| **活性化パラメータ数** | 推論時に実際に動くパラメータ | DeepSeek-V3: 671B → 37B活性化 |
| **推論時FLOPs** | 1トークン生成に必要な演算量 | MoEで大幅削減 |
| **KVキャッシュサイズ** | 長文処理時のメモリ消費 | MLA/GQAで削減 |
| **Test-Time Compute** | 推論時にどれだけ計算するか | o1/R1スタイル |

### 5.2 アーキテクチャの「密度（Density）」とコストのバランス

高性能なモデルは、以下のバランスを最適化しています：

- **計算効率**：MoEによる疎な活性化
- **メモリ効率**：GQA/MLAによるKVキャッシュ削減
- **推論品質**：Test-Time Computeによる深い思考

### 5.3 次世代アーキテクチャへの期待

Transformerを超える試みも進行中です：

- **SSM（State Space Models）**：Mamba[^mamba-paper]など、線形計算量で長文処理
- **Hybrid Architectures**：TransformerとSSMの融合（Jamba[^jamba-paper]など）
- **Hardware-Aware Design**：GPUやTPUの特性に最適化された構造

[^mamba-paper]: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
[^jamba-paper]: [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887)

LLMのアーキテクチャは、2020年の「巨大な一枚岩」から、2026年の「高度な分業システム」へと進化しました。この進化は今後も加速していくでしょう。

## 付録：主要モデルのスペック変遷表

| モデル | リリース | 総パラメータ | 活性化パラメータ | Attention | 位置埋め込み | 最大コンテキスト | 特徴的技術 |
|--------|----------|--------------|------------------|-----------|--------------|------------------|------------|
| GPT-3 | 2020/06 | 175B | 175B（全活性） | MHA | Learned | 2k | 巨大な一枚岩 |
| GPT-4 | 2023/03 | ~1.8T（推定） | ~220B（推定） | GQA | Learned | 8k→128k | MoE（推定） |
| Llama 3.1 | 2024/07 | 405B | 405B | GQA | RoPE | 128k | Dense Model |
| DeepSeek-V3 | 2024/12 | 671B | 37B | MLA | RoPE | 128k | Fine-grained MoE |
| GPT-5.2 | 2025/12 | 非公開 | 非公開 | 非公開 | 非公開 | 1M+ | Test-Time Compute |
| Claude 4.6 | 2026/01 | 非公開 | 非公開 | 非公開 | 非公開 | 500k | 長文処理特化 |
| DeepSeek-R1 | 2025/01 | 671B | 37B | MLA | RoPE | 128k | Pure RL推論 |

:::message
GPT-4、GPT-5.2、Claude 4.6の詳細スペックは公式に未公開です。表中の値は公開情報や技術レポートからの推定を含みます。
:::

## 参考文献

主要な論文・技術レポート：

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原論文
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - GPT-3
- [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538) - MoE
- [DeepSeek-V3 Technical Report](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf)
- [DeepSeek-R1 Technical Report](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)
- [RoFormer](https://arxiv.org/abs/2104.09864) - RoPE
- [GQA Paper](https://arxiv.org/abs/2305.13245)
- [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/) - OpenAI o1

---

この記事が、LLMアーキテクチャの進化を理解する一助となれば幸いです。技術的な誤りや補足事項があれば、コメントでお知らせください。
