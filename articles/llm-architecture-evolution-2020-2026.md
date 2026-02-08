---
title: "GPT-3で知識が止まっている人のための、LLMアーキテクチャ進化論（2020-2026）"
emoji: "🧠"
type: "tech"
topics: ["llm", "transformer", "deeplearning", "ai", "architecture"]
published: true
---

## はじめに

「LLMって結局、次の単語を予測してるだけでしょ？」

その通りです。基本原理は2020年のGPT-3から変わっていません。でも、その中身はこの6年で別物になりました。

GPT-3は1750億パラメータを持ち、推論時には全パラメータが動いていました。一方、2025-2026年の最新モデル（GPT-5.2、Claude Opus 4.6、Gemini 3など）は、必要な部分だけを動かす疎な活性化、長文処理でメモリを食わない効率的なAttention、答えを出す前に裏で試行錯誤する推論時計算を使っています。

この記事では、GPT-3時代で知識が止まっている人向けに、LLMアーキテクチャの進化を整理します。

:::message
この記事は主に**アーキテクチャの変遷**に焦点を当てています。学習データの質向上、RLHF/DPOなどのアライメント手法、マルチモーダル化などの話題は別の観点として扱います。
:::

:::message
**最新モデルについて（2026年2月時点）**
本記事で言及している最新モデルは、公式発表に基づく実在のモデルです：
- [GPT-5.2](https://openai.com/index/introducing-gpt-5-2/)（2025年12月リリース）
- [Claude Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6)（2026年2月リリース）
- [Gemini 3](https://blog.google/products/gemini/gemini-3/)（2025年11月リリース）
- [Claude 3.7 Sonnet](https://www.anthropic.com/news/claude-3-7-sonnet)（2025年2月リリース）
:::

## 1. 「巨大な一枚岩」の終焉：MoE（Mixture of Experts）

### 1.1 全パラメータ活性化の限界

GPT-3は1750億パラメータすべてを推論時に使います。1トークン生成するのに膨大な計算量が必要で、GPUメモリの帯域幅がボトルネックになります。パラメータを増やせば増やすほど、コストは線形以上に跳ね上がります。

### 1.2 MoEの基本原理

Mixture of Experts（MoE）は、この問題を「専門家の分業」で解決します[^moe-paper]。

```
入力トークン → Router（ルーター） → Expert選択 → 計算 → 出力
                    ↓
              Top-k Experts のみ活性化
```

具体的には：

まずRouter（ゲーティング関数）が入力トークンを見て、どの専門家に任せるかを決めます：

$$
G(x) = \text{Softmax}(x \cdot W_g)
$$

スコアが高いTop-k個の専門家だけが活性化され、それぞれの出力を重み付き和で合成します：

$$
y = \sum_{i \in \text{Top-k}} \frac{G(x)_i}{\sum_{j \in \text{Top-k}} G(x)_j} \cdot \text{Expert}_i(x)
$$

分母で正規化して、重みの合計を1にしています。これで総パラメータ数は巨大でも、推論時に動くのは一部だけという疎な活性化が実現できます。

[^moe-paper]: [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)

### 1.3 最新の工夫：DeepSeekMoE

2024年にDeepSeekが提案した**DeepSeekMoE**[^deepseek-moe]は、専門家を64〜256個に細分化し（従来は8〜16個）、全トークン共通の「共有専門家」を別途用意しました。専門家間の負荷分散のため、補助損失関数も導入しています。

DeepSeek-V3では671Bの総パラメータのうち、1トークンあたり37Bだけが活性化されます（活性化率5.5%）[^deepseek-v3]。

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

ここで、K（Key）とV（Value）のキャッシュが推論時のメモリを圧迫します。長文処理では、このKVキャッシュのサイズが爆発的に増えます。

例えば、32個のAttentionヘッドを持つモデルで128kトークンを処理する場合：
- 各ヘッドのKVキャッシュ：`128k tokens × 128 dim × 2 (K,V) × 2 bytes (FP16) = 64 MB`
- 全ヘッド合計：`64 MB × 32 heads = 2 GB`
- 複数レイヤーでさらに増加

### 2.2 GQA（Grouped-Query Attention）

**GQA**[^gqa-paper]は、複数のQueryヘッドで1つのKey/Valueペアを共有する仕組みです。

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

これで、KVキャッシュのサイズをヘッド数分の1に削減できます（上記の例では32→8で4分の1）。

GPT-4やLlama 3.1などの最新モデルはGQAを採用しています。

[^gqa-paper]: [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)

### 2.3 MLA（Multi-head Latent Attention）

DeepSeek-V3が導入した**MLA**[^mla-paper]は、さらに進んだ圧縮手法です：

$$
K = W_K^{down} W_K^{up} \cdot x, \quad V = W_V^{down} W_V^{up} \cdot x
$$

低ランク行列分解により、KとVを一度低次元空間（潜在空間）に圧縮してからキャッシュします。これで、GQAよりもさらに小さいKVキャッシュで済みます。

DeepSeek-V3では、従来のMHAと比較して約93%のKVキャッシュ削減を達成しています（512kトークンで380GBのVRAMを節約）[^deepseek-v3]。

[^mla-paper]: [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)

### 2.4 RoPE（Rotary Positional Embedding）

従来のTransformerは絶対位置埋め込み（各トークンの位置を固定ベクトルで表現）を使っていました。これだと学習時の最大長を超えられないし、長文での表現力が低くなります。

**RoPE**[^rope-paper]は、位置情報を回転行列として埋め込みます。2次元の場合：

$$
f(x_m, m) = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_m^{(1)} \\ x_m^{(2)} \end{pmatrix}
$$

ここで、$m$はトークン位置、$\theta$は回転角度です。

:::details 実際の実装では
$d$次元ベクトルを$d/2$個の2次元ペアに分割し、各ペアに異なる周波数 $\theta_i = 10000^{-2i/d}$ の回転を適用します。低次元には高周波数（短距離情報）、高次元には低周波数（長距離情報）を割り当てて、多スケールの位置情報を表現します。
:::

これにより、トークン間の相対位置が直接計算でき、学習時より長い文章でも位置情報が破綻しません（外挿性）。

Llama、GPT-NeoX、DeepSeekなど多くのモデルがRoPEを採用しています。YaRN[^yarn-paper]やABF[^abf-paper]などの拡張で、100万トークン超の超長文処理も可能になりました。

[^rope-paper]: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
[^yarn-paper]: [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)
[^abf-paper]: [Attention Bridge: Instruction-Tuned Attention for Long Context](https://arxiv.org/abs/2502.06526)

:::message
**なぜ回転なのか？**
RoPEの核心は、異なる位置のトークンに異なる回転を適用することで、内積が相対位置のみに依存するようになる点です。位置mと位置nに対して：

$$
(R_m q)^T (R_n k) = q^T R_{n-m} k
$$

内積の結果が相対位置 $(n-m)$ のみに依存します。回転行列の直交性により、絶対位置情報は消失し、Attentionに必要な相対的な位置関係だけが保存されます。
:::

## 3. 「反射」から「熟考」へ：推論時計算（Test-Time Compute）

### 3.1 System 1とSystem 2

心理学者ダニエル・カーネマンの「ファスト&スロー」に登場する概念を、LLMの挙動に比喩的に当てはめると：

- System 1（直感）：即座に答えを出す。速いが浅い。
- System 2（論理）：じっくり考えて答えを出す。遅いが深い。

:::message
**注:** LLMに実際に「直感」や「論理」があるわけではありません。これはモデルの挙動を理解するための比喩です。
:::

従来のLLMは「System 1型」でした。入力を受け取ったら即座にトークンを生成します。でも、数学の証明や複雑な論理推論では「じっくり考える」時間が必要です。

### 3.2 OpenAI o1の「思考プロセス」

**OpenAI o1**（2024年9月リリース）[^openai-o1]は、推論時に内部で試行錯誤する仕組みを導入しました：

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

この「内部思考」はChain of Thought（CoT）の発展形で、ユーザーには見せずに裏で複数の推論パスを探索します。

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

つまり、答えを出すまでにどれだけ計算リソースを使うかが性能を左右します。OpenAIのGreg Brockman氏は「推論時計算を10倍にすると、100倍のデータで学習したのと同等の性能向上が得られる」と述べています[^test-time-scaling]。

[^test-time-scaling]: [Reasoning and System 2 Thinking](https://www.youtube.com/watch?v=fGqFVOr5pGE)

:::message alert
**注意：o1の詳細は非公開**
OpenAIはo1の内部実装を公開していません。ここで述べた「思考プロセス」は、公開された挙動やサンプル出力から推測されるメカニズムです。
:::

### 3.4 DeepSeek-R1とオープンソースの挑戦

2025年1月、DeepSeekは**DeepSeek-R1**[^deepseek-r1]をリリースしました。これはo1スタイルの推論時計算をオープンソース化したもので、純粋な強化学習（Pure RL）でCoTを学習し（教師データなし）、RLの過程で自発的に推論パターンを獲得します（Aha Moment）。数学・コード・推論タスクでGPT-4oやClaude 3.5 Sonnetを上回る性能を出しています。

DeepSeek-R1の登場で、推論時計算の仕組みが学術的にも検証可能になりました。

[^deepseek-r1]: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)

## 4. 高速化の裏側：先読みと圧縮

### 4.1 Speculative Decoding（投機的実行）

LLMの生成は逐次的（シーケンシャル）です。1トークンずつしか生成できず、並列化が困難でした。

**Speculative Decoding**[^speculative-decoding]は、CPUの投機的実行にヒントを得た手法です：

1. 小さな高速なモデル（Draft Model）が複数トークンを先読み生成
2. 大きなモデル（Target Model）がそれを一括検証
3. 正しければ採用、間違っていればその時点から大きなモデルで再生成

```
Draft Model（速い）: "The cat sat on the mat"（推測生成）
                              ↓（一括検証）
Target Model（遅い）: "The cat sat on the"まで正解
                      ✓   ✓   ✓   ✓  ✓   ×
                      → 最初の5トークンを採用、6トークン目から再生成
```

この手法で、大きなモデル単体よりも2〜3倍高速に生成できます。

[^speculative-decoding]: [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)

### 4.2 MTP（Multi-Token Prediction）

従来の言語モデルは「次の1トークン」だけを予測していました。**MTP**[^mtp-paper]は、複数先のトークンを同時に予測します：

```
入力: "The cat sat on"

従来: 次の1トークンだけ予測
  → "the" のみ

MTP: 次の5トークンを同時予測
  → "the", "mat", "and", "looked", "around"
```

これにより、1サンプルから複数の予測タスクを学習でき（学習の効率化）、数トークン先を見据えた文脈理解が可能になり（長期依存関係の学習）、Speculative Decodingとの相性も良くなります（推論の高速化）。

Meta（Facebook）の研究では、7Bモデルで最大3倍の学習効率向上が報告されています[^mtp-paper]。

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

### 5.2 アーキテクチャの「密度」とコストのバランス

高性能なモデルは、計算効率（MoEの疎な活性化）、メモリ効率（GQA/MLAのKVキャッシュ削減）、推論品質（Test-Time Computeの深い思考）のバランスを取っています。

### 5.3 次世代アーキテクチャへの期待

Transformerを超える試みも進んでいます。Mamba[^mamba-paper]のようなSSM（State Space Models）は線形計算量で長文を処理できますし、Jamba[^jamba-paper]のようなハイブリッドアーキテクチャも出てきました。GPU/TPUの特性に最適化されたハードウェア対応設計も重要になっています。

[^mamba-paper]: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
[^jamba-paper]: [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887)

LLMは2020年の「巨大な一枚岩」から、2026年の「高度な分業システム」へ進化しました。この流れは今後も続くでしょう。

## 付録：主要モデルのスペック変遷表

| モデル | リリース | 総パラメータ | 活性化パラメータ | Attention | 位置埋め込み | 最大コンテキスト | 特徴的技術 |
|--------|----------|--------------|------------------|-----------|--------------|------------------|------------|
| GPT-3 | 2020/06 | 175B | 175B（全活性） | MHA | Learned | 2k | 巨大な一枚岩 |
| GPT-4 | 2023/03 | ~1.8T（推定） | ~220B（推定） | GQA | Learned | 8k→128k | MoE（推定） |
| Llama 3.1 | 2024/07 | 405B | 405B | GQA | RoPE | 128k | Dense Model |
| DeepSeek-V3 | 2024/12 | 671B | 37B | MLA | RoPE | 128k | Fine-grained MoE |
| GPT-5.2 | 2025/12 | 非公開 | 非公開 | 非公開 | 非公開 | 1M+ | Test-Time Compute |
| Claude Opus 4.6 | 2026/02 | 非公開 | 非公開 | 非公開 | 非公開 | 1M | Agent Teams |
| Gemini 3 | 2025/11 | 非公開 | 非公開 | 非公開 | 非公開 | 非公開 | Multimodal AGI |
| DeepSeek-R1 | 2025/01 | 671B | 37B | MLA | RoPE | 128k | Pure RL推論 |
| Claude 3.7 Sonnet | 2025/02 | 非公開 | 非公開 | 非公開 | 非公開 | 非公開 | Extended Thinking |

:::message
GPT-4、GPT-5.2、Claude Opus 4.6、Gemini 3、Claude 3.7 Sonnetの詳細スペックは公式に未公開です。最大コンテキストなど一部の値は公式発表に基づきます。
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
