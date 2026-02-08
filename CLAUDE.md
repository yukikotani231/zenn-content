# Zenn記事管理リポジトリ - Claude Codeコンテキスト

## リポジトリについて

このリポジトリはZenn（日本の技術ブログプラットフォーム）の記事を管理するためのものです。
記事はMarkdown形式で書かれ、GitHubリポジトリと連携して公開されます。

## ディレクトリ構造

- `articles/`: 記事ファイル（Markdown）を格納
  - ファイル名: `{slug}.md`（slugは記事のURL識別子）
- `books/`: 本（複数記事をまとめたもの）を格納
- `.github/workflows/`: CI/CD設定

## 記事フォーマット

各記事は以下のフロントマターで始まります：

```markdown
---
title: "記事タイトル"
emoji: "📝"
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["topic1", "topic2"] # 最大5個
published: true # true: 公開 / false: 下書き
---
```

## 文体とスタイル

- **文体**: ですます調で統一
- **コードブロック**: 言語指定を必ず行う（```python など）
- **見出し**: `##` から開始（`#` はタイトルで使用済み）
- **リンク**: 外部リンクは必要に応じてリンクカードを使用

## Zenn固有の記法

- メッセージボックス: `:::message` `:::alert` `:::details`
- 数式: `$$...$$`（KaTeX記法）
- 埋め込み: `@[youtube](動画ID)` など

## レビュー時の注意事項

- 技術的正確性を最優先に確認
- 読者が初学者でも理解できるかを考慮
- コード例は実際に動作するものか検証
- 日本語として自然な表現か確認
- Zennの記事として適切なトピック選定か確認

## 参考リンク

- [Zenn CLIガイド](https://zenn.dev/zenn/articles/zenn-cli-guide)
- [ZennのMarkdown記法](https://zenn.dev/zenn/articles/markdown-guide)
