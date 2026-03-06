---
title: "ターミナルからコードレビューできるようにしました"
emoji: "🧪"
type: "tech"
topics: ["github", "cli", "go", "tui", "review"]
published: true
---

# CLIでコードレビュー、したくない．．．？ 

最近Claude CodeやCodexがアツいですよね。メインの作業場がVSCodeからターミナルになったという方も多いのではないでしょうか。
筆者もその一人で、エディタさえもneovimに移行してしまいました。そしていつしか、「ターミナルから出たくねえな．．．全てターミナルで済むようになれば楽なのにな．．．」という思いを抱くようになりました。
しかし、世の中にはターミナルの世界から一歩踏み出さなければならない場面がまだまだ多くあります。
例えば、**コードレビュー**はその一つとして大きいものではないでしょうか。

ということで、GitHub CLIの拡張機能 `gh-pr-review` を作ってみました。

![gh-pr-review main](/images/gh-pr-review_main.png)

https://github.com/yukikotani231/gh-pr-review

## 使用方法

以下は`gh`コマンドが入っている前提になるので、まだ入ってない方は導入と`gh auth login`まで済ませて下さい。
まずは以下コマンドで拡張機能をインストールします。

```bash
gh extension install yukikotani231/gh-pr-review
```

そして以下で起動します。

```bash
# PR番号を指定して起動
gh pr-review 123

# 引数なしで起動（PR選択UIが開きます）
gh pr-review
```

起動後の操作

- `j` / `k` で移動
- `c` で行コメント
- `r` でスレッド返信
- `R` で resolve / unresolve
- `S` でレビュー送信

メイン画面

![gh-pr-review main](/images/gh-pr-review_main.png)

PR選択画面（引数なし起動時）

![gh-pr-review select pr](/images/gh-pr-review_select-pr.png)

コメント入力

![gh-pr-review comment](/images/gh-pr-review_comment.png)

## できること

- diff 閲覧（hunk移動、スレッドジャンプ）
- 行コメント追加（`c`）
- 返信（`r`）/ resolve 切り替え（`R`）
- Approve / Request Changes / Comment 送信（`S`）
- ファイル既読管理（`v`）

## 導入時の注意

- `gh auth login` が未実施だと API 呼び出しで失敗する
- 権限不足のリポジトリでは PR 情報を取得できない場合がある

## 制限

画面幅が狭い環境だと、レイアウトがやや窮屈に見えることがあります。
また現状は、GitHubのブラウザ差分表示ほどの見やすさにはまだ届いていない部分があります。
このあたりは今後のバージョンで改善していく予定です。

## 最後に

ターミナルから出ない生活をしたい方はぜひお使いください。
バグや改善点はリポジトリにIssueとしてあげていただけると非常に助かりますので、よろしくお願いいたします。

https://github.com/yukikotani231/gh-pr-review
