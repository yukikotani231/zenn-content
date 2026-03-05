---
title: "SQLの静的検査ツール（CLI/LSP）を公開しました"
emoji: "🧹"
type: "tech"
topics: ["sql", "rust", "ci", "rails", "prisma"]
published: true
---

SQLって、実行時まで壊れ方が見えにくくて地味に緊張感が高いですよね。
たとえばテーブル・列名のタイポ、型の不一致、JOIN条件のミスなどは、実行するまで気づきにくいことがあります^[DB接続して実行前検証する手段はありますが、もっと軽く回せる静的検査の選択肢も欲しい、というモチベーションです。]。

「実行せずに不正なSQL検出できないかな．．．」と思ってそういったツールを探したのですが意外と見つからず。
ということで、作ってみました。CLIとLSPで使えます。

GitHub
https://github.com/yukikotani231/sqlsift

npm (`sqlsift-cli`)
https://www.npmjs.com/package/sqlsift-cli

VS Code Extension
https://marketplace.visualstudio.com/items?itemName=sqlsift.sqlsift

CLI の実行イメージ

![sqlsift cli demo](/images/sample_cli.gif)

LSP の表示イメージ

![sqlsift lsp demo](/images/sample_lsp.gif)

## CLIで試す

とりあえず試したい方は、以下をそのままコピペで実行してみてください。（仮のファイルを作成するので注意）

```bash
cat > schema.sql <<'SQL'
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  email TEXT UNIQUE
);
SQL

cat > query.sql <<'SQL'
SELECT naem, user_id FROM users;
SQL

npx sqlsift-cli check --schema schema.sql query.sql
```

実行結果の例

```text
error[E0002]: Column 'naem' not found
  = help: Did you mean 'name'?

error[E0002]: Column 'user_id' not found
```

## 検出できるもの

記事公開時点で検出できるエラーは以下の通りです。

- `E0001` テーブル未定義
- `E0002` カラム未定義（タイポ候補を提案）
- `E0003` 型不一致（比較、算術、INSERT/UPDATE代入など）
- `E0004` NOT NULL制約違反の可能性（明示的なNULL代入）
- `E0005` INSERTの列数不一致
- `E0006` 曖昧なカラム参照
- `E0007` JOIN条件の型不一致

## 導入方法

CLI、LSPの2形態で配布をしています。

### CLI

単発で試すだけなら `npx` で十分です。

```bash
npx sqlsift-cli check --schema db/structure.sql "queries/**/*.sql"
```

マイグレーションディレクトリを使う場合

```bash
npx sqlsift-cli check --schema-dir prisma/migrations "queries/**/*.sql"
```

グローバルに入れて使う場合

```bash
npm install -g sqlsift-cli
sqlsift check --schema db/structure.sql "queries/**/*.sql"
```

### LSP

VS Code 拡張を入れると、編集中のSQLに診断を出せます。

VS Code Extension
https://marketplace.visualstudio.com/items?itemName=sqlsift.sqlsift

### 設定ファイルについて

`sqlsift.toml` をプロジェクトルートに置くと、スキーマや対象SQLを毎回指定せずに運用できます。

```toml
schema = [
  "db/structure.sql",
]

files = [
  "app/queries/**/*.sql",
  "db/queries/**/*.sql",
]
```

### CI（SARIF）

SARIF出力にも対応しているので、例えばGithub Actionsに入れるとエラー箇所を差分上で見られたりします。
GitHub Code Scanning に載せる最小例です。

```yaml
name: SQL Lint
on: [push, pull_request]
jobs:
  sqlsift:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    steps:
      - uses: actions/checkout@v4
      - run: npx sqlsift-cli check -s schema.sql -f sarif queries/*.sql > results.sarif
      - uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: results.sarif
```

Code Scanning 上での表示イメージ

![sqlsift code scanning](/images/codescanning1.png)
![sqlsift code scanning 2](/images/codescanning2.png)

## 最後に

まだ足りない機能が多い状態ではありますが、ぜひ使ってみていただいて、そして改善点などをissueにあげていただけると泣いて喜びます。

https://github.com/yukikotani231/sqlsift/issues
