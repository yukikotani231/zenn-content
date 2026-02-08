---
title: "Google AI Studio と Vertex AI を両立する Gemini クライアント実装"
emoji: "🤖"
type: "tech"
topics: ["gemini", "vertexai", "googlecloud", "typescript", "nodejs"]
published: false
---

## はじめに

Gemini API を使ったアプリケーションを開発する際、以下の2つの認証方式を使い分けたいケースがあります。

| 認証方式 | 用途 | メリット |
|---------|------|---------|
| **Google AI Studio（APIキー）** | 開発・プロトタイプ | セットアップが簡単 |
| **Vertex AI（サービスアカウント）** | 本番環境 | エンタープライズ向けセキュリティ、SLA |

本記事では、これらを統一的に扱える共通クライアントの実装方法を紹介します。

## 背景：なぜ両立が必要か

### 開発環境での課題

- サービスアカウントの認証情報を全開発者に配布するのはセキュリティリスク
- ローカル開発では API キーの方が手軽

### 本番環境での要件

- Vertex AI の方がセキュリティ・監査面で優れている
- GCP のサービスアカウントで統一的な権限管理が可能
- SLA が提供される

### 追加の考慮事項：リージョンとモデル可用性

Vertex AI では、モデルによって利用可能なリージョンが異なります。

```
gemini-2.5-flash      → asia-northeast1 ✅
gemini-2.5-flash-lite → asia-northeast1 ❌（global, US, EU のみ）
```

この問題を回避するため、`global` リージョンの使用を推奨します。

## 実装

### 依存パッケージ

```bash
npm install @google/genai @google-cloud/storage
```

`@google/genai` は Google AI Studio と Vertex AI の両方に対応した公式 SDK です。

### 型定義

```typescript:types.ts
import type { GoogleGenAI } from '@google/genai'

export type GenAIClientOptions = {
  /** サービスアカウント認証情報ファイルのパス（Vertex AI用） */
  credentialsPath?: string
  /** Gemini API キー（Google AI Studio用） */
  apiKey?: string
  /** Vertex AI のロケーション（デフォルト: asia-northeast1） */
  location?: string
  /** ログ出力関数 */
  logger?: (message: string) => void
}

export type GenAIClientResult = {
  client: GoogleGenAI
  isVertexAI: boolean
}

/** アップロードされたファイル情報（GCS or Files API） */
export type GeminiFileInfo =
  | { type: 'gcs'; gcsUri: string; gcsPath: string; mimeType: string }
  | { type: 'uri'; name: string; uri: string; mimeType: string }
```

### クライアントファクトリ

```typescript:client.ts
import * as fs from 'fs'
import { GoogleGenAI } from '@google/genai'
import type { GenAIClientOptions, GenAIClientResult } from './types'

/**
 * GoogleGenAI クライアントを作成する
 * 優先順位: Vertex AI > Gemini API
 */
export function createGenAIClient(options?: GenAIClientOptions): GenAIClientResult {
  const {
    credentialsPath = process.env.GOOGLE_APPLICATION_CREDENTIALS,
    apiKey = process.env.GEMINI_API_KEY,
    location = process.env.GCP_LOCATION || 'asia-northeast1',
    logger = console.log,
  } = options ?? {}

  // Vertex AI（サービスアカウント認証）
  if (credentialsPath && fs.existsSync(credentialsPath)) {
    const credentialsJson = fs.readFileSync(credentialsPath, 'utf-8')

    let credentials: { project_id?: string }
    try {
      credentials = JSON.parse(credentialsJson)
    } catch (e) {
      throw new Error(
        `認証情報ファイルの解析に失敗しました: ${e instanceof Error ? e.message : String(e)}`
      )
    }

    const projectId = credentials.project_id
    if (!projectId) {
      throw new Error('認証情報ファイルに project_id が見つかりません')
    }

    logger(`[Vertex AI] project: ${projectId}, location: ${location}`)
    return {
      client: new GoogleGenAI({ vertexai: true, project: projectId, location }),
      isVertexAI: true,
    }
  }

  // Gemini API（APIキー認証）
  if (apiKey) {
    logger('[Gemini API] Initializing with API key')
    return {
      client: new GoogleGenAI({ apiKey }),
      isVertexAI: false,
    }
  }

  throw new Error(
    'Gemini認証情報が設定されていません。' +
    'GOOGLE_APPLICATION_CREDENTIALS または GEMINI_API_KEY を設定してください。'
  )
}
```

### ファイルアップロードの抽象化

Vertex AI と Gemini API ではファイルアップロードの方法が異なります。

| 認証方式 | ファイルアップロード |
|---------|-------------------|
| Vertex AI | GCS（Google Cloud Storage）経由 |
| Gemini API | Files API |

```typescript:file.ts
import { type GoogleGenAI, type Part, createPartFromUri } from '@google/genai'
import { Storage } from '@google-cloud/storage'
import type { GeminiFileInfo } from './types'

// Storage インスタンスをシングルトンで再利用
let storageInstance: Storage | null = null
function getStorage(): Storage {
  if (!storageInstance) {
    storageInstance = new Storage()
  }
  return storageInstance
}

/**
 * Buffer を GCS にアップロード（Vertex AI用）
 */
async function uploadBufferToGCS(
  buffer: Buffer,
  mimeType: string,
  bucketName: string
): Promise<{ gcsUri: string; gcsPath: string }> {
  const storage = getStorage()
  const bucket = storage.bucket(bucketName)
  const extension = mimeType.split('/')[1] || 'bin'
  const fileName = `temp/${Date.now()}-${Math.random().toString(36).substring(7)}.${extension}`
  const file = bucket.file(fileName)

  await file.save(buffer, { contentType: mimeType, resumable: false })

  return {
    gcsUri: `gs://${bucketName}/${fileName}`,
    gcsPath: fileName,
  }
}

/**
 * Buffer を Gemini Files API にアップロード
 */
async function uploadBufferToGemini(
  ai: GoogleGenAI,
  buffer: Buffer,
  mimeType: string
): Promise<{ uri: string; name: string; mimeType: string }> {
  const blob = new Blob([buffer], { type: mimeType })
  const uploaded = await ai.files.upload({ file: blob, config: { mimeType } })

  if (!uploaded.uri || !uploaded.name) {
    throw new Error('Failed to upload file to Gemini Files API')
  }

  return { uri: uploaded.uri, mimeType, name: uploaded.name }
}

/**
 * 認証方式に応じてファイルをアップロード
 */
export async function uploadBuffer(
  ai: GoogleGenAI,
  buffer: Buffer,
  mimeType: string,
  options: { isVertexAI: boolean; bucketName?: string }
): Promise<GeminiFileInfo> {
  const { isVertexAI, bucketName = process.env.VERTEX_AI_GCS_BUCKET } = options

  if (isVertexAI) {
    if (!bucketName) {
      throw new Error('VERTEX_AI_GCS_BUCKET is required for Vertex AI')
    }
    const result = await uploadBufferToGCS(buffer, mimeType, bucketName)
    return { type: 'gcs', ...result, mimeType }
  } else {
    const result = await uploadBufferToGemini(ai, buffer, mimeType)
    return { type: 'uri', ...result }
  }
}

/**
 * GeminiFileInfo から Part を作成
 */
export function createFilePart(fileInfo: GeminiFileInfo): Part {
  if (fileInfo.type === 'gcs') {
    return {
      fileData: {
        mimeType: fileInfo.mimeType,
        fileUri: fileInfo.gcsUri,
      },
    }
  } else {
    return createPartFromUri(fileInfo.uri, fileInfo.mimeType)
  }
}

/**
 * アップロード済みファイルを削除
 */
export async function deleteFile(
  ai: GoogleGenAI,
  fileInfo: GeminiFileInfo,
  bucketName?: string
): Promise<void> {
  if (fileInfo.type === 'gcs') {
    const storage = getStorage()
    const bucket = bucketName || process.env.VERTEX_AI_GCS_BUCKET
    if (bucket) {
      await storage.bucket(bucket).file(fileInfo.gcsPath).delete().catch(() => {})
    }
  } else {
    const name = fileInfo.name.startsWith('files/') ? fileInfo.name : `files/${fileInfo.name}`
    await ai.files.delete({ name }).catch(() => {})
  }
}
```

## 使用例

### 画像解析の実装例

```typescript:analyze.ts
import { createGenAIClient, uploadBuffer, createFilePart, deleteFile } from './genai-client'

async function analyzeImage(imageBuffer: Buffer, mimeType: string): Promise<string> {
  const { client: ai, isVertexAI } = createGenAIClient()

  let fileInfo = null
  try {
    // ファイルをアップロード
    fileInfo = await uploadBuffer(ai, imageBuffer, mimeType, { isVertexAI })

    // Gemini で画像解析
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: [
        {
          role: 'user',
          parts: [
            { text: 'この画像の内容を説明してください。' },
            createFilePart(fileInfo),
          ],
        },
      ],
    })

    return response.text ?? ''
  } finally {
    // 一時ファイルを削除
    if (fileInfo) {
      await deleteFile(ai, fileInfo)
    }
  }
}
```

## 環境変数

```bash
# Gemini API（開発環境）
GEMINI_API_KEY=your-api-key

# Vertex AI（本番環境）
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GCP_LOCATION=global
VERTEX_AI_GCS_BUCKET=your-bucket-name
```

:::message
`GCP_LOCATION=global` を推奨します。`gemini-2.5-flash-lite` など一部モデルは特定リージョン（asia-northeast1 等）では利用できません。
:::

## まとめ

この実装により、以下のメリットが得られます。

1. **環境に応じた自動切り替え** - 環境変数だけで認証方式を切り替え可能
2. **コードの共通化** - 呼び出し側は認証方式を意識する必要がない
3. **ファイルアップロードの抽象化** - GCS / Files API の違いを隠蔽
4. **本番環境でのセキュリティ** - Vertex AI のサービスアカウント認証を活用

開発環境では API キーで手軽に、本番環境では Vertex AI で堅牢に運用できる柔軟な構成が実現できます。

## 参考リンク

- [Google AI for Developers - Gemini API](https://ai.google.dev/)
- [Vertex AI - Generative AI](https://cloud.google.com/vertex-ai/generative-ai/docs/overview)
- [@google/genai - npm](https://www.npmjs.com/package/@google/genai)
- [Gemini 2.5 Flash-Lite | Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash-lite)
