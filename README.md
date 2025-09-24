## 注意

**このリポジトリのコードやドキュメントのほとんどは Claude Code や Codex によって生成されており、不正確なコメント等が多数含まれている可能性があります。**
**あくまで学習済みモデルがどのように構築されたか確認可能にするために公開しており、コードの実用は推奨しません。**

# japanese-hubert-base-phoneme-ctc

日本語音声から音素列を推定する HuBERT ベースのモデル。

## セットアップ

```bash
uv sync
```

## 学習
```bash
make train
```

## モデル

学習済みモデル: [prj-beatrice/japanese-hubert-base-phoneme-ctc-v3](https://huggingface.co/prj-beatrice/japanese-hubert-base-phoneme-ctc-v3)
