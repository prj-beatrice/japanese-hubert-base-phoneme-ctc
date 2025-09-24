"""
HuggingFace transformersを使用した日本語音素認識用HuBERTモデル
"""

import json
import os
import tempfile

from transformers import (
    HubertConfig,
    HubertForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

from .phoneme_labeling import PHONEME_VOCAB


def create_hubert_model_and_processor(
    model_name: str = "rinna/japanese-hubert-base", freeze_feature_encoder: bool = True
):
    """日本語音素認識用のHuBERTモデルとプロセッサーを作成"""

    # 一時語彙ファイルでトークナイザーを作成
    vocab_dict = {phoneme: i for i, phoneme in enumerate(PHONEME_VOCAB)}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(vocab_dict, f)
        temp_vocab_path = f.name

    try:
        tokenizer = Wav2Vec2CTCTokenizer(
            temp_vocab_path,
            unk_token="UNK",
            pad_token="PAD",
            word_delimiter_token="UNK",
            bos_token="SOS",
            eos_token="EOS",
        )
    finally:
        os.unlink(temp_vocab_path)

    # 事前学習モデルから特徴抽出器を作成
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )

    # ベースHuBERTモデル設定を読み込み
    hubert_config = HubertConfig.from_pretrained(model_name)

    # CTC用に設定を更新
    hubert_config.vocab_size = len(PHONEME_VOCAB)
    hubert_config.pad_token_id = 0  # PAD token id
    hubert_config.word_delimiter_token_id = 1  # UNK token id

    model = HubertForCTC.from_pretrained(
        model_name,
        config=hubert_config,
        ignore_mismatched_sizes=True,  # 語彙サイズの不一致を許可
    )

    # 指定されていれば特徴エンコーダーを凍結
    if freeze_feature_encoder:
        model.hubert.feature_extractor._freeze_parameters()
        model.hubert.feature_projection.requires_grad_(False)

    return model, processor
