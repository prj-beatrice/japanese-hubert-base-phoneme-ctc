"""
HuBERTを使用した日本語音素認識用の簡略化HuggingFace Trainerセットアップ
"""

from typing import Optional

import torch
from transformers import Trainer, TrainingArguments

from .dataset import DataCollatorCTCWithPadding, prepare_dataset
from .model import create_hubert_model_and_processor
from .phoneme_candidates import CandidatePhoneme, CandidateScorer


def create_trainer(
    # モデル設定
    model_name: str = "rinna/japanese-hubert-base",
    freeze_feature_encoder: bool = True,
    # データセット設定
    max_audio_length_samples: int = 250000,
    cache_dir: str = "./transcriptions",
    # カスタム学習設定
    head_learning_rate: Optional[float] = None,
    **training_args_kwargs,
) -> Trainer:
    """
    日本語音素認識用のHuggingFace Trainerを作成

    Args:
        model_name: ファインチューニングするベースHuBERTモデル
        freeze_feature_encoder: 特徴エンコーダー層を凍結するか
        max_audio_length_samples: サンプル単位での最大音声長
        cache_dir: 書き起こしファイルを含むディレクトリ
        head_learning_rate: 新しく追加されたヘッド層（CTCヘッド）の学習率。Noneの場合、learning_rateと同じ値を使用
        **training_args_kwargs: TrainingArgumentsに渡される引数（output_dir、learning_rateなどを含む）

    Returns:
        設定済みHuggingFace Trainer
    """
    model, processor = create_hubert_model_and_processor(
        model_name=model_name,
        freeze_feature_encoder=freeze_feature_encoder,
    )

    # データセットを準備
    train_dataset = prepare_dataset(
        processor=processor,
        max_audio_length_samples=max_audio_length_samples,
        cache_dir=cache_dir,
    )
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # ストリーミングデータセット用の適切なデフォルト値で学習引数をセットアップ
    training_args_defaults = {
        "output_dir": "outputs",
        "dataloader_num_workers": 0,  # ストリーミング用にシングルスレッド
        "dataloader_prefetch_factor": 8,
        "group_by_length": False,  # ストリーミングデータセットには適用不可
        "hub_private_repo": True,
        "dataloader_drop_last": True,  # accelerateバッチサイズ不一致の修正
        "ddp_find_unused_parameters": True,  # 分散学習用
        "accelerator_config": {"dispatch_batches": False},
        "remove_unused_columns": False,
    }

    training_args_defaults.update(training_args_kwargs)
    training_args = TrainingArguments(**training_args_defaults)

    # 必要に応じて差分学習率でオプティマイザーを作成
    optimizers = (None, None)  # デフォルト: Trainerにオプティマイザー作成を委ねる

    if head_learning_rate is not None:
        # 必要なオプティマイザーパラメータが提供されていることを確認
        required_params = [
            "learning_rate",
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "weight_decay",
        ]
        missing_params = [p for p in required_params if p not in training_args_kwargs]
        if missing_params:
            raise ValueError(
                f"When using head_learning_rate, the following parameters must be specified: {missing_params}"
            )

        # パラメータをグループに分離
        pretrained_params = []
        head_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                # lm_headはCTCヘッド（新しく追加された層）
                if "lm_head" in name:
                    head_params.append(param)
                else:
                    pretrained_params.append(param)

        # 異なる学習率でパラメータグループを作成
        optimizer_grouped_parameters = []

        if pretrained_params:
            optimizer_grouped_parameters.append(
                {
                    "params": pretrained_params,
                    "lr": training_args_kwargs["learning_rate"],
                }
            )

        if head_params:
            optimizer_grouped_parameters.append(
                {
                    "params": head_params,
                    "lr": head_learning_rate,
                }
            )

        # オプティマイザーを作成
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            betas=(
                training_args_kwargs["adam_beta1"],
                training_args_kwargs["adam_beta2"],
            ),
            eps=training_args_kwargs["adam_epsilon"],
            weight_decay=training_args_kwargs["weight_decay"],
        )

        optimizers = (optimizer, None)  # (optimizer, lr_scheduler=None)

        # パラメータグループをログ出力
        print(
            f"Created optimizer with {len(optimizer_grouped_parameters)} parameter groups:"
        )
        for i, group in enumerate(optimizer_grouped_parameters):
            print(f"  Group {i}: {len(group['params'])} parameters, lr={group['lr']}")

    # トレーナーを作成して返す
    trainer = HubertTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        optimizers=optimizers,
        candidate_device=training_args.device,
    )

    print("Created HuBERT trainer")
    print(f"Model vocab size: {model.config.vocab_size}")

    return trainer


class HubertTrainer(Trainer):
    """候補音素のスコアリングをメインスレッドで行うTrainer"""

    def __init__(
        self, *args, candidate_device: Optional[torch.device] = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        device = candidate_device or self.args.device
        self._candidate_device = torch.device(device)
        self._candidate_scorer: Optional[CandidateScorer] = None

    def _ensure_candidate_scorer(self) -> CandidateScorer:
        if self._candidate_scorer is None:
            self._candidate_scorer = CandidateScorer(device=self._candidate_device)
        return self._candidate_scorer

    def _prepare_inputs(self, inputs):
        candidate_batches_raw = inputs.pop("candidate_batches", None)

        candidate_input_values = inputs.pop("candidate_input_values", None)
        candidate_attention_mask = inputs.pop("candidate_attention_mask", None)

        assert candidate_input_values is not None

        candidate_batches = [
            [CandidatePhoneme(**entry) for entry in batch]
            for batch in candidate_batches_raw
        ]

        scorer = self._ensure_candidate_scorer()
        best_candidates = scorer.select_best(
            candidate_input_values=candidate_input_values,
            candidate_attention_mask=candidate_attention_mask,
            batch_candidates=candidate_batches,
        )

        tokenizer = self.data_collator.processor.tokenizer

        max_length = max(len(candidate["phonemes"]) for candidate in best_candidates)
        labels = torch.full(
            (len(best_candidates), max_length),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )

        for idx, candidate in enumerate(best_candidates):
            phoneme_ids = tokenizer.convert_tokens_to_ids(candidate["phonemes"])
            seq_length = len(phoneme_ids)
            labels[idx, :seq_length] = torch.tensor(
                phoneme_ids,
                dtype=torch.long,
            )

        inputs["labels"] = labels.masked_fill(labels == tokenizer.pad_token_id, -100)

        return super()._prepare_inputs(inputs)


def save_model_with_processor(trainer: Trainer, save_directory: str):
    """
    モデルとプロセッサーの両方をディレクトリに保存

    Args:
        trainer: HuggingFace Trainer
        save_directory: 保存先ディレクトリ
    """
    # モデルを保存
    trainer.save_model(save_directory)

    # プロセッサーを保存（data_collatorからアクセスする必要がある）
    processor = trainer.data_collator.processor
    processor.save_pretrained(save_directory)

    print(f"Model and processor saved to {save_directory}")
