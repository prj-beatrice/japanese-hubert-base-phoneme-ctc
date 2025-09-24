#!/usr/bin/env python3
"""簡略化されたHuggingFace TrainerでPhonemeHubertモデルを学習"""

import argparse
import logging
import os

import torch.multiprocessing as mp
from transformers.trainer_utils import get_last_checkpoint

from src.trainer import create_trainer, save_model_with_processor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(
        description="HuggingFace TrainerでPhonemeHubertモデルを学習"
    )

    # データセット引数
    parser.add_argument(
        "--max_audio_length_samples",
        type=int,
        default=250000,
        help="サンプル単位での最大音声長（デフォルト: 250000 = 16kHzで約15.6秒）",
    )

    # モデル引数
    parser.add_argument(
        "--model_name",
        type=str,
        default="rinna/japanese-hubert-base",
        help="ファインチューニングするベースHuBERTモデル",
    )
    parser.add_argument(
        "--freeze_feature_encoder",
        action="store_true",
        help="特徴エンコーダー層を凍結",
    )

    # 学習引数 - TrainingArgumentsに直接渡される
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="モデル出力を保存するディレクトリ",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="学習ステップ数",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="デバイスあたりの学習バッチサイズ",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="逆伝播/更新パスを実行する前に蓄積する更新ステップ数",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="事前学習済み層の学習率",
    )
    parser.add_argument(
        "--head_learning_rate",
        type=float,
        default=None,
        help="新しく追加されたヘッド層（CTCヘッド）の学習率。指定されない場合、learning_rateと同じ値を使用",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="重み減衰",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Adamオプティマイザーのbeta1パラメータ",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Adamオプティマイザーのbeta2パラメータ",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Adamオプティマイザーのepsilonパラメータ",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="ウォームアップステップ数",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
        help="学習率スケジューラーのタイプ",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Xステップごとにメトリクスをログ出力",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Xステップごとにチェックポイントを保存",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="保持するチェックポイントの最大数",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="16ビット浮動小数点精度を使用",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="メモリ節約のためにグラディエントチェックポイントを使用",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="データ読み込み用ワーカー数（0=シングルスレッド、2-4推奨）",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Hugging Face Hubにモデルをプッシュ",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Hugging Face Hub上のモデルID（例: username/model-name）",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["tensorboard", "none"],
        help="ログプラットフォーム",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="ログ用の実行名",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="再現性のためのランダムシード",
    )

    return parser.parse_args()


def main():
    """メイン学習関数"""
    args = parse_args()

    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # create_trainerとTrainingArguments用の引数を分離
    trainer_args = {
        "model_name": args.model_name,
        "freeze_feature_encoder": args.freeze_feature_encoder,
        "max_audio_length_samples": args.max_audio_length_samples,
        "head_learning_rate": args.head_learning_rate,
    }

    # その他はTrainingArgumentsに渡す
    training_args = {
        "output_dir": args.output_dir,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_epsilon": args.adam_epsilon,
        "warmup_steps": args.warmup_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "dataloader_num_workers": args.dataloader_num_workers,
        "push_to_hub": args.push_to_hub,
        "hub_model_id": args.hub_model_id,
        "report_to": args.report_to,
        "run_name": args.run_name,
        "seed": args.seed,
    }

    logger.info("Creating HuBERT trainer...")
    trainer = create_trainer(**trainer_args, **training_args)

    logger.info("Starting training...")

    resume_checkpoint = get_last_checkpoint(args.output_dir)
    if resume_checkpoint is not None:
        logger.info("Resuming from checkpoint: %s", resume_checkpoint)
    else:
        logger.info("No checkpoint found, starting fresh.")

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    trainer.save_model()
    save_model_with_processor(trainer, args.output_dir)

    # 指定されていればHubにプッシュ
    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError(
                "--hub_model_id must be specified when using --push_to_hub"
            )

        logger.info(f"Saving model and processor to {args.output_dir}")
        save_model_with_processor(trainer, args.output_dir)

        logger.info(f"Pushing model to Hub: {args.hub_model_id}")
        # モデルとプロセッサーの両方をHubにプッシュ
        trainer.model.push_to_hub(
            args.hub_model_id, commit_message="Add Japanese phoneme recognition model"
        )
        trainer.data_collator.processor.push_to_hub(
            args.hub_model_id, commit_message="Add processor config"
        )

    logger.info("Training completed!")
    logger.info(f"Final training loss: {train_result.training_loss:.4f}")


if __name__ == "__main__":
    main()
