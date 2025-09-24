#!/usr/bin/env python3
"""
Hugging Face Hub上の既存モデルに不足しているプロセッサー設定を修正
"""

import argparse

from src.model import create_hubert_model_and_processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hub_model_id", help="修正するHugging Face Hub モデルID")
    parser.add_argument(
        "--base-model", default="rinna/japanese-hubert-base", help="ベースモデル名"
    )
    args = parser.parse_args()

    print(f"Creating processor for {args.hub_model_id}...")

    # 学習時と同じ設定でプロセッサーを作成
    _, processor = create_hubert_model_and_processor(
        model_name=args.base_model, freeze_feature_encoder=False
    )

    # プロセッサーをHubにプッシュ
    print(f"Pushing processor to {args.hub_model_id}...")
    processor.push_to_hub(args.hub_model_id, commit_message="Add processor config")

    print("✅ Done! You can now use the model with infer.py")


if __name__ == "__main__":
    main()
