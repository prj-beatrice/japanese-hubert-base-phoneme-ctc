.PHONY: format test train infer fix-hub-model

# コードフォーマット＆自動修正
format:
	@echo "🔄 コードフォーマット実行中..."
	uv run ruff format src/ tests/ *.py
	uv run ruff check --fix src/ tests/ *.py
	@echo "✅ フォーマット完了"

# 全テスト実行
test:
	@echo "🔄 全テスト実行中..."
	uv run pytest tests/ -v
	@echo "✅ テスト完了"

# 学習実行
train:
	@echo "🚀 学習開始..."
	uv run accelerate launch train.py \
		--output_dir outputs/japanese-hubert-base-phoneme-ctc-v5 \
		--per_device_train_batch_size 32 \
		--max_steps 800000 \
		--learning_rate 2e-6 \
		--head_learning_rate 2e-5 \
		--adam_beta1 0.9 \
		--adam_beta2 0.98 \
		--adam_epsilon 1e-6 \
		--warmup_steps 10000 \
		--lr_scheduler_type cosine \
		--logging_steps 1000 \
		--save_steps 10000 \
		--fp16 \
		--max_audio_length_samples 250000 \
		--dataloader_num_workers 8 \
		--report_to tensorboard \
		--push_to_hub \
		--hub_model_id prj-beatrice/japanese-hubert-base-phoneme-ctc-v5
	@echo "✅ 学習フロー終了"

# 推論実行
# 例: make infer AUDIO=VOICEACTRESS100_001.wav MODEL=outputs/custom-model
infer:
ifndef AUDIO
	$(error AUDIO変数を指定してください。例: make infer AUDIO=VOICEACTRESS100_001.wav)
endif
	@echo "🔍 推論を実行します..."
	uv run python infer.py $(AUDIO) $(if $(MODEL),--model-path $(MODEL))
	@echo "✅ 推論完了"

# Hubモデルのプロセッサー修正
# 例: make fix-hub-model HUB_MODEL=prj-beatrice/japanese-hubert-base-phoneme-ctc-v5 BASE_MODEL=rinna/japanese-hubert-base
fix-hub-model:
ifndef HUB_MODEL
	$(error HUB_MODEL変数を指定してください。例: make fix-hub-model HUB_MODEL=prj-beatrice/japanese-hubert-base-phoneme-ctc-v5)
endif
	@echo "🛠️ Hubモデルを修正します..."
	uv run python fix_hub_model.py $(HUB_MODEL) $(if $(BASE_MODEL),--base-model $(BASE_MODEL))
	@echo "✅ Hubモデル修正完了"
