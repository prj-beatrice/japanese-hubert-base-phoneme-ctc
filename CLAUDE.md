# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development workflow
- `make format` - Format code using Ruff and apply auto-fixes
- `make test` - Run all tests with pytest

### Training and execution
- `make train` - Full training run with production settings (800k steps, all data)
- `make infer` - Run phoneme inference (`AUDIO=...` / optional `MODEL=...`)
- `uv run python train.py --output_dir test_outputs` - Custom training with parameters

### Additional tools
- `make fix-hub-model` - Push a regenerated processor to an existing Hub model
- `uv run accelerate launch train.py ...` - Distributed training across multiple GPUs

### Package management
- `uv sync` - Install dependencies (uses uv for fast package management)

## Architecture

### Core Components
- **HuBERT-based phoneme recognition**: Uses `rinna/japanese-hubert-base` as the foundation model with a CTC head for Japanese phoneme classification
- **48-phoneme vocabulary**: Comprehensive set including special tokens, vowels, devoiced vowels, consonants, and foreign sounds
- **Streaming ReazonSpeech dataset**: Uses HuggingFace datasets with streaming support for large-scale Japanese speech data
- **CTC (Connectionist Temporal Classification)**: For sequence-to-sequence alignment without explicit segmentation

### Key Files
- `src/model.py`: HuBERT model creation, processor setup, and WER metrics computation
- `src/dataset.py`: ReazonSpeech streaming dataset, transcription handling, and data collation for CTC
- `src/phoneme_labeling.py`: Japanese text-to-phoneme conversion using pyopenjtalk-plus
- `src/trainer.py`: HuggingFace Trainer configuration and training logic
- `train.py`: Main training script with command-line interface

### Dataset Architecture
- **ReazonSpeech full dataset**: training split with all available audio leveraged via streaming
- **Transcription preprocessing**: Downloads TSV files, converts to directory-based JSON cache for efficient lookup
- **Audio preprocessing**: Handles variable-length audio with max length filtering (default: 250k samples ≈ 15.6s)
- **Phoneme conversion**: Text → pyopenjtalk → phoneme strings → token IDs

### Training Features
- **Mixed precision (fp16)** and gradient checkpointing for memory efficiency
- **Frozen feature encoder** option to reduce training parameters
- **Distributed training** support with proper synchronization for transcription preparation
- **HuggingFace Hub integration** for model saving and sharing

### Model Configuration
- Default model: `rinna/japanese-hubert-base`
- CTC head with 48 phoneme vocabulary
- Optional feature encoder freezing for faster convergence
- Configurable audio length limits and batch sizes for different hardware setups

### Hardware Optimization
- Default configuration optimized for A100 setup (batch size 32)
- Test configuration uses smaller batch sizes (batch size 4) for development
- Supports distributed training with accelerate launch
- Mixed precision (fp16) and gradient checkpointing for memory efficiency

### Development Notes
- Uses `uv` for fast package management and dependency resolution
- Makefile commands are in Japanese for the development team
- All training configurations push to HuggingFace Hub by default
- Test runs create models under `prj-beatrice/phoneme-hubert-test-*` namespace
- Production models are pushed to `prj-beatrice/japanese-hubert-base-phoneme-ctc`
- Coverage reports are generated in `htmlcov/` directory when using `make test-cov`
- Supports TensorBoard logging via HuggingFace `report_to` flag
- Makefile includes `make help` command (in Japanese) for displaying available tasks
- Training supports differential learning rates: lower for base model (2e-6), higher for CTC head (2e-5)
