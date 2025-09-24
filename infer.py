#!/usr/bin/env python3

import argparse

import librosa
import numpy as np
import torch
from transformers import HubertForCTC, Wav2Vec2Processor


def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    audio, sr = librosa.load(audio_path, sr=target_sr)
    audio = audio.astype(np.float32)
    # パディングを追加してモデルに適合させる
    return np.concatenate(
        [np.zeros(sr, dtype=np.float32), audio, np.zeros(sr // 2, dtype=np.float32)]
    )


def infer_phonemes(
    model: HubertForCTC, processor: Wav2Vec2Processor, audio: np.ndarray
):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_ids = outputs.logits.argmax(-1)
        print(f"Predicted IDs shape: {predicted_ids.shape}")
        phonemes = processor.decode(
            predicted_ids.squeeze(0), spaces_between_special_tokens=True
        )
        return phonemes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", help="音声ファイルのパス")
    parser.add_argument(
        "--model-path",
        default="prj-beatrice/japanese-hubert-base-phoneme-ctc-v3",
        help="モデルのパス",
    )
    args = parser.parse_args()

    audio = load_audio(args.audio_path)
    model = HubertForCTC.from_pretrained(args.model_path)
    processor = Wav2Vec2Processor.from_pretrained(args.model_path)

    phonemes = infer_phonemes(model, processor, audio)
    print(phonemes)


if __name__ == "__main__":
    main()
