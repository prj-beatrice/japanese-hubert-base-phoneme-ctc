#!/usr/bin/env python3

import argparse

import librosa
import numpy as np
import torch
from torchaudio.models.decoder import ctc_decoder
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
        log_probs = outputs.logits.log_softmax(dim=-1).cpu()
    id2tok = {i: tok for tok, i in processor.tokenizer.get_vocab().items()}
    labels = [id2tok[i] for i in range(len(id2tok))]
    blank_token = id2tok[processor.tokenizer.pad_token_id]
    decoder = ctc_decoder(
        lexicon=None,
        tokens=labels,
        lm=None,
        nbest=1,
        beam_size=10,
        beam_threshold=10,
        log_add=True,
        blank_token=blank_token,
        sil_token="sil",
    )
    hyps = decoder(log_probs)
    best = hyps[0][0]
    id_seq = best.tokens
    phonemes = [labels[i] for i in id_seq]
    BOUNDARY = {"sil"}
    while phonemes and phonemes[0] in BOUNDARY:
        phonemes.pop(0)
    while phonemes and phonemes[-1] in BOUNDARY:
        phonemes.pop()
    phonemes = " ".join(phonemes)
    return phonemes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", help="音声ファイルのパス")
    parser.add_argument(
        "--model-path",
        default="prj-beatrice/japanese-hubert-base-phoneme-ctc-v4",
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
