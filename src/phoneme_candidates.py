"""音素候補生成とHuBERTスコアリングユーティリティ"""

from __future__ import annotations

import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import fugashi
import numpy as np
import pyopenjtalk
import torch
import torch.nn.functional as F
from transformers import HubertForCTC, Wav2Vec2Processor


@dataclass
class CandidatePhoneme:
    phonemes: list[str]
    mecab_cost: int
    is_digit_expanded: bool


class CandidatePreprocessor:
    """HuBERT を使用しない音素候補前処理."""

    def __init__(self):
        dictionary_dir = Path(pyopenjtalk.__file__).parent / "dictionary"
        assert dictionary_dir.exists()

        # openjtalk には設定がハードコードされていて dicrc ファイルが存在しない
        with open(dictionary_dir / "dicrc", "w") as f:
            f.write(r"""cost-factor = 800
bos-feature = BOS/EOS,*,*,*,*,*,*,*,*
eval-size = 8
unk-eval-size = 4
node-format-yomi = %pS%f[7]
unk-format-yomi = %M
eos-format-yomi = \n
node-format-simple = %m\t%F-[0,1,2,3]\n
eos-format-simple = EOS\n
node-format-chasen = %m\t%f[7]\t%f[6]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\n
unk-format-chasen = %m\t%m\t%m\t%F-[0,1,2,3]\t\t\n
eos-format-chasen = EOS\n
node-format-chasen2 = %M\t%f[7]\t%f[6]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\n
unk-format-chasen2 = %M\t%m\t%m\t%F-[0,1,2,3]\t\t\n
eos-format-chasen2 = EOS\n
""")

        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as tmp:
            while True:
                try:
                    self.tagger = fugashi.GenericTagger(
                        f"-r {tmp.name} -d {dictionary_dir}"
                    )
                    break
                except RuntimeError as e:
                    print(f"Failed to create MeCab tagger, retrying: {e}")
                    from time import sleep

                    sleep(1)
            while True:
                try:
                    self.tagger_p = fugashi.GenericTagger(
                        f"-r {tmp.name} -d {dictionary_dir} -p -F 0 -E %pc"
                    )
                    break
                except RuntimeError as e:
                    print(f"Failed to create MeCab tagger with -p, retrying: {e}")
                    from time import sleep

                    sleep(1)

        self.hankaku_to_zenkaku_table = str.maketrans(
            {chr(i): chr(i + 0xFEE0) for i in range(33, 127)}
        )

    def generate_candidates(self, text: str, num: int) -> list[CandidatePhoneme]:
        """テキストから音素候補を生成."""
        results = self._run_mecab(text, num)
        self._add_mecab_costs(results)
        self._add_njd_result(results)
        self._expand_digit_reading(results)
        self._add_phonemes(results)

        candidates: list[CandidatePhoneme] = []
        for candidate in results["candidates"]:
            phonemes = candidate.get("phonemes", [])
            mecab_cost = candidate.get("mecab_cost", 0)
            is_digit_expanded = candidate.get("is_digit_expanded", False)
            if not phonemes:
                continue
            candidates.append(
                CandidatePhoneme(
                    phonemes=list(phonemes),
                    mecab_cost=int(mecab_cost),
                    is_digit_expanded=bool(is_digit_expanded),
                )
            )
        return candidates

    def _run_mecab(self, text: str, num: int) -> dict:
        """MeCab を N-best で実行"""
        text = text.translate(self.hankaku_to_zenkaku_table)
        nbest: str = self.tagger.nbest(text, num=num)
        candidates = []
        seen_raw_candidates = set()
        for raw_candidate in (nbest + "\n").split("\nEOS\n"):
            if not raw_candidate:
                continue
            if raw_candidate in seen_raw_candidates:
                continue
            seen_raw_candidates.add(raw_candidate)
            features = [line.replace("\t", ",") for line in raw_candidate.splitlines()]
            raw_candidate += "\nEOS"
            candidate = {
                "raw": raw_candidate,
                "features": features,
            }
            candidates.append(candidate)
        return {"candidates": candidates}

    def _add_mecab_costs(self, results: dict):
        """Mecab コストを追加"""
        for candidate in results["candidates"]:
            raw_candidate = candidate["raw"]
            mecab_cost = int(self.tagger_p.parse(raw_candidate).lstrip("0"))
            candidate["mecab_cost"] = mecab_cost

    def _add_njd_result(self, results: dict):
        """NJD による処理の結果を追加"""
        for candidate in results["candidates"]:
            njd_result = pyopenjtalk.run_njd_from_mecab(candidate["features"])
            for i, feature in enumerate(njd_result):
                if (
                    (feature["string"] in ["いう", "言う"])
                    and feature["pos"] == "動詞"
                    and feature["pos_group1"] == "自立"
                    and feature["ctype"] == "サ変・スル"
                    and feature["cform"] == "基本形"
                    and (feature["orig"] in ["いう", "言う"])
                    and feature["read"] == "イウ"
                    and feature["pron"] == "イウ"
                ):
                    feature["pron"] = "ユー"
                if "チャンピオン" in feature["pron"]:
                    feature["pron"] = feature["pron"].replace(
                        "チャンピオン", "チャンピョン"
                    )
                if feature["pron"].endswith("ティ") or feature["pron"].endswith("ディ"):
                    feature["pron"] += "ー"
                if (
                    feature["pron"].replace("’", "")
                    in {
                        "ウェイター",
                        "ウェーター",
                        "ウェイトレス",
                        "ウェートレス",
                        "ウェディング",
                        "ウェイトリフティング",
                        "ウェートリフティング",
                        "ウェイトトレーニング",
                        "ウェートトレーニング",
                        "ウェスト",
                        "ウェストミンスター",
                        "ウェスタン",
                        "ウェットティッシュ",
                        "ウェットシート",
                        "ウェットタオル",
                        "ウェットスーツ",
                        "デラウェア",
                    }
                    or (
                        i + 1 < len(njd_result)
                        and feature["pron"] in ["ウェイト", "ウェート"]
                        and njd_result[i + 1]["pron"]
                        in ["リフティング", "トレーニング"]
                    )
                    or (
                        i + 1 < len(njd_result)
                        and feature["pron"] == "ウェット"
                        and njd_result[i + 1]["pron"]
                        in ["ティッシュ", "シート", "タオル", "スーツ"]
                    )
                ):
                    feature["pron"] = feature["pron"].replace("ウェ", "ウエ")
                if feature["pron"] in ["キレイ", "キレイゴト", "キレイドコロ"]:
                    feature["pron"] = feature["pron"].replace("キレイ", "キレー")
                if "エイ" in feature["pron"] and (
                    "A" in feature["string"] or "Ａ" in feature["string"]
                ):
                    feature["pron"] = feature["pron"].replace("エイ", "エー")
                if "ジェイ" in feature["pron"] and (
                    "J" in feature["string"] or "Ｊ" in feature["string"]
                ):
                    feature["pron"] = feature["pron"].replace("ジェイ", "ジェー")
                if "ケイ" in feature["pron"] and (
                    "K" in feature["string"] or "Ｋ" in feature["string"]
                ):
                    feature["pron"] = feature["pron"].replace("ケイ", "ケー")

            candidate["njd_result"] = njd_result

    def _expand_digit_reading(self, results: dict):
        """数字の読みを複数通りにする"""
        new_candidates = []
        DIGIT_CANDIDATES = {
            "四": [("シ", "シ", 1, 1), ("ヨン", "ヨン", 1, 2)],
            "七": [("シチ", "シチ", 2, 2), ("ナナ", "ナナ", 1, 2)],
            "八": [("ハチ", "ハチ", 2, 2)],
            "九": [("キュウ", "キュー", 1, 2), ("ク", "ク", 1, 1)],
        }
        for candidate in results["candidates"]:
            candidate: dict[str, Any]
            njd_result: list[dict[str, Any]] = candidate["njd_result"]
            expanded = [njd_result]
            for idx_feature, feature in enumerate(njd_result):
                if (
                    feature["pos_group1"] == "数"
                    and feature["pos"] == "名詞"
                    and feature["string"] in DIGIT_CANDIDATES
                ):
                    digit_features = DIGIT_CANDIDATES[feature["string"]]
                    expanded_old = expanded
                    expanded = expanded_old.copy()
                    for read, pron, acc, mora_size in digit_features:
                        if feature["pron"] != pron:
                            for e in expanded_old:
                                e = e.copy()
                                e[idx_feature] = e[idx_feature].copy()
                                e[idx_feature]["read"] = read
                                e[idx_feature]["pron"] = pron
                                e[idx_feature]["acc"] = acc
                                e[idx_feature]["mora_size"] = mora_size
                                e[idx_feature]["chain_rule"] = "*"
                                expanded.append(e)
            assert len(expanded) >= 1
            assert expanded[0] is njd_result
            candidate["digit_expanded"] = njd_result
            candidate["is_digit_expanded"] = False
            new_candidates.append(candidate)
            for e in expanded[1:]:
                new_candidate = candidate.copy()
                new_candidate["digit_expanded"] = e
                new_candidate["is_digit_expanded"] = True
                new_candidates.append(new_candidate)
        results["candidates"] = new_candidates

    def _add_phonemes(self, results: dict):
        """音素列等を追加"""
        for candidate in results["candidates"]:
            postprocessed = deepcopy(candidate["digit_expanded"])
            # modify_kanji_yomi は使わない
            postprocessed = pyopenjtalk.modify_filler_accent(postprocessed)
            postprocessed = pyopenjtalk.retreat_acc_nuc(postprocessed)
            postprocessed = pyopenjtalk.modify_acc_after_chaining(postprocessed)
            postprocessed = pyopenjtalk.process_odori_features(postprocessed)
            labels = pyopenjtalk.make_label(postprocessed)
            phonemes = list(map(lambda s: s.split("-")[1].split("+")[0], labels[1:-1]))
            candidate["postprocessed"] = postprocessed
            candidate["phonemes"] = phonemes


class CandidateScorer:
    """HuBERT を用いて音素候補をスコアリング"""

    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        HUBERT_MODEL_NAME = "prj-beatrice/japanese-hubert-base-phoneme-ctc-v3"
        REVISION = "076b30425082827844a0b2be2f009b5768272761"  # 50k iter
        self.model = HubertForCTC.from_pretrained(
            HUBERT_MODEL_NAME, revision=REVISION
        ).to(device)
        self.wav2vec_processor: Wav2Vec2Processor = Wav2Vec2Processor.from_pretrained(
            HUBERT_MODEL_NAME
        )
        self.model.eval()

    @torch.no_grad()
    def score_batch(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        batch_candidates: Iterable[Iterable[CandidatePhoneme]],
    ) -> list[list[dict]]:
        """
        HuBERT 出力を用いて各候補の CTC loss とスコアを計算
        """
        input_values = input_values.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self.model(input_values=input_values, attention_mask=attention_mask)

        log_probs = F.log_softmax(outputs.logits, dim=-1).transpose(0, 1)

        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=1)
            input_lengths = self.model._get_feat_extract_output_lengths(
                input_lengths
            ).to(self.device)
        else:
            input_lengths = torch.full(
                (input_values.size(0),),
                log_probs.size(0),
                dtype=torch.long,
                device=self.device,
            )

        sequence_index_lookup: dict[tuple[int, str], int] = {}
        target_lengths: list[int] = []
        sequence_sample_indices: list[int] = []
        sequence_label_ids: list[int] = []
        candidate_groups: list[list[tuple[CandidatePhoneme, int]]] = []

        for batch_index, candidates in enumerate(batch_candidates):
            group_entries: list[tuple[CandidatePhoneme, int]] = []
            for candidate in candidates:
                phonemes = candidate.phonemes
                if not phonemes:
                    continue
                key = (batch_index, " ".join(phonemes))
                seq_index = sequence_index_lookup.get(key)
                if seq_index is None:
                    candidate_phoneme_ids = (
                        self.wav2vec_processor.tokenizer.convert_tokens_to_ids(phonemes)
                    )
                    seq_index = len(sequence_index_lookup)
                    sequence_index_lookup[key] = seq_index
                    sequence_sample_indices.append(batch_index)
                    target_lengths.append(len(candidate_phoneme_ids))
                    sequence_label_ids.extend(candidate_phoneme_ids)

                group_entries.append((candidate, seq_index))

            if not group_entries:
                raise ValueError("No valid phoneme candidates were produced.")

            candidate_groups.append(group_entries)

        with torch.backends.cudnn.flags(enabled=False):
            sample_indices_tensor = torch.tensor(
                sequence_sample_indices, device=self.device
            )
            selected_log_probs = log_probs.index_select(1, sample_indices_tensor)
            selected_input_lengths = input_lengths.index_select(
                0, sample_indices_tensor
            )
            losses = F.ctc_loss(
                selected_log_probs,
                torch.tensor(sequence_label_ids, device=self.device),
                selected_input_lengths,
                torch.tensor(target_lengths, device=self.device),
                blank=self.model.config.pad_token_id,
                reduction="none",
            )

        loss_values = losses.detach().cpu().tolist()

        scored_batch: list[list[dict]] = []
        for candidate_pairs in candidate_groups:
            scored_candidates: list[dict] = []
            for candidate, seq_index in candidate_pairs:
                loss_value = loss_values[seq_index]
                score = loss_value * 0.35295 + candidate.mecab_cost / 800.0 * 0.3284
                scored_candidates.append(
                    {
                        "phonemes": candidate.phonemes,
                        "mecab_cost": candidate.mecab_cost,
                        "is_digit_expanded": candidate.is_digit_expanded,
                        "ctc_loss": float(loss_value),
                        "score": float(score),
                    }
                )

            if not scored_candidates:
                raise ValueError("No valid phoneme candidates were produced.")

            scored_batch.append(scored_candidates)

        return scored_batch

    def select_best(
        self,
        candidate_input_values: torch.Tensor,
        candidate_attention_mask: Optional[torch.Tensor],
        batch_candidates: Iterable[Iterable[CandidatePhoneme]],
    ) -> list[dict]:
        scored_batch = self.score_batch(
            input_values=candidate_input_values,
            attention_mask=candidate_attention_mask,
            batch_candidates=batch_candidates,
        )
        best = []
        for candidates in scored_batch:
            candidate = min(candidates, key=lambda c: c["score"])
            best.append(candidate)
        return best


class CandidateGenerator:
    """従来の API を維持するための合成クラス"""

    def __init__(
        self,
        device: str = "cpu",
    ):
        self.scorer = CandidateScorer(device)
        self.preprocessor = CandidatePreprocessor()

    def generate(self, text: str, audio_16khz: np.ndarray, num: int) -> dict:
        candidates = self.preprocessor.generate_candidates(text, num)
        if not candidates:
            raise ValueError("No phoneme candidates generated.")

        sampling_rate = 16000
        padded_audio = np.concatenate(
            [np.zeros(sampling_rate), audio_16khz, np.zeros(sampling_rate // 2)]
        )
        batch = self.scorer.wav2vec_processor(
            padded_audio,
            sampling_rate=sampling_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        scored = self.scorer.score_batch(
            input_values=batch.input_values,
            attention_mask=batch["attention_mask"],
            batch_candidates=[candidates],
        )[0]

        return {"candidates": scored}


if __name__ == "__main__":
    import sys

    import librosa

    device = "cuda" if torch.cuda.is_available() else "cpu"
    candidate_generator = CandidateGenerator(device)
    text = sys.argv[1]
    audio_file = Path(sys.argv[2])
    audio_16khz, sr = librosa.load(audio_file, sr=16000)
    results = candidate_generator.generate(text, audio_16khz, num=20)

    for candidate in results["candidates"]:
        print(
            f"Score: {candidate['score']:.3f}, MeCab Cost: {candidate['mecab_cost']}, CTC Loss: {candidate['ctc_loss']:.3f}, Phonemes: {' '.join(candidate['phonemes'])}"
        )
    print(results)
