"""
HuggingFaceベストプラクティスに従ったReazonSpeechデータセット実装
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from torch.utils.data import IterableDataset
from transformers import Wav2Vec2Processor

from .phoneme_candidates import CandidatePhoneme, CandidatePreprocessor
from .phoneme_labeling import filter_text


def create_dataset_logger(log_dir: str = "./dataset_logs"):
    """データセット処理用のシンプルなロガーを作成"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    rank = int(os.environ.get("LOCAL_RANK", 0))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = log_dir / f"dataset_rank{rank}_{timestamp}.jsonl"

    def log_event(event_type: str, sample_id: str, **details):
        """単一イベントをJSONLファイルにログ出力"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "rank": rank,
            "event_type": event_type,
            "sample_id": sample_id,
            **details,
        }

        # ファイルにシンプル追加（各プロセスは独自ファイルを持つ）
        with open(log_file, "a") as f:
            json.dump(log_entry, f)
            f.write("\n")

    return log_event, log_file


def download_and_prepare_transcriptions(cache_dir: str = "./transcriptions"):
    """
    ReazonSpeech TSVファイルをダウンロードし、ディレクトリ別に書き起こしJSONファイルを準備

    Args:
        cache_dir: 書き起こしファイルを保存するディレクトリ

    Returns:
        準備された書き起こし数
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    # 既に処理済みかチェック
    marker_file = cache_path / "processed.marker"
    if marker_file.exists():
        print(f"Transcriptions already prepared in {cache_dir}")
        return

    # 分散学習：rank 0のみダウンロードする
    try:
        # 環境変数で分散モードかチェック
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        if world_size > 1 and local_rank != 0:
            # rank 0のダウンロード完了を待機
            print(f"Rank {local_rank} waiting for rank 0 to prepare transcriptions...")
            while not marker_file.exists():
                time.sleep(1)
            print(f"Rank {local_rank}: Transcriptions ready!")
            return
        elif world_size > 1:
            print(f"Rank {local_rank} will download transcriptions for all processes")
    except Exception:
        # 分散モードではない、通常処理
        pass

    # TSVファイルをダウンロード
    tsv_url = "https://corpus.reazon-research.org/reazonspeech-v2/tsv/all.tsv"
    tsv_path = cache_path / "all.tsv"

    # TSVファイルが既に存在するかチェック
    if not tsv_path.exists():
        print(f"Downloading {tsv_url}...")
        try:
            urllib.request.urlretrieve(tsv_url, tsv_path)
            print(f"Downloaded {tsv_path}")
        except Exception as e:
            print(f"Failed to download {tsv_url}: {e}")
            return 0
    else:
        print(f"Using existing TSV file: {tsv_path}")

    # TSVファイルを処理しディレクトリ別にグループ化
    print(f"Processing {tsv_path}...")
    transcription_count = 0
    dir_transcriptions = {}  # ディレクトリ別の書き起こしを保持する辞書

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                flac_path, text = row[0], row[1]

                # ディレクトリとファイル名を抽出
                # e.g., "000/000734dcb35d6.flac" -> dir="000", filename="000734dcb35d6"
                parts = flac_path.split("/")
                if len(parts) == 2:
                    dir_name = parts[0]
                    filename = parts[1].replace(".flac", "")

                    # ディレクトリ辞書に追加
                    if dir_name not in dir_transcriptions:
                        dir_transcriptions[dir_name] = {}

                    dir_transcriptions[dir_name][filename] = text
                    transcription_count += 1

                if transcription_count % 100000 == 0:
                    print(f"Processed {transcription_count} transcriptions...")

    # ディレクトリ別に書き起こしを保存
    for dir_name, transcriptions in dir_transcriptions.items():
        json_file = cache_path / f"{dir_name}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(transcriptions, f, ensure_ascii=False, indent=2)

    # マーカーファイルを作成
    with open(marker_file, "w") as f:
        f.write(f"Processed {transcription_count} transcriptions")

    print(f"Prepared {transcription_count} transcriptions in {cache_dir}")

    # TSVファイルを清理
    os.remove(tsv_path)

    return transcription_count


# 読み込み済み書き起こし辞書のキャッシュ
_transcription_cache = {}


def load_transcription_from_key(key: str, cache_dir: str = "./transcriptions") -> str:
    """
    サンプルキーを使用して保存されたJSONファイルから書き起こしテキストを読み込む

    Args:
        key: サンプルキー (例: "000/000734dcb35d6")
        cache_dir: 書き起こしファイルを含むディレクトリ

    Returns:
        書き起こしテキストまたは見つからない場合は空文字列
    """
    # キーからディレクトリとファイル名を抽出
    parts = key.split("/")
    if len(parts) != 2:
        return ""

    dir_name, filename = parts[0], parts[1]

    # キャッシュにこのディレクトリがあるかチェック
    cache_key = f"{cache_dir}/{dir_name}"
    if cache_key not in _transcription_cache:
        json_file = Path(cache_dir) / f"{dir_name}.json"

        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    _transcription_cache[cache_key] = json.load(f)
            except Exception as e:
                print(f"Error reading transcription file {json_file}: {e}")
                _transcription_cache[cache_key] = {}
        else:
            _transcription_cache[cache_key] = {}

    # キャッシュから書き起こしを返す
    return _transcription_cache[cache_key].get(filename, "")


class DataCollatorCTCWithPadding:
    """候補音素情報を保持したままCTC学習用バッチを構築"""

    def __init__(self, processor: Wav2Vec2Processor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        candidate_batches: List[List[Dict[str, Any]]] = []

        input_features = []
        for feature in features:
            candidate_batches.append(feature.pop("candidate_data"))
            input_features.append({"input_values": feature.pop("input_values")})
            feature.pop("text")
            feature.pop("sample_id")

        batch = self.processor.pad(
            input_features=input_features,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        batch["candidate_batches"] = candidate_batches

        return batch


class ReazonSpeechDataset(IterableDataset):
    """
    ReazonSpeech用のHuggingFace互換ストリーミングデータセット
    """

    def __init__(
        self,
        processor: Wav2Vec2Processor,
        streaming: bool = True,
        max_samples: Optional[int] = None,
        max_audio_length_samples: int = 250000,
        cache_dir: str = "./transcriptions",
        log_dir: str = "./dataset_logs",
    ):
        """
        ReazonSpeechデータセットを初期化

        Args:
            processor: 音声前処理用のHuggingFaceプロセッサー
            streaming: ストリーミングモードを使用するかどうか
            max_samples: 処理するサンプルの最大数（テスト用）
            max_audio_length_samples: サンプル単位の最大音声長（デフォルト: 250000）
            cache_dir: 事前保存された書き起こしファイルを含むディレクトリ
            log_dir: データセット処理ログ用のディレクトリ
        """
        self.processor = processor
        self.streaming = streaming
        self.max_samples = max_samples
        self.max_audio_length_samples = max_audio_length_samples
        self.cache_dir = cache_dir
        self.candidate_preprocessor: Optional[CandidatePreprocessor] = None
        self.log_dir = log_dir
        self.log_event = None
        self.log_file = None

        split = "train"
        subset = "all"
        self.split = split
        self.subset = subset

        # 自動音声デコードを無効にしてデータセットをロード
        self.dataset = load_dataset(
            "litagin/reazon-speech-v2-clone",
            split=split,
            streaming=streaming,
            trust_remote_code=True,
            storage_options={
                "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
            },
        )

        # FLACエラーを手動で処理するため自動音声デコードを無効化
        # これによりデコードされた音声の代わりに生バイトを取得
        if hasattr(self.dataset, "cast_column"):
            self.dataset = self.dataset.cast_column("flac", Audio(decode=False))

        print(
            f"Initialized ReazonSpeech dataset (split: {split}, subset: {subset}, streaming: {streaming})"
        )

    def _decode_flac_safely(self, flac_data: bytes, sample_key: str) -> tuple:
        """エラーハンドリング付きでFLACデータを安全にデコード"""
        try:
            # soundfileでFLACデータをデコード
            with io.BytesIO(flac_data) as flac_file:
                audio_array, sampling_rate = sf.read(flac_file)
                return audio_array, sampling_rate, None
        except Exception as e:
            # 例外を発生させる代わりにエラー情報を返す
            return None, None, str(e)

    def _ensure_logger(self):
        if self.log_event is None:
            self.log_event, self.log_file = create_dataset_logger(self.log_dir)
            self.log_event(
                "dataset_init",
                "init",
                split=self.split,
                subset=self.subset,
                streaming=self.streaming,
                max_samples=self.max_samples,
                pid=os.getpid(),
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["log_event"] = None
        state["log_file"] = None
        state["candidate_preprocessor"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.log_event = None
        self.log_file = None
        self.candidate_preprocessor = None

    def __iter__(self):
        """ストリーミングデータセット用のイテレーター"""
        sample_count = 0
        yielded_count = 0

        # カウンターを初期化
        stats = {
            "total": 0,
            "yielded": 0,
            "tar_error": 0,
            "audio_too_short": 0,
            "audio_too_long": 0,
            "no_key": 0,
            "no_transcription": 0,
            "text_filter": 0,
            "no_candidates": 0,
            "processing_error": 0,
        }

        self._ensure_logger()
        self.log_event("iteration_start", "start")

        if self.candidate_preprocessor is None:
            # Delay heavy fugashi/pyopenjtalk initialization until workers start
            self.candidate_preprocessor = CandidatePreprocessor()

        dataset_iter = iter(self.dataset)
        while True:
            try:
                sample = next(dataset_iter)
            except StopIteration:
                self.log_event("iteration_end", "end", **stats)
                print(f"Dataset iteration completed: {stats}")
                break
            except Exception as e:
                stats["tar_error"] += 1
                self.log_event(
                    "skip", f"sample_{sample_count}", reason="tar_error", error=str(e)
                )
                continue  # tar/ネットワークが壊れていたらスキップ

            sample_count += 1
            stats["total"] += 1

            # max_samplesに達したら停止
            if self.max_samples and yielded_count >= self.max_samples:
                self.log_event("max_samples_reached", "end", **stats)
                break

            try:
                # サンプルキーを取得
                sample_key = sample.get("__key__", f"sample_{sample_count}")

                # 生のFLACデータを取得（decode=Falseのためバイトであるべき）
                flac_info = sample["flac"]
                if isinstance(flac_info, dict) and "bytes" in flac_info:
                    flac_data = flac_info["bytes"]
                else:
                    # フォールバック：古い形式の場合はスキップ
                    stats["processing_error"] += 1
                    self.log_event(
                        "skip",
                        sample_key,
                        reason="processing_error",
                        error="Unable to get raw FLAC bytes",
                    )
                    continue

                # エラーハンドリング付きでFLACデータを手動デコード
                audio_array, sampling_rate, decode_error = self._decode_flac_safely(
                    flac_data, sample_key
                )

                # FLACデコードに失敗した場合はスキップ
                if decode_error is not None:
                    stats["tar_error"] += 1
                    self.log_event(
                        "skip",
                        sample_key,
                        reason="tar_error",
                        error=f"flac decoder error: {decode_error}",
                    )
                    continue

                # 音声が空または短すぎる場合はスキップ（16kHzで0.1秒未満）
                min_audio_length = 1600  # 16kHzで0.1秒
                if len(audio_array) < min_audio_length:
                    stats["audio_too_short"] += 1
                    self.log_event(
                        "skip",
                        sample_key,
                        reason="audio_too_short",
                        audio_length=len(audio_array),
                    )
                    continue

                # 音声が長すぎる場合はスキップ
                if len(audio_array) > self.max_audio_length_samples:
                    stats["audio_too_long"] += 1
                    self.log_event(
                        "skip",
                        sample_key,
                        reason="audio_too_long",
                        audio_length=len(audio_array),
                    )
                    continue

                # サンプルキーを使って事前保存されたテキストファイルから書き起こしを取得
                if not sample_key or sample_key.startswith("sample_"):
                    stats["no_key"] += 1
                    self.log_event("skip", f"sample_{sample_count}", reason="no_key")
                    continue

                text = load_transcription_from_key(sample_key, self.cache_dir)
                if not text:
                    stats["no_transcription"] += 1
                    self.log_event("skip", sample_key, reason="no_transcription")
                    continue

                filtering_result = filter_text(text, sample_key)
                if filtering_result is not None:
                    stats["text_filter"] += 1
                    self.log_event(
                        "skip",
                        sample_key,
                        reason="text_filter",
                        error=filtering_result,
                    )
                    continue

                # 候補音素列を生成
                try:
                    candidates: List[CandidatePhoneme] = (
                        self.candidate_preprocessor.generate_candidates(text, num=20)
                    )
                except Exception as e:
                    stats["processing_error"] += 1
                    self.log_event(
                        "skip",
                        sample_key,
                        reason="candidate_generation_failed",
                        error=str(e),
                    )
                    continue

                if not candidates:
                    stats["no_candidates"] += 1
                    self.log_event(
                        "skip",
                        sample_key,
                        reason="no_candidate_phonemes",
                    )
                    continue

                processed_audio = self.processor(
                    audio_array,
                    sampling_rate=sampling_rate,
                    return_tensors="pt",
                    padding=False,
                )

                yielded_count += 1
                stats["yielded"] += 1

                # 成功したサンプルをログ出力
                self.log_event(
                    "success",
                    sample_key,
                    audio_length=len(audio_array),
                    phoneme_count=len(candidates[0].phonemes),
                    candidate_count=len(candidates),
                    text_length=len(text),
                )

                # 1000サンプルごとに統計を表示
                if stats["total"] % 1000 == 0:
                    print(
                        f"[Rank {int(os.environ.get('LOCAL_RANK', 0))}] Processed {stats['total']} samples: {stats}"
                    )

                if yielded_count <= 5:
                    print(
                        f"Yielding sample {yielded_count}: audio_len={len(audio_array)}, candidates={len(candidates)}, phonemes_len_first={len(candidates[0].phonemes)}, text='{text[:30]}...'"
                    )

                yield {
                    "input_values": processed_audio.input_values.squeeze(0).tolist(),
                    "candidate_data": [
                        {
                            "phonemes": candidate.phonemes,
                            "mecab_cost": candidate.mecab_cost,
                            "is_digit_expanded": candidate.is_digit_expanded,
                        }
                        for candidate in candidates
                    ],
                    "text": text,
                    "sample_id": sample_key,
                }

            except Exception as e:
                stats["processing_error"] += 1
                self.log_event(
                    "skip",
                    sample.get("__key__", f"sample_{sample_count}"),
                    reason="processing_error",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                continue


def prepare_dataset(
    processor: Any,
    max_audio_length_samples: int = 250000,
    cache_dir: str = "./transcriptions",
):
    """
    HuggingFace Trainer用の学習および検証データセットを準備
    必要に応じて書き起こしをダウンロードして準備する
    """

    # データセット作成前に書き起こしをダウンロード・準備
    print("Preparing transcriptions...")
    download_and_prepare_transcriptions(cache_dir=cache_dir)

    # 学習データセットを作成
    train_dataset = ReazonSpeechDataset(
        processor=processor,
        streaming=True,
        max_samples=None,
        max_audio_length_samples=max_audio_length_samples,
        cache_dir=cache_dir,
    )

    return train_dataset
