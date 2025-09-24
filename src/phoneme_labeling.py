from typing import Optional

import pyopenjtalk

# pyopenjtalk-plus出力に基づく音素語彙定義
# 日本語固有音と外来音をカバーする包括的最小セット
PHONEME_VOCAB = [
    # 特殊トークン
    "PAD",
    "UNK",
    "SOS",
    "EOS",
    # 母音
    "a",
    "i",
    "u",
    "e",
    "o",
    # 無声化母音
    # 特定の音韻環境で生じる。例: すき→sUki, ひと→hIto
    "I",  # 無声化i
    "U",  # 無声化u
    # 基本子音
    "k",
    "g",
    "s",
    "z",
    "t",
    "d",
    "n",
    "h",
    "b",
    "p",
    "m",
    "y",
    "r",
    "w",
    "f",
    "j",
    "v",  # 外来音（ヴ）
    # 特殊子音
    "N",  # 撥音（ん）
    "cl",  # 促音（っ）
    # 子音組み合わせ
    "sh",  # しゃ行
    "ch",  # ちゃ行
    "ts",  # つ、ツァ行
    # 口蓋化子音
    "ky",
    "gy",
    "hy",
    "by",
    "py",
    "my",
    "ny",
    "ry",
    "fy",
    "ty",
    "dy",
    # 外来子音組み合わせ
    "kw",  # クヮ
    "gw",  # グヮ
    # 特殊記号
    "pau",  # 休止
    "sil",  # 無音
]


def filter_text(text: str) -> Optional[str]:
    """学習から除外するテキストかどうかチェック"""

    if not text:
        return "Empty text"
    for word in ["9人", "９人", "九人", "今シーズン", "今大会"]:
        if word in text:
            return f"'{word}' in text"
    if "0" in text or "０" in text or "十" in text:
        njd_features = pyopenjtalk.run_frontend(text)
        labels = pyopenjtalk.make_label(njd_features)
        prons = "-".join(map(lambda s: s.split("-")[1].split("+")[0], labels[1:-1]))
        if "j-u-cl-" in prons:
            return "'十' in text"
