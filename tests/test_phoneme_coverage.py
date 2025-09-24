import random

import pyopenjtalk

from src.phoneme_labeling import PHONEME_VOCAB


class TestPhonemeComprehensiveness:
    def test_phoneme_vocabulary_completeness(self):
        """Test that phoneme vocabulary covers pyopenjtalk-plus output."""
        katakana_candidates = "ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴ"

        random.seed(42)
        test_texts = [
            "".join(random.choice(katakana_candidates) for _ in range(10))
            for _ in range(1000)
        ]

        unknown_phonemes = set()

        for text in test_texts:
            phonemes = pyopenjtalk.g2p(text, join=False)
            for phoneme in phonemes:
                if phoneme not in PHONEME_VOCAB:
                    unknown_phonemes.add(phoneme)

        # All phonemes should be in vocabulary
        assert len(unknown_phonemes) == 0, f"Unknown phonemes found: {unknown_phonemes}"

    def test_no_duplicate_phonemes(self):
        """Test that there are no duplicate phonemes in vocabulary."""
        assert len(PHONEME_VOCAB) == len(set(PHONEME_VOCAB))
