from src.phoneme_labeling import PHONEME_VOCAB


class TestPhonemeVocabFunctions:
    def test_vocab_properties(self):
        assert len(PHONEME_VOCAB) > 40  # Should have reasonable number of phonemes
        assert PHONEME_VOCAB[0] == "PAD"  # Padding token should be first
        assert PHONEME_VOCAB[1] == "UNK"  # Unknown token should be second
        assert "a" in PHONEME_VOCAB  # Should contain basic vowels
        assert "k" in PHONEME_VOCAB  # Should contain basic consonants
