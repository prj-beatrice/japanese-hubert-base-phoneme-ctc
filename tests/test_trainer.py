from unittest.mock import MagicMock


class TestTrainerUtilities:
    def test_save_model_with_processor(self):
        from src.trainer import save_model_with_processor

        # Mock trainer with data collator
        mock_trainer = MagicMock()
        mock_processor = MagicMock()
        mock_trainer.data_collator.processor = mock_processor

        # Test save_model_with_processor
        save_model_with_processor(mock_trainer, "/test/save/path")

        # Check that both model and processor were saved
        mock_trainer.save_model.assert_called_once_with("/test/save/path")
        mock_processor.save_pretrained.assert_called_once_with("/test/save/path")
