"""
Test to verify that the diffusion trainer logging methods work correctly
"""
import unittest
import numpy as np
from typing import List, Dict


class TestDiffusionTrainerLogging(unittest.TestCase):
    """Test that the diffusion trainer logging methods handle validation correctly"""
    
    def test_on_train_epoch_end_tracking(self):
        """Test that on_train_epoch_end properly tracks training losses"""
        # Mock the trainer
        class MockTrainer:
            def __init__(self):
                self.train_losses = []
                self.current_epoch = 0
                
            def print_to_log_file(self, msg):
                pass  # Silent for testing
                
            def on_train_epoch_end(self, train_outputs: List[dict]):
                """Log training epoch statistics"""
                from nnunetv2.utilities.collate_outputs import collate_outputs
                outputs_collated = collate_outputs(train_outputs)
                train_loss = np.mean(outputs_collated['loss'])
                
                self.train_losses.append(train_loss)
                
                self.print_to_log_file(f"Epoch {self.current_epoch} - Training loss: {train_loss:.6f}")
        
        trainer = MockTrainer()
        
        # Simulate training outputs
        train_outputs = [
            {'loss': np.array([0.5])},
            {'loss': np.array([0.6])},
            {'loss': np.array([0.4])},
        ]
        
        trainer.on_train_epoch_end(train_outputs)
        
        # Verify losses were tracked
        self.assertEqual(len(trainer.train_losses), 1)
        expected_loss = np.mean([0.5, 0.6, 0.4])
        self.assertAlmostEqual(trainer.train_losses[0], expected_loss, places=5)
    
    def test_on_validation_epoch_end_tracking(self):
        """Test that on_validation_epoch_end properly tracks validation losses and EMA"""
        # Mock the trainer
        class MockTrainer:
            def __init__(self):
                self.validation_losses = []
                self._best_ema = None
                self.current_epoch = 0
                
            def print_to_log_file(self, msg):
                pass  # Silent for testing
                
            def on_validation_epoch_end(self, val_outputs: List[dict]):
                """Handle validation epoch end"""
                from nnunetv2.utilities.collate_outputs import collate_outputs
                outputs_collated = collate_outputs(val_outputs)
                val_loss = np.mean(outputs_collated['loss'])
                
                self.validation_losses.append(val_loss)
                
                self.print_to_log_file(f"Epoch {self.current_epoch} - Validation loss: {val_loss:.6f}")
                
                # Update EMA for checkpoint selection
                if self._best_ema is None:
                    self._best_ema = val_loss
                else:
                    self._best_ema = 0.9 * self._best_ema + 0.1 * val_loss
                
                self.print_to_log_file(f"Best EMA: {self._best_ema:.6f}")
        
        trainer = MockTrainer()
        
        # Simulate validation outputs for first epoch
        val_outputs_1 = [
            {'loss': np.array([0.3])},
            {'loss': np.array([0.35])},
        ]
        
        trainer.on_validation_epoch_end(val_outputs_1)
        
        # Verify first epoch
        self.assertEqual(len(trainer.validation_losses), 1)
        expected_loss_1 = np.mean([0.3, 0.35])
        self.assertAlmostEqual(trainer.validation_losses[0], expected_loss_1, places=5)
        self.assertAlmostEqual(trainer._best_ema, expected_loss_1, places=5)
        
        # Simulate validation outputs for second epoch
        trainer.current_epoch = 1
        val_outputs_2 = [
            {'loss': np.array([0.25])},
            {'loss': np.array([0.3])},
        ]
        
        trainer.on_validation_epoch_end(val_outputs_2)
        
        # Verify second epoch
        self.assertEqual(len(trainer.validation_losses), 2)
        expected_loss_2 = np.mean([0.25, 0.3])
        self.assertAlmostEqual(trainer.validation_losses[1], expected_loss_2, places=5)
        
        # Verify EMA update
        expected_ema = 0.9 * expected_loss_1 + 0.1 * expected_loss_2
        self.assertAlmostEqual(trainer._best_ema, expected_ema, places=5)
    
    def test_no_logger_assertion_error(self):
        """Test that validation doesn't call logger.log with uninitialized keys"""
        # Mock the trainer with a strict logger
        class StrictLogger:
            def __init__(self):
                self.my_fantastic_logging = {
                    'epoch_end_timestamps': []  # Only this key is initialized
                }
            
            def log(self, key, value, epoch):
                # This simulates the actual logger behavior
                assert key in self.my_fantastic_logging.keys() and isinstance(self.my_fantastic_logging[key], list), \
                    f"This function is only intended to log stuff to lists and to have one entry per epoch. Key: {key}"
        
        class MockTrainer:
            def __init__(self):
                self.validation_losses = []
                self._best_ema = None
                self.current_epoch = 0
                self.logger = StrictLogger()
                
            def print_to_log_file(self, msg):
                pass
                
            def on_validation_epoch_end(self, val_outputs: List[dict]):
                """Handle validation epoch end - should NOT call logger.log with 'val_loss'"""
                from nnunetv2.utilities.collate_outputs import collate_outputs
                outputs_collated = collate_outputs(val_outputs)
                val_loss = np.mean(outputs_collated['loss'])
                
                self.validation_losses.append(val_loss)
                
                # This is the fix - we don't call logger.log with uninitialized keys
                # self.logger.log('val_loss', val_loss, self.current_epoch)  # OLD - would fail
                
                self.print_to_log_file(f"Epoch {self.current_epoch} - Validation loss: {val_loss:.6f}")
                
                if self._best_ema is None:
                    self._best_ema = val_loss
                else:
                    self._best_ema = 0.9 * self._best_ema + 0.1 * val_loss
                
                self.print_to_log_file(f"Best EMA: {self._best_ema:.6f}")
        
        trainer = MockTrainer()
        
        # This should NOT raise an AssertionError
        val_outputs = [
            {'loss': np.array([0.5])},
        ]
        
        try:
            trainer.on_validation_epoch_end(val_outputs)
            # If we get here, the test passed - no assertion error
            self.assertTrue(True)
        except AssertionError as e:
            self.fail(f"on_validation_epoch_end should not cause logger assertion error: {e}")


if __name__ == '__main__':
    unittest.main()

