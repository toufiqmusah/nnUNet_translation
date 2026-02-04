"""
Unit tests for nnUNetDiffusionTrainer to verify channel handling
"""
import unittest
import torch
import tempfile
import json
import os
from pathlib import Path

# Mock necessary components
class MockPlansManager:
    def __init__(self):
        pass
    
    def get_label_manager(self, dataset_json):
        return MockLabelManager()

class MockLabelManager:
    def __init__(self):
        self.num_segmentation_heads = 2
        self.foreground_labels = [1]

class MockConfigurationManager:
    def __init__(self):
        self.network_arch_class_name = "dynamic_network_architectures.architectures.unet.PlainConvUNet"
        self.network_arch_init_kwargs = {
            'n_stages': 5,
            'features_per_stage': [32, 64, 128, 256, 320],
            'conv_op': 'torch.nn.modules.conv.Conv3d',
            'kernel_sizes': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            'strides': [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            'n_conv_per_stage': [2, 2, 2, 2, 2],
            'n_conv_per_stage_decoder': [2, 2, 2, 2],
            'conv_bias': True,
            'norm_op': 'torch.nn.modules.instancenorm.InstanceNorm3d',
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None,
            'dropout_op_kwargs': None,
            'nonlin': 'torch.nn.LeakyReLU',
            'nonlin_kwargs': {'inplace': True},
        }
        self.network_arch_init_kwargs_req_import = [
            'conv_op', 'norm_op', 'dropout_op', 'nonlin'
        ]
        self.previous_stage_name = None


class TestDiffusionTrainerChannels(unittest.TestCase):
    """Test that the diffusion trainer correctly handles input channels"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')  # Use CPU for testing
        
        # Create a temporary directory for test artifacts
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal dataset_json
        self.dataset_json = {
            'channel_names': {
                '0': 'MR'
            },
            'labels': {
                'background': 0,
                'CT': 1
            },
            'file_ending': '.nii.gz',
            'numTraining': 10
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_single_channel_input_doubles_to_two(self):
        """Test that 1 input channel becomes 2 after concatenation conditioning"""
        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
        
        # Setup mock objects
        plans_manager = MockPlansManager()
        config_manager = MockConfigurationManager()
        
        # Test determine_num_input_channels (base function)
        base_channels = determine_num_input_channels(
            plans_manager,
            config_manager,
            self.dataset_json
        )
        
        # Verify base channels is 1 (single MR modality)
        self.assertEqual(base_channels, 1, 
                        f"Expected 1 base channel for single modality, got {base_channels}")
        
        # Verify that after concatenation, we should have 2 channels
        expected_network_channels = base_channels * 2
        self.assertEqual(expected_network_channels, 2,
                        f"Expected 2 network input channels after concatenation, got {expected_network_channels}")
    
    def test_multi_channel_input_doubles_correctly(self):
        """Test that N input channels become 2*N after concatenation conditioning"""
        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
        
        # Setup mock objects
        plans_manager = MockPlansManager()
        config_manager = MockConfigurationManager()
        
        # Create dataset with 2 modalities
        dataset_json_multi = {
            'channel_names': {
                '0': 'T1',
                '1': 'T2'
            },
            'labels': {
                'background': 0,
                'CT': 1
            },
            'file_ending': '.nii.gz',
            'numTraining': 10
        }
        
        # Test determine_num_input_channels
        base_channels = determine_num_input_channels(
            plans_manager,
            config_manager,
            dataset_json_multi
        )
        
        # Verify base channels is 2 (two modalities)
        self.assertEqual(base_channels, 2,
                        f"Expected 2 base channels for two modalities, got {base_channels}")
        
        # Verify that after concatenation, we should have 4 channels
        expected_network_channels = base_channels * 2
        self.assertEqual(expected_network_channels, 4,
                        f"Expected 4 network input channels after concatenation, got {expected_network_channels}")
    
    def test_concat_conditioning_doubles_channels(self):
        """Test that ConcatConditioning correctly concatenates along channel dimension"""
        from nnunetv2.training.diffusion.conditioning.concat_conditioning import ConcatConditioning
        
        conditioning_method = ConcatConditioning()
        
        # Create mock tensors
        batch_size = 2
        channels = 1
        spatial_dims = (8, 8, 8)
        
        x_t = torch.randn(batch_size, channels, *spatial_dims)
        source = torch.randn(batch_size, channels, *spatial_dims)
        
        # Prepare and apply conditioning
        conditioning = conditioning_method.prepare_conditioning(source)
        model_input = conditioning_method.apply_conditioning(x_t, conditioning)
        
        # Verify shapes
        expected_shape = (batch_size, channels * 2, *spatial_dims)
        self.assertEqual(model_input.shape, expected_shape,
                        f"Expected shape {expected_shape}, got {model_input.shape}")
        
        # Verify concatenation is correct (first channels should be x_t, second should be source)
        self.assertTrue(torch.allclose(model_input[:, :channels], x_t),
                       "First channels should match x_t")
        self.assertTrue(torch.allclose(model_input[:, channels:], source),
                       "Second channels should match source")


if __name__ == '__main__':
    unittest.main()
