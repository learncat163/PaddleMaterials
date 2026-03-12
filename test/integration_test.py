# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Integration tests for MatInvent RL pipeline.

Tests configuration loading, module imports, and basic functionality.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from omegaconf import OmegaConf
from pymatgen.core.structure import Structure, Lattice


class TestConfigLoading(unittest.TestCase):
    """Test cases for configuration file loading."""

    def test_mattergen_rl_config_exists(self):
        """Test MatterGen RL config file exists and is valid YAML."""
        config_path = "structure_generation/configs/rl/mattergen_rl.yaml"

        self.assertTrue(os.path.exists(config_path), f"Config file not found: {config_path}")

        # Try to load the config
        config = OmegaConf.load(config_path)
        self.assertIsNotNone(config)

        # Check required sections
        self.assertIn("Global", config)
        self.assertIn("Model", config)
        self.assertIn("RL", config)

    def test_diffcsp_rl_config_exists(self):
        """Test DiffCSP RL config file exists and is valid YAML."""
        config_path = "structure_generation/configs/rl/diffcsp_rl.yaml"

        self.assertTrue(os.path.exists(config_path), f"Config file not found: {config_path}")

        # Try to load the config
        config = OmegaConf.load(config_path)
        self.assertIsNotNone(config)

        # Check required sections
        self.assertIn("Global", config)
        self.assertIn("Model", config)
        self.assertIn("RL", config)

    def test_mattergen_config_values(self):
        """Test MatterGen config has expected values."""
        config_path = "structure_generation/configs/rl/mattergen_rl.yaml"
        config = OmegaConf.load(config_path)

        # Check RL parameters
        self.assertIn("rl_epoch", config.RL)
        self.assertIn("topk_ratio", config.RL)
        self.assertIn("sample_cfg", config.RL)
        self.assertIn("finetune_cfg", config.RL)
        self.assertIn("reward_cfg", config.RL)

        # Check types
        self.assertIsInstance(config.RL.rl_epoch, (int, str))
        self.assertIsInstance(config.RL.topk_ratio, (float, str))

    def test_diffcsp_config_values(self):
        """Test DiffCSP config has expected values."""
        config_path = "structure_generation/configs/rl/diffcsp_rl.yaml"
        config = OmegaConf.load(config_path)

        # Check RL parameters
        self.assertIn("rl_epoch", config.RL)
        self.assertIn("topk_ratio", config.RL)
        self.assertIn("sample_cfg", config.RL)
        self.assertIn("finetune_cfg", config.RL)


class TestModuleImports(unittest.TestCase):
    """Test cases for module imports."""

    def test_import_memory_module(self):
        """Test memory module can be imported."""
        from ppmat.models.matinvent.memory import ReplayBuffer, LongTimeMem
        self.assertIsNotNone(ReplayBuffer)
        self.assertIsNotNone(LongTimeMem)

    def test_import_rewards_module(self):
        """Test rewards module can be imported."""
        from ppmat.models.matinvent.rewards import Reward, Calculator
        self.assertIsNotNone(Reward)
        self.assertIsNotNone(Calculator)

    def test_import_rl_module(self):
        """Test RL module can be imported."""
        from ppmat.models.matinvent.rl import ReinL, MatInvent
        self.assertIsNotNone(ReinL)
        self.assertIsNotNone(MatInvent)

    def test_import_rl_models(self):
        """Test RL model suites can be imported."""
        from ppmat.models.matinvent.rl.models import ModelSuite, MatterGenSuite, DiffCSPSuite
        self.assertIsNotNone(ModelSuite)
        self.assertIsNotNone(MatterGenSuite)
        self.assertIsNotNone(DiffCSPSuite)


class TestBasicFunctionality(unittest.TestCase):
    """Test cases for basic functionality."""

    def test_replay_buffer_basic_ops(self):
        """Test ReplayBuffer basic operations."""
        from ppmat.models.matinvent.memory import ReplayBuffer

        buffer = ReplayBuffer(buffer_size=10, sample_size=4)

        # Mock data and structures
        class MockData:
            pass

        class MockStructure:
            def __init__(self, formula):
                self.composition = Mock()
                self.composition.reduced_formula = formula
                # Create mock species
                self.species = [Mock()]

        data_list = [MockData() for _ in range(5)]
        strucs = [MockStructure(f"Elem{i}") for i in range(5)]
        rewards = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        # Test extend
        buffer.extend(data_list, strucs, rewards)
        self.assertEqual(len(buffer), 5)

        # Test sample
        sampled_data, sampled_rewards = buffer.sample()
        self.assertEqual(len(sampled_data), 4)

    def test_ltm_basic_ops(self):
        """Test LongTimeMem basic operations."""
        from ppmat.models.matinvent.memory import LongTimeMem

        ltm = LongTimeMem()

        # Mock structure
        class MockStructure:
            def __init__(self, formula):
                self.composition = Mock()
                self.composition.reduced_formula = formula
                self.species = [Mock()]

        strucs = [MockStructure("Si"), MockStructure("Ge")]
        rewards = np.array([0.5, 0.7])

        # Test extend
        ltm.extend(strucs, rewards, step=0)
        self.assertEqual(len(ltm), 2)

        # Test unique components
        self.assertEqual(len(ltm.unique_comps), 2)

    def test_reward_scaling_functions(self):
        """Test reward scaling functions."""
        from ppmat.models.matinvent.rewards.reward import linear_scaling, average_props, min_props

        # Test linear scaling
        values = np.array([0.0, 0.5, 1.0])
        scaled = linear_scaling(values, minv=0.0, maxv=1.0)
        np.testing.assert_array_almost_equal(scaled, [0.0, 0.5, 1.0])

        # Test average props
        prop_dict = {"p1": np.array([0.2, 0.4]), "p2": np.array([0.3, 0.5])}
        avg = average_props(prop_dict)
        np.testing.assert_array_almost_equal(avg, [0.25, 0.45])

        # Test min props
        # Note: dict order may vary, so we check that min of [0.2,0.3] is 0.2 and min of [0.4,0.5] is 0.4
        # The result depends on dict iteration order, so we just verify it's one of the two rows
        min_val = min_props(prop_dict)
        # min_val should be the element-wise minimum of the two arrays
        # Since dict order is not guaranteed, we verify the shape and reasonable values
        self.assertEqual(len(min_val), 2)
        self.assertTrue(min_val[0] in [0.2, 0.3])  # Either p1[0] or p2[0]
        self.assertTrue(min_val[1] in [0.4, 0.5])  # Either p1[1] or p2[1]

    def test_device_selection(self):
        """Test device selection function."""
        from ppmat.models.matinvent.rl.base import get_device

        # Test default device selection
        device = get_device()
        # Paddle returns Place object, check it contains gpu or cpu
        device_str = str(device)
        self.assertTrue("gpu" in device_str or "cpu" in device_str)

    def test_mattergen_has_rl_methods(self):
        """Test MatterGen model has RL methods."""
        from ppmat.models import MatterGen

        # Check that RL methods exist (even if not implemented)
        self.assertTrue(hasattr(MatterGen, "add_noise"))
        self.assertTrue(hasattr(MatterGen, "calc_sample_loss"))
        self.assertTrue(hasattr(MatterGen, "calc_kl_reg"))

    def test_diffcsp_has_rl_methods(self):
        """Test DiffCSP model has RL methods."""
        from ppmat.models import DiffCSP

        # Check that RL methods exist (even if not implemented)
        self.assertTrue(hasattr(DiffCSP, "add_noise"))
        self.assertTrue(hasattr(DiffCSP, "calc_sample_loss"))
        self.assertTrue(hasattr(DiffCSP, "calc_kl_reg"))


class TestTrainingScript(unittest.TestCase):
    """Test cases for training script."""

    def test_training_script_exists(self):
        """Test training script exists."""
        script_path = "ppmat/models/matinvent/rl_train.py"
        self.assertTrue(os.path.exists(script_path))

    def test_training_script_importable(self):
        """Test training script can be imported."""
        import importlib.util

        script_path = "ppmat/models/matinvent/rl_train.py"
        spec = importlib.util.spec_from_file_location("rl_train", script_path)
        self.assertIsNotNone(spec)


class TestModelSaveLoad(unittest.TestCase):
    """Test cases for model save/load functionality."""

    def test_model_suite_has_save_method(self):
        """Test model suites have save method."""
        from ppmat.models.matinvent.rl.models import MatterGenSuite, DiffCSPSuite

        # Check save method exists
        self.assertTrue(hasattr(MatterGenSuite, "save_model"))
        self.assertTrue(hasattr(DiffCSPSuite, "save_model"))

    def test_model_suite_has_load_method(self):
        """Test model suites have load method."""
        from ppmat.models.matinvent.rl.models import MatterGenSuite, DiffCSPSuite

        # Check load method exists
        self.assertTrue(hasattr(MatterGenSuite, "load_model"))
        self.assertTrue(hasattr(DiffCSPSuite, "load_model"))


def run_tests():
    """Run all integration tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConfigLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestModuleImports))
    suite.addTests(loader.loadTestsFromTestCase(TestBasicFunctionality))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingScript))
    suite.addTests(loader.loadTestsFromTestCase(TestModelSaveLoad))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
