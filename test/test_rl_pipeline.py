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
Unit tests for RL pipeline module.

Tests for ReinL and MatInvent classes.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import paddle
from pymatgen.core.structure import Structure, Lattice

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppmat.rl.base import ReinL, get_device
from ppmat.rl.mat_invent import MatInvent


class TestGetDevice(unittest.TestCase):
    """Test cases for get_device function."""

    @patch('paddle.is_compiled_with_cuda')
    def test_get_device_with_cuda(self, mock_cuda):
        """Test device selection with CUDA available."""
        mock_cuda.return_value = True

        device = get_device()

        self.assertEqual(device, "gpu")

    @patch('paddle.is_compiled_with_cuda')
    def test_get_device_without_cuda(self, mock_cuda):
        """Test device selection without CUDA."""
        mock_cuda.return_value = False

        device = get_device()

        self.assertEqual(device, "cpu")

    @patch('paddle.is_compiled_with_cuda')
    @patch('paddle.set_device')
    def test_get_device_explicit(self, mock_set_device, mock_cuda):
        """Test explicit device selection."""
        device = get_device(device="cpu")

        mock_set_device.assert_called_once()


class TestReinL(unittest.TestCase):
    """Test cases for ReinL base class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create mock components
        self.mock_model_suite = Mock()
        self.mock_model_suite.sample_cfg = {}
        self.mock_model_suite.finetune_cfg = {}
        self.mock_model_suite.get_sampler = Mock(return_value=Mock())

        self.mock_reward = Mock()
        self.mock_reward.threshold = 0.5
        self.mock_reward.scoring = Mock(return_value=(
            np.array([0.3, 0.5, 0.7]),
            {"prop1": np.array([0.1, 0.2, 0.3])},
            np.array([False, False, False])
        ))

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_reinl_initialization(self):
        """Test ReinL initialization."""
        reinl = ReinL(
            rl_epoch=10,
            model_suite=self.mock_model_suite,
            reward=self.mock_reward,
            sample_cfg={},
            finetune_cfg={},
            save_dir=self.temp_dir,
            save_freq=5,
            device="cpu",
        )

        self.assertEqual(reinl.rl_epoch, 10)
        self.assertEqual(reinl.step, 0)
        self.assertEqual(reinl.cost, 0)

        # Check directories were created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "models")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "samples")))

    def test_reinl_with_replay(self):
        """Test ReinL with replay buffer enabled."""
        reinl = ReinL(
            rl_epoch=10,
            model_suite=self.mock_model_suite,
            reward=self.mock_reward,
            sample_cfg={},
            finetune_cfg={},
            save_dir=self.temp_dir,
            save_freq=5,
            replay=True,
            replay_args={"buffer_size": 100, "sample_size": 10},
            device="cpu",
        )

        self.assertIsNotNone(reinl.replay)

    def test_reward_step(self):
        """Test reward calculation step."""
        reinl = ReinL(
            rl_epoch=10,
            model_suite=self.mock_model_suite,
            reward=self.mock_reward,
            sample_cfg={},
            finetune_cfg={},
            save_dir=self.temp_dir,
            save_freq=5,
            device="cpu",
        )

        # Create mock samples
        sample_data = [Mock(), Mock(), Mock()]
        sample_struc = [Mock(), Mock(), Mock()]
        xyz_path = "/tmp/test.xyz"

        success_data, success_struc, success_rewards, success_prop = reinl.reward_step(
            sample_data, sample_struc, xyz_path, "test_label"
        )

        self.assertEqual(len(success_data), 3)
        self.assertEqual(len(success_struc), 3)
        self.assertEqual(len(success_rewards), 3)
        self.assertEqual(reinl.cost, 3)

    def test_reward_step_with_failures(self):
        """Test reward step with some failed samples."""
        # Mock reward with failures
        self.mock_reward.scoring = Mock(return_value=(
            np.array([0.3, 0.0, 0.7]),  # Middle one failed (reward = 0)
            {"prop1": np.array([0.1, 0.2, 0.3])},
            np.array([False, True, False])  # Middle one failed
        ))

        reinl = ReinL(
            rl_epoch=10,
            model_suite=self.mock_model_suite,
            reward=self.mock_reward,
            sample_cfg={},
            finetune_cfg={},
            save_dir=self.temp_dir,
            save_freq=5,
            device="cpu",
        )

        sample_data = [Mock(), Mock(), Mock()]
        sample_struc = [Mock(), Mock(), Mock()]
        xyz_path = "/tmp/test.xyz"

        success_data, success_struc, success_rewards, success_prop = reinl.reward_step(
            sample_data, sample_struc, xyz_path, "test_label"
        )

        # Only 2 samples should succeed
        self.assertEqual(len(success_data), 2)
        self.assertEqual(len(success_struc), 2)
        self.assertEqual(len(success_rewards), 2)


class TestMatInvent(unittest.TestCase):
    """Test cases for MatInvent class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create mock components
        self.mock_model_suite = Mock()
        self.mock_model_suite.sample_cfg = {}
        self.mock_model_suite.finetune_cfg = {
            "batch_size": 8,
            "lr": 0.0005,
            "epochs": 1,
            "timesteps": 10,
            "accum_steps": 2,
            "sigma": 1.0,
        }
        self.mock_model_suite.get_sampler = Mock(return_value=Mock())

        self.mock_reward = Mock()
        self.mock_reward.threshold = 0.5
        self.mock_reward.scoring = Mock(return_value=(
            np.array([0.3, 0.5, 0.7]),
            {"prop1": np.array([0.1, 0.2, 0.3])},
            np.array([False, False, False])
        ))

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_mat_invent_initialization(self):
        """Test MatInvent initialization."""
        with patch.object(MatInvent, 'load_model'):
            mat_invent = MatInvent(
                rl_epoch=10,
                model_suite=self.mock_model_suite,
                reward=self.mock_reward,
                sample_cfg={},
                finetune_cfg={},
                topk_ratio=0.5,
                save_dir=self.temp_dir,
                save_freq=5,
                device="cpu",
            )

            self.assertEqual(mat_invent.topk_ratio, 0.5)
            self.assertFalse(mat_invent.div_filter)

    def test_mat_invent_with_div_filter(self):
        """Test MatInvent with diversity filter enabled."""
        with patch.object(MatInvent, 'load_model'):
            mat_invent = MatInvent(
                rl_epoch=10,
                model_suite=self.mock_model_suite,
                reward=self.mock_reward,
                sample_cfg={},
                finetune_cfg={},
                topk_ratio=0.5,
                save_dir=self.temp_dir,
                save_freq=5,
                div_filter=True,
                df_args={"tol": 10, "buff": 20, "method": "composition"},
                device="cpu",
            )

            self.assertTrue(mat_invent.div_filter)
            self.assertIsNotNone(mat_invent.df_args)

    def test_mat_invent_invalid_topk_ratio(self):
        """Test MatInvent with invalid topk_ratio."""
        with patch.object(MatInvent, 'load_model'):
            with self.assertRaises(AssertionError):
                MatInvent(
                    rl_epoch=10,
                    model_suite=self.mock_model_suite,
                    reward=self.mock_reward,
                    sample_cfg={},
                    finetune_cfg={},
                    topk_ratio=1.5,  # Invalid: > 1.0
                    save_dir=self.temp_dir,
                    save_freq=5,
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
