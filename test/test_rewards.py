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
Unit tests for rewards module.

Tests for Reward class and property scaling functions.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppmat.models.matinvent.rewards.reward import Reward, linear_scaling, average_props, min_props


class TestLinearScaling(unittest.TestCase):
    """Test cases for linear_scaling function."""

    def test_linear_scaling_basic(self):
        """Test basic linear scaling."""
        values = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
        scaled = linear_scaling(values, minv=0.0, maxv=10.0)

        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(scaled, expected)

    def test_linear_scaling_clip(self):
        """Test clipping of values outside range."""
        values = np.array([-5.0, 0.0, 5.0, 10.0, 15.0])
        scaled = linear_scaling(values, minv=0.0, maxv=10.0)

        # Values below min should be 0, above max should be 1
        self.assertEqual(scaled[0], 0.0)
        self.assertEqual(scaled[-1], 1.0)

    def test_linear_scaling_descending(self):
        """Test scaling with inverted values (descending)."""
        values = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
        scaled = linear_scaling(-values, minv=-10.0, maxv=0.0)

        # For descending: larger original values should give smaller scores
        # -values = [-0.0, -2.5, -5.0, -7.5, -10.0]
        # scaled = [1.0, 0.75, 0.5, 0.25, 0.0] (descending order)
        expected = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
        np.testing.assert_array_almost_equal(scaled, expected)


class TestAverageProps(unittest.TestCase):
    """Test cases for average_props function."""

    def test_average_two_props(self):
        """Test averaging two properties."""
        prop_dict = {
            "prop1": np.array([0.2, 0.4, 0.6, 0.8]),
            "prop2": np.array([0.3, 0.5, 0.7, 0.9]),
        }

        result = average_props(prop_dict)

        expected = np.array([0.25, 0.45, 0.65, 0.85])
        np.testing.assert_array_almost_equal(result, expected)

    def test_average_three_props(self):
        """Test averaging three properties."""
        prop_dict = {
            "prop1": np.array([0.1, 0.2, 0.3]),
            "prop2": np.array([0.2, 0.3, 0.4]),
            "prop3": np.array([0.3, 0.4, 0.5]),
        }

        result = average_props(prop_dict)

        expected = np.array([0.2, 0.3, 0.4])
        np.testing.assert_array_almost_equal(result, expected)


class TestMinProps(unittest.TestCase):
    """Test cases for min_props function."""

    def test_min_two_props(self):
        """Test taking minimum of two properties."""
        prop_dict = {
            "prop1": np.array([0.2, 0.5, 0.8]),
            "prop2": np.array([0.3, 0.4, 0.7]),
        }

        result = min_props(prop_dict)

        expected = np.array([0.2, 0.4, 0.7])
        np.testing.assert_array_almost_equal(result, expected)


class TestReward(unittest.TestCase):
    """Test cases for Reward class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create mock property config with proper string name
        prop1 = Mock()
        prop1.name = "prop1"
        prop1.target = "ascending"
        prop1.minv = 0.0
        prop1.maxv = 1.0
        prop1.calculator = Mock()
        prop1.calculator.calc = Mock(return_value=np.array([0.2, 0.5, 0.8]))
        
        prop2 = Mock()
        prop2.name = "prop2"
        prop2.target = "descending"
        prop2.minv = 0.0
        prop2.maxv = 1.0
        prop2.calculator = Mock()
        prop2.calculator.calc = Mock(return_value=np.array([0.7, 0.4, 0.1]))
        
        self.mock_prop_cfg = [prop1, prop2]

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_reward_initialization(self):
        """Test Reward initialization."""
        reward = Reward(
            root_dir=self.temp_dir,
            prop_cfg=self.mock_prop_cfg,
            reward_threshold=0.5,
            reduce="mean",
        )

        self.assertEqual(reward.threshold, 0.5)
        self.assertEqual(reward.reduce, "mean")
        self.assertTrue(os.path.exists(self.temp_dir))

    def test_calc_props(self):
        """Test property calculation."""
        reward = Reward(
            root_dir=self.temp_dir,
            prop_cfg=self.mock_prop_cfg,
            reward_threshold=0.5,
            reduce="mean",
        )

        samples = (None, "test_label")
        prop_dict, failed_mask = reward.calc_props(samples, "test_label")

        self.assertIn("prop1", prop_dict)
        self.assertIn("prop2", prop_dict)
        self.assertEqual(len(prop_dict["prop1"]), 3)
        self.assertEqual(len(prop_dict["prop2"]), 3)

    def test_scoring_ascending(self):
        """Test scoring with ascending target."""
        mock_prop_cfg = [
            Mock(
                name="prop1",
                target="ascending",
                minv=0.0,
                maxv=1.0,
                calculator=Mock(calc=Mock(return_value=np.array([0.2, 0.5, 0.8]))),
            ),
        ]

        reward = Reward(
            root_dir=self.temp_dir,
            prop_cfg=mock_prop_cfg,
            reward_threshold=0.5,
            reduce="mean",
        )

        samples = (None, "test_label")
        rewards, prop_dict, failed_mask = reward.scoring(samples, "test_label")

        # Check that rewards are in [0, 1] range
        self.assertTrue(np.all(rewards >= 0.0))
        self.assertTrue(np.all(rewards <= 1.0))

    def test_scoring_descending(self):
        """Test scoring with descending target."""
        mock_prop_cfg = [
            Mock(
                name="prop1",
                target="descending",
                minv=0.0,
                maxv=1.0,
                calculator=Mock(calc=Mock(return_value=np.array([0.2, 0.5, 0.8]))),
            ),
        ]

        reward = Reward(
            root_dir=self.temp_dir,
            prop_cfg=mock_prop_cfg,
            reward_threshold=0.5,
            reduce="mean",
        )

        samples = (None, "test_label")
        rewards, prop_dict, failed_mask = reward.scoring(samples, "test_label")

        # Check that rewards are in [0, 1] range
        self.assertTrue(np.all(rewards >= 0.0))
        self.assertTrue(np.all(rewards <= 1.0))

    def test_scoring_reduce_mean(self):
        """Test scoring with mean reduction."""
        reward = Reward(
            root_dir=self.temp_dir,
            prop_cfg=self.mock_prop_cfg,
            reward_threshold=0.5,
            reduce="mean",
        )

        samples = (None, "test_label")
        rewards, prop_dict, failed_mask = reward.scoring(samples, "test_label")

        self.assertEqual(len(rewards), 3)

    def test_scoring_reduce_min(self):
        """Test scoring with min reduction."""
        reward = Reward(
            root_dir=self.temp_dir,
            prop_cfg=self.mock_prop_cfg,
            reward_threshold=0.5,
            reduce="min",
        )

        samples = (None, "test_label")
        rewards, prop_dict, failed_mask = reward.scoring(samples, "test_label")

        self.assertEqual(len(rewards), 3)

    def test_scoring_with_nan(self):
        """Test scoring with NaN values."""
        prop = Mock()
        prop.name = "prop1"
        prop.target = "ascending"
        prop.minv = 0.0
        prop.maxv = 1.0
        prop.calculator = Mock()
        prop.calculator.calc = Mock(return_value=np.array([0.2, np.nan, 0.8]))
        
        mock_prop_cfg = [prop]

        reward = Reward(
            root_dir=self.temp_dir,
            prop_cfg=mock_prop_cfg,
            reward_threshold=0.5,
            reduce="mean",
        )

        samples = (None, "test_label")
        rewards, prop_dict, failed_mask = reward.scoring(samples, "test_label")

        # NaN should be replaced with 0.0 and marked as failed
        self.assertTrue(failed_mask[1])  # Middle value was NaN
        self.assertEqual(prop_dict["prop1"][1], 0.0)


if __name__ == "__main__":
    unittest.main()
