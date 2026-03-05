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
Unit tests for memory module.

Tests for ReplayBuffer and LongTimeMem classes.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure, Lattice

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppmat.models.matinvent.memory.replay_buffer import ReplayBuffer
from ppmat.models.matinvent.memory.ltm import LongTimeMem


class TestReplayBuffer(unittest.TestCase):
    """Test cases for ReplayBuffer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.buffer = ReplayBuffer(
            buffer_size=10,
            sample_size=4,
            reward_cutoff=0.0,
        )

    def create_mock_structure(self, formula: str = "Si") -> Structure:
        """Create a mock structure for testing."""
        # Map of test names to real elements
        element_map = {
            "Si": "Si", "Ge": "Ge", "C": "C",
            "Elem0": "Al", "Elem1": "Mg", "Elem2": "Ca",
            "Elem3": "Ti", "Elem4": "Fe", "Elem5": "Co",
            "Elem6": "Ni", "Elem7": "Cu", "Elem8": "Zn",
        }
        element = element_map.get(formula, formula)
        
        lattice_params = {
            "Si": 5.43, "Ge": 5.66, "C": 3.57,
            "Al": 4.05, "Mg": 3.21, "Ca": 5.59,
            "Ti": 4.73, "Fe": 2.87, "Co": 3.55,
            "Ni": 3.52, "Cu": 3.61, "Zn": 4.59,
        }
        a = lattice_params.get(element, 4.0)
        lattice = Lattice.cubic(a)
        return Structure(lattice, [element], [[0, 0, 0]])

    def create_mock_data(self):
        """Create mock data object."""
        class MockData:
            def __init__(self):
                self.x = np.array([[1.0, 2.0, 3.0]])
                self.edge_index = np.array([[0], [0]])
        return MockData()

    def test_buffer_initialization(self):
        """Test buffer initialization."""
        self.assertEqual(self.buffer.buffer_size, 10)
        self.assertEqual(self.buffer.sample_size, 4)
        self.assertEqual(len(self.buffer), 0)

    def test_buffer_extend(self):
        """Test adding samples to buffer."""
        # Use 5 different structures to avoid deduplication
        data_list = [self.create_mock_data() for _ in range(5)]
        strucs = [self.create_mock_structure(f"Elem{i}") for i in range(5)]
        rewards = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        self.buffer.extend(data_list, strucs, rewards)

        self.assertEqual(len(self.buffer), 5)
        self.assertEqual(self.buffer.buffer["reward"].max(), 0.9)

    def test_buffer_deduplication(self):
        """Test deduplication by composition."""
        data_list = [self.create_mock_data() for _ in range(5)]
        # Add duplicate structures
        strucs = [
            self.create_mock_structure("Si"),
            self.create_mock_structure("Si"),  # Duplicate
            self.create_mock_structure("Ge"),
            self.create_mock_structure("Si"),  # Duplicate
            self.create_mock_structure("Ge"),  # Duplicate
        ]
        rewards = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        self.buffer.extend(data_list, strucs, rewards)

        # Should keep only 2 unique compositions (Si and Ge)
        self.assertLessEqual(len(self.buffer), 2)

    def test_buffer_sample(self):
        """Test sampling from buffer."""
        data_list = [self.create_mock_data() for _ in range(5)]
        strucs = [self.create_mock_structure(f"Elem{i}") for i in range(5)]
        rewards = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        self.buffer.extend(data_list, strucs, rewards)

        sampled_data, sampled_rewards = self.buffer.sample()

        self.assertEqual(len(sampled_data), 4)  # sample_size
        self.assertEqual(len(sampled_rewards), 4)

    def test_buffer_empty_sample(self):
        """Test sampling from empty buffer."""
        sampled_data, sampled_rewards = self.buffer.sample()

        self.assertEqual(len(sampled_data), 0)
        self.assertEqual(len(sampled_rewards), 0)

    def test_memory_purge(self):
        """Test purging samples from buffer."""
        data_list = [self.create_mock_data() for _ in range(3)]
        strucs = [
            self.create_mock_structure("Si"),
            self.create_mock_structure("Ge"),
            self.create_mock_structure("C"),
        ]
        rewards = np.array([0.5, 0.7, 0.9])

        self.buffer.extend(data_list, strucs, rewards)

        # Purge Si structure
        self.buffer.memory_purge([strucs[0]])

        # Buffer should no longer contain Si
        self.assertNotIn("Si", self.buffer.buffer["comp"].values)


class TestLongTimeMem(unittest.TestCase):
    """Test cases for LongTimeMem class."""

    def setUp(self):
        """Set up test fixtures."""
        self.ltm = LongTimeMem()

    def create_mock_structure(self, formula: str = "Si") -> Structure:
        """Create a mock structure for testing."""
        # Map of test names to real elements
        element_map = {
            "Si": "Si", "Ge": "Ge", "C": "C",
            "Elem0": "Al", "Elem1": "Mg", "Elem2": "Ca",
            "Elem3": "Ti", "Elem4": "Fe", "Elem5": "Co",
            "Elem6": "Ni", "Elem7": "Cu", "Elem8": "Zn",
        }
        element = element_map.get(formula, formula)
        
        lattice_params = {
            "Si": 5.43, "Ge": 5.66, "C": 3.57,
            "Al": 4.05, "Mg": 3.21, "Ca": 5.59,
            "Ti": 4.73, "Fe": 2.87, "Co": 3.55,
            "Ni": 3.52, "Cu": 3.61, "Zn": 4.59,
        }
        a = lattice_params.get(element, 4.0)
        lattice = Lattice.cubic(a)
        return Structure(lattice, [element], [[0, 0, 0]])

    def test_ltm_initialization(self):
        """Test LTM initialization."""
        self.assertEqual(len(self.ltm), 0)
        self.assertEqual(len(self.ltm.unique_comps), 0)

    def test_ltm_extend(self):
        """Test adding samples to LTM."""
        strucs = [
            self.create_mock_structure("Si"),
            self.create_mock_structure("Ge"),
            self.create_mock_structure("C"),
        ]
        rewards = np.array([0.5, 0.7, 0.9])

        self.ltm.extend(strucs, rewards, step=0)

        self.assertEqual(len(self.ltm), 3)
        self.assertEqual(len(self.ltm.unique_comps), 3)

    def test_ltm_div_filter(self):
        """Test diversity filter."""
        # Add some initial samples
        strucs_init = [self.create_mock_structure("Si") for _ in range(15)]
        rewards_init = np.ones(15) * 0.5
        self.ltm.extend(strucs_init, rewards_init, step=0)

        # Test diversity filter
        strucs_test = [self.create_mock_structure("Si") for _ in range(5)]
        rewards_test = np.ones(5) * 0.8

        new_rewards, penalty_idx, tol_n, buff_n = self.ltm.div_filter(
            strucs_test, rewards_test, tol=10, buff=20
        )

        # Some samples should be penalized
        self.assertGreater(tol_n + buff_n, 0)
        self.assertEqual(len(new_rewards), 5)
        self.assertEqual(len(penalty_idx), buff_n)

    def test_ltm_calc_metrics(self):
        """Test metric calculation."""
        strucs = [
            self.create_mock_structure("Si"),
            self.create_mock_structure("Ge"),
            self.create_mock_structure("C"),
        ]
        rewards = np.array([0.5, 0.7, 0.9])

        self.ltm.extend(strucs, rewards, step=0)

        burden, div_ratio = self.ltm.calc_metrics(thred=0.5)

        # burden should be None (not enough candidates)
        # div_ratio should be calculated
        self.assertIsNone(burden)
        self.assertIsNotNone(div_ratio)

    def test_ltm_get_baseline(self):
        """Test baseline calculation."""
        strucs = [self.create_mock_structure("Si") for _ in range(5)]
        rewards = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

        self.ltm.extend(strucs, rewards, step=0)
        self.ltm.extend(strucs, rewards * 1.1, step=1)
        self.ltm.extend(strucs, rewards * 1.2, step=2)

        baseline = self.ltm.get_baseline(step=2, prev=2)

        # Baseline should be the mean reward from steps 0-1
        self.assertIsNotNone(baseline)

    def test_ltm_save(self):
        """Test saving LTM to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "ltm_test.csv")

            strucs = [self.create_mock_structure("Si"), self.create_mock_structure("Ge")]
            rewards = np.array([0.5, 0.7])

            self.ltm.extend(strucs, rewards, step=0)
            self.ltm.save(save_path)

            # Check file was created
            self.assertTrue(os.path.exists(save_path))

            # Check file content
            df = pd.read_csv(save_path)
            self.assertEqual(len(df), 2)


if __name__ == "__main__":
    unittest.main()
