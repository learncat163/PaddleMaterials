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
Test script to download and verify MatterGen and DiffCSP model weights.

This script tests the native model loading functionality by:
1. Downloading pre-trained weights from BOS server
2. Loading model configurations
3. Building models with weights
"""

import os
import sys
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ppmat.models import build_model_from_name


class TestModelDownload(unittest.TestCase):
    """Test cases for downloading and loading pre-trained models."""

    def test_mattergen_mp20_download(self):
        """Test downloading and loading MatterGen MP20 model."""
        print("\n" + "=" * 60)
        print("Testing MatterGen MP20 model download and loading...")
        print("=" * 60)

        model_name = "mattergen_mp20"

        try:
            model, config = build_model_from_name(model_name)

            self.assertIsNotNone(model, "Model should not be None")
            self.assertIsNotNone(config, "Config should not be None")

            # Check model has parameters
            params = list(model.parameters())
            self.assertGreater(len(params), 0, "Model should have parameters")

            # Check total parameters count
            total_params = sum(p.numel().item() for p in params)
            print(f"✓ MatterGen MP20 loaded successfully")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Model type: {type(model).__name__}")

        except Exception as e:
            self.fail(f"Failed to load MatterGen MP20: {e}")

    def test_mattergen_alex_mp20_download(self):
        """Test downloading and loading MatterGen Alex MP20 model."""
        print("\n" + "=" * 60)
        print("Testing MatterGen Alex MP20 model download and loading...")
        print("=" * 60)

        model_name = "mattergen_alex_mp20"

        try:
            model, config = build_model_from_name(model_name)

            self.assertIsNotNone(model, "Model should not be None")
            self.assertIsNotNone(config, "Config should not be None")

            params = list(model.parameters())
            total_params = sum(p.numel().item() for p in params)
            print(f"✓ MatterGen Alex MP20 loaded successfully")
            print(f"  Total parameters: {total_params:,}")

        except Exception as e:
            self.fail(f"Failed to load MatterGen Alex MP20: {e}")

    def test_diffcsp_mp20_download(self):
        """Test downloading and loading DiffCSP MP20 model."""
        print("\n" + "=" * 60)
        print("Testing DiffCSP MP20 model download and loading...")
        print("=" * 60)

        model_name = "diffcsp_mp20"

        try:
            model, config = build_model_from_name(model_name)

            self.assertIsNotNone(model, "Model should not be None")
            self.assertIsNotNone(config, "Config should not be None")

            params = list(model.parameters())
            total_params = sum(p.numel().item() for p in params)
            print(f"✓ DiffCSP MP20 loaded successfully")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Model type: {type(model).__name__}")

        except Exception as e:
            self.fail(f"Failed to load DiffCSP MP20: {e}")

    def test_mattergen_conditional_models(self):
        """Test downloading conditional MatterGen models."""
        print("\n" + "=" * 60)
        print("Testing conditional MatterGen models...")
        print("=" * 60)

        conditional_models = [
            "mattergen_mp20_chemical_system",
            "mattergen_mp20_dft_band_gap",
        ]

        for model_name in conditional_models:
            try:
                model, config = build_model_from_name(model_name)
                params = list(model.parameters())
                total_params = sum(p.numel().item() for p in params)
                print(f"✓ {model_name} loaded successfully ({total_params:,} params)")
            except Exception as e:
                print(f"✗ {model_name} failed: {e}")

    def test_weights_cache_location(self):
        """Test that weights are cached in the correct location."""
        print("\n" + "=" * 60)
        print("Testing weights cache location...")
        print("=" * 60)

        # Default cache location
        cache_dir = os.path.expanduser("~/.paddlemat/weights")
        self.assertTrue(os.path.exists(cache_dir), f"Cache dir should exist: {cache_dir}")

        print(f"✓ Weights cache directory: {cache_dir}")

        # List cached models
        if os.path.exists(cache_dir):
            cached_items = os.listdir(cache_dir)
            print(f"  Cached items: {len(cached_items)}")

            # Check for specific model directories
            model_dirs = [d for d in cached_items if os.path.isdir(os.path.join(cache_dir, d))]
            for model_dir in sorted(model_dirs):
                print(f"    - {model_dir}")


def run_tests():
    """Run all download tests."""
    print("\n" + "=" * 70)
    print("MODEL DOWNLOAD TEST SUITE")
    print("=" * 70)
    print("\nThis test will download pre-trained model weights if not already cached.")
    print("Download size may be large (several GB).")
    print("")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestModelDownload))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
