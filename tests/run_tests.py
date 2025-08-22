"""Test runner for the frame_transforms package."""

import unittest
import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all test modules
from test_transforms import TestTransform
from test_pose import TestPose  
from test_registry import TestRegistry
from test_utils import TestUtils
from test_integration import TestIntegration


def run_all_tests():
    """Run all tests in the test suite."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTransform))
    suite.addTests(loader.loadTestsFromTestCase(TestPose))
    suite.addTests(loader.loadTestsFromTestCase(TestRegistry))
    suite.addTests(loader.loadTestsFromTestCase(TestUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
