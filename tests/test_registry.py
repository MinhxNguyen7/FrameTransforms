"""Tests for the Registry class."""

import unittest
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation

from frame_transforms import Transform, Registry, InvalidTransformError


class TestRegistry(unittest.TestCase):
    """Test cases for the Registry class."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = Registry("world")

        self.simple_transform = Transform(
            np.array([1.0, 2.0, 3.0]),
            Rotation.from_euler("xyz", [0, 0, 90], degrees=True),
        )

        self.complex_transform = Transform(
            np.array([0.5, -0.5, 1.0]),
            Rotation.from_euler("xyz", [45, 30, 60], degrees=True),
        )

    def test_init(self):
        """Test Registry initialization."""
        registry = Registry("test_world")

        # Should have the world frame
        self.assertIn("test_world", registry._adjacencies)
        self.assertEqual(registry._parents["test_world"], None)

    def test_add_transform_valid(self):
        """Test adding a valid transform."""
        self.registry.add_transform("world", "base", self.simple_transform)

        # Both frames should exist
        self.assertIn("world", self.registry._adjacencies)
        self.assertIn("base", self.registry._adjacencies)

        # Bidirectional connection should exist
        self.assertIn("base", self.registry._adjacencies["world"])
        self.assertIn("world", self.registry._adjacencies["base"])

    def test_add_transform_with_numpy_array(self):
        """Test adding transform with numpy array."""
        matrix = self.simple_transform.as_matrix()
        self.registry.add_transform("world", "base", matrix)

        # Should work the same as with Transform object
        retrieved = self.registry.get_transform("world", "base")
        np.testing.assert_array_almost_equal(
            retrieved.translation, self.simple_transform.translation
        )

    def test_add_transform_both_frames_exist(self):
        """Test adding transform when both frames already exist."""
        self.registry.add_transform("world", "base", self.simple_transform)

        with self.assertRaises(InvalidTransformError):
            self.registry.add_transform("world", "base", self.complex_transform)

    def test_add_transform_neither_frame_exists(self):
        """Test adding transform when neither frame exists."""
        with self.assertRaises(InvalidTransformError):
            self.registry.add_transform("base", "camera", self.simple_transform)

    def test_add_transform_invalid_matrix_shape(self):
        """Test adding transform with invalid matrix shape."""
        invalid_matrix = np.eye(3)  # Wrong shape

        with self.assertRaises(ValueError):
            self.registry.add_transform("world", "base", invalid_matrix)

    def test_add_transform_invalid_type(self):
        """Test adding transform with invalid type."""
        with self.assertRaises(ValueError):
            self.registry.add_transform("world", "base", "invalid")

    def test_get_transform_same_frame(self):
        """Test getting transform from frame to itself."""
        transform = self.registry.get_transform("world", "world")

        # Should be identity transform
        np.testing.assert_array_almost_equal(
            transform.translation, np.array([0.0, 0.0, 0.0])
        )
        np.testing.assert_array_almost_equal(transform.rotation.as_matrix(), np.eye(3))

    def test_get_transform_direct_connection(self):
        """Test getting transform with direct connection."""
        self.registry.add_transform("world", "base", self.simple_transform)

        # Forward direction
        transform = self.registry.get_transform("world", "base")
        np.testing.assert_array_almost_equal(
            transform.translation, self.simple_transform.translation
        )
        np.testing.assert_array_almost_equal(
            transform.rotation.as_matrix(), self.simple_transform.rotation.as_matrix()
        )

        # Reverse direction (should be inverse)
        inverse_transform = self.registry.get_transform("base", "world")
        expected_inverse_matrix = np.linalg.inv(self.simple_transform.as_matrix())
        actual_inverse_matrix = inverse_transform.as_matrix()
        np.testing.assert_array_almost_equal(
            actual_inverse_matrix, expected_inverse_matrix
        )

    def test_get_transform_transitive(self):
        """Test getting transform through intermediate frame."""
        # Create chain: world -> base -> camera
        self.registry.add_transform("world", "base", self.simple_transform)
        self.registry.add_transform("base", "camera", self.complex_transform)

        # Get transform from world to camera
        transform = self.registry.get_transform("world", "camera")

        # Should be composition of transforms
        expected_matrix = (
            self.simple_transform.as_matrix() @ self.complex_transform.as_matrix()
        )
        actual_matrix = transform.as_matrix()
        np.testing.assert_array_almost_equal(actual_matrix, expected_matrix)

    def test_get_transform_nonexistent_frame(self):
        """Test getting transform to nonexistent frame."""
        with self.assertRaises(InvalidTransformError):
            self.registry.get_transform("world", "nonexistent")

    def test_update_transform_valid(self):
        """Test updating an existing transform."""
        self.registry.add_transform("world", "base", self.simple_transform)

        # Update the transform
        new_transform = Transform(
            np.array([5.0, 6.0, 7.0]),
            Rotation.from_euler("xyz", [30, 45, 60], degrees=True),
        )
        self.registry.update("world", "base", new_transform)

        # Retrieved transform should be the new one
        retrieved = self.registry.get_transform("world", "base")
        np.testing.assert_array_almost_equal(
            retrieved.translation, new_transform.translation
        )
        np.testing.assert_array_almost_equal(
            retrieved.rotation.as_matrix(), new_transform.rotation.as_matrix()
        )

    def test_update_transform_with_numpy_array(self):
        """Test updating transform with numpy array."""
        self.registry.add_transform("world", "base", self.simple_transform)

        new_matrix = self.complex_transform.as_matrix()
        self.registry.update("world", "base", new_matrix)

        retrieved = self.registry.get_transform("world", "base")
        np.testing.assert_array_almost_equal(
            retrieved.translation, self.complex_transform.translation
        )

    def test_update_nonexistent_from_frame(self):
        """Test updating transform with nonexistent from_frame."""
        with self.assertRaises(InvalidTransformError):
            self.registry.update("nonexistent", "world", self.simple_transform)

    def test_update_nonexistent_to_frame(self):
        """Test updating transform with nonexistent to_frame."""
        with self.assertRaises(InvalidTransformError):
            self.registry.update("world", "nonexistent", self.simple_transform)

    def test_update_unconnected_frames(self):
        """Test updating transform between unconnected frames."""
        self.registry.add_transform("world", "base", self.simple_transform)

        # Add another frame not connected to base
        registry2 = Registry("other_world")
        registry2.add_transform("other_world", "other_base", self.complex_transform)

        # Try to update unconnected frames in original registry
        with self.assertRaises(InvalidTransformError):
            self.registry.update("world", "other_base", self.simple_transform)

    def test_update_invalid_matrix_shape(self):
        """Test updating with invalid matrix shape."""
        self.registry.add_transform("world", "base", self.simple_transform)

        invalid_matrix = np.eye(3)  # Wrong shape
        with self.assertRaises(ValueError):
            self.registry.update("world", "base", invalid_matrix)

    def test_update_invalid_type(self):
        """Test updating with invalid type."""
        self.registry.add_transform("world", "base", self.simple_transform)

        with self.assertRaises(ValueError):
            self.registry.update("world", "base", "invalid")

    def test_complex_frame_hierarchy(self):
        """Test complex frame hierarchy with multiple branches."""
        # Create hierarchy:
        #       world
        #      /     \
        #   base1   base2
        #    |       |
        # camera1  camera2

        base1_transform = Transform(
            np.array([1.0, 0.0, 0.0]),
            Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
        )
        base2_transform = Transform(
            np.array([0.0, 1.0, 0.0]),
            Rotation.from_euler("xyz", [0, 0, 90], degrees=True),
        )
        camera1_transform = Transform(
            np.array([0.0, 0.0, 1.0]),
            Rotation.from_euler("xyz", [90, 0, 0], degrees=True),
        )
        camera2_transform = Transform(
            np.array([0.0, 0.0, 1.0]),
            Rotation.from_euler("xyz", [0, 90, 0], degrees=True),
        )

        self.registry.add_transform("world", "base1", base1_transform)
        self.registry.add_transform("world", "base2", base2_transform)
        self.registry.add_transform("base1", "camera1", camera1_transform)
        self.registry.add_transform("base2", "camera2", camera2_transform)

        # Test transforms across branches
        transform_camera1_to_camera2 = self.registry.get_transform("camera1", "camera2")
        self.assertIsInstance(transform_camera1_to_camera2, Transform)

        # Test round trip
        forward = self.registry.get_transform("camera1", "camera2")
        backward = self.registry.get_transform("camera2", "camera1")

        # Forward @ backward should be approximately identity
        composed = forward.as_matrix() @ backward.as_matrix()
        np.testing.assert_array_almost_equal(composed, np.eye(4), decimal=10)

    def test_thread_safety_concurrent_reads(self):
        """Test thread safety with concurrent reads."""
        # Add some transforms
        self.registry.add_transform("world", "base", self.simple_transform)
        self.registry.add_transform("base", "camera", self.complex_transform)

        results = []
        errors = []

        def read_transform():
            try:
                for _ in range(100):
                    transform = self.registry.get_transform("world", "camera")
                    results.append(transform.translation.copy())
                    time.sleep(0.001)  # Small delay to encourage race conditions
            except Exception as e:
                errors.append(e)

        # Start multiple reader threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=read_transform)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should have no errors
        self.assertEqual(len(errors), 0)

        # All results should be the same
        expected = self.registry.get_transform("world", "camera").translation
        for result in results:
            np.testing.assert_array_almost_equal(result, expected)

    def test_thread_safety_concurrent_writes(self):
        """Test thread safety with concurrent writes."""
        self.registry.add_transform("world", "base", self.simple_transform)

        errors = []

        def update_transform(value):
            try:
                for i in range(10):
                    new_transform = Transform(
                        np.array([value, value, value]),
                        Rotation.from_euler("xyz", [0, 0, value], degrees=True),
                    )
                    self.registry.update("world", "base", new_transform)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        # Start multiple writer threads with different values
        threads = []
        for i in range(3):
            thread = threading.Thread(target=update_transform, args=(i + 1,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should have no errors (though final state is unpredictable)
        self.assertEqual(len(errors), 0)

        # Registry should still be in a valid state
        final_transform = self.registry.get_transform("world", "base")
        self.assertIsInstance(final_transform, Transform)

    def test_thread_safety_mixed_operations(self):
        """Test thread safety with mixed read/write operations."""
        self.registry.add_transform("world", "base", self.simple_transform)

        errors = []
        read_results = []

        def reader():
            try:
                for _ in range(50):
                    transform = self.registry.get_transform("world", "base")
                    read_results.append(transform.translation.copy())
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(25):
                    new_transform = Transform(
                        np.array([i, i, i]),
                        Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
                    )
                    self.registry.update("world", "base", new_transform)
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        # Start mixed threads
        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=reader))
        for _ in range(2):
            threads.append(threading.Thread(target=writer))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should have no errors
        self.assertEqual(len(errors), 0)

        # Should have some read results
        self.assertGreater(len(read_results), 0)


if __name__ == "__main__":
    unittest.main()
