"""Tests for the utils module."""

import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from frame_transforms.utils import make_3d_transformation


class TestUtils(unittest.TestCase):
    """Test cases for the utils module."""

    def test_make_3d_transformation_identity(self):
        """Test creating identity transformation matrix."""
        translation = np.array([0.0, 0.0, 0.0])
        rotation = Rotation.from_euler("xyz", [0, 0, 0], degrees=True)

        matrix = make_3d_transformation(translation, rotation)

        expected = np.eye(4)
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_make_3d_transformation_translation_only(self):
        """Test creating transformation matrix with translation only."""
        translation = np.array([1.0, 2.0, 3.0])
        rotation = Rotation.from_euler("xyz", [0, 0, 0], degrees=True)

        matrix = make_3d_transformation(translation, rotation)

        expected = np.eye(4)
        expected[:3, 3] = [1.0, 2.0, 3.0]
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_make_3d_transformation_rotation_only(self):
        """Test creating transformation matrix with rotation only."""
        translation = np.array([0.0, 0.0, 0.0])
        rotation = Rotation.from_euler("xyz", [90, 0, 0], degrees=True)

        matrix = make_3d_transformation(translation, rotation)

        expected = np.eye(4)
        expected[:3, :3] = rotation.as_matrix()
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_make_3d_transformation_combined(self):
        """Test creating transformation matrix with both translation and rotation."""
        translation = np.array([1.5, -2.3, 4.7])
        rotation = Rotation.from_euler("xyz", [45, 30, 60], degrees=True)

        matrix = make_3d_transformation(translation, rotation)

        # Check shape
        self.assertEqual(matrix.shape, (4, 4))

        # Check homogeneous coordinates
        np.testing.assert_array_almost_equal(matrix[3, :], [0, 0, 0, 1])

        # Check translation part
        np.testing.assert_array_almost_equal(matrix[:3, 3], translation)

        # Check rotation part
        np.testing.assert_array_almost_equal(matrix[:3, :3], rotation.as_matrix())

    def test_make_3d_transformation_invalid_translation_shape(self):
        """Test creating transformation with invalid translation shape."""
        translation = np.array([1.0, 2.0])  # Wrong shape
        rotation = Rotation.from_euler("xyz", [0, 0, 0], degrees=True)

        with self.assertRaises(AssertionError):
            make_3d_transformation(translation, rotation)

    def test_make_3d_transformation_invalid_translation_type(self):
        """Test creating transformation with invalid translation type."""
        translation = [1.0, 2.0, 3.0]  # List instead of numpy array
        rotation = Rotation.from_euler("xyz", [0, 0, 0], degrees=True)

        with self.assertRaises(AttributeError):  # .shape attribute doesn't exist
            make_3d_transformation(translation, rotation)

    def test_make_3d_transformation_various_rotations(self):
        """Test creating transformations with various rotation representations."""
        translation = np.array([1.0, 1.0, 1.0])

        # Euler angles
        rotation_euler = Rotation.from_euler("xyz", [30, 45, 60], degrees=True)
        matrix_euler = make_3d_transformation(translation, rotation_euler)

        # Quaternion (same rotation)
        quat = rotation_euler.as_quat()
        rotation_quat = Rotation.from_quat(quat)
        matrix_quat = make_3d_transformation(translation, rotation_quat)

        # Should be the same
        np.testing.assert_array_almost_equal(matrix_euler, matrix_quat)

    def test_make_3d_transformation_matrix_properties(self):
        """Test that created matrix has proper transformation matrix properties."""
        translation = np.array([2.0, -1.5, 3.7])
        rotation = Rotation.from_euler("xyz", [15, 25, 35], degrees=True)

        matrix = make_3d_transformation(translation, rotation)

        # Should be 4x4
        self.assertEqual(matrix.shape, (4, 4))

        # Bottom row should be [0, 0, 0, 1]
        np.testing.assert_array_almost_equal(matrix[3, :], [0, 0, 0, 1])

        # Rotation part should be orthogonal
        rotation_part = matrix[:3, :3]
        should_be_identity = rotation_part @ rotation_part.T
        np.testing.assert_array_almost_equal(should_be_identity, np.eye(3))

        # Determinant of rotation part should be 1
        self.assertAlmostEqual(np.linalg.det(rotation_part), 1.0, places=10)

    def test_make_3d_transformation_edge_cases(self):
        """Test edge cases for transformation creation."""
        # Very small translation
        small_translation = np.array([1e-15, 1e-15, 1e-15])
        rotation = Rotation.from_euler("xyz", [0, 0, 0], degrees=True)

        matrix = make_3d_transformation(small_translation, rotation)
        self.assertEqual(matrix.shape, (4, 4))

        # Very large translation
        large_translation = np.array([1e6, -1e6, 1e6])
        matrix_large = make_3d_transformation(large_translation, rotation)
        np.testing.assert_array_almost_equal(matrix_large[:3, 3], large_translation)

    def test_make_3d_transformation_consistency_with_transform_class(self):
        """Test that utils function creates same matrix as Transform.as_matrix()."""
        from frame_transforms import Transform

        translation = np.array([1.2, -3.4, 5.6])
        rotation = Rotation.from_euler("xyz", [78, 123, 45], degrees=True)

        # Create using utils function
        matrix_utils = make_3d_transformation(translation, rotation)

        # Create using Transform class
        transform = Transform(translation, rotation)
        matrix_transform = transform.as_matrix()

        # Should be identical
        np.testing.assert_array_almost_equal(matrix_utils, matrix_transform)

    def test_make_3d_transformation_multiple_calls_consistent(self):
        """Test that multiple calls with same inputs produce same result."""
        translation = np.array([0.5, 1.5, 2.5])
        rotation = Rotation.from_euler("xyz", [10, 20, 30], degrees=True)

        matrix1 = make_3d_transformation(translation, rotation)
        matrix2 = make_3d_transformation(translation, rotation)

        np.testing.assert_array_equal(matrix1, matrix2)

    def test_make_3d_transformation_copy_safety(self):
        """Test that modifying input arrays doesn't affect the result."""
        translation = np.array([1.0, 2.0, 3.0])
        rotation = Rotation.from_euler("xyz", [0, 0, 0], degrees=True)

        matrix = make_3d_transformation(translation, rotation)
        original_matrix = matrix.copy()

        # Modify input translation
        translation[0] = 999.0

        # Create new matrix with modified input
        new_matrix = make_3d_transformation(translation, rotation)

        # Original matrix should be unchanged
        np.testing.assert_array_equal(matrix, original_matrix)

        # New matrix should be different
        self.assertFalse(np.array_equal(matrix[:3, 3], new_matrix[:3, 3]))


if __name__ == "__main__":
    unittest.main()
