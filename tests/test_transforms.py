"""Tests for the Transform class."""

import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from frame_transforms import Transform, Pose, Registry, InvalidTransformError


class TestTransform(unittest.TestCase):
    """Test cases for the Transform class."""

    def setUp(self):
        """Set up test fixtures."""
        self.identity_transform = Transform(
            np.array([0.0, 0.0, 0.0]),
            Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
        )

        self.translation_transform = Transform(
            np.array([1.0, 2.0, 3.0]),
            Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
        )

        self.rotation_transform = Transform(
            np.array([0.0, 0.0, 0.0]),
            Rotation.from_euler("xyz", [90, 0, 0], degrees=True),
        )

        self.combined_transform = Transform(
            np.array([1.0, 2.0, 3.0]),
            Rotation.from_euler("xyz", [90, 45, 30], degrees=True),
        )

    def test_init_valid(self):
        """Test valid Transform initialization."""
        transform = Transform(
            np.array([1.0, 2.0, 3.0]),
            Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
        )
        np.testing.assert_array_equal(transform.translation, [1.0, 2.0, 3.0])
        self.assertIsInstance(transform.rotation, Rotation)

    def test_init_invalid_translation_shape(self):
        """Test Transform initialization with invalid translation shape."""
        with self.assertRaises(ValueError):
            Transform(
                np.array([1.0, 2.0]),  # Wrong shape
                Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
            )

    def test_init_invalid_rotation_type(self):
        """Test Transform initialization with invalid rotation type."""
        with self.assertRaises(TypeError):
            Transform(np.array([1.0, 2.0, 3.0]), "not_a_rotation")  # Wrong type

    def test_translation_property_immutable(self):
        """Test that translation property returns a copy."""
        transform = self.translation_transform
        translation = transform.translation
        translation[0] = 999.0  # Modify the returned array
        # Original should be unchanged
        np.testing.assert_array_equal(transform.translation, [1.0, 2.0, 3.0])

    def test_rotation_property(self):
        """Test rotation property access."""
        transform = self.rotation_transform
        rotation = transform.rotation
        self.assertIsInstance(rotation, Rotation)

    def test_as_matrix_identity(self):
        """Test conversion to matrix for identity transform."""
        matrix = self.identity_transform.as_matrix()
        expected = np.eye(4)
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_as_matrix_translation_only(self):
        """Test conversion to matrix for translation-only transform."""
        matrix = self.translation_transform.as_matrix()
        expected = np.eye(4)
        expected[:3, 3] = [1.0, 2.0, 3.0]
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_as_matrix_rotation_only(self):
        """Test conversion to matrix for rotation-only transform."""
        matrix = self.rotation_transform.as_matrix()
        expected = np.eye(4)
        expected[:3, :3] = Rotation.from_euler(
            "xyz", [90, 0, 0], degrees=True
        ).as_matrix()
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_matmul_with_transform(self):
        """Test matrix multiplication with another Transform."""
        result = self.translation_transform @ self.rotation_transform

        self.assertIsInstance(result, Transform)
        # Translation should be combined
        expected_translation = (
            self.translation_transform.translation + self.rotation_transform.translation
        )
        np.testing.assert_array_almost_equal(result.translation, expected_translation)

        # Rotation should be composed
        expected_rotation = (
            self.translation_transform.rotation * self.rotation_transform.rotation
        )
        np.testing.assert_array_almost_equal(
            result.rotation.as_matrix(), expected_rotation.as_matrix()
        )

    def test_matmul_with_3d_vector(self):
        """Test matrix multiplication with a 3D vector."""
        vector = np.array([1.0, 0.0, 0.0])
        result = self.combined_transform @ vector

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3,))

        # Should apply rotation then translation
        expected = (
            self.combined_transform.rotation.apply(vector)
            + self.combined_transform.translation
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_matmul_with_4x4_matrix(self):
        """Test matrix multiplication with a 4x4 matrix."""
        matrix = np.eye(4)
        matrix[:3, 3] = [5.0, 6.0, 7.0]  # Add some translation

        result = self.combined_transform @ matrix

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4, 4))

        # Should be equivalent to matrix multiplication
        expected = self.combined_transform.as_matrix() @ matrix
        np.testing.assert_array_almost_equal(result, expected)

    def test_matmul_with_pose(self):
        """Test matrix multiplication with a Pose."""
        registry = Registry("world")
        pose = Pose(self.translation_transform, "world", registry)

        result = self.rotation_transform @ pose

        self.assertIsInstance(result, Pose)
        self.assertEqual(result.parent_frame, "world")
        self.assertIs(result.registry, registry)

    def test_matmul_invalid_shape(self):
        """Test matrix multiplication with invalid array shape."""
        invalid_array = np.array([1.0, 2.0])  # Wrong shape

        with self.assertRaises(ValueError):
            _ = self.translation_transform @ invalid_array

    def test_matmul_invalid_type(self):
        """Test matrix multiplication with invalid type."""
        with self.assertRaises(TypeError):
            _ = self.translation_transform @ "invalid"

    def test_transform_composition_associative(self):
        """Test that transform composition is associative."""
        t1 = self.translation_transform
        t2 = self.rotation_transform
        t3 = self.combined_transform

        # (t1 @ t2) @ t3 should equal t1 @ (t2 @ t3)
        left_assoc = (t1 @ t2) @ t3
        right_assoc = t1 @ (t2 @ t3)

        np.testing.assert_array_almost_equal(
            left_assoc.translation, right_assoc.translation
        )
        np.testing.assert_array_almost_equal(
            left_assoc.rotation.as_matrix(), right_assoc.rotation.as_matrix()
        )

    def test_frozen_dataclass(self):
        """Test that Transform is immutable (frozen dataclass)."""
        transform = self.translation_transform

        with self.assertRaises(Exception):  # Should be FrozenInstanceError
            transform._translation = np.array([999.0, 999.0, 999.0])

        with self.assertRaises(Exception):  # Should be FrozenInstanceError
            transform._rotation = Rotation.from_euler("xyz", [180, 0, 0], degrees=True)


if __name__ == "__main__":
    unittest.main()
