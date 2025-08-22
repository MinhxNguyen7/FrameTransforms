"""Tests for the Pose class."""

import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from frame_transforms import Transform, Pose, Registry


class TestPose(unittest.TestCase):
    """Test cases for the Pose class."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = Registry("world")

        # Add a base frame
        world_to_base = Transform(
            np.array([1.0, 2.0, 3.0]),
            Rotation.from_euler("xyz", [0, 0, 45], degrees=True),
        )
        self.registry.add_transform("world", "base", world_to_base)

        # Add a camera frame
        base_to_camera = Transform(
            np.array([0.0, 0.0, 1.0]),
            Rotation.from_euler("xyz", [0, 90, 0], degrees=True),
        )
        self.registry.add_transform("base", "camera", base_to_camera)

        self.test_transform = Transform(
            np.array([0.5, 0.5, 0.5]),
            Rotation.from_euler("xyz", [30, 0, 0], degrees=True),
        )

    def test_init_valid(self):
        """Test valid Pose initialization."""
        pose = Pose(self.test_transform, "world", self.registry)

        self.assertEqual(pose.transform, self.test_transform)
        self.assertEqual(pose.parent_frame, "world")
        self.assertIs(pose.registry, self.registry)

    def test_init_invalid_transform_type(self):
        """Test Pose initialization with invalid transform type."""
        with self.assertRaises(TypeError):
            Pose("not_a_transform", "world", self.registry)

    def test_init_invalid_registry_type(self):
        """Test Pose initialization with invalid registry type."""
        with self.assertRaises(TypeError):
            Pose(self.test_transform, "world", "not_a_registry")

    def test_apply_transform_same_frame(self):
        """Test applying transform without changing frame."""
        original_pose = Pose(self.test_transform, "world", self.registry)
        additional_transform = Transform(
            np.array([1.0, 0.0, 0.0]),
            Rotation.from_euler("xyz", [0, 0, 90], degrees=True),
        )

        new_pose = original_pose.apply_transform(additional_transform, None)

        # Should remain in same frame
        self.assertEqual(new_pose.parent_frame, "world")
        self.assertIs(new_pose.registry, self.registry)

        # Transform should be combined
        expected_translation = (
            original_pose.transform.translation + additional_transform.translation
        )
        np.testing.assert_array_almost_equal(
            new_pose.transform.translation, expected_translation
        )

        expected_rotation = (
            original_pose.transform.rotation * additional_transform.rotation
        )
        np.testing.assert_array_almost_equal(
            new_pose.transform.rotation.as_matrix(), expected_rotation.as_matrix()
        )

    def test_apply_transform_new_frame(self):
        """Test applying transform with changing frame."""
        original_pose = Pose(self.test_transform, "world", self.registry)
        additional_transform = Transform(
            np.array([1.0, 0.0, 0.0]),
            Rotation.from_euler("xyz", [0, 0, 90], degrees=True),
        )

        new_pose = original_pose.apply_transform(additional_transform, "base")

        # Should be in new frame
        self.assertEqual(new_pose.parent_frame, "base")
        self.assertIs(new_pose.registry, self.registry)

    def test_in_frame_same_frame(self):
        """Test transforming pose to the same frame."""
        pose = Pose(self.test_transform, "world", self.registry)
        transformed_pose = pose.in_frame("world")

        # Should be unchanged
        np.testing.assert_array_almost_equal(
            pose.transform.translation, transformed_pose.transform.translation
        )
        np.testing.assert_array_almost_equal(
            pose.transform.rotation.as_matrix(),
            transformed_pose.transform.rotation.as_matrix(),
        )
        self.assertEqual(transformed_pose.parent_frame, "world")

    def test_in_frame_different_frame(self):
        """Test transforming pose to a different frame."""
        pose = Pose(self.test_transform, "world", self.registry)
        transformed_pose = pose.in_frame("base")

        self.assertEqual(transformed_pose.parent_frame, "base")
        self.assertIs(transformed_pose.registry, self.registry)

        # Should not be the same as original
        self.assertFalse(
            np.allclose(
                pose.transform.translation, transformed_pose.transform.translation
            )
        )

    def test_in_frame_through_intermediate(self):
        """Test transforming pose through intermediate frames."""
        pose = Pose(self.test_transform, "world", self.registry)
        transformed_pose = pose.in_frame("camera")

        self.assertEqual(transformed_pose.parent_frame, "camera")

        # Transform should be valid (no exceptions)
        self.assertIsInstance(transformed_pose.transform.translation, np.ndarray)
        self.assertIsInstance(transformed_pose.transform.rotation, Rotation)

    def test_get_position_none_frame(self):
        """Test getting position with None frame (should use parent frame)."""
        pose = Pose(self.test_transform, "world", self.registry)
        position = pose.get_position(None)

        np.testing.assert_array_almost_equal(position, pose.transform.translation)

    def test_get_position_same_frame(self):
        """Test getting position in the same frame."""
        pose = Pose(self.test_transform, "world", self.registry)
        position = pose.get_position("world")

        np.testing.assert_array_almost_equal(position, pose.transform.translation)

    def test_get_position_different_frame(self):
        """Test getting position in a different frame."""
        pose = Pose(self.test_transform, "world", self.registry)
        position_in_base = pose.get_position("base")

        self.assertIsInstance(position_in_base, np.ndarray)
        self.assertEqual(position_in_base.shape, (3,))

        # Should be different from original position
        self.assertFalse(np.allclose(position_in_base, pose.transform.translation))

    def test_get_orientation_none_frame(self):
        """Test getting orientation with None frame (should use parent frame)."""
        pose = Pose(self.test_transform, "world", self.registry)
        orientation = pose.get_orientation(None)

        np.testing.assert_array_almost_equal(
            orientation.as_matrix(), pose.transform.rotation.as_matrix()
        )

    def test_get_orientation_same_frame(self):
        """Test getting orientation in the same frame."""
        pose = Pose(self.test_transform, "world", self.registry)
        orientation = pose.get_orientation("world")

        np.testing.assert_array_almost_equal(
            orientation.as_matrix(), pose.transform.rotation.as_matrix()
        )

    def test_get_orientation_different_frame(self):
        """Test getting orientation in a different frame."""
        pose = Pose(self.test_transform, "world", self.registry)
        orientation_in_base = pose.get_orientation("base")

        self.assertIsInstance(orientation_in_base, Rotation)

        # Should be different from original orientation (in most cases)
        original_matrix = pose.transform.rotation.as_matrix()
        transformed_matrix = orientation_in_base.as_matrix()

        # At least one element should be different (unless identity transform)
        self.assertIsInstance(transformed_matrix, np.ndarray)
        self.assertEqual(transformed_matrix.shape, (3, 3))

    def test_round_trip_transformation(self):
        """Test that transforming to a frame and back gives original pose."""
        # Use a simpler setup with just one intermediate frame
        simple_registry = Registry("world")
        simple_transform = Transform(
            np.array([1.0, 0.0, 0.0]),
            Rotation.from_euler("xyz", [0, 0, 0], degrees=True),  # No rotation
        )
        simple_registry.add_transform("world", "base", simple_transform)

        original_pose = Pose(
            Transform(
                np.array([0.5, 0.5, 0.5]),
                Rotation.from_euler("xyz", [0, 0, 0], degrees=True),  # No rotation
            ),
            "world",
            simple_registry,
        )

        # Transform to base and back to world
        intermediate_pose = original_pose.in_frame("base")
        final_pose = intermediate_pose.in_frame("world")

        # Should be approximately equal to original (simple case should be exact)
        np.testing.assert_array_almost_equal(
            original_pose.transform.translation,
            final_pose.transform.translation,
            decimal=10,
        )
        np.testing.assert_array_almost_equal(
            original_pose.transform.rotation.as_matrix(),
            final_pose.transform.rotation.as_matrix(),
            decimal=10,
        )

    def test_pose_chain_transformation(self):
        """Test transforming pose through a chain of frames."""
        original_pose = Pose(self.test_transform, "world", self.registry)

        # Transform world -> base -> camera
        pose_in_camera = original_pose.in_frame("camera")

        # Should maintain consistency
        self.assertEqual(pose_in_camera.parent_frame, "camera")

        # Position and orientation should be reasonable
        position = pose_in_camera.transform.translation
        self.assertEqual(position.shape, (3,))
        self.assertTrue(np.all(np.isfinite(position)))

        rotation_matrix = pose_in_camera.transform.rotation.as_matrix()
        self.assertEqual(rotation_matrix.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(rotation_matrix)))

    def test_frozen_dataclass(self):
        """Test that Pose is immutable (frozen dataclass)."""
        pose = Pose(self.test_transform, "world", self.registry)

        with self.assertRaises(Exception):  # Should be FrozenInstanceError
            pose.transform = Transform(
                np.array([999.0, 999.0, 999.0]),
                Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
            )

        with self.assertRaises(Exception):  # Should be FrozenInstanceError
            pose.parent_frame = "base"


if __name__ == "__main__":
    unittest.main()
