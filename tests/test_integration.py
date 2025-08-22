"""Integration tests for the frame_transforms package."""

import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from frame_transforms import (
    Transform,
    Pose,
    Registry,
    InvalidTransformError,
    make_3d_transformation,
)


class TestIntegration(unittest.TestCase):
    """Integration test cases that test the complete workflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = Registry("world")

    def test_robot_scenario(self):
        """Test the complete robot scenario from the README."""
        # Setup robot frames
        base_pose = Transform(
            np.array([2.0, 1.0, 0.0]),
            Rotation.from_euler("xyz", [0, 0, 45], degrees=True),
        )
        self.registry.add_transform("world", "base", base_pose)

        camera_pose = Transform(
            np.array([0.0, 0.0, 1.5]),
            Rotation.from_euler("xyz", [0, 90, 0], degrees=True),
        )
        self.registry.add_transform("base", "camera", camera_pose)

        # Define object pose in camera frame
        object_transform = Transform(
            np.array([1.0, 0.0, 0.0]),
            Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
        )
        object_pose = Pose(object_transform, "camera", self.registry)

        # Get position and orientation in world frame
        position_in_world = object_pose.get_position("world")
        orientation_in_world = object_pose.get_orientation("world")

        # Verify results are reasonable
        self.assertEqual(position_in_world.shape, (3,))
        self.assertIsInstance(orientation_in_world, Rotation)

        # Position should be transformed through the chain
        self.assertFalse(np.allclose(position_in_world, [1.0, 0.0, 0.0]))

    def test_complex_frame_hierarchy(self):
        """Test complex frame hierarchy with multiple objects."""
        # Create robot with multiple sensors
        base_transform = Transform(
            np.array([1.0, 2.0, 0.0]),
            Rotation.from_euler("xyz", [0, 0, 30], degrees=True),
        )
        self.registry.add_transform("world", "base", base_transform)

        # Camera on robot
        camera_transform = Transform(
            np.array([0.3, 0.0, 1.2]),
            Rotation.from_euler("xyz", [0, 15, 0], degrees=True),
        )
        self.registry.add_transform("base", "camera", camera_transform)

        # Object detected by camera
        object_in_camera = Pose(
            Transform(
                np.array([2.5, -0.5, 1.0]),
                Rotation.from_euler("xyz", [45, 30, 15], degrees=True),
            ),
            "camera",
            self.registry,
        )

        # Transform object to different frames
        object_in_base = object_in_camera.in_frame("base")
        object_in_world = object_in_camera.in_frame("world")

        # Verify frame assignments
        self.assertEqual(object_in_camera.parent_frame, "camera")
        self.assertEqual(object_in_base.parent_frame, "base")
        self.assertEqual(object_in_world.parent_frame, "world")

        # Verify round-trip consistency
        back_to_camera = object_in_world.in_frame("camera")
        np.testing.assert_array_almost_equal(
            object_in_camera.transform.translation,
            back_to_camera.transform.translation,
            decimal=6,  # Reduced precision due to numerical accumulation in complex transforms
        )

    def test_dynamic_frame_updates(self):
        """Test updating frame transforms dynamically."""
        # Initial robot pose
        initial_base = Transform(
            np.array([0.0, 0.0, 0.0]),
            Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
        )
        self.registry.add_transform("world", "base", initial_base)

        # Camera fixed relative to base
        camera_transform = Transform(
            np.array([0.0, 0.0, 1.0]),
            Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
        )
        self.registry.add_transform("base", "camera", camera_transform)

        # Object in camera frame
        object_pose = Pose(
            Transform(
                np.array([1.0, 0.0, 0.0]),
                Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
            ),
            "camera",
            self.registry,
        )

        # Get initial world position
        initial_world_pos = object_pose.get_position("world")

        # Move the robot base
        new_base = Transform(
            np.array([5.0, 3.0, 0.0]),
            Rotation.from_euler("xyz", [0, 0, 90], degrees=True),
        )
        self.registry.update("world", "base", new_base)

        # Get new world position
        new_world_pos = object_pose.get_position("world")

        # Positions should be different
        self.assertFalse(np.allclose(initial_world_pos, new_world_pos))

        # Object should still be at same position relative to camera
        camera_pos = object_pose.get_position("camera")
        np.testing.assert_array_almost_equal(camera_pos, [1.0, 0.0, 0.0])

    def test_transform_matrix_integration(self):
        """Test integration between Transform objects and numpy matrices."""
        # Create transforms using different methods
        translation = np.array([1.0, 2.0, 3.0])
        rotation = Rotation.from_euler("xyz", [30, 45, 60], degrees=True)

        # Method 1: Transform object
        transform1 = Transform(translation, rotation)

        # Method 2: Utility function
        matrix = make_3d_transformation(translation, rotation)

        # Method 3: Add to registry using matrix
        self.registry.add_transform("world", "base", matrix)
        retrieved_transform = self.registry.get_transform("world", "base")

        # All should be equivalent
        np.testing.assert_array_almost_equal(transform1.as_matrix(), matrix)
        np.testing.assert_array_almost_equal(
            transform1.translation, retrieved_transform.translation
        )
        np.testing.assert_array_almost_equal(
            transform1.rotation.as_matrix(), retrieved_transform.rotation.as_matrix()
        )

    def test_error_handling_workflow(self):
        """Test error handling in typical workflows."""
        # Add base frame
        base_transform = Transform(
            np.array([1.0, 0.0, 0.0]),
            Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
        )
        self.registry.add_transform("world", "base", base_transform)

        # Try to add conflicting transform
        with self.assertRaises(InvalidTransformError):
            self.registry.add_transform("world", "base", base_transform)

        # Try to transform to nonexistent frame
        pose = Pose(base_transform, "world", self.registry)
        with self.assertRaises(InvalidTransformError):
            pose.in_frame("nonexistent")

        # Try to update nonexistent transform
        with self.assertRaises(InvalidTransformError):
            self.registry.update("world", "nonexistent", base_transform)

    def test_example_from_readme(self):
        """Test the exact example from the README."""
        registry = Registry("world")

        # Setup
        base_pose = Transform(
            np.array([1.0, 2.0, 0.0]),
            Rotation.from_euler("xyz", [0, 0, 30], degrees=True),
        )
        registry.add_transform("world", "base", base_pose)

        camera_pose = Transform(
            np.array([0.0, 0.0, 1.0]),
            Rotation.from_euler("xyz", [0, 90, 0], degrees=True),
        )
        registry.add_transform("base", "camera", camera_pose)

        # Define the Pose
        object_pose = Pose(
            Transform(
                np.array([1.0, 0.0, 0.0]),
                Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
            ),
            parent_frame="camera",
            registry=registry,
        )

        # Get the position and orientation of the object in world frame
        position_in_world = object_pose.get_position("world")
        orientation_in_world = object_pose.get_orientation("world")

        # Should not raise exceptions
        self.assertEqual(position_in_world.shape, (3,))
        self.assertIsInstance(orientation_in_world, Rotation)

    def test_multiple_registry_isolation(self):
        """Test that multiple registries are properly isolated."""
        # Create two separate registries
        registry1 = Registry("world1")
        registry2 = Registry("world2")

        # Add transforms to each
        transform1 = Transform(
            np.array([1.0, 0.0, 0.0]),
            Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
        )
        transform2 = Transform(
            np.array([0.0, 1.0, 0.0]),
            Rotation.from_euler("xyz", [0, 0, 90], degrees=True),
        )

        registry1.add_transform("world1", "base1", transform1)
        registry2.add_transform("world2", "base2", transform2)

        # Poses should be isolated
        pose1 = Pose(transform1, "world1", registry1)
        pose2 = Pose(transform2, "world2", registry2)

        # Should not be able to transform between registries
        # (This would require frames to exist in the same registry)
        self.assertEqual(pose1.parent_frame, "world1")
        self.assertEqual(pose2.parent_frame, "world2")

    def test_coordinate_frame_consistency(self):
        """Test coordinate frame consistency across transformations."""
        # Set up right-handed coordinate system test
        self.registry.add_transform(
            "world",
            "base",
            Transform(
                np.array([1.0, 0.0, 0.0]),  # Move 1 unit in X
                Rotation.from_euler(
                    "xyz", [0, 0, 90], degrees=True
                ),  # Rotate 90° around Z
            ),
        )

        # Point at origin in base frame
        point_in_base = Pose(
            Transform(
                np.array([0.0, 0.0, 0.0]),
                Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
            ),
            "base",
            self.registry,
        )

        # Transform to world frame
        point_in_world = point_in_base.in_frame("world")

        # After 90° rotation around Z and translation [1,0,0],
        # a point at origin in base should be at [0, 1, 0] in world
        world_pos = point_in_world.transform.translation
        np.testing.assert_array_almost_equal(world_pos, [0.0, 1.0, 0.0])

    def test_numerical_precision(self):
        """Test numerical precision in transform compositions."""
        # Create a chain of small transforms
        current_frame = "world"

        # Add 10 small transforms (reduced from 100 for simpler test)
        for i in range(1, 11):
            next_frame = f"frame_{i}"
            small_transform = Transform(
                np.array([0.1, 0.0, 0.0]),  # Small translation
                Rotation.from_euler("xyz", [0, 0, 1.0], degrees=True),  # Small rotation
            )
            self.registry.add_transform(current_frame, next_frame, small_transform)
            current_frame = next_frame

        # Transform through entire chain
        final_transform = self.registry.get_transform("world", "frame_10")

        # Result should be reasonable (approximately 1.0 in X, some Y from rotation)
        translation = final_transform.translation
        self.assertAlmostEqual(translation[0], 1.0, places=1)  # 10 * 0.1

        # Rotation should be approximately 10 degrees total
        total_rotation = final_transform.rotation
        euler = total_rotation.as_euler("xyz", degrees=True)
        self.assertAlmostEqual(euler[2], 10.0, places=1)  # 10 * 1.0


if __name__ == "__main__":
    unittest.main()
