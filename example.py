from enum import Enum

import numpy as np
from scipy.spatial.transform import Rotation

from frame_transforms import (
    Pose,
    Transform,
    Registry,
    InvalidTransformError,
)


class Frame(Enum):
    WORLD = "world"
    BASE = "base"
    CAMERA = "camera"


def make_example_registry():
    """
    Creates an example XYZ registry of a robot at (0, 1, 0) with a camera one unit
    above the base, rotated 90 degrees around the Y-axis.
    """
    registry = Registry(Frame.WORLD)

    # Add transformations between frames

    world_to_base = Transform(
        np.array([0, 1, 0]), Rotation.from_euler("xyz", [0, 0, 0], degrees=True)
    )
    registry.add_transform(Frame.WORLD, Frame.BASE, world_to_base)

    base_to_camera = Transform(
        np.array([0, 0, 1]), Rotation.from_euler("xyz", [0, 90, 0], degrees=True)
    )
    registry.add_transform(Frame.BASE, Frame.CAMERA, base_to_camera)

    return registry


def add_cycle_example():
    """
    Demonstrates adding a transform that would create a cycle in the registry,
    raising an InvalidTransformError.
    """
    registry = make_example_registry()

    # Attempt to add a transform that creates a cycle
    try:
        registry.add_transform(Frame.CAMERA, Frame.WORLD, np.zeros(4))
    except InvalidTransformError:
        print(
            "Caught invalid transform because there is already a path between CAMERA and WORLD."
        )


def transitive_transform_example():
    """
    Demonstrates getting a transform from one frame to another through an intermediate frame.
    """
    registry = make_example_registry()

    expected = Transform(
        np.array([0, 1, 1]), Rotation.from_euler("xyz", [0, 90, 0], degrees=True)
    )
    actual = registry.get_transform(Frame.WORLD, Frame.CAMERA)
    assert np.allclose(actual.translation, expected.translation), "Position mismatch"
    assert np.allclose(
        actual.rotation.as_matrix(), expected.rotation.as_matrix()
    ), "Rotation mismatch"
    print("Transform from WORLD to CAMERA is correct.")


def update_transform_example():
    """
    Demonstrates updating an existing transform in the registry,
    specifically, moving the base on which the camera sits.
    """
    registry = make_example_registry()

    # Update the transform from WORLD to BASE
    new_transform = Transform(
        np.array([0, 2, 0]), Rotation.from_euler("xyz", [0, 0, 0], degrees=True)
    )
    registry.update(Frame.WORLD, Frame.BASE, new_transform)

    # Attempt to add instead of update the transform
    try:
        registry.add_transform(Frame.WORLD, Frame.BASE, new_transform)
    except InvalidTransformError:
        print(
            "Caught invalid transform because both frames already exist in the registry."
        )

    # Check the updated transform
    expected = Transform(
        np.array([0, 2, 1]), Rotation.from_euler("xyz", [0, 90, 0], degrees=True)
    )
    actual = registry.get_transform(Frame.WORLD, Frame.CAMERA)
    assert np.allclose(
        actual.translation, expected.translation
    ), "Position mismatch after update"
    assert np.allclose(
        actual.rotation.as_matrix(), expected.rotation.as_matrix()
    ), "Rotation mismatch after update"
    print("Transform from WORLD to CAMERA updated correctly.")


def pose_example():
    """
    Demonstrates creating and using a Pose object.
    """
    registry = make_example_registry()

    object_pose = Pose(
        Transform(
            np.array([1, 0, 0]),
            Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
        ),
        parent_frame=Frame.CAMERA,
        registry=registry,
    )

    position_in_world = object_pose.get_position(Frame.WORLD)
    print("Object position in WORLD frame:", position_in_world)

    orientation_in_world = object_pose.get_orientation(Frame.WORLD)
    print("Object orientation in WORLD frame:", orientation_in_world.as_matrix())


if __name__ == "__main__":
    add_cycle_example()
    transitive_transform_example()
    update_transform_example()
    pose_example()
