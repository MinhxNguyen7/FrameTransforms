from enum import Enum

import numpy as np
from scipy.spatial.transform import Rotation

from frame_transforms import Registry, make_3d_transformation, InvaidTransformationError


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

    world_to_base_transform = make_3d_transformation(
        np.array([0, 1, 0]), Rotation.from_euler("xyz", [0, 0, 0], degrees=True)
    )
    registry.add_transform(Frame.WORLD, Frame.BASE, world_to_base_transform)

    base_to_camera_transform = make_3d_transformation(
        np.array([0, 0, 1]), Rotation.from_euler("xyz", [0, 90, 0], degrees=True)
    )
    registry.add_transform(Frame.BASE, Frame.CAMERA, base_to_camera_transform)

    return registry


def add_cycle_example():
    """
    Demonstrates adding a transformation that would create a cycle in the registry,
    raising an InvaidTransformationError.
    """
    registry = make_example_registry()

    # Attempt to add a transformation that creates a cycle
    try:
        registry.add_transform(Frame.CAMERA, Frame.WORLD, np.zeros(4))
    except InvaidTransformationError:
        print(
            "Caught invalid transformation because there is already a path between CAMERA and WORLD."
        )


def transitive_transformation_example():
    """
    Demonstrates getting a transformation from one frame to another through an intermediate frame.
    """
    registry = make_example_registry()

    expected = make_3d_transformation(
        np.array([0, 1, 1]), Rotation.from_euler("xyz", [0, 90, 0], degrees=True)
    )
    actual = registry.get_transform(Frame.WORLD, Frame.CAMERA)
    assert np.allclose(actual[:3], expected[:3]), "Position mismatch"
    assert np.allclose(actual[3:], expected[3:]), "Rotation mismatch"
    print("Transformation from WORLD to CAMERA is correct.")


def update_transformation_example():
    """
    Demonstrates updating an existing transformation in the registry,
    specifically, moving the base on which the camera sits.
    """
    registry = make_example_registry()

    # Update the transformation from WORLD to BASE
    new_transform = make_3d_transformation(
        np.array([0, 2, 0]), Rotation.from_euler("xyz", [0, 0, 0], degrees=True)
    )
    registry.update(Frame.WORLD, Frame.BASE, new_transform)

    # Attempt to add instead of update the transformation
    try:
        registry.add_transform(Frame.WORLD, Frame.BASE, new_transform)
    except InvaidTransformationError:
        print(
            "Caught invalid transformation because both frames already exist in the registry."
        )

    # Check the updated transformation
    expected = make_3d_transformation(
        np.array([0, 2, 1]), Rotation.from_euler("xyz", [0, 90, 0], degrees=True)
    )
    actual = registry.get_transform(Frame.WORLD, Frame.CAMERA)
    assert np.allclose(actual[:3], expected[:3]), "Position mismatch after update"
    assert np.allclose(actual[3:], expected[3:]), "Rotation mismatch after update"
    print("Transformation from WORLD to CAMERA updated correctly.")


if __name__ == "__main__":
    add_cycle_example()
    transitive_transformation_example()
    update_transformation_example()
