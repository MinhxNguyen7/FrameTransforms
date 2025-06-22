from enum import Enum

import numpy as np
from scipy.spatial.transform import Rotation

from frame_transforms import Registry, make_3d_transformation, InvaidTransformationError


class Frame(Enum):
    WORLD = "world"
    BASE = "base"
    CAMERA = "camera"


def make_example_registry():
    registry = Registry(Frame.WORLD)

    # Add transformations between frames

    world_to_base_transform = make_3d_transformation(
        np.array([1, 0, 0]), Rotation.from_euler("xyz", [0, 0, 0], degrees=True)
    )
    registry.add_transform(Frame.WORLD, Frame.BASE, world_to_base_transform)

    base_to_camera_transform = make_3d_transformation(
        np.array([0, 1, 0]), Rotation.from_euler("xyz", [0, 90, 0], degrees=True)
    )
    registry.add_transform(Frame.BASE, Frame.CAMERA, base_to_camera_transform)

    return registry


if __name__ == "__main__":
    registry = make_example_registry()

    # Attempt to add a transformation that creates a cycle
    try:
        registry.add_transform(Frame.CAMERA, Frame.WORLD, np.zeros(4))
    except InvaidTransformationError:
        print(
            "Caught invalid transformation because there is already a path between CAMERA and WORLD."
        )

    # Get a transitive transformation from world to camera
    expected = make_3d_transformation(
        np.array([1, 1, 0]), Rotation.from_euler("xyz", [0, 90, 0], degrees=True)
    )
    actual = registry.get_transform(Frame.WORLD, Frame.CAMERA)
    assert np.allclose(actual[:3], expected[:3]), "Position mismatch"
    assert np.allclose(actual[3:], expected[3:]), "Rotation mismatch"
    print("Transformation from WORLD to CAMERA is correct.")
