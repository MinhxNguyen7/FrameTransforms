import numpy as np
from scipy.spatial.transform import Rotation


def make_3d_transformation(translation: np.ndarray, rotation: Rotation) -> np.ndarray:
    """
    Helper to create a 3D transformation matrix from translation and rotation.

    Args:
        translation: A 3-element array representing the translation.
        rotation: A scipy Rotation object representing the rotation.

    Returns:
        A 4x4 transformation matrix.
    """
    assert translation.shape == (3,), "Translation must be a 3-element array."
    
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation.as_matrix()
    transform_matrix[:3, 3] = translation
    return transform_matrix
