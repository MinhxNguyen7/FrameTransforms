"""Tests that reveal bugs in the current Transform implementation.

These tests demonstrate that the current Transform composition logic is incorrect.
The @ operator for Transform objects doesn't properly compose 3D transformations.
"""

import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from frame_transforms import Transform


class TestTransformCompositionBugs(unittest.TestCase):
    """Test cases that reveal bugs in the current implementation."""

    def test_transform_composition_simple_case(self):
        """Test that reveals the translation composition bug."""
        # First transform: translate [1,0,0] then rotate 90° around Z
        t1 = Transform(
            np.array([1.0, 0.0, 0.0]),
            Rotation.from_euler("xyz", [0, 0, 90], degrees=True),
        )

        # Second transform: translate [1,0,0], no rotation
        t2 = Transform(
            np.array([1.0, 0.0, 0.0]),
            Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
        )

        # Current (incorrect) implementation
        current_result = t1 @ t2

        # Correct result using matrix composition
        correct_matrix = t1.as_matrix() @ t2.as_matrix()
        expected_translation = correct_matrix[:3, 3]
        expected_rotation = Rotation.from_matrix(correct_matrix[:3, :3])

        print(f"Current result: {current_result.translation}")
        print(f"Expected result: {expected_translation}")

        # This test should now PASS with the fixed implementation
        np.testing.assert_array_almost_equal(
            current_result.translation,
            expected_translation,
            err_msg="Transform composition should now be correct",
        )

    def test_vector_transformation_bug(self):
        """Test that reveals the vector transformation bug."""
        transform = Transform(
            np.array([1.0, 2.0, 3.0]),
            Rotation.from_euler("xyz", [0, 0, 90], degrees=True),
        )

        vector = np.array([1.0, 0.0, 0.0])

        # Current (incorrect) implementation
        current_result = transform @ vector

        # Correct transformation: rotate first, then translate
        correct_result = transform.rotation.apply(vector) + transform.translation

        print(f"Current vector result: {current_result}")
        print(f"Expected vector result: {correct_result}")

        # This test should now PASS with the fixed implementation
        np.testing.assert_array_almost_equal(
            current_result,
            correct_result,
            err_msg="Vector transformation should now be correct",
        )

    def test_matrix_vs_transform_composition(self):
        """Test showing matrix composition gives different results than Transform @."""
        t1 = Transform(
            np.array([2.0, 1.0, 0.0]),
            Rotation.from_euler("xyz", [0, 0, 45], degrees=True),
        )

        t2 = Transform(
            np.array([0.0, 0.0, 1.0]),
            Rotation.from_euler("xyz", [0, 90, 0], degrees=True),
        )

        # Transform composition (current implementation)
        transform_result = t1 @ t2

        # Matrix composition (correct)
        matrix_result_matrix = t1.as_matrix() @ t2.as_matrix()
        matrix_result = Transform(
            matrix_result_matrix[:3, 3],
            Rotation.from_matrix(matrix_result_matrix[:3, :3]),
        )

        print("Transform @ result:")
        print(f"  Translation: {transform_result.translation}")
        print(f"  Rotation: {transform_result.rotation.as_euler('xyz', degrees=True)}")

        print("Matrix @ result:")
        print(f"  Translation: {matrix_result.translation}")
        print(f"  Rotation: {matrix_result.rotation.as_euler('xyz', degrees=True)}")

        # These should now be the same with the fixed implementation
        translation_same = np.allclose(
            transform_result.translation, matrix_result.translation
        )
        rotation_same = np.allclose(
            transform_result.rotation.as_matrix(), matrix_result.rotation.as_matrix()
        )

        self.assertTrue(
            translation_same and rotation_same,
            "Transform @ and matrix @ should give same result with fixed implementation",
        )

    def test_identity_composition_bug(self):
        """Test showing that even identity composition can be wrong."""
        # Non-identity transform
        transform = Transform(
            np.array([1.0, 1.0, 0.0]),
            Rotation.from_euler("xyz", [0, 0, 45], degrees=True),
        )

        # Identity transform
        identity = Transform(
            np.array([0.0, 0.0, 0.0]),
            Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
        )

        # These should give the same result as the original transform
        result1 = transform @ identity
        result2 = identity @ transform

        print("Original transform translation:", transform.translation)
        print("transform @ identity:", result1.translation)
        print("identity @ transform:", result2.translation)

        # Both should now pass with the fixed implementation
        same1 = np.allclose(result1.translation, transform.translation)
        same2 = np.allclose(result2.translation, transform.translation)

        if not same1:
            print("BUG: transform @ identity != transform")
        if not same2:
            print("BUG: identity @ transform != transform")

        # Both should now work with the fixed implementation
        self.assertTrue(same1, "transform @ identity should equal transform")
        self.assertTrue(same2, "identity @ transform should equal transform")


if __name__ == "__main__":
    unittest.main()
