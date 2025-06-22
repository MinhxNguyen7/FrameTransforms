from typing import Generic, Hashable, TypeVar

import numpy as np


# Key to identify coordinate frames in the registry.
FrameID_T = TypeVar("FrameID_T", bound=Hashable)


class InvaidTransformationError(Exception):
    pass


class Registry(Generic[FrameID_T]):
    """
    Registry of coordinate frames and corresponding transforms.

    Automatically computes transitive tramsforms between frames if possible by
    maintaining a directed acyclic graph (DAG) of relationships.

    Made for use with 4x4 3D transformation matrices.

    TODO: Support concurrency
    """

    def __init__(self, world_frame: FrameID_T):
        """
        Args:
            world_frame: Identifier of the frame that serves as the root of the registry.
        """
        self._adjacencies: dict[FrameID_T, dict[FrameID_T, np.ndarray]] = {
            world_frame: {}
        }

        self._parents: dict[FrameID_T, FrameID_T | None] = {world_frame: None}

        # Paths between frames for quickly retrieving the path between two frames.
        self._paths: dict[FrameID_T, dict[FrameID_T, list[FrameID_T]]] = {
            world_frame: {world_frame: [world_frame]}
        }

    def get_transform(self, from_frame: FrameID_T, to_frame: FrameID_T) -> np.ndarray:
        """
        Gets the transformation matrix from one frame to another.

        Args:
            from_frame: The source frame.
            to_frame: The destination frame.

        Returns:
            The transformation matrix from `from_frame` to `to_frame`.
        """
        path = self._get_path(from_frame, to_frame)

        transformation = np.eye(4)
        for i in range(len(path) - 1):
            current_frame = path[i]
            next_frame = path[i + 1]
            transformation = (
                transformation @ self._adjacencies[current_frame][next_frame]
            )

        return transformation

    def add_transform(
        self, from_frame: FrameID_T, to_frame: FrameID_T, transform: np.ndarray
    ):
        """
        Adds a transformation from one frame to another.

        Exactly *one* of the frames must exist in the registry.

        Args:
            from_frame: The source frame.
            to_frame: The destination frame.
            transform: The transformation matrix from `from_frame` to `to_frame`.
        """
        if from_frame in self._adjacencies and to_frame in self._adjacencies:
            raise InvaidTransformationError(
                "Both frames already exist in the registry."
            )

        if from_frame not in self._adjacencies and to_frame not in self._adjacencies:
            raise InvaidTransformationError(
                "At least one of the frames must exist in the registry."
            )

        if from_frame not in self._adjacencies:
            self._adjacencies[from_frame] = {to_frame: transform}
            self._adjacencies[to_frame][from_frame] = np.linalg.inv(transform)
            self._update_paths(from_frame)
        else:
            self._adjacencies[from_frame][to_frame] = transform
            self._adjacencies[to_frame] = {from_frame: np.linalg.inv(transform)}
            self._update_paths(to_frame)

    def update(self, from_frame: FrameID_T, to_frame: FrameID_T, transform: np.ndarray):
        """
        Updates the transforms of an existing frame.
        In effect, this moves all children of the given frame as well (e.g., moving a robot base).

        Note that `from_frame` and `to_frame` must have been added together in the registry,
        i.e., they are attached to each other. However, they can be in any order.

        Args:
            from_frame: The source frame whose transformation is being updated.
            to_frame: The destination frame (should be the parent of `from_frame`).
            transform: The new transformation matrix from `from_frame` to `to_frame`.
        """
        if from_frame not in self._adjacencies:
            raise InvaidTransformationError(
                f"Frame {from_frame} does not exist in the registry."
            )

        if to_frame not in self._adjacencies:
            raise InvaidTransformationError(
                f"Frame {to_frame} does not exist in the registry."
            )

        if to_frame not in self._adjacencies[from_frame]:
            raise InvaidTransformationError(
                f"Frame {to_frame} is not attached to {from_frame}."
            )

        self._adjacencies[from_frame][to_frame] = transform
        self._adjacencies[to_frame][from_frame] = np.linalg.inv(transform)

    def _update_paths(self, new_frame: FrameID_T):
        """
        Updates the paths in the registry after adding a new frame.

        This method computes the shortest paths from the new frame to all other frames
        and updates the paths dictionary accordingly.

        Args:
            new_frame: The newly added frame.
        """
        self._paths[new_frame] = {new_frame: [new_frame]}

        # A new frame only has a path to its parent frame.
        parent = next(iter(self._adjacencies[new_frame].keys()))
        self._parents[new_frame] = parent

        # So that the dict doens't change size during iteration.
        self._paths[new_frame][parent] = [new_frame, parent]
        self._paths[parent][new_frame] = [parent, new_frame]

        # Connect the new frame to all existing frames and vice versa.
        for to_frame, path in self._paths[parent].items():
            if to_frame == new_frame or to_frame == parent:
                continue

            self._paths[new_frame][to_frame] = [new_frame] + path
            self._paths[to_frame][new_frame] = list(reversed(path)) + [new_frame]

    def _get_path(self, from_frame: FrameID_T, to_frame: FrameID_T) -> list[FrameID_T]:
        """
        Retrieves the path between two frames.

        Args:
            from_frame: The source frame.
            to_frame: The destination frame.

        Returns:
            A list of frames representing the path from `from_frame` to `to_frame`.
        """
        if from_frame not in self._adjacencies:
            raise InvaidTransformationError(
                f"Frame {from_frame} does not exist in the registry."
            )

        if to_frame not in self._adjacencies:
            raise InvaidTransformationError(
                f"Frame {to_frame} does not exist in the registry."
            )

        return self._paths[from_frame][to_frame]
