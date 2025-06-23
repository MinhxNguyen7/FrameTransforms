from typing import Any, Callable, Generic, Hashable, TypeVar
from threading import Lock

import numpy as np


# Key to identify coordinate frames in the registry.
FrameID_T = TypeVar("FrameID_T", bound=Hashable)
Ret_T = TypeVar("Ret_T", bound=Any)


class InvaidTransformationError(Exception):
    pass


class Registry(Generic[FrameID_T]):
    """
    Registry of coordinate frames and corresponding transforms.

    Automatically computes transitive tramsforms between frames if possible by
    maintaining a directed acyclic graph (DAG) of relationships.

    Made for use with 4x4 3D transformation matrices.
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

        # For thread safety, implement as third readers-writers problem (no starvation).
        # Reference: https://en.wikipedia.org/wiki/Readers%E2%80%93writers_problem#Third_readers%E2%80%93writers_problem
        self._read_count = 0

        self._resource_lock = Lock()
        self._counts_lock = Lock()
        self._service_queue = Lock()

    def get_transform(self, from_frame: FrameID_T, to_frame: FrameID_T) -> np.ndarray:
        """
        Gets the transformation matrix from one frame to another.

        Args:
            from_frame: The source frame.
            to_frame: The destination frame.

        Returns:
            The transformation matrix from `from_frame` to `to_frame`.
        """
        return self._concurrent_read(
            lambda: self._get_transform_unsafe(from_frame, to_frame)
        )

    def _get_transform_unsafe(
        self, from_frame: FrameID_T, to_frame: FrameID_T
    ) -> np.ndarray:
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
        self._concurrent_write(
            lambda: self._add_transform_unsafe(from_frame, to_frame, transform)
        )

    def _add_transform_unsafe(
        self, from_frame: FrameID_T, to_frame: FrameID_T, transform: np.ndarray
    ):
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
        self._concurrent_write(
            lambda: self._update_unsafe(from_frame, to_frame, transform)
        )

    def _update_unsafe(
        self, from_frame: FrameID_T, to_frame: FrameID_T, transform: np.ndarray
    ):
        """
        Internal method to update the transformation between two frames.
        This is used by the `update` method to ensure that the frames are already connected.

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

    def _concurrent_read(self, func: Callable[[], Ret_T]) -> Ret_T:
        """
        Wrapper to execute a synchrnous, thread-unsafe function that reads from the registry.
        """
        self._service_queue.acquire()
        self._counts_lock.acquire()

        self._read_count += 1
        if self._read_count == 1:
            self._resource_lock.acquire()

        self._service_queue.release()
        self._counts_lock.release()

        try:
            return func()
        finally:
            with self._counts_lock:
                self._read_count -= 1
                if self._read_count == 0:
                    self._resource_lock.release()

    def _concurrent_write(self, func: Callable[[], Ret_T]) -> Ret_T:
        """
        Wrapper to execute a synchronous, thread-unsafe function that writes to the registry.
        """
        with self._service_queue:
            self._resource_lock.acquire()

        try:
            return func()
        finally:
            self._resource_lock.release()

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
        try:
            return self._paths[from_frame][to_frame]
        except KeyError:
            raise InvaidTransformationError(
                f"Either {from_frame} or {to_frame} does not exist in the registry."
            )
