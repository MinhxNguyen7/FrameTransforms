from threading import Lock
from typing import Any, Callable, Generic, Hashable, TypeVar
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

# Key to identify coordinate frames in the registry.
FrameID_T = TypeVar("FrameID_T", bound=Hashable)
Ret_T = TypeVar("Ret_T", bound=Any)


class InvalidTransformError(Exception):
    pass


TransformTarget_t = TypeVar("TransformTarget_t", bound="Transform|Pose|np.ndarray")


@dataclass(frozen=True)
class Transform:
    """
    Similarly to posetree, this represents an action ("to transform")
    to be applied to a pose.
    """

    _translation: np.ndarray
    _rotation: Rotation

    def __post_init__(self):
        if self._translation.shape != (3,):
            raise ValueError("Translation must be a 3D vector.")
        if not isinstance(self._rotation, Rotation):
            raise TypeError(
                "Rotation must be an instance of scipy.spatial.transform.Rotation."
            )

    @property
    def translation(self) -> np.ndarray:
        """
        Returns the translation vector.
        """
        return self._translation.copy()

    @property
    def rotation(self) -> Rotation:
        """
        Returns the rotation.
        """
        return self._rotation

    def as_matrix(self) -> np.ndarray:
        """
        Converts the transformation to a 4x4 transformation matrix.
        """
        matrix = np.eye(4)
        matrix[:3, :3] = self._rotation.as_matrix()
        matrix[:3, 3] = self._translation
        return matrix

    def _apply_to_pose(self, pose: "Pose[FrameID_T]") -> "Pose[FrameID_T]":
        """
        Applies this transformation to a given pose and returns a new pose.
        """
        new_translation = self._translation + pose.transform.translation
        new_rotation = self._rotation * pose.transform.rotation

        return Pose(
            Transform(new_translation, new_rotation),
            pose.parent_frame,
            pose.registry,
        )

    def __matmul__(self, other: TransformTarget_t) -> TransformTarget_t:
        """
        Applies the transformation to another Transform, Pose, 3D vector (point),
        or a 4x4 homogeneous transformation matrix.
        """
        if isinstance(other, Transform):
            new_translation = self._translation + other.translation
            new_rotation = self._rotation * other.rotation
            return Transform(new_translation, new_rotation)  # type: ignore[return-value]

        elif isinstance(other, Pose):
            return self._apply_to_pose(other)

        elif isinstance(other, np.ndarray):
            match other.shape:
                case (4, 4):
                    return self.as_matrix() @ other
                case (3,):
                    return np.array(self._rotation.apply(other + self._translation))  # type: ignore[return-value]
                case _:
                    raise ValueError(
                        "Invalid shape for transformation application. Expected (4, 4) or (3,)."
                    )
        else:
            raise TypeError(
                "Unsupported type for transformation application. Expected Transform, Pose, or np.ndarray."
            )


@dataclass(frozen=True)
class Pose(Generic[FrameID_T]):
    """
    Represents a 3D pose with position and orientation.
    """

    transform: Transform
    parent_frame: FrameID_T
    registry: "Registry[FrameID_T]"

    def __post_init__(self):
        if not isinstance(self.transform, Transform):
            raise TypeError("Transform must be an instance of Transform.")
        if not isinstance(self.registry, Registry):
            raise TypeError("Registry must be an instance of Registry.")

    def apply_transform(
        self, transform: Transform, new_frame: FrameID_T | None
    ) -> "Pose":
        """
        Applies a transformation to the pose and returns a new pose.

        The new pose will be attached to the specified new frame, if provided.
        If no new frame is specified, the pose remains in its current frame.
        """
        new_translation = self.transform.translation + transform.translation
        new_rotation = self.transform.rotation * transform.rotation

        return Pose(
            Transform(new_translation, new_rotation),
            new_frame or self.parent_frame,
            self.registry,
        )

    def in_frame(self, frame: FrameID_T) -> "Pose":
        """
        Transforms the pose to the specified frame.
        I.e., where is the pose in the specified frame?
        """
        transform = self.registry.get_transform(self.parent_frame, frame)
        new_transform = self.transform @ transform
        return Pose(new_transform, frame, self.registry)

    def get_position(self, frame: FrameID_T | None) -> np.ndarray:
        """
        Gets the position of the pose in the specified frame.

        If frame is None, the position is returned in the pose's parent frame.
        """
        if frame is None:
            frame = self.parent_frame

        return self.in_frame(
            frame if frame is not None else self.parent_frame
        ).transform.translation

    def get_orientation(self, frame: FrameID_T | None) -> Rotation:
        """
        Gets the orientation of the pose in the specified frame.

        If frame is None, the orientation is returned in the pose's parent frame.
        """
        if frame is None:
            frame = self.parent_frame

        return self.in_frame(
            frame if frame is not None else self.parent_frame
        ).transform.rotation


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

    def get_transform(self, from_frame: FrameID_T, to_frame: FrameID_T) -> Transform:
        """
        Gets the transformation matrix from one frame to another.

        Args:
            from_frame: The source frame.
            to_frame: The destination frame.

        Returns:
            The transformation from `from_frame` to `to_frame`.
        """
        return self._concurrent_read(
            lambda: self._get_transform_unsafe(from_frame, to_frame)
        )

    def _get_transform_unsafe(
        self, from_frame: FrameID_T, to_frame: FrameID_T
    ) -> Transform:
        path = self._get_path(from_frame, to_frame)

        trans = np.eye(4)
        for i in range(len(path) - 1):
            current_frame = path[i]
            next_frame = path[i + 1]
            trans = trans @ self._adjacencies[current_frame][next_frame]

        return Transform(trans[:3, 3], Rotation.from_matrix(trans[:3, :3]))

    def add_transform(
        self,
        from_frame: FrameID_T,
        to_frame: FrameID_T,
        transform: Transform | np.ndarray,
    ):
        """
        Adds a transformation from one frame to another.

        Exactly *one* of the frames must exist in the registry.

        Args:
            from_frame: The source frame.
            to_frame: The destination frame.
            transform: The Transform or 4x4 transformation matrix from `from_frame` to `to_frame`.
        """
        self._concurrent_write(
            lambda: self._add_transform_unsafe(from_frame, to_frame, transform)
        )

    def _add_transform_unsafe(
        self,
        from_frame: FrameID_T,
        to_frame: FrameID_T,
        transform: Transform | np.ndarray,
    ):
        if from_frame in self._adjacencies and to_frame in self._adjacencies:
            raise InvalidTransformError("Both frames already exist in the registry.")

        if from_frame not in self._adjacencies and to_frame not in self._adjacencies:
            raise InvalidTransformError(
                "At least one of the frames must exist in the registry."
            )

        if isinstance(transform, Transform):
            transform = transform.as_matrix()
        elif not (isinstance(transform, np.ndarray) and transform.shape == (4, 4)):
            raise ValueError("Transform must be a Transform or a 4x4 numpy array.")

        if from_frame not in self._adjacencies:
            self._adjacencies[from_frame] = {to_frame: transform}
            self._adjacencies[to_frame][from_frame] = np.linalg.inv(transform)
            self._update_paths(from_frame)
        else:
            self._adjacencies[from_frame][to_frame] = transform
            self._adjacencies[to_frame] = {from_frame: np.linalg.inv(transform)}
            self._update_paths(to_frame)

    def update(
        self,
        from_frame: FrameID_T,
        to_frame: FrameID_T,
        transform: Transform | np.ndarray,
    ):
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
        self,
        from_frame: FrameID_T,
        to_frame: FrameID_T,
        transform: Transform | np.ndarray,
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
            raise InvalidTransformError(
                f"Frame {from_frame} does not exist in the registry."
            )

        if to_frame not in self._adjacencies:
            raise InvalidTransformError(
                f"Frame {to_frame} does not exist in the registry."
            )

        if to_frame not in self._adjacencies[from_frame]:
            raise InvalidTransformError(
                f"Frame {to_frame} is not attached to {from_frame}."
            )

        if isinstance(transform, Transform):
            transform = transform.as_matrix()
        elif not (isinstance(transform, np.ndarray) and transform.shape == (4, 4)):
            raise ValueError("Transform must be a Transform or a 4x4 numpy array.")

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
            raise InvalidTransformError(
                f"Either {from_frame} or {to_frame} does not exist in the registry."
            )
