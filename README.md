# frame-transforms
`frame-transforms` is a lightweight, thread-safe, Python-native package to simplify frame transforms in robotics. With it, you can manage and translate between coordinate frames with ease. It features:

1. Automatic computation of transitive transforms.
2. Registration and update of relative coordinate frames.
3. An intuitive, object-oriented API. 

Though in beta, the library is extensively tested.

This package was inspired by the interface of `posetree` and shares much of its functionality but offers a more batteries-included experience. Similarly to [posetree](https://github.com/robobenjie/posetree?tab=readme-ov-file#philosophy-of-transforms-poses-and-frames)'s nomenclature, `Pose` is a location and orientation in space, whereas `Transform` is an action that describes the change in position and orientation to get from one `Pose` to another.

## Installation

```bash
pip install frame-transforms
```

## Application
Consider a simple robot consisting of a mobile base and a camera mounted on a gimbal. 

The camera detects an object in its coordinate frame. Where is it in world frame?

```python
# Setup
registry.update(Frame.WORLD, Frame.BASE, base_transform)
registry.update(Frame.BASE, Frame.CAMERA, camera_transform)

# Define the Pose
object_pose = Pose(
    Transform(
        np.array([1, 0, 0]),
        Rotation.from_euler("xyz", [0, 0, 0], degrees=True),
    ),
    parent_frame=Frame.CAMERA,
    registry=registry,
)

# Get the position and orientation of the object in world frame
position_in_world = object_pose.get_position(Frame.WORLD)
orientation_in_world = object_pose.get_orientation(Frame.WORLD)
```

## ROS Integration

Simply wrap the `Register` in a ROS node, subscribing to pose updates and publishing/service-calling the poses. The `Register` is thread-safe, so callbacks are simple.

## Alternatives
- `tf2`: Heavyweight, requires ROS.
- `posetree`: Requires `PoseTree` subclass implementation.

# [Examples](https://github.com/MinhxNguyen7/FrameTransforms/blob/main/example.py)

# Development Setup

- Clone and `cd` into this repository.
- Set up virtual environment.
  - `python -m venv venv`
  - `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
- Install package with dev and test dependencies
  - `pip install -e '.[dev,test]'`

# Implementation Details

- Transforms are stored as a tree starting from the world frame (provided at `Registry` initialization).
- To optimize run-time performance, the paths between all pairs of frames are eagerly precomputed and stored when a frame is added.
  - Therefore, the runtime complexity on request is proportional to the shortest path, instead of all frames in the case of a full graph search.
- Transitive transforms themselves are only computed on-demand because an intermediate transform can change.
  - This is preferred because poses often change more often than they are requested.
