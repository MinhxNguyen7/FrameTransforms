# Description
FrameTransforms is a lightweight, native Python pacakge to simplify frame transforms. It supports:

1. Registration and update of relative coordinate frames.
2. Automatic computation of transitive transforms.
3. Multithreaded access.

## Application
Consider a simple robot consisting of a mobile base and a camera mounted on a gimbal. 

The camera detects an object in its coordinate frame. Where is it in world frame?

```python
# Setup
registry.update(Frame.WORLD, Frame.BASE, base_pose)
registry.update(Frame.BASE, Frame.CAMERA, camera_pose)

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

# [Examples](https://github.com/MinhxNguyen7/FrameTransforms/blob/main/example.py)
