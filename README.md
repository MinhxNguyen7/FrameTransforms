# Description
FrameTransforms is a lightweight, native Python pacakge to simplify frame transformations. It supports:

1. Registration and update of relative coordinate frames.
2. Automatic computation of transitive transformations.
3. Multithreaded access.

## Application
Consider a simple robot consisting of a mobile base and a camera mounted on a gimbal. 

The camera detects an obstacle in its coordinate frame. Where is it in world frame?

```python
registry.update(Frame.WORLD, Frame.BASE, base_pose)
registry.update(Frame.BASE, Frame.CAMERA, camera_pose)

# Locations are in homogenous coordinates
obstacle_in_world = registry.get_transform(Frame.CAMERA, Frame.WORLD) @ obstacle_in_camera
```

# [Examples](https://github.com/MinhxNguyen7/FrameTransforms/blob/main/example.py)