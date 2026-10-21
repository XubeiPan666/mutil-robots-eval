"""Custom actor builders for RoboFactory."""
import numpy as np
import sapien
from typing import Optional, Union, Any
from mani_skill.utils.structs.types import Array
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building.actors.common import _build_by_type


def build_square_region(
    scene: Any,  # ManiSkillScene - using Any to avoid import issues
    half_size: Union[float, list],
    thickness: float,
    name: str,
    color: Optional[list] = None,
    body_type: str = "kinematic",
    add_collision: bool = False,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
    label_number: Optional[int] = None,
):
    """Build a flat square/rectangular region marker with optional number label.
    
    Args:
        scene: The ManiSkill scene
        half_size: Half size of the square. If float, creates a square. 
                   If list [x, y], creates a rectangle with half_size_x and half_size_y
        thickness: Thickness (height) of the region marker
        name: Name of the actor
        color: RGBA color [r, g, b, a], default is semi-transparent red
        body_type: Body type ("kinematic", "static", or "dynamic")
        add_collision: Whether to add collision shape
        scene_idxs: Scene indices
        initial_pose: Initial pose of the actor
        label_number: Optional number (1, 2, 3, etc.) to display above the region
        label_height: Height offset for the label above the region
    """
    if color is None:
        color = [1.0, 0.0, 0.0, 0.5]  # Default: semi-transparent red
    
    # Handle half_size: if float, convert to [x, y, z]; if list, use as [x, y]
    if isinstance(half_size, (int, float)):
        half_size_x = half_size_y = float(half_size)
    elif isinstance(half_size, list):
        if len(half_size) >= 2:
            half_size_x = float(half_size[0])
            half_size_y = float(half_size[1])
        else:
            half_size_x = half_size_y = float(half_size[0])
    else:
        raise ValueError(f"half_size must be float or list, got {type(half_size)}")
    
    half_size_z = thickness / 2.0
    
    builder = scene.create_actor_builder()
    
    # Add box visual for the square region
    builder.add_box_visual(
        half_size=[half_size_x, half_size_y, half_size_z],
        material=sapien.render.RenderMaterial(base_color=color),
    )
    
    # Add number label as flat squares arranged to form the digit
    if label_number is not None:
        label_color = [1.0, 1.0, 1.0, 1.0]  # White for visibility
        label_size = min(half_size_x, half_size_y) * 0.25  # Increased label size (was 0.15)
        # label_z = half_size_z + label_height  # Position above the region
        label_z = half_size_z
        
        # Create digit pattern using flat squares
        # Each digit is represented by a 3x5 grid pattern
        # Note: coordinates are (dx, dy) where dx is horizontal, dy is vertical
        digit_patterns = {
            1: [
                (0, 2), (0, 1), (0, 0), (0, -1), (0, -2)
            ],
            2: [
                (-1, 2), (0, 2), (1, 2),
                (1, 1), (0, 0), (-1, -1),
                (-1, -2), (0, -2), (1, -2)
            ],
            3: [
                (-1, 2), (0, 2), (1, 2),
                (1, 1), (0, 0), (1, -1),
                (-1, -2), (0, -2), (1, -2)
            ],
            4: [
                (-1, 2), (-1, 1), (-1, 0),  # 左竖线
                (0, 0),  # 横线中间
                (1, 2), (1, 1), (1, 0), (1, -1), (1, -2)  # 右竖线
            ]
        }
        
        pattern = digit_patterns.get(label_number, [])
        # Flat square size - same thickness as the region, but larger
        square_half_size = label_size * 0.35  # Increased size (was 0.25)
        square_thickness = half_size_z  # Same thickness as the region
        
        # Rotate 90 degrees in XY plane: (x, y) -> (-y, x)
        # This rotates the digit pattern 90 degrees while keeping it flat
        for dx, dy in pattern:
            # Original position relative to center of region
            pos_x_orig = dx * label_size * 0.5  # Increased spacing (was 0.4)
            pos_y_orig = dy * label_size * 0.5
            
            # Rotate 90 degrees in XY plane: (x, y) -> (-y, x)
            pos_x = -pos_y_orig
            pos_y = pos_x_orig
            
            # Add flat square visual (no collision) - no rotation, just position rotated
            builder.add_box_visual(
                half_size=[square_half_size, square_half_size, square_thickness],
                pose=sapien.Pose(p=[pos_x, pos_y, label_z]),
                material=sapien.render.RenderMaterial(base_color=label_color),
            )
    
    if add_collision:
        builder.add_box_collision(
            half_size=[half_size_x, half_size_y, half_size_z],
        )
    
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def build_outer_insertion_object(
    scene: Any,  # ManiSkillScene - using Any to avoid import issues
    inner_radius: float,
    inner_half_length: float,
    wall_thickness: float = 0.002,  # Thickness of the outer object's wall
    clearance: float = 0.001,  # Gap between outer and inner objects (default 1mm)
    name: str = "outer_object",
    color: Optional[list] = None,
    body_type: str = "kinematic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    """Build a hollow outer object (box with inner cavity) for tight insertion fit.
    
    Creates a hollow box with four walls (front, back, left, right) that can accommodate 
    a cylindrical inner object (stick) with a small clearance gap. The top and bottom 
    are open, allowing the stick to be inserted from above.
    
    Args:
        scene: The ManiSkill scene
        inner_radius: Radius of the inner cylindrical object (stick) that will fit inside
        inner_half_length: Half length (half height) of the inner cylindrical object
        wall_thickness: Thickness of the outer object's walls (default: 0.002m = 2mm)
        clearance: Gap between outer inner cavity and inner object (default: 0.001m = 1mm)
        name: Name of the outer object
        color: RGBA color [r, g, b, a], default is semi-transparent blue
        body_type: Body type ("kinematic", "static", or "dynamic")
        add_collision: Whether to add collision shape
        scene_idxs: Scene indices
        initial_pose: Initial pose of the actor
    
    Returns:
        Actor: The outer object actor (hollow box with four walls)
    """
    if color is None:
        color = [0.0, 0.0, 1.0, 1]  # Semi-transparent blue
    
    # Calculate outer dimensions: inner dimensions + clearance + wall thickness
    # For a cylindrical inner object, we create a square/rectangular outer box
    # The inner cavity should accommodate the cylinder with clearance
    inner_cavity_half_x = inner_radius + clearance
    inner_cavity_half_y = inner_radius + clearance
    inner_cavity_half_z = inner_half_length + clearance
    
    # Outer box dimensions (only for the four side walls)
    outer_half_x = inner_cavity_half_x + wall_thickness
    outer_half_y = inner_cavity_half_y + wall_thickness
    
    # Build hollow box using 4 walls (front, back, left, right) - top and bottom are open
    builder = scene.create_actor_builder()
    material = sapien.render.RenderMaterial(base_color=color)
    
    # Front wall (positive Y)
    builder.add_box_visual(
        half_size=[outer_half_x, wall_thickness / 2, inner_cavity_half_z],
        pose=sapien.Pose(p=[0, outer_half_y - wall_thickness / 2, 0]),
        material=material,
    )
    if add_collision:
        builder.add_box_collision(
            half_size=[outer_half_x, wall_thickness / 2, inner_cavity_half_z],
            pose=sapien.Pose(p=[0, outer_half_y - wall_thickness / 2, 0]),
        )
    
    # Back wall (negative Y)
    builder.add_box_visual(
        half_size=[outer_half_x, wall_thickness / 2, inner_cavity_half_z],
        pose=sapien.Pose(p=[0, -outer_half_y + wall_thickness / 2, 0]),
        material=material,
    )
    if add_collision:
        builder.add_box_collision(
            half_size=[outer_half_x, wall_thickness / 2, inner_cavity_half_z],
            pose=sapien.Pose(p=[0, -outer_half_y + wall_thickness / 2, 0]),
        )
    
    # Left wall (negative X)
    builder.add_box_visual(
        half_size=[wall_thickness / 2, inner_cavity_half_y, inner_cavity_half_z],
        pose=sapien.Pose(p=[-outer_half_x + wall_thickness / 2, 0, 0]),
        material=material,
    )
    if add_collision:
        builder.add_box_collision(
            half_size=[wall_thickness / 2, inner_cavity_half_y, inner_cavity_half_z],
            pose=sapien.Pose(p=[-outer_half_x + wall_thickness / 2, 0, 0]),
        )
    
    # Right wall (positive X)
    builder.add_box_visual(
        half_size=[wall_thickness / 2, inner_cavity_half_y, inner_cavity_half_z],
        pose=sapien.Pose(p=[outer_half_x - wall_thickness / 2, 0, 0]),
        material=material,
    )
    if add_collision:
        builder.add_box_collision(
            half_size=[wall_thickness / 2, inner_cavity_half_y, inner_cavity_half_z],
            pose=sapien.Pose(p=[outer_half_x - wall_thickness / 2, 0, 0]),
        )
    
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def build_inner_insertion_object(
    scene: Any,  # ManiSkillScene - using Any to avoid import issues
    inner_radius: float,
    inner_half_length: float,
    clearance: float = 0.001,  # Gap between outer and inner objects (default 1mm)
    name: str = "inner_object",
    color: Optional[list] = None,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    """Build a box-shaped inner object (stick) for tight insertion fit.
    
    Creates a box-shaped object that can be inserted into the hollow outer object with a small clearance gap.
    The inner object's dimensions are calculated to fit snugly inside the outer object's cavity.
    
    Args:
        scene: The ManiSkill scene
        inner_radius: Radius (half-width) of the inner object in X and Y directions.
                      This should match the inner_radius parameter used for the outer object.
        inner_half_length: Half length (half height) of the inner object along Z axis.
                          This should match the inner_half_length parameter used for the outer object.
        clearance: Gap between outer inner cavity and inner object (default: 0.001m = 1mm).
                  This should match the clearance parameter used for the outer object.
        name: Name of the inner object
        color: RGBA color [r, g, b, a], default is semi-transparent green
        body_type: Body type ("kinematic", "static", or "dynamic")
        add_collision: Whether to add collision shape
        scene_idxs: Scene indices
        initial_pose: Initial pose of the actor
    
    Returns:
        Actor: The inner object actor (box-shaped stick)
    """
    if color is None:
        color = [0.0, 1.0, 0.0, 1]  # Semi-transparent green
    
    # Calculate inner object dimensions to fit inside the outer object's cavity
    # The outer object's inner cavity is: inner_radius + clearance
    # So the inner object should be: inner_radius (to leave clearance gap)
    inner_half_x = inner_radius
    inner_half_y = inner_radius
    inner_half_z = inner_half_length
    
    # Build box-shaped stick
    builder = scene.create_actor_builder()
    material = sapien.render.RenderMaterial(base_color=color)
    
    # Add box visual and collision
    builder.add_box_visual(
        half_size=[inner_half_x, inner_half_y, inner_half_z],
        material=material,
    )
    if add_collision:
        builder.add_box_collision(
            half_size=[inner_half_x, inner_half_y, inner_half_z],
        )
    
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def build_hollow_cube(
    scene: Any,  # ManiSkillScene - using Any to avoid import issues
    half_size: float,
    edge_thickness: float = 0.005,  # Thickness of each edge
    name: str = "hollow_cube",
    color: Optional[list] = None,
    body_type: str = "kinematic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    """Build a hollow cube (wireframe cube) consisting of 12 edges only.
    
    Creates a cube frame with 12 edges:
    - 4 vertical edges (along Z-axis) at the 4 corners
    - 4 horizontal edges (along X-axis) on top and bottom faces
    - 4 horizontal edges (along Y-axis) on top and bottom faces
    
    Args:
        scene: The ManiSkill scene
        half_size: Half size of the cube (cube extends from -half_size to +half_size in all dimensions)
        edge_thickness: Thickness (radius) of each edge (default: 0.005m = 5mm)
        name: Name of the actor
        color: RGBA color [r, g, b, a], default is semi-transparent gray
        body_type: Body type ("kinematic", "static", or "dynamic")
        add_collision: Whether to add collision shapes
        scene_idxs: Scene indices
        initial_pose: Initial pose of the actor
    
    Returns:
        Actor: The hollow cube actor (wireframe with 12 edges)
    """
    if color is None:
        color = [0.5, 0.5, 0.5, 0.8]  # Default: semi-transparent gray
    
    builder = scene.create_actor_builder()
    material = sapien.render.RenderMaterial(base_color=color)
    
    s = half_size  # Shorthand for half_size
    t = edge_thickness / 2.0  # Half thickness of edge
    
    # Define the 8 vertices of the cube
    vertices = [
        [-s, -s, -s],  # 0: bottom-back-left
        [ s, -s, -s],  # 1: bottom-back-right
        [ s,  s, -s],  # 2: bottom-front-right
        [-s,  s, -s],  # 3: bottom-front-left
        [-s, -s,  s],  # 4: top-back-left
        [ s, -s,  s],  # 5: top-back-right
        [ s,  s,  s],  # 6: top-front-right
        [-s,  s,  s],  # 7: top-front-left
    ]
    
    # Define the 12 edges as (start_vertex, end_vertex) pairs
    edges = [
        # 4 vertical edges (Z direction)
        (0, 4),  # back-left vertical
        (1, 5),  # back-right vertical
        (2, 6),  # front-right vertical
        (3, 7),  # front-left vertical
        # 4 bottom horizontal edges (X direction)
        (0, 1),  # bottom-back horizontal
        (2, 3),  # bottom-front horizontal
        # 4 bottom horizontal edges (Y direction)
        (1, 2),  # bottom-right horizontal
        (0, 3),  # bottom-left horizontal
        # 4 top horizontal edges (X direction)
        (4, 5),  # top-back horizontal
        (6, 7),  # top-front horizontal
        # 4 top horizontal edges (Y direction)
        (5, 6),  # top-right horizontal
        (4, 7),  # top-left horizontal
    ]
    
    # Create each edge as a box
    for start_idx, end_idx in edges:
        v0 = np.array(vertices[start_idx])
        v1 = np.array(vertices[end_idx])
        
        # Calculate center and direction
        center = (v0 + v1) / 2.0
        direction = v1 - v0
        length = np.linalg.norm(direction)
        
        if length < 1e-6:
            continue
        
        # Normalize direction
        direction = direction / length
        
        # Create a box along the edge
        # The box extends along the direction vector
        half_length = length / 2.0
        
        # Determine box orientation
        # For edges along X, Y, or Z axes, we can use simple orientations
        if abs(direction[0]) > 0.9:  # X direction
            half_size_box = [half_length, t, t]
            pose = sapien.Pose(p=center)
        elif abs(direction[1]) > 0.9:  # Y direction
            half_size_box = [t, half_length, t]
            pose = sapien.Pose(p=center)
        elif abs(direction[2]) > 0.9:  # Z direction
            half_size_box = [t, t, half_length]
            pose = sapien.Pose(p=center)
        else:
            # For diagonal edges, we need to compute rotation
            # Use a more general approach: create box aligned with edge
            half_size_box = [half_length, t, t]
            # Compute rotation to align with edge direction
            z_axis = np.array([0, 0, 1])
            if np.abs(np.dot(direction, z_axis)) > 0.99:
                # Already aligned with Z
                pose = sapien.Pose(p=center)
            else:
                # Compute rotation axis and angle
                axis = np.cross(z_axis, direction)
                if np.linalg.norm(axis) < 1e-6:
                    pose = sapien.Pose(p=center)
                else:
                    axis = axis / np.linalg.norm(axis)
                    angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
                    from transforms3d.axangles import axangle2quat
                    quat = axangle2quat(axis, angle)
                    pose = sapien.Pose(p=center, q=quat)
        
        # Add visual
        builder.add_box_visual(
            half_size=half_size_box,
            pose=pose,
            material=material,
        )
        
        # Add collision if requested
        if add_collision:
            builder.add_box_collision(
                half_size=half_size_box,
                pose=pose,
            )
    
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def build_welding_sphere(
    scene: Any,  # ManiSkillScene - using Any to avoid import issues
    radius: float,
    name: str = "welding_sphere",
    color: Optional[list] = None,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
    return_actor_data: bool = True,
):
    """Build a sphere that can be grasped and moved to simulate welding process.
    
    Creates a dynamic sphere with contact points for robot grasping. The sphere can be
    placed at the corners of a hollow cube to simulate welding operations.
    
    Args:
        scene: The ManiSkill scene
        radius: Radius of the sphere
        name: Name of the actor
        color: RGBA color [r, g, b, a], default is red
        body_type: Body type ("kinematic", "static", or "dynamic")
        add_collision: Whether to add collision shape
        scene_idxs: Scene indices
        initial_pose: Initial pose of the actor
        return_actor_data: If True, returns tuple (actor, actor_data). If False, returns only actor.
    
    Returns:
        Actor or tuple (Actor, dict): The sphere actor, and optionally actor_data dict with contact_points_pose
    """
    if color is None:
        color = [1.0, 0.0, 0.0, 0.8]  # Default: red
    
    builder = scene.create_actor_builder()
    material = sapien.render.RenderMaterial(base_color=color)
    
    # Add sphere visual
    builder.add_sphere_visual(
        sapien.Pose(),
        radius=radius,
        material=material,
    )
    
    # Add collision if requested
    if add_collision:
        builder.add_sphere_collision(
            sapien.Pose(),
            radius=radius,
        )
    
    actor = _build_by_type(builder, name, body_type, scene_idxs, initial_pose)
    
    if not return_actor_data:
        return actor
    
    # Create actor_data with contact points for grasping
    # For a sphere, we define contact points on the surface
    # We'll define multiple contact points around the sphere for flexible grasping
    
    # Contact point 0: top of sphere (Z+ direction)
    contact_0_matrix = np.eye(4)
    contact_0_matrix[:3, 3] = [0, 0, radius]  # Position at top
    # Orientation: Z-axis pointing outward (up), Y-axis forward
    contact_0_matrix[:3, :3] = np.array([
        [1, 0, 0],  # X axis
        [0, 1, 0],  # Y axis
        [0, 0, 1],  # Z axis (pointing outward, up)
    ])
    
    # Contact point 1: bottom of sphere (Z- direction)
    contact_1_matrix = np.eye(4)
    contact_1_matrix[:3, 3] = [0, 0, -radius]  # Position at bottom
    # Orientation: Z-axis pointing outward (down), Y-axis forward
    contact_1_matrix[:3, :3] = np.array([
        [1, 0,  0],  # X axis
        [0, 1,  0],  # Y axis
        [0, 0, -1],  # Z axis (pointing outward, down)
    ])
    
    actor_data = {
        "contact_points_pose": [
            contact_0_matrix.tolist(),
            contact_1_matrix.tolist(),
        ],
        "scale": 1.0,
    }
    
    return actor, actor_data






