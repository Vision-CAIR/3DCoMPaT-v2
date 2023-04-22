"""
General rendering utilities.
"""
import enum
from collections import defaultdict
from math import radians

import bpy
import mathutils as mu
import numpy as np

RENDER_PARAMS = {
    # Render engine
    "render_engine": "CYCLES",
    "cycles_device": "CPU",
    # Compression
    "use_compression": True,
    "compression": 5,
    # Render resolution
    "resolution_x": 256,
    "resolution_y": 256,
    # Masks color depth
    "mask_color_depth": 8,
    # Number of views
    "n_canonical_views": 4,
    "n_random_views": 4,
    # Camera parameters
    "cam_location": (0, -5, 6),
    "cam_base_angle": 40,
    "cam_phi_range": [0, 360],
    "cam_theta_range": [-50, 50],
}


class RenderMode(enum.Enum):
    RENDERING = "rendering"
    SEGMENTATION = "segmentation"
    MODEL = "model"

    def __str__(self):
        return self.value


"""
BPY Primitives.
"""


def get_camera():
    """
    Return the camera object.
    """
    return bpy.data.objects["#_cam_1"]


def get_objects(key_start):
    """
    Return all objects that start with a given key.
    """
    return [obj for key, obj in bpy.data.objects.items() if key.startswith(key_start)]


def select_objects(object_names):
    """
    Select objects by name.
    """
    # De-select anything
    bpy.ops.object.select_all(action="DESELECT")

    # Select
    for n in object_names:
        bpy.ops.object.select_pattern(pattern=n + "*")


def delete_objects(object_names):
    """
    Delete objects by name.
    """
    select_objects(object_names)
    bpy.ops.object.delete()


def get_mesh_map(parts_remap):
    """
    Get a mesh_name -> mesh_object map.
    """
    mesh_map = defaultdict(list)

    for obj in bpy.context.scene.objects:
        if obj.type == "MESH" and obj.name[0] != "#":
            part_name, part_idx = obj.name.rsplit("_", 1)
            part_idx = int(part_idx)

            if parts_remap is not None:
                part_name = parts_remap[part_name]
            mesh_map[part_name] += [obj]

    return mesh_map


def rotate_selection(rotate_value, orient_axis):
    """
    Rotate the selected objects.
    """
    bpy.ops.transform.rotate(
        value=rotate_value,
        orient_axis=orient_axis,
        orient_type="GLOBAL",
        orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        orient_matrix_type="GLOBAL",
        mirror=True,
        use_proportional_edit=False,
        proportional_edit_falloff="SMOOTH",
        proportional_size=1,
        use_proportional_connected=False,
        use_proportional_projected=False,
        release_confirm=True,
    )


def rotate_model(model_parts, angle):
    """
    Rotate all model parts.
    """
    select_objects(model_parts)
    rotate_selection(radians(angle), "Z")


def rotate_point(p, c, theta, phi):
    """
    Rotate a point around a center.
    """
    x = radians(theta)
    z = radians(phi)

    p = mu.Vector(p)
    c = mu.Vector(c)
    p = p - c
    p.rotate(mu.Euler((x, 0, z)))
    p = p + c
    return p


def rotate_camera(cam, angle):
    """
    Rotate all model parts.
    """
    _, obj_center, _ = get_object_stats()
    cam.location = rotate_point(cam.location, obj_center, radians(angle))


def load_config(blender_config):
    """
    Load a blender configuration.
    """
    bpy.ops.wm.read_homefile(app_template=blender_config)

    # Applying renderer configuration
    bpy.context.scene.render.engine = RENDER_PARAMS["render_engine"]
    bpy.context.scene.cycles.device = RENDER_PARAMS["cycles_device"]
    bpy.context.scene.render.resolution_x = RENDER_PARAMS["resolution_x"]
    bpy.context.scene.render.resolution_y = RENDER_PARAMS["resolution_y"]

    if RENDER_PARAMS["use_compression"]:
        bpy.context.scene.render.image_settings.compression = RENDER_PARAMS[
            "compression"
        ]


"""
Centering and scaling models.
"""


def get_object_bbox(object):
    """
    Get the bounding box of an object.
    """
    # Get bounding Box
    object_bb = [
        object.matrix_world
        @ mu.Vector(
            (object.bound_box[k][0], object.bound_box[k][1], object.bound_box[k][2])
        )
        for k in range(8)
    ]
    return np.asarray(object_bb)


def get_object_stats():
    """
    Get the loaded model's properties.
    """
    bbs = []
    for obj in bpy.data.objects.values():
        if obj.type != "MESH" or obj.name[0] == "#":
            continue

        # Get the object bbox in the world coordinates
        bbox_points = np.asarray(get_object_bbox(obj))
        bbs.append(bbox_points)

    bbs = np.vstack(bbs)
    x_max, x_min = np.max(bbs[:, 0]), np.min(bbs[:, 0])
    y_max, y_min = np.max(bbs[:, 1]), np.min(bbs[:, 1])
    z_max, z_min = np.max(bbs[:, 2]), np.min(bbs[:, 2])

    # Get the largest coordinate difference
    max_axis = np.max(np.asarray([x_max - x_min, y_max - y_min, z_max - z_min]))

    c_x = x_min + (x_max - x_min) / 2
    c_y = y_min + (y_max - y_min) / 2
    c_z = z_min + (z_max - z_min) / 2
    bbox_center = np.asarray((c_x, c_y, c_z))

    return max_axis, bbox_center, z_min


def move_to_z_up():
    """
    Move object to z-up.
    """
    (_, _, z_min) = get_object_stats()
    bpy.ops.transform.translate(
        value=(0, 0, z_min * -1),
        orient_type="GLOBAL",
        orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        orient_matrix_type="GLOBAL",
        mirror=True,
        use_proportional_edit=False,
        proportional_edit_falloff="SMOOTH",
        proportional_size=1,
        use_proportional_connected=False,
        use_proportional_projected=False,
    )


"""
Camera information.
"""


def get_camera_parameters(cam):
    """
    Compute the 3D to 2D plane projection matrix
    based on Blender parameters.
    """
    width = RENDER_PARAMS["resolution_x"]
    height = RENDER_PARAMS["resolution_y"]

    aspect_ratio = width / height
    fx = -width / 2 / np.tan(cam.data.angle / 2)
    fy = height / 2 / np.tan(cam.data.angle / 2) * aspect_ratio
    mx = width / 2
    my = height / 2

    K = np.array(
        [[fx, 0, mx, 0], [0, fy, my, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )

    M = np.array(cam.matrix_world.to_4x4())
    M = np.linalg.inv(M)

    # Vertically concatenate K and M
    K_M = np.concatenate((K, M), axis=0)

    return K_M
