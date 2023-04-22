"""
Blender rendering class definitions.
"""
import json
import os
import random

import bpy
import mathutils as mu
import numpy as np
import utils.depth as depth
import utils.masks as masks
import utils.render as render
import utils.utils as utils
import utils.zip_write as zip_write
from PIL import Image
from shader import Shader


class BlenderIO:
    def __init__(self):
        pass

    @staticmethod
    def load_model(data_path, models_path, model_id, extension):
        model_path = os.path.join(data_path, models_path, model_id + f".{extension}")

        if extension == "gltf" or extension == "glb":
            bpy.ops.import_scene.gltf(filepath=model_path)
        elif extension == "fbx":
            bpy.ops.import_scene.fbx(filepath=model_path)
        elif extension == "obj":
            bpy.ops.import_scene.obj(filepath=model_path)


class MassRenderer:
    """
    Render a set of models with a set of styles.
    Generate matching depth maps and segmentation masks.
    """

    def __init__(self, args, zip_output, zip_path, tmp_dir):
        # TODO: Refactor properly, use root class
        self.conf = args
        self.conf.zip_output = zip_output
        self.conf.zip_path = zip_path
        self.conf.tmp_dir = tmp_dir

        # Create names -> 8bits colors mappings
        self.create_mappings(args)

        # Loaded materials cache
        self.loaded_materials = {}

        # Read the styles JSON file
        self.styles = json.load(open(os.path.join(self.conf.json_file)))
        self.model_id = os.path.basename(self.conf.json_file).split(".")[0]

    def create_mappings(self, args):
        """
        Map part/material names to 8-bits codes.
        """
        # Part to 8-bits value mapping
        self.part_to_idx = {p: i + 1 for i, p in enumerate(args.parts_list)}

        # Fine material to 8-bits value mapping
        self.mat_to_fine_idx = {}
        for mat_cat in args.mat_cats:
            for mat in args.mat_cats[mat_cat]:
                self.mat_to_fine_idx[mat] = args.mat_cats[mat_cat].index(mat) + 1

        # Coarse material to 8-bits value mapping
        mat_list_coarse = list(args.mat_cats.keys())
        mat_list_fine = [
            mat for mat_cat in args.mat_cats for mat in args.mat_cats[mat_cat]
        ]
        self.mat_to_coarse_idx = {
            mat: mat_list_coarse.index(mat.split("_")[0]) + 1 for mat in mat_list_fine
        }

    def get_part_to_mat(self, col_map, mat_map):
        """
        Get the part to 8-bits coarse and fine material codes for a given model style.
        """
        # Get the part to 8-bits coarse and fine material codes
        part_to_mat_coarse_idx = {
            p: self.mat_to_coarse_idx[mat_map[p]] for p in col_map
        }
        part_to_mat_fine_idx = {p: self.mat_to_fine_idx[mat_map[p]] for p in col_map}

        return part_to_mat_coarse_idx, part_to_mat_fine_idx

    def sample_camera(
        self, obj_center, view_code, random_view=False, view_per_model=True
    ):
        """
        Sample a camera position and orientation, with the seed fixed by the provided view identifier.
        """
        cam = render.get_camera()
        model_id, view_id, style_id = view_code

        # Fixing seed
        seed_str = model_id + str(view_id)
        if not view_per_model:
            seed_str += str(style_id)
        random.seed(seed_str)

        # Defining phi/theta angles
        phi_angle = (
            random.uniform(*render.RENDER_PARAMS["cam_phi_range"])
            if random_view
            else view_id * 90
        )
        phi_angle = render.RENDER_PARAMS["cam_base_angle"] + phi_angle
        theta_angle = (
            random.uniform(*render.RENDER_PARAMS["cam_theta_range"])
            if random_view
            else 0
        )

        # Rotating the camera
        base_location = mu.Vector(render.RENDER_PARAMS["cam_location"])
        cam.location = render.rotate_point(
            p=base_location, c=obj_center, theta=theta_angle, phi=phi_angle
        )

        # Pointing the camera to the object
        direction = cam.location - mu.Vector(obj_center)
        rot = direction.to_track_quat("Z", "Y").to_matrix().to_4x4()
        loc = mu.Matrix.Translation(cam.location - 0.75 * direction)
        cam.matrix_world = loc @ rot

        return render.get_camera_parameters(cam)

    def render(self, model_id, style_id, color_map=None, part_maps=None):
        """
        Render all views for a model in a given style.
        """
        scene = bpy.context.scene

        # Defining object center
        _, obj_center, _ = render.get_object_stats()

        # Iterating for each view point
        n_canonical_views = render.RENDER_PARAMS["n_canonical_views"]
        n_random_views = render.RENDER_PARAMS["n_random_views"]
        r_views = [False] * n_canonical_views + [True] * n_random_views

        for view_id, random_view in zip(
            range(n_canonical_views + n_random_views), r_views
        ):
            view_code = (model_id, view_id, style_id)
            cam_param = self.sample_camera(
                obj_center, view_code=view_code, random_view=random_view
            )

            # Define the base path for the output
            for scene in bpy.data.scenes:
                for node in scene.node_tree.nodes:
                    if node.type == "OUTPUT_FILE":
                        node.base_path = self.conf.tmp_dir

            # Defining the output filepath
            if self.conf.rendering_mode:
                tmp_file = os.path.join(self.conf.tmp_dir, "render_%d" % view_id)
                bpy.context.scene.render.use_file_extension = True
                bpy.context.scene.render.filepath = tmp_file
            else:
                tmp_file = os.path.join(self.conf.tmp_dir, "file_dump")
                bpy.context.scene.render.use_file_extension = True
                bpy.context.scene.render.filepath = tmp_file
                bpy.context.window.workspace = bpy.data.workspaces["Compositing"]

            # Rendering the image
            with utils.stdout_redirected():
                bpy.ops.render.render(write_still=True)

            # Pack images into the zip file
            model_dir = model_id + "_" + style_id
            if self.conf.seg_mode:
                for is_seg, (file_name, tmp_file) in enumerate(
                    zip(
                        ["depth_%d.exr", "seg_%d.png"], ["depth0000.exr", "seg0000.png"]
                    )
                ):
                    # Defining input/zip output paths
                    tmp_file = os.path.join(self.conf.tmp_dir, tmp_file)
                    file_name = file_name % view_id
                    file_path = os.path.join(model_dir, file_name)

                    if is_seg:
                        # Remapping colors to uint8-RGB values
                        cv_img = masks.remap_colors(
                            tmp_file, color_map, self.part_to_idx, *part_maps
                        )
                        zip_write.write_cv_img(self.conf.zip_output, cv_img, file_path)
                    else:
                        # Loading the OpenEXR image
                        exr_img = depth.read_exr(tmp_file)
                        zip_write.write_exr_img(
                            self.conf.zip_output, exr_img, file_path
                        )

                    utils.log_msg("MassRenderer", "Written to: [%s]." % file_path)

                # Writing camera parameters
                npy_file_path = os.path.join(model_dir, "cam_%d.npy" % view_id)
                zip_write.write_npy(self.conf.zip_output, cam_param, npy_file_path)

                utils.log_msg("MassRenderer", "Written to: [%s]." % npy_file_path)
            else:
                pil_img = Image.open(tmp_file + ".png").convert("RGB")
                file_path = os.path.join(model_dir, "render_%d.png" % view_id)
                zip_write.write_pil_img(self.conf.zip_output, pil_img, file_path)

        return True

    def run(self):
        """
        Loop over all models and styles and render them.

        TODO: Factorize with ModelExporter.
        """
        file_extensions = os.listdir(
            os.path.join(self.conf.data_path, self.conf.models_path)
        )
        file_names = [file.split(".")[0] for file in file_extensions]

        # Creating the output directory
        utils.create_dir(os.path.join(self.conf.data_path, self.conf.output_folder))

        # Loading the Blender config
        render.load_config(self.conf.blender_config)

        # Loop over the styles and models
        for model_code, part_styles in self.styles.items():
            model_id, style_id = model_code.split("__")
            class_part_remap = None
            if self.conf.parts_remap is not None:
                class_part_remap = self.conf.parts_remap[model_id[:2]]

            if not self.conf.seg_mode:
                utils.log_msg(
                    "MassRenderer", "Rendering: [%s, %s]" % (model_id, style_id)
                )
            else:
                utils.log_msg("MassRenderer", "Rendering: [%s]" % model_id)

            # Loop over styles and apply materials
            file_ext = file_extensions[file_names.index(model_id)].split(".")[1]
            BlenderIO.load_model(
                self.conf.data_path, self.conf.models_path, model_id, file_ext
            )
            utils.log_msg(
                "MassRenderer", "Loaded model file: [%s.%s]." % (model_id, file_ext)
            )

            # Center and scale the model
            render.move_to_z_up()

            # Mapping part names to mesh objects
            mesh_map = render.get_mesh_map(class_part_remap)

            # Configuring color coding for segmentation masks
            if self.conf.seg_mode:
                color_map = {}
                max_color = 2 ** render.RENDER_PARAMS["mask_color_depth"] - 1

                # Equally sample colors from range [max_color//4, max_color]
                part_colors = np.linspace(
                    max_color // 4, max_color, len(part_styles), dtype=np.int32
                )
                for part_color, (part_name, _) in zip(part_colors, part_styles.items()):
                    color_map[part_name] = part_color
                    # Setting index pass
                    for obj in mesh_map[part_name]:
                        obj.pass_index = color_map[part_name]

                # Render the views
                part_to_mat_coarse_idx, part_to_mat_fine_idx = self.get_part_to_mat(
                    color_map, part_styles
                )
                self.render(
                    model_id,
                    style_id,
                    color_map=color_map,
                    part_maps=(part_to_mat_coarse_idx, part_to_mat_fine_idx),
                )
            else:
                loaded_materials = {}
                for part_name, material_id in part_styles.items():
                    # Load all material textures if not loaded before
                    if material_id not in loaded_materials:
                        loaded_materials[material_id] = Shader.load_pbr_material(
                            self.conf.data_path, material_id
                        )
                    # Applying materials to each part
                    for mesh_obj in mesh_map[part_name]:
                        Shader.apply_material(mesh_obj, loaded_materials[material_id])

                # Render the views
                self.render(model_id, style_id)

            # Reset config to avoid any side effects
            render.load_config(self.conf.blender_config)

            # Flushing output zip file
            self.conf.zip_output = zip_write.zip_flush(
                self.conf.zip_output, self.conf.zip_path
            )

        # Printing a success mesage
        utils.log_msg("MassRenderer", "Success!")


class ModelExporter:
    """
    Export a stylized model to a GLB file.
    """

    def __init__(self, args, zip_output, zip_path, tmp_dir):
        self.conf = args
        self.conf.zip_output = zip_output
        self.conf.zip_path = zip_path
        self.conf.tmp_dir = tmp_dir

        # Read the styles JSON file
        self.styles = json.load(open(os.path.join(self.conf.json_file)))

    def export_glb(self, model_id, mode="GLTF_SEPARATE", reverse_map=None):
        """
        Export a stylized GLB model.
        """
        # Rendering to a GLB file
        glb_file_path = os.path.join(f"{model_id}.gltf")
        tmp_file = os.path.join(self.conf.tmp_dir, "tmp_render")
        tmp_gltf_file, tmp_bin_file = tmp_file + ".gltf", tmp_file + ".bin"

        # Exporting the separated GLTF file
        with utils.stdout_redirected():
            bpy.ops.export_scene.gltf(
                filepath=tmp_gltf_file,
                export_format=mode,
                ui_tab="GENERAL",
                export_copyright="",
                export_image_format="AUTO",
                export_texture_dir=self.conf.texture_dir,
                export_texcoords=True,
                export_normals=True,
                export_draco_mesh_compression_enable=False,
                export_draco_mesh_compression_level=6,
                export_draco_position_quantization=14,
                export_draco_normal_quantization=10,
                export_draco_texcoord_quantization=12,
                export_draco_color_quantization=10,
                export_draco_generic_quantization=12,
                export_tangents=False,
                export_materials="EXPORT",
                export_colors=True,
                use_mesh_edges=False,
                use_mesh_vertices=False,
                export_cameras=False,
                use_selection=False,
                use_visible=False,
                use_renderable=False,
                use_active_collection=False,
                export_extras=False,
                export_yup=True,
                export_apply=False,
                export_animations=False,
                export_frame_range=False,
                export_frame_step=1,
                export_force_sampling=True,
                export_nla_strips=True,
                export_def_bones=False,
                export_current_frame=False,
                export_skins=False,
                export_all_influences=False,
                export_morph=False,
                export_morph_normal=False,
                export_morph_tangent=False,
                export_lights=False,
                will_save_settings=False,
            )

        if mode == "GLTF_SEPARATE":
            # Merging the GLTF with the BIN
            zip_write.merge_gltf(tmp_gltf_file, tmp_bin_file)

            # Replace material names with part names
            if reverse_map:
                gltf_file = json.load(open(tmp_gltf_file))
                for img in gltf_file["images"]:
                    img["mimeType"] = "type_placeholder"

                    res_name = img["name"]
                    if "metalness" in res_name:
                        map_type = "metallic-roughness"
                        mat_name = res_name.split("__")[0]
                    else:
                        mat_name, map_type = res_name.split("__")

                    img["name"] = reverse_map[mat_name]
                    img["uri"] = map_type

                # Write back to the file
                with open(tmp_gltf_file, "w") as f:
                    json.dump(gltf_file, f)

            # Writing the GLTF file to zip
            gltf_file = open(tmp_gltf_file, "rb").read()
            zip_write.write_bin(self.conf.zip_output, gltf_file, glb_file_path)
            utils.log_msg("MassRenderer", "Written to: [%s]." % glb_file_path)
        elif mode == "GLTF_EMBEDDED":
            # Writing the GLTF file to zip
            gltf_file = open(tmp_gltf_file, "rb").read()
            zip_write.write_bin(self.conf.zip_output, gltf_file, glb_file_path)
            utils.log_msg("MassRenderer", "Written to: [%s]." % glb_file_path)

    def run(self):
        """
        Loop over all models and styles and export them.
        """
        file_extensions = os.listdir(
            os.path.join(self.conf.data_path, self.conf.models_path)
        )
        file_names = [file.split(".")[0] for file in file_extensions]

        # Creating the output directory
        utils.create_dir(os.path.join(self.conf.data_path, self.conf.output_folder))

        # Loop over the styles and models
        for model_code, part_styles in self.styles.items():
            model_id, style_id = model_code.split("__")
            class_part_remap = None
            if self.conf.parts_remap is not None:
                class_part_remap = self.conf.parts_remap[model_id[:2]]
            loaded_materials = {}
            reverse_map = {}

            utils.log_msg("MassRenderer", "Exporting: [%s]" % model_id)

            # Loop over styles and apply materials
            file_ext = file_extensions[file_names.index(model_id)].split(".")[1]
            BlenderIO.load_model(
                self.conf.data_path, self.conf.models_path, model_id, file_ext
            )
            utils.log_msg(
                "MassRenderer", "Loaded model file: [%s.%s]." % (model_id, file_ext)
            )

            # Mapping part names to mesh objects
            mesh_map = render.get_mesh_map(class_part_remap)

            for part_name, material_id in part_styles.items():
                reverse_map[material_id] = part_name
                # Load all material textures if not loaded before
                if material_id not in loaded_materials:
                    loaded_materials[material_id] = Shader.load_pbr_material(
                        self.conf.data_path, material_id
                    )
                # Applying materials to each part
                for mesh_obj in mesh_map[part_name]:
                    Shader.apply_material(mesh_obj, loaded_materials[material_id])

            # Export the models
            self.export_glb(model_id, mode=self.conf.gltf_mode, reverse_map=reverse_map)

            # Reset homefile to avoid any side effects
            bpy.ops.wm.read_homefile(app_template=self.conf.blender_config)

            # Flushing output zip file
            self.conf.zip_output = zip_write.zip_flush(
                self.conf.zip_output, self.conf.zip_path
            )
