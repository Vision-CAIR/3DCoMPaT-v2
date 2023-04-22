"""
Defining Blender shaders.
"""
import os

import bpy
import utils.render as render


class Shader:
    """
    Shader class: manipulate the Blender shader editor
    to stylize the 3D models.
    """

    def __init__(self):
        pass

    @staticmethod
    def get_bsdf(mat):
        return mat.node_tree.nodes["Principled BSDF"]

    @staticmethod
    def create_shader_node(mat):
        return mat.node_tree.nodes.new("ShaderNodeTexImage")

    @staticmethod
    def create_bump_node(mat):
        return mat.node_tree.nodes.new("ShaderNodeBump")

    @staticmethod
    def create_normal_map_node(mat):
        return mat.node_tree.nodes.new("ShaderNodeNormalMap")

    @staticmethod
    def create_multiply_node(mat):
        return mat.node_tree.nodes.new("ShaderNodeMath")

    @staticmethod
    def set_node_name(node, name):
        node.label = name
        node.name = name

    @staticmethod
    def create_ambient_occlusion_node(ao_image, diffuse_image, mat, link_to_base=True):
        # Get the principled node
        principled_node = Shader.get_bsdf(mat)

        # Create diffuse (color) node
        color_node = Shader.create_shader_node(mat)
        color_node.image = diffuse_image

        # Create clearcoat normal node
        ao_node = Shader.create_shader_node(mat)

        # Load the image
        ao_node.image = ao_image

        # Set up the label
        Shader.set_node_name(color_node, "diffuse_node")
        Shader.set_node_name(ao_node, "ambient_occlusion_node")

        # Create the normal map that will be connected to the principled node
        multiply = Shader.create_multiply_node(mat)
        multiply.operation = "MULTIPLY"

        # Link to base material if applicable
        if link_to_base:
            node_link = mat.node_tree.links.new
            node_link(color_node.outputs["Color"], multiply.inputs[0])
            node_link(ao_node.outputs["Color"], multiply.inputs[1])
            node_link(multiply.outputs[0], principled_node.inputs["Base Color"])

        return True

    @staticmethod
    def create_normal_node(
        normal_image, mat, link_to_base=True, node_name="normal", bsdf_input="Normal"
    ):
        # Get the principled node
        principled_node = Shader.get_bsdf(mat)

        # Create clearcoat normal node
        normal_texture_node = Shader.create_shader_node(mat)
        normal_texture_node.image = normal_image
        normal_texture_node.image.colorspace_settings.name = "Non-Color"

        # Set up the labels
        Shader.set_node_name(normal_texture_node, "%s_node" % node_name)

        # Create the normal map that will be connected to the principled node
        normal_map = Shader.create_normal_map_node(mat)
        normal_map.uv_map = "UVMap"

        # Link to base material if applicable
        if link_to_base:
            node_link = mat.node_tree.links.new
            node_link(normal_texture_node.outputs["Color"], normal_map.inputs["Color"])
            node_link(normal_map.outputs["Normal"], principled_node.inputs[bsdf_input])

        return True

    @staticmethod
    def create_clearcoat_normal_node(normal_image, mat, link_to_base=True):
        Shader.create_normal_node(
            normal_image,
            mat,
            link_to_base,
            node_name="clearcoat_normal",
            bsdf_input="Clearcoat Normal",
        )

        return True

    @staticmethod
    def create_diffuse_node(diffuse_image, mat, link_to_base=True):
        # Get the principled node
        principled_node = Shader.get_bsdf(mat)

        # Create diffuse (color) node
        color_node = Shader.create_shader_node(mat)
        color_node.image = diffuse_image

        # Set up the label
        Shader.set_node_name(color_node, "diffuse_node")

        # Link to base material if applicable
        if link_to_base:
            node_link = mat.node_tree.links.new
            node_link(color_node.outputs["Color"], principled_node.inputs["Base Color"])

        # Set up the label
        Shader.set_node_name(color_node, "diffuse_node_{}".format(mat.name))

        return True

    @staticmethod
    def create_height_node(height_image, mat, link_to_base=True):
        # Get the principled node
        principled_node = Shader.get_bsdf(mat)

        # Create height node
        height_node = Shader.create_shader_node(mat)
        height_node.image = height_image
        height_node.image.colorspace_settings.name = "Non-Color"

        # Create bump node
        bump_node = Shader.create_bump_node(mat)

        # Set up the label
        Shader.set_node_name(height_node, "height_node")

        # Link to base material if applicable
        if link_to_base:
            node_link = mat.node_tree.links.new
            node_link(height_node.outputs["Color"], bump_node.inputs["Height"])
            node_link(bump_node.outputs["Normal"], principled_node.inputs["Normal"])

        return True

    @staticmethod
    def create_metalness_node(metallic_image, mat, link_to_base=True):
        # Get the principled node
        principled_node = Shader.get_bsdf(mat)

        # Create metallic node
        metallic_node = Shader.create_shader_node(mat)
        metallic_node.image = metallic_image
        metallic_node.image.colorspace_settings.name = "Non-Color"

        # Set up the label
        Shader.set_node_name(metallic_node, "metalness_node")

        # Link to base material if applicable
        if link_to_base:
            node_link = mat.node_tree.links.new
            node_link(
                metallic_node.outputs["Color"], principled_node.inputs["Metallic"]
            )

        return True

    @staticmethod
    def create_roughness_node(roughness_image, mat, link_to_base=True):
        # Get the principled node
        principled_node = Shader.get_bsdf(mat)

        # Create metallic node
        roughnness_node = Shader.create_shader_node(mat)
        roughnness_node.image = roughness_image
        roughnness_node.image.colorspace_settings.name = "Non-Color"

        # Set up the label
        Shader.set_node_name(roughnness_node, "roughness_node")

        # Link to base material if applicable
        if link_to_base:
            node_link = mat.node_tree.links.new
            node_link(
                roughnness_node.outputs["Color"], principled_node.inputs["Roughness"]
            )

        return True

    @staticmethod
    def create_specular_node(specular_image, mat, link_to_base=True):
        # Get the principled node
        principled_node = Shader.get_bsdf(mat)

        # Create metallic node
        specular_node = Shader.create_shader_node(mat)
        specular_node.image = specular_image
        specular_node.image.colorspace_settings.name = "Non-Color"

        # Set up the label
        Shader.set_node_name(specular_node, "specular_node")

        # Link to base material if applicable
        if link_to_base:
            node_link = mat.node_tree.links.new
            node_link(
                specular_node.outputs["Color"], principled_node.inputs["Specular"]
            )

        return True

    @staticmethod
    def create_material_slot(mesh):
        # Create material slot
        render.select_objects([mesh.name])
        mesh_name = "mat_{}".format(mesh.name)
        mat = bpy.data.materials.new(name=mesh_name)
        mat.use_nodes = True

        # Assign it to object
        mesh.data.materials.append(mat)

        return mat

    @staticmethod
    def apply_material(mesh_obj, mat_data):
        mat = Shader.create_material_slot(mesh_obj)

        # Creating PBR material nodes
        for pbr_property, pbr_image in mat_data.items():
            PBR_CREATE[pbr_property](*pbr_image, mat, link_to_base=True)

    @staticmethod
    def load_pbr_material(data_path, mat_id):
        material_path = os.path.join(data_path, "materials", mat_id)
        pbr_map = {}

        # List all the files in the material folder without extensions
        pbr_images = {
            os.path.splitext(os.path.basename(f).split("__")[1])[0]: f
            for f in os.listdir(material_path)
        }

        for pbr_property, pbr_image in pbr_images.items():
            if pbr_property in PBR_CREATE:
                pbr_path = os.path.join(material_path, pbr_image)
                pbr_map[pbr_property] = [bpy.data.images.load(pbr_path)]

        # If ambient_occlusion is in the map, add diffuse map to it
        if "ambient_occlusion" in pbr_map:
            pbr_map["ambient_occlusion"] = [
                *pbr_map["ambient_occlusion"],
                *pbr_map["diffuse"],
            ]
            # Delete the 'diffuse' entry
            del pbr_map["diffuse"]

        return pbr_map


PBR_CREATE = {
    "ambient_occlusion": Shader.create_ambient_occlusion_node,
    "clearcoat_normal": Shader.create_clearcoat_normal_node,
    "diffuse": Shader.create_diffuse_node,
    "height": Shader.create_height_node,
    "metalness": Shader.create_metalness_node,
    "normal": Shader.create_normal_node,
    "roughness": Shader.create_roughness_node,
    "specular": Shader.create_specular_node,
}
