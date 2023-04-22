"""
GLTF 3D models utilities.
"""
import json
import trimesh


class ZipTextureResolver(trimesh.resolvers.FilePathResolver):
    """
    Resolve texture files from the input zip.
    """
    def __init__(self, zip_f):
        self.zip_f = zip_f

    def get(self, file_path):
        return self.zip_f.open(file_path, "r").read()


def load_gltf(model_id, zip_file, models_dir):
    """
    Load a gltf file and return a dictionary with the contents.
    """
    gltf_f = models_dir + model_id + ".gltf"
    return zip_file.open(gltf_f, "r")

def edit_gltf_entry(img, mat_name, textures_file_map):
    """
    Edit a gltf entry to point to a specific texture.
    """
    map_type = img['uri']

    map_file, map_ext = textures_file_map[mat_name][map_type]
    map_name = mat_name + "__" + map_type

    img['mimeType'] = "image/" + map_ext[1:]
    img['name'] = map_name
    img['uri'] = map_file

def apply_style(gltf_f, style_entry, textures_file_map, shape_part_remap=None):
    """
    Apply a style to a gltf file.
    """
    # Replacing texture paths in the GLTF file
    gltf_data = json.load(gltf_f)
    for img in gltf_data['images']:
        part_name = img['name']
        if shape_part_remap:
            part_name = shape_part_remap[part_name]
        mat_name = style_entry[part_name]
        edit_gltf_entry(img, mat_name, textures_file_map)

    # Get a file stream from the gltf_data
    gltf_str = json.dumps(gltf_data)
    return trimesh.util.wrap_as_stream(gltf_str)

def apply_placeholder(gltf_f, textures_file_map):
    """
    Apply a placeholder style to a gltf file.
    """
    # Replacing texture paths in the GLTF file
    gltf_data = json.load(gltf_f)
    for img in gltf_data['images']:
        edit_gltf_entry(img, 'placeholder', textures_file_map)

    # Get a file stream from the gltf_data
    gltf_str = json.dumps(gltf_data)
    return trimesh.util.wrap_as_stream(gltf_str)


def load_style_json(split, comp_k, sem_level, zip_file, styles_dir):
    """
    Load a json file with all the styles for a split, level and composition.
    """
    styles_f = styles_dir + "%s/comp_%s_%d.json" % (split, sem_level, comp_k)
    return json.load(zip_file.open(styles_f))

def load_styles(shape_id, style_id, split, comp_k, sem_level, zip_file, styles_dir):
    """
    Load a specific style for a given model.
    """
    styles_f = load_style_json(split,
                               comp_k,
                               sem_level,
                               zip_file,
                               styles_dir)
    return styles_f[shape_id + '__' + style_id]
