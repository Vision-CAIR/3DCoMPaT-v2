#!/usr/bin/env python3
"""
Script to visualize 3DCoMPaT models with part and material annotations either in surface mesh or sampled point cloud mode in an interactive viewer.

assumes the following files stored under data-root/:
3DCoMPaT_ZIP.zip : the original dataset zip archive
- models/ : GLTF mesh files named <shape_id>.gltf
- textures/ : texture image files referenced by the GLTF files
- styles/ : style definition JSON files organized in subdirectories per split (train, valid)
- textures_map.json : mapping of texture names to image files

Example usage:
    python visualize_3dcompat.py --shape-id 00_00a --html 00_00a.html //
    python visualize_3dcompat.py --shape-id 00_00a --polyscope //
    python visualize_3dcompat.py --shape-id 00_00a --source dir --polyscope //
    python visualize_3dcompat.py --shape-id 00_00a --source zip --polyscope //

Additional package dependencies (install via pip):
- plotly (for HTML export)
- polyscope (for Polyscope viewer)

"""
import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.colors import hsv_to_rgb

# Ensure we can import the loader utilities without modifying PYTHONPATH globally.
BASE_DIR = Path(__file__).resolve().parent
UTILS_DIR = BASE_DIR / "loaders" / "3D"
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

import utils3D.gltf as gltf_utils  # noqa: E402
import utils3D.pointcloud as pc_utils  # noqa: E402
from utils3D.semantic_levels import SemanticLevel  # noqa: E402


_DISTINCT_PART_COLOR_HEX = [
    "#E58606",
    "#5D69B1",
    "#52BCA3",
    "#99C945",
    "#CC61B0",
    "#24796C",
    "#DAA51B",
    "#2F8AC4",
    "#764E9F",
    "#ED645A",
    "#A5AA99",
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
    "#3366CC",
    "#DC3912",
    "#FF9900",
    "#109618",
    "#990099",
    "#0099C6",
    "#DD4477",
    "#66AA00",
    "#B82E2E",
    "#316395",
    "#4C78A8",
    "#F58518",
    "#E45756",
    "#72B7B2",
    "#54A24B",
    "#EECA3B",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#AA0DFE",
    "#3283FE",
    "#85660D",
    "#782AB6",
    "#565656",
    "#1C8356",
    "#16FF32",
    "#F7E1A0",
    "#E2E2E2",
    "#1CBE4F",
    "#C4451C",
    "#DEA0FD",
    "#FE00FA",
    "#325A9B",
    "#FEAF16",
    "#F8A19F",
    "#90AD1C",
    "#F6222E",
    "#1CFFCE",
    "#2ED9FF",
    "#B10DA1",
    "#C075A6",
    "#FC1CBF",
    "#B00068",
    "#FBE426",
    "#FA0087",
    "#2E91E5",
    "#E15F99",
    "#1CA71C",
    "#FB0D0D",
    "#DA16FF",
    "#222A2A",
    "#B68100",
    "#750D86",
    "#EB663B",
    "#511CFB",
    "#00A08B",
    "#FB00D1",
    "#FC0080",
    "#B2828D",
    "#6C7C32",
    "#778AAE",
    "#862A16",
    "#A777F1",
    "#620042",
    "#1616A7",
    "#DA60CA",
    "#6C4516",
    "#0D2A63",
    "#AF0038",
    "#FD3216",
    "#00FE35",
    "#6A76FC",
    "#FED4C4",
    "#FE00CE",
    "#0DF9FF",
    "#F6F926",
    "#FF9616",
    "#479B55",
    "#EEA6FB",
    "#DC587D",
    "#D626FF",
    "#6E899C",
    "#00B5F7",
    "#B68E00",
    "#C9FBE5",
    "#FF0092",
    "#22FFA7",
    "#E3EE9E",
    "#86CE00",
    "#BC7196",
    "#7E7DCD",
    "#FC6955",
    "#E48F72",
    "#7F3C8D",
    "#11A579",
    "#3969AC",
    "#F2B701",
    "#E73F74",
    "#80BA5A",
    "#E68310",
    "#008695",
    "#CF1C90",
    "#F97B72",
    "#88CCEE",
    "#CC6677",
    "#DDCC77",
    "#117733",
    "#332288",
    "#AA4499",
    "#44AA99",
    "#999933",
    "#882255",
    "#661100",
    "#888888",
    "#855C75",
    "#D9AF6B",
    "#AF6458",
    "#736F4C",
    "#526A83",
    "#625377",
    "#68855C",
    "#9C9C5E",
    "#A06177",
    "#8C7853",
    "#7C7C7C",
    "#8DD3C7",
    "#FFFFB3",
    "#BEBADA",
    "#FB8072",
    "#80B1D3",
    "#FDB462",
    "#B3DE69",
    "#FCCDE5",
    "#D9D9D9",
    "#BC80BD",
    "#CCEBC5",
    "#FFED6F",
]

_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949


def _normalize_key(key):
    if isinstance(key, np.generic):
        return key.item()
    return key


def _stable_hash(value):
    return int(hashlib.sha1(repr(value).encode("utf-8")).hexdigest(), 16)


def _hex_to_rgb_array(hex_color):
    stripped = hex_color.lstrip("#")
    if len(stripped) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    r = int(stripped[0:2], 16)
    g = int(stripped[2:4], 16)
    b = int(stripped[4:6], 16)
    return np.array([r, g, b], dtype=np.float32) / 255.0


def _generate_extra_color(index):
    hue = (0.15 + index * _GOLDEN_RATIO_CONJUGATE) % 1.0
    saturation = 0.55 + 0.25 * ((index * 37) % 100) / 100.0
    value = 0.75 + 0.2 * ((index * 53) % 100) / 100.0
    color = hsv_to_rgb([hue, min(saturation, 1.0), min(value, 1.0)])
    return np.asarray(color, dtype=np.float32)


def _color_from_lookup(lookup, key, cmap_name="tab20"):
    key_norm = _normalize_key(key)
    if lookup and key_norm in lookup:
        return np.asarray(lookup[key_norm], dtype=np.float32)
    cmap = plt.get_cmap(cmap_name)
    idx = hash(key_norm) % cmap.N
    return np.asarray(cmap(idx)[:3], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a 3DCoMPaT GLTF mesh with semantic part labels and materials.")
    parser.add_argument("--zip-path", default=str(BASE_DIR / "data" / "3DCoMPaT_ZIP.zip"), help="Path to the 3DCoMPaT zip archive (contains models, textures, styles).")
    parser.add_argument("--data-root", default=str(BASE_DIR / "data"), help="Path to the unzipped 3DCoMPaT data directory (containing models/, textures/, styles/).")
    parser.add_argument("--metadata-dir", default=str(BASE_DIR / "metadata"), help="Path to the metadata directory (parts, materials, splits).")
    parser.add_argument("--shape-id", required=True, help="Shape identifier, e.g. 00_00a.")
    parser.add_argument("--style-id", help="Style identifier to apply (matches the comp id, e.g. 373). If omitted, the first available style for the shape is used.")
    parser.add_argument("--split", default="train", choices=["train", "valid"], help="Data split to look up style definitions.")
    parser.add_argument("--semantic-level", default="fine", choices=["fine", "medium", "coarse"], help="Segmentation level to visualize.")
    parser.add_argument("--points", type=int, default=20000, help="Number of points to sample from the mesh surface.")
    parser.add_argument("--source", choices=["auto", "zip", "dir"], default="auto", help="Preferred data source: auto selects unzipped data when available, otherwise falls back to the ZIP.")
    parser.add_argument("--html", help="Optional path to export an interactive HTML viewer of the stylized mesh.")
    parser.add_argument("--polyscope", action="store_true", help="Launch an interactive Polyscope window with multiple color quantities.")
    parser.add_argument("--legend-max", type=int, default=15, help="Maximum number of legend entries per legend panel (additional entries are summarised).")
    return parser.parse_args()


def _find_style_zip(zip_f, split, semantic_level, shape_id, style_id=None):
    styles_dir = f"styles/{split}/"
    if style_id is not None:
        comp_k = int(style_id)
        style_key = f"{shape_id}__{style_id}"
        file_name = f"{styles_dir}comp_{semantic_level}_{comp_k}.json"
        try:
            styles_json = json.load(zip_f.open(file_name))
        except KeyError as exc:
            raise FileNotFoundError(f"Style file not found in zip: {file_name}") from exc
        if style_key not in styles_json:
            raise KeyError(f"Style {style_key} not found in {file_name}.")
        return comp_k, styles_json[style_key]

    semantic_tag = f"comp_{semantic_level}_"
    for name in zip_f.namelist():
        if not name.startswith(styles_dir) or not name.endswith(".json"):
            continue
        if semantic_tag not in name:
            continue
        styles_json = json.load(zip_f.open(name))
        for key, entry in styles_json.items():
            if key.startswith(f"{shape_id}__"):
                sid = key.split("__")[1]
                return int(sid), entry
    raise RuntimeError(
        f"Could not automatically locate a style for shape {shape_id} in split {split}."
        " Provide --style-id explicitly."
    )


def _find_style_dir(data_root, split, semantic_level, shape_id, style_id=None):
    styles_dir = Path(data_root) / "styles" / split
    if not styles_dir.exists():
        raise FileNotFoundError(f"Styles directory not found: {styles_dir}")

    if style_id is not None:
        comp_k = int(style_id)
        style_key = f"{shape_id}__{style_id}"
        file_path = styles_dir / f"comp_{semantic_level}_{comp_k}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Style file not found: {file_path}")
        styles_json = json.load(open(file_path))
        if style_key not in styles_json:
            raise KeyError(f"Style {style_key} not present in {file_path}")
        return comp_k, styles_json[style_key]

    pattern = f"comp_{semantic_level}_*.json"
    for file_path in sorted(styles_dir.glob(pattern)):
        styles_json = json.load(open(file_path))
        for key, entry in styles_json.items():
            if key.startswith(f"{shape_id}__"):
                sid = key.split("__")[1]
                return int(sid), entry
    raise RuntimeError(
        f"Could not automatically locate a style for shape {shape_id} in split {split} under {styles_dir}."
        " Provide --style-id explicitly."
    )


class DirectoryTextureResolver(trimesh.resolvers.FilePathResolver):
    """
    Resolve texture files from an unpacked dataset directory.
    """

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)

    def get(self, file_path):
        file_path = Path(file_path)
        candidate = self.base_dir / file_path
        if not candidate.exists():
            raise FileNotFoundError(f"Missing texture file: {candidate}")
        return candidate.read_bytes()


def _load_mesh_zip(zip_f, shape_id, style_entry, textures_map, shape_part_remap):
    gltf_f = gltf_utils.load_gltf(
        model_id=shape_id,
        zip_file=zip_f,
        models_dir="models/",
    )
    if style_entry is not None:
        gltf_stream = gltf_utils.apply_style(
            gltf_f,
            style_entry,
            textures_file_map=textures_map,
            shape_part_remap=shape_part_remap,
        )
    else:
        gltf_stream = gltf_utils.apply_placeholder(
            gltf_f,
            textures_file_map=textures_map,
        )
    return trimesh.load(
        gltf_stream,
        file_type=".gltf",
        force="scene",
        resolver=gltf_utils.ZipTextureResolver(zip_f=zip_f),
    )


def _load_mesh_dir(data_root, shape_id, style_entry, textures_map, shape_part_remap):
    data_root = Path(data_root)
    gltf_path = data_root / "models" / f"{shape_id}.gltf"
    if not gltf_path.exists():
        raise FileNotFoundError(f"GLTF file not found: {gltf_path}")

    with open(gltf_path, "r") as gltf_f:
        if style_entry is not None:
            gltf_stream = gltf_utils.apply_style(
                gltf_f,
                style_entry,
                textures_file_map=textures_map,
                shape_part_remap=shape_part_remap,
            )
        else:
            gltf_stream = gltf_utils.apply_placeholder(
                gltf_f,
                textures_file_map=textures_map,
            )

    resolver = DirectoryTextureResolver(base_dir=data_root)
    return trimesh.load(gltf_stream, file_type=".gltf", force="scene", resolver=resolver)


def _labels_to_colors(labels, cmap_name=None, *, palette=None, disperse=True):
    if len(labels) == 0:
        return np.zeros((0, 3), dtype=np.float32), {}

    unique_labels = sorted(set(labels))
    assignment_order = (sorted(unique_labels, key=_stable_hash) if disperse else unique_labels)

    palette_rgb = None
    if palette is not None:
        palette_rgb = [_hex_to_rgb_array(color) for color in palette]

    cmap = None
    if cmap_name is not None:
        cmap = plt.get_cmap(cmap_name, max(1, len(unique_labels)))

    lookup = {}
    extra_idx = 0

    for idx, label in enumerate(assignment_order):
        color = None
        if palette_rgb is not None and idx < len(palette_rgb):
            color = palette_rgb[idx]
        elif palette_rgb is not None:
            color = _generate_extra_color(extra_idx)
            extra_idx += 1
        elif cmap is not None:
            color = np.asarray(cmap(idx % cmap.N)[:3], dtype=np.float32)
        else:
            color = _generate_extra_color(extra_idx)
            extra_idx += 1
        lookup[label] = np.asarray(color, dtype=np.float32)

    colors = np.array([lookup[label] for label in labels], dtype=np.float32)
    return colors, lookup


def _rgb_to_hex(rgb):
    rgb = np.clip(np.asarray(rgb), 0.0, 1.0)
    return "#{:02x}{:02x}{:02x}".format(
        int(round(rgb[0] * 255)),
        int(round(rgb[1] * 255)),
        int(round(rgb[2] * 255)),
    )


def _colors_to_hex_list(colors):
    return [_rgb_to_hex(color) for color in np.asarray(colors)]


def _legend_annotations(title, entries, x_position, legend_max, center_y=0.5, step=0.035):
    annotations = []
    if not entries:
        return annotations

    shown = entries[:legend_max]
    extra = max(len(entries) - legend_max, 0)
    total_lines = 1 + len(shown) + (1 if extra else 0)
    start_y = center_y + (total_lines - 1) * step / 2.0
    start_y = min(start_y, 0.95)
    start_y = max(start_y, 0.05)

    annotations.append(
        dict(
            text=f"<b>{title}</b>",
            x=x_position,
            y=start_y,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="right",
            font=dict(size=12),
        )
    )

    for idx, entry in enumerate(shown, start=1):
        y = start_y - idx * step
        annotations.append(
            dict(
                text=f"<span style='color:{entry['color_hex']}'>■</span> {entry['label']}",
                x=x_position,
                y=y,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="right",
                font=dict(size=10),
            )
        )

    if extra:
        y = start_y - (len(shown) + 1) * step
        annotations.append(
            dict(
                text=f"+{extra} more",
                x=x_position,
                y=y,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="right",
                font=dict(size=10, color="#666666"),
            )
        )
    return annotations


def _prepare_surface_data(
    mesh_map,
    part_material_map,
    part_color_lookup,
    material_color_lookup,
    *,
    part_to_coarse_idx=None,
    coarse_color_lookup=None,
    coarse_default_color=(0.6, 0.6, 0.6),
    part_stylized_lookup=None,
    stylized_default_color=(0.5, 0.5, 0.5),
):
    meshes = []
    face_part_colors = []
    face_material_colors = []
    face_part_ids = []
    face_material_names = []
    face_coarse_colors = [] if part_to_coarse_idx and coarse_color_lookup else None
    face_coarse_ids = [] if face_coarse_colors is not None else None
    coarse_default = np.asarray(coarse_default_color, dtype=np.float32)
    face_stylized_colors = [] if part_stylized_lookup else None
    stylized_default = np.asarray(stylized_default_color, dtype=np.float32)

    for part_id, mesh in mesh_map.items():
        mesh_copy = mesh.copy()
        n_faces = len(mesh_copy.faces)
        if n_faces == 0:
            continue

        part_color = _color_from_lookup(part_color_lookup, part_id)
        mat_name = part_material_map.get(_normalize_key(part_id), "unknown")
        mat_color = _color_from_lookup(material_color_lookup, mat_name)

        face_part_colors.append(np.tile(part_color, (n_faces, 1)))
        face_material_colors.append(np.tile(mat_color, (n_faces, 1)))
        face_part_ids.append(np.full(n_faces, _normalize_key(part_id), dtype=np.int32))
        face_material_names.extend([mat_name] * n_faces)
        if face_coarse_colors is not None and face_coarse_ids is not None:
            coarse_idx = part_to_coarse_idx.get(_normalize_key(part_id))
            if coarse_idx is None or coarse_idx not in coarse_color_lookup:
                coarse_color = coarse_default
                coarse_face_id = -1
            else:
                coarse_color = _color_from_lookup(coarse_color_lookup, coarse_idx)
                coarse_face_id = int(coarse_idx)
            face_coarse_colors.append(np.tile(coarse_color, (n_faces, 1)))
            face_coarse_ids.append(np.full(n_faces, coarse_face_id, dtype=np.int32))
        if face_stylized_colors is not None:
            stylized_color = part_stylized_lookup.get(_normalize_key(part_id))
            if stylized_color is None:
                stylized_color = stylized_default
            face_stylized_colors.append(np.tile(np.asarray(stylized_color, dtype=np.float32), (n_faces, 1)))
        meshes.append(mesh_copy)

    if not meshes:
        return None

    combined_mesh = trimesh.util.concatenate(meshes)
    face_part_colors = np.vstack(face_part_colors)
    face_material_colors = np.vstack(face_material_colors)
    face_part_ids = np.concatenate(face_part_ids)

    unique_materials = sorted(set(face_material_names))
    material_to_idx = {name: idx for idx, name in enumerate(unique_materials)}
    face_material_ids = np.array([material_to_idx[name] for name in face_material_names], dtype=np.int32)

    result = {
        "mesh": combined_mesh,
        "face_part_colors": face_part_colors,
        "face_material_colors": face_material_colors,
        "face_part_ids": face_part_ids,
        "face_material_ids": face_material_ids,
        "material_to_idx": material_to_idx,
    }
    if face_coarse_colors is not None and face_coarse_ids is not None:
        result["face_coarse_colors"] = np.vstack(face_coarse_colors)
        result["face_coarse_ids"] = np.concatenate(face_coarse_ids)
    if face_stylized_colors is not None:
        result["face_stylized_colors"] = np.vstack(face_stylized_colors)
    return result


def main():
    args = parse_args()
    zip_path = Path(args.zip_path)
    data_root = Path(args.data_root)
    metadata_dir = Path(args.metadata_dir)
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

    if args.source == "zip":
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip archive not found: {zip_path}")
        source_mode = "zip"
    elif args.source == "dir":
        if not data_root.exists():
            raise FileNotFoundError(f"Data directory not found: {data_root}")
        source_mode = "dir"
    else:
        prefer_dir = (
            data_root.exists()
            and (data_root / "models" / f"{args.shape_id}.gltf").exists()
            and (data_root / "textures_map.json").exists()
        )
        if prefer_dir:
            source_mode = "dir"
        elif zip_path.exists():
            source_mode = "zip"
        else:
            raise FileNotFoundError(
                "Unable to locate data. Provide a valid --data-root or --zip-path."
            )

    semantic_level = SemanticLevel(args.semantic_level)
    semantic_level_label = args.semantic_level.replace("_", " ")
    semantic_level_title = semantic_level_label.title()
    parts_list = semantic_level.get_parts(metadata_dir)
    part_to_idx = {name: idx for idx, name in enumerate(parts_list)}
    part_id_to_name = {idx: name for name, idx in part_to_idx.items()}

    shape_label = int(args.shape_id[:2], 16)
    remap_dict = semantic_level.get_remap(metadata_dir)
    shape_part_remap = remap_dict.get(shape_label) if remap_dict is not None else None

    zip_f = None
    try:
        if source_mode == "zip":
            zip_f = zipfile.ZipFile(zip_path, "r")
            try:
                textures_map = json.load(zip_f.open("textures_map.json"))
            except KeyError as exc:
                raise FileNotFoundError("textures_map.json missing from the dataset archive.") from exc
            comp_k, style_entry = _find_style_zip(zip_f, split=args.split, semantic_level=str(semantic_level),
                                                  shape_id=args.shape_id, style_id=args.style_id)
            mesh_scene = _load_mesh_zip(zip_f, shape_id=args.shape_id, style_entry=style_entry, 
                                        textures_map=textures_map, shape_part_remap=shape_part_remap)
        else:
            textures_map_path = data_root / "textures_map.json"
            if not textures_map_path.exists():
                raise FileNotFoundError(f"textures_map.json not found in {data_root}")
            textures_map = json.load(open(textures_map_path))
            comp_k, style_entry = _find_style_dir(data_root, split=args.split, semantic_level=str(semantic_level),
                                                  shape_id=args.shape_id, style_id=args.style_id)
            mesh_scene = _load_mesh_dir(data_root, shape_id=args.shape_id, style_entry=style_entry,
                                        textures_map=textures_map, shape_part_remap=shape_part_remap)

        mesh_map = pc_utils.map_meshes(mesh_scene, part_to_idx=part_to_idx, part_remap=shape_part_remap)
        sample = pc_utils.sample_pointcloud(in_mesh=mesh_map, n_points=args.points, sample_color=True,
                                            shape_only=False, get_normals=False)
    finally:
        if zip_f is not None:
            zip_f.close()

    p_xyz, p_seg, p_col = sample
    styled_colors = np.clip(p_col / 255.0, 0.0, 1.0)
    part_colors, part_color_lookup = _labels_to_colors(p_seg.tolist(), cmap_name="hsv", 
                                                       palette=_DISTINCT_PART_COLOR_HEX)
    part_color_lookup = {_normalize_key(k): np.asarray(v, dtype=np.float32) for k, v in part_color_lookup.items()}

    unique_part_ids = np.unique(p_seg)
    part_stylized_lookup = {}
    for pid in unique_part_ids.tolist():
        mask = p_seg == pid
        if np.any(mask):
            part_stylized_lookup[int(pid)] = np.asarray(styled_colors[mask].mean(axis=0), dtype=np.float32)

    coarse_point_ids = None
    coarse_point_colors = None
    coarse_color_lookup = {}
    coarse_legend = []
    part_to_coarse_idx = {}
    coarse_idx_to_name = {}

    try:
        coarse_level = SemanticLevel("coarse")
        coarse_parts_list = coarse_level.get_parts(metadata_dir)
        coarse_part_to_idx = {name: idx for idx, name in enumerate(coarse_parts_list)}
        coarse_idx_to_name = {idx: name for name, idx in coarse_part_to_idx.items()}
        coarse_remap_dict = coarse_level.get_remap(metadata_dir)
        coarse_shape_remap = coarse_remap_dict.get(shape_label, {}) if coarse_remap_dict else {}
    except FileNotFoundError:
        coarse_part_to_idx = None
        coarse_shape_remap = {}

    if coarse_part_to_idx:
        for pid, part_name in part_id_to_name.items():
            coarse_name = coarse_shape_remap.get(part_name, part_name)
            coarse_idx = coarse_part_to_idx.get(coarse_name)
            if coarse_idx is not None:
                part_to_coarse_idx[pid] = coarse_idx

        if part_to_coarse_idx:
            coarse_point_ids = np.array(
                [part_to_coarse_idx.get(int(pid), -1) for pid in p_seg.tolist()],
                dtype=np.int32,
            )
            valid_mask = coarse_point_ids >= 0
            if np.any(valid_mask):
                _, coarse_color_lookup_raw = _labels_to_colors(
                    coarse_point_ids[valid_mask].tolist(),
                    cmap_name="hsv",
                    palette=_DISTINCT_PART_COLOR_HEX,
                )
                coarse_color_lookup = {
                    _normalize_key(k): np.asarray(v, dtype=np.float32)
                    for k, v in coarse_color_lookup_raw.items()
                }
                coarse_point_colors = np.zeros_like(part_colors)
                for idx, coarse_id in enumerate(coarse_point_ids):
                    if coarse_id >= 0:
                        coarse_point_colors[idx] = _color_from_lookup(coarse_color_lookup, int(coarse_id))
                    else:
                        coarse_point_colors[idx] = np.array([0.6, 0.6, 0.6], dtype=np.float32)
                coarse_counts = Counter(coarse_point_ids[valid_mask].tolist())
                sorted_coarse_ids = sorted(
                    coarse_counts.keys(),
                    key=lambda cid: coarse_counts[cid],
                    reverse=True,
                )
                for cid in sorted_coarse_ids:
                    cid_int = int(cid)
                    coarse_name = coarse_idx_to_name.get(cid_int, str(cid_int))
                    color = _color_from_lookup(coarse_color_lookup, cid_int)
                    coarse_legend.append(
                        {
                            "label": f"{coarse_name} ({cid_int})",
                            "color_rgb": color,
                            "color_hex": _rgb_to_hex(color),
                        }
                    )
            else:
                coarse_point_ids = None
                part_to_coarse_idx = {}
                coarse_color_lookup = {}

    if style_entry is not None:
        part_material_map = {
            part_to_idx[name]: material for name, material in style_entry.items() if name in part_to_idx
        }
    else:
        part_material_map = {}
    point_material_labels = [part_material_map.get(int(part_id), "unknown") for part_id in p_seg.tolist()]
    material_colors, material_color_lookup = _labels_to_colors(point_material_labels, cmap_name="tab20")
    material_color_lookup = {_normalize_key(k): np.asarray(v, dtype=np.float32) for k, v in material_color_lookup.items()}
    surface_data = _prepare_surface_data(
        mesh_map,
        part_material_map,
        part_color_lookup,
        material_color_lookup,
        part_to_coarse_idx=part_to_coarse_idx if part_to_coarse_idx else None,
        coarse_color_lookup=coarse_color_lookup if coarse_color_lookup else None,
        part_stylized_lookup=part_stylized_lookup if part_stylized_lookup else None,
    )

    unique_parts, counts = np.unique(p_seg, return_counts=True)
    material_hex_lookup = {mat: _rgb_to_hex(col) for mat, col in material_color_lookup.items()}
    source_desc = args.zip_path if source_mode == "zip" else args.data_root
    print(f"\nShape {args.shape_id} | semantic level: {args.semantic_level} | style comp: {comp_k} | source: {source_mode} ({source_desc})")
    print(f"{'PartID':>6}  {'Part Name':<30}  {'Material':<20}  {'Color':<8}  {'Points':>8}")
    for part_id, count in zip(unique_parts, counts):
        part_name = part_id_to_name.get(int(part_id), f"part_{part_id}")
        material_name = part_material_map.get(int(part_id), "unknown")
        color_hex = material_hex_lookup.get(material_name, "#aaaaaa")
        print(f"{int(part_id):6d}  {part_name:<30}  {material_name:<20}  {color_hex:<8}  {int(count):8d}")

    part_counts = dict(zip(unique_parts.tolist(), counts.tolist()))
    sorted_parts = sorted(unique_parts.tolist(), key=lambda pid: part_counts[pid], reverse=True)
    part_legend = []
    for pid in sorted_parts:
        pid_int = int(pid)
        part_color = _color_from_lookup(part_color_lookup, pid_int)
        part_legend.append(
            {
                "label": f"{part_id_to_name.get(pid_int, str(pid_int))} ({pid_int})",
                "color_rgb": part_color,
                "color_hex": _rgb_to_hex(part_color),
            }
        )

    material_counts = Counter(point_material_labels)
    material_legend = []
    for mat, _ in sorted(material_counts.items(), key=lambda item: item[1], reverse=True):
        material_color = _color_from_lookup(material_color_lookup, mat)
        material_legend.append(
            {
                "label": f"{mat} ({material_counts[mat]})",
                "color_rgb": material_color,
                "color_hex": _rgb_to_hex(material_color),
            }
        )

    if len(part_legend) > args.legend_max:
        print(
            f"{semantic_level_title} parts legend truncated to top {args.legend_max} entries "
            f"(total {len(part_legend)})."
        )
    if coarse_legend and len(coarse_legend) > args.legend_max:
        print(
            f"Coarse parts legend truncated to top {args.legend_max} entries "
            f"(total {len(coarse_legend)})."
        )
    if len(material_legend) > args.legend_max:
        print(f"Materials legend truncated to top {args.legend_max} entries (total {len(material_legend)}).")

    if args.html:
        try:
            import plotly.graph_objects as go
            from plotly.offline import plot as plotly_plot
        except ImportError as exc:
            raise RuntimeError(
                "HTML export requires the `plotly` package. Install it via `pip install plotly`."
            ) from exc

        html_path = Path(args.html)
        html_path.parent.mkdir(parents=True, exist_ok=True)

        styled_hex = _colors_to_hex_list(styled_colors)
        part_hex = _colors_to_hex_list(part_colors)
        material_hex = _colors_to_hex_list(material_colors)
        coarse_hex = (_colors_to_hex_list(coarse_point_colors) if coarse_point_colors is not None else None)

        traces = []

        def _add_trace(trace):
            traces.append(trace)
            return len(traces) - 1

        idx_stylized = _add_trace(
            go.Scatter3d(
                x=p_xyz[:, 0],
                y=p_xyz[:, 1],
                z=p_xyz[:, 2],
                mode="markers",
                marker=dict(size=2, color=styled_hex, opacity=0.9),
                name="Stylized colors (points)",
                visible=False,
                showlegend=False,
            )
        )
        idx_part_points = _add_trace(
            go.Scatter3d(
                x=p_xyz[:, 0],
                y=p_xyz[:, 1],
                z=p_xyz[:, 2],
                mode="markers",
                marker=dict(size=2, color=part_hex, opacity=0.9),
                name=f"{semantic_level_title} part labels (points)",
                visible=False,
                showlegend=False,
            )
        )
        idx_coarse_points = None
        if coarse_hex is not None:
            idx_coarse_points = _add_trace(
                go.Scatter3d(
                    x=p_xyz[:, 0],
                    y=p_xyz[:, 1],
                    z=p_xyz[:, 2],
                    mode="markers",
                    marker=dict(size=2, color=coarse_hex, opacity=0.9),
                    name="Coarse part labels (points)",
                    visible=False,
                    showlegend=False,
                )
            )
        idx_material_points = _add_trace(
            go.Scatter3d(
                x=p_xyz[:, 0],
                y=p_xyz[:, 1],
                z=p_xyz[:, 2],
                mode="markers",
                marker=dict(size=2, color=material_hex, opacity=0.9),
                name="Material labels (points)",
                visible=False,
                showlegend=False,
            )
        )

        idx_stylized_surface = None
        idx_part_surface = None
        idx_material_surface = None
        idx_coarse_surface = None
        if surface_data is not None:
            mesh = surface_data["mesh"]
            faces = mesh.faces
            vertices = mesh.vertices
            part_surface_hex = _colors_to_hex_list(surface_data["face_part_colors"])
            material_surface_hex = _colors_to_hex_list(surface_data["face_material_colors"])
            coarse_surface_hex = (
                _colors_to_hex_list(surface_data["face_coarse_colors"])
                if surface_data.get("face_coarse_colors") is not None
                else None
            )
            stylized_surface_hex = (
                _colors_to_hex_list(surface_data["face_stylized_colors"])
                if surface_data.get("face_stylized_colors") is not None
                else None
            )

            if stylized_surface_hex is not None:
                idx_stylized_surface = _add_trace(
                    go.Mesh3d(
                        x=vertices[:, 0],
                        y=vertices[:, 1],
                        z=vertices[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        facecolor=stylized_surface_hex,
                        name="Stylized colors (surface)",
                        visible=False,
                        opacity=1.0,
                        showscale=False,
                        flatshading=True,
                        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.9),
                        showlegend=False,
                    )
                )
            idx_part_surface = _add_trace(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    facecolor=part_surface_hex,
                    name=f"{semantic_level_title} part labels (surface)",
                    visible=False,
                    opacity=1.0,
                    showscale=False,
                    flatshading=True,
                    lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.9),
                    showlegend=False,
                )
            )
            if coarse_surface_hex is not None:
                idx_coarse_surface = _add_trace(
                    go.Mesh3d(
                        x=vertices[:, 0],
                        y=vertices[:, 1],
                        z=vertices[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        facecolor=coarse_surface_hex,
                        name="Coarse part labels (surface)",
                        visible=False,
                        opacity=1.0,
                        showscale=False,
                        flatshading=True,
                        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.9),
                        showlegend=False,
                    )
                )
            idx_material_surface = _add_trace(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    facecolor=material_surface_hex,
                    name="Material labels (surface)",
                    visible=False,
                    opacity=1.0,
                    showscale=False,
                    flatshading=True,
                    lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.9),
                    showlegend=False,
                )
            )

        def _visibility(indices):
            vis = [False] * len(traces)
            for idx in indices:
                if idx is not None:
                    vis[idx] = True
            return vis

        part_surface_label = f"{semantic_level_title} part surfaces"
        part_surface_title = f"{args.shape_id} – {semantic_level_label} part labels (faces)"

        part_points_label = f"{semantic_level_title} part points"
        part_points_title = f"{args.shape_id} – {semantic_level_label} part labels (points)"

        button_modes = []
        if idx_stylized_surface is not None:
            button_modes.append(("Stylized surfaces", [idx_stylized_surface], f"{args.shape_id} – stylized colors (faces)"))
        if idx_part_surface is not None:
            button_modes.append((part_surface_label, [idx_part_surface], part_surface_title))
        if idx_coarse_surface is not None:
            button_modes.append(
                ("Coarse surfaces", [idx_coarse_surface], f"{args.shape_id} – coarse part labels (faces)")
            )
        if idx_material_surface is not None:
            button_modes.append(
                ("Material surfaces", [idx_material_surface], f"{args.shape_id} – material labels (faces)")
            )
        button_modes.append(
            ("Stylized points", [idx_stylized], f"{args.shape_id} – stylized colors (points)")
        )
        button_modes.append((part_points_label, [idx_part_points], part_points_title))
        if idx_coarse_points is not None:
            button_modes.append(
                ("Coarse points", [idx_coarse_points], f"{args.shape_id} – coarse part labels (points)")
            )
        button_modes.append(
            ("Material points", [idx_material_points], f"{args.shape_id} – material labels (points)")
        )

        buttons = [
            dict(label=label, method="update", args=[{"visible": _visibility(indices)}, {"title": title}])
            for label, indices, title in button_modes
        ]

        fig_html = go.Figure(data=traces)
        initial_visible = _visibility(button_modes[0][1])
        for idx, trace in enumerate(fig_html.data):
            trace.visible = initial_visible[idx]

        legend_panels = []
        if part_legend:
            legend_panels.append((f"{semantic_level_title} parts", part_legend))
        if coarse_legend:
            legend_panels.append(("Coarse parts", coarse_legend))
        if material_legend:
            legend_panels.append(("Materials", material_legend))

        annotations = []
        if legend_panels:
            if len(legend_panels) == 1:
                centers = [0.6]
            else:
                centers = np.linspace(0.75, 0.25, len(legend_panels))
            for (title, entries), center in zip(legend_panels, centers):
                annotations.extend(
                    _legend_annotations(
                        title,
                        entries,
                        1.02,
                        args.legend_max,
                        center_y=float(center),
                    )
                )
        fig_html.update_layout(
            title=button_modes[0][2],
            scene=dict(
                aspectmode="data",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="X"),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="Y"),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="Z"),
            ),
            margin=dict(l=0, r=260, t=60, b=0),
            legend=dict(itemsizing="constant"),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    buttons=buttons,
                    pad=dict(r=10, t=10),
                    showactive=True,
                    x=0.5,
                    xanchor="center",
                    y=1.15,
                    yanchor="top",
                )
            ],
            annotations=annotations,
        )

        plotly_plot(fig_html, filename=str(html_path), auto_open=False, include_plotlyjs="cdn")
        print(f"Saved interactive HTML viewer to {html_path}")

    if args.polyscope:
        try:
            import polyscope as ps
            from polyscope import imgui
        except ImportError as exc:
            raise RuntimeError(
                "Polyscope visualisation requested but the polyscope package is not installed. "
                "Install it via `pip install polyscope`."
            ) from exc

        ps.set_allow_headless_backends(True)
        ps.init()
        if ps.is_headless():
            raise RuntimeError(
                "Polyscope initialised in headless mode (no active display). Launch with a display available to use --polyscope."
            )
        cloud = ps.register_point_cloud("3DCoMPaT sample", p_xyz, enabled=False)
        point_quantities = {}
        point_quantities["stylized"] = cloud.add_color_quantity("Stylized colors", styled_colors, enabled=False)
        point_quantities["parts"] = cloud.add_color_quantity(
            f"{semantic_level_title} part labels", part_colors, enabled=False
        )
        if coarse_point_colors is not None:
            point_quantities["coarse"] = cloud.add_color_quantity("Coarse part labels", coarse_point_colors, enabled=False)
        else:
            point_quantities["coarse"] = None
        point_quantities["materials"] = cloud.add_color_quantity("Material labels", material_colors, enabled=False)

        surface = None
        surface_quantities = {}
        if surface_data is not None:
            surface_mesh = surface_data["mesh"]
            surface = ps.register_surface_mesh("3DCoMPaT mesh", surface_mesh.vertices, surface_mesh.faces, enabled=False)
            surface_quantities["parts"] = surface.add_color_quantity(
                f"{semantic_level_title} part colors (faces)",
                surface_data["face_part_colors"],
                defined_on="faces",
                enabled=False,
            )
            if surface_data.get("face_stylized_colors") is not None:
                surface_quantities["stylized"] = surface.add_color_quantity(
                    "Stylized colors (faces)",
                    surface_data["face_stylized_colors"],
                    defined_on="faces",
                    enabled=True,
                )
            else:
                surface_quantities["stylized"] = None
            if surface_data.get("face_coarse_colors") is not None:
                surface_quantities["coarse"] = surface.add_color_quantity(
                    "Coarse part colors (faces)",
                    surface_data["face_coarse_colors"],
                    defined_on="faces",
                    enabled=False,
                )
            else:
                surface_quantities["coarse"] = None
            surface_quantities["materials"] = surface.add_color_quantity(
                "Material colors (faces)",
                surface_data["face_material_colors"],
                defined_on="faces",
                enabled=False,
            )

        def _default_view(quantities):
            for key in ("stylized", "parts", "coarse", "materials"):
                if quantities.get(key) is not None:
                    return key
            return None

        display_state = {
            "mode": "surface" if surface is not None else "points",
            "surface_view": _default_view(surface_quantities) if surface is not None else None,
            "point_view": _default_view(point_quantities),
        }

        def _apply_surface_view(selection=None):
            if surface is None:
                return
            key = selection or display_state["surface_view"]
            if key is None or surface_quantities.get(key) is None:
                key = _default_view(surface_quantities)
            if key is None:
                display_state["surface_view"] = None
                return
            for name, quantity in surface_quantities.items():
                if quantity is not None:
                    quantity.set_enabled(name == key)
            display_state["surface_view"] = key
            try:
                ps.request_redraw()
            except Exception:  # pragma: no cover - fallback if request_redraw unavailable
                pass

        def _apply_point_view(selection=None):
            key = selection or display_state["point_view"]
            if key is None or point_quantities.get(key) is None:
                key = _default_view(point_quantities)
            if key is None:
                display_state["point_view"] = None
                return
            for name, quantity in point_quantities.items():
                if quantity is not None:
                    quantity.set_enabled(name == key)
            display_state["point_view"] = key
            try:
                ps.request_redraw()
            except Exception:  # pragma: no cover
                pass

        def _apply_display_mode(mode):
            """
            Enforce a single active structure and (re)apply the active view for that mode.
            This is called after any UI interaction and also once per frame to override
            manual toggles in the Polyscope 'Structures' panel.
            """
            want_surface = (mode == "surface" and surface is not None)
            # Enforce mutual exclusivity regardless of what the user did in the Structures panel.
            if surface is not None:
                surface.set_enabled(want_surface)
            cloud.set_enabled(not want_surface)
            if want_surface:
                _apply_surface_view()
            else:
                _apply_point_view()

        def _transfer_view(prev_mode, new_mode):
            if prev_mode == new_mode:
                return
            if new_mode == "surface" and surface is None:
                return
            if prev_mode == "surface":
                source_view = display_state.get("surface_view")
                if source_view is None or point_quantities.get(source_view) is None:
                    source_view = _default_view(point_quantities)
                display_state["point_view"] = source_view
            else:
                source_view = display_state.get("point_view")
                if surface is None:
                    return
                if source_view is None or surface_quantities.get(source_view) is None:
                    source_view = _default_view(surface_quantities)
                display_state["surface_view"] = source_view

        if display_state["mode"] == "surface" and surface is not None:
            _apply_surface_view(display_state["surface_view"])
        else:
            _apply_point_view(display_state["point_view"])
        _apply_display_mode(display_state["mode"])

        part_legend_ui = part_legend[: args.legend_max]
        coarse_legend_ui = coarse_legend[: args.legend_max]
        material_legend_ui = material_legend[: args.legend_max]

        def _legend_ui():
            width, height = ps.get_window_size()
            panel_width = 280
            sections = []
            if part_legend_ui:
                sections.append((f"{semantic_level_title} parts", part_legend_ui, part_legend))
            if coarse_legend_ui:
                sections.append(("Coarse parts", coarse_legend_ui, coarse_legend))
            if material_legend_ui:
                sections.append(("Materials", material_legend_ui, material_legend))
            total_entries = sum(len(entries) for _, entries, _ in sections)
            extra_rows = 10 if surface is not None else 6
            panel_height = min(max(220, 24 * (total_entries + extra_rows)), height - 40)
            x_pos = max(width - panel_width - 20, 20)
            y_pos = max(height * 0.5 - panel_height / 2, 20)
            imgui.SetNextWindowPos((float(x_pos), float(y_pos)), cond=imgui.ImGuiCond_Always)
            imgui.SetNextWindowSize((float(panel_width), float(panel_height)), cond=imgui.ImGuiCond_Always)

            opened, _ = imgui.Begin(
                "3DCoMPaT Legends",
                True,
                flags=imgui.ImGuiWindowFlags_NoCollapse | imgui.ImGuiWindowFlags_NoResize,
            )
            if opened:
                for section_idx, (title, entries_ui, entries_full) in enumerate(sections):
                    if section_idx > 0:
                        imgui.Separator()
                    imgui.Text(title)
                    for idx, entry in enumerate(entries_ui):
                        r, g, b = (
                            float(entry["color_rgb"][0]),
                            float(entry["color_rgb"][1]),
                            float(entry["color_rgb"][2]),
                        )
                        imgui.PushID(f"{title}_{idx}")
                        imgui.ColorButton(
                            "##legend_color",
                            (r, g, b, 1.0),
                            flags=imgui.ImGuiColorEditFlags_NoTooltip | imgui.ImGuiColorEditFlags_NoBorder,
                            size=(14, 14),
                        )
                        imgui.SameLine()
                        imgui.Text(entry["label"])
                        imgui.PopID()
                    if len(entries_full) > len(entries_ui):
                        imgui.TextDisabled(f"+{len(entries_full) - len(entries_ui)} more")
                if sections:
                    imgui.Separator()
                imgui.Text("Display mode")
                if surface is not None:
                    previous_mode = display_state["mode"]
                    if imgui.RadioButton("Surface mesh", display_state["mode"] == "surface"):
                        display_state["mode"] = "surface"
                    if imgui.RadioButton("Point cloud", display_state["mode"] == "points"):
                        display_state["mode"] = "points"
                    if display_state["mode"] != previous_mode:
                        _transfer_view(previous_mode, display_state["mode"])
                        _apply_display_mode(display_state["mode"])
                else:
                    imgui.RadioButton("Point cloud", True)
                imgui.End()

            _apply_display_mode(display_state["mode"])

        ps.set_user_callback(_legend_ui)
        ps.show()


if __name__ == "__main__":
    main()