"""
Generating style JSONs for renders and segmentation masks.
"""

import argparse
import json
import os
import random
import sys
import tqdm

import utils.semantic_levels as sem_levels
import utils.style_combinations as style_comb
from itertools import chain


def parse_args(argv):
    """
    Parsing input arguments.
    """

    # Arguments
    parser = argparse.ArgumentParser(
        description="Generating style JSONs for input 3D models."
    )

    # data
    parser.add_argument(
        "--model-list",
        type=str,
        required=True,
        help="List of models IDs to generate styles for, in JSON format",
    )
    parser.add_argument(
        "--output-folder", type=str, required=True, help="Output folder name"
    )
    parser.add_argument(
        "--meta-folder",
        type=str,
        default="metadata",
        help="Input metadata folder (default=%(default)s)",
    )

    # styles
    parser.add_argument(
        "--styles-count",
        type=int,
        default=1,
        help="Target average number of styles per model to generate (default=%(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Starting random seed (default=%(default)s)"
    )

    # model mode
    parser.add_argument(
        "--model-mode",
        action="store_true",
        help="Generate PBR GLTF models (default=%(default)s)",
    )
    parser.add_argument(
        "--bake-mode",
        action="store_true",
        help="Sample all materials to fetch Blender baked textures (default=%(default)s)",
    )
    parser.add_argument(
        "--seg-mode",
        action="store_true",
        help="Generate segmentation views (empty styles) (default=%(default)s)",
    )

    # semantic level
    parser.add_argument(
        "--semantic-level",
        type=str,
        required=True,
        help="Semantic level (default=%(default)s)",
        choices=["fine", "medium", "coarse"],
    )

    args = parser.parse_args(argv)

    # Fixing random seed
    random.seed(args.seed)

    # Defining semantic level
    args.semantic_level = sem_levels.SemanticLevel(args.semantic_level)

    # Printing input arguments
    print("Input arguments:")
    print(args)

    return args


def sample_mat(category_map, mat_list):
    """
    Randomly sampling a material from a category of materials.
    """
    # If there are more than one material category, randomly sample one
    if len(mat_list) > 1:
        mat_cat = random.choice(mat_list)
    else:
        mat_cat = mat_list[0]
    return random.choice(category_map[mat_cat])


def gen_styles(model_id, style_count, part_mat_map, category_map):
    """
    Generating a style entry for a given model.
    """
    styles = []
    sampled_styles = set()
    for k in range(style_count):
        entry = {}
        while len(entry) == 0 or tuple(entry.items()) in sampled_styles:
            for part, mat_list in part_mat_map[model_id].items():
                entry[part] = sample_mat(category_map, mat_list)
        sampled_styles.add(tuple(entry.items()))
        styles.append({model_id + "__" + str(k): entry})

    return styles


def enum_styles(model_id, part_mat_map, category_map, sample_single=False):
    """
    Enumerate all materials on a given model.
    """
    k = 0
    # Concatenate all materials into a single list
    mat_list = list(chain.from_iterable(category_map.values()))
    mat_list.sort()
    mat_it = iter(mat_list)

    styles = {}
    all_seen = False
    while not all_seen:
        entry = {}
        for part, _ in part_mat_map[model_id].items():
            try:
                entry[part] = next(mat_it)
            except StopIteration:
                entry[part] = mat_list[-1]
                all_seen = True
        styles[model_id + "__" + str(k)] = entry
        k += 1
        if sample_single:
            return entry

    return styles


def main(argv):
    # Parsing arguments
    args = parse_args(argv[1:])

    # Opening input metadata
    model_list = json.load(open(args.model_list))
    part_mat_map = args.semantic_level.get_part_mat_map(args.meta_folder)
    category_map = json.load(
        open(os.path.join(args.meta_folder, "mat_categories.json"))
    )
    model_list.sort()

    # Generating JSON files
    output_folder = os.path.realpath(args.output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if args.bake_mode:
        # Sampling all materials to fetch Blender baked textures
        model_id = model_list[0]
        dummy_samples = enum_styles(model_id, part_mat_map, category_map)
        json.dump(
            dummy_samples,
            open(os.path.join(output_folder, "bake_shuffle.json"), "w"),
            indent=4,
        )
    elif args.model_mode:
        # Sampling dummy materials for PBR GLTF exports
        pbr_models = {}
        for model_id in tqdm.tqdm(model_list):
            pbr_models[model_id + "__0"] = enum_styles(
                model_id, part_mat_map, category_map, sample_single=True
            )
        json.dump(
            pbr_models,
            open(os.path.join(output_folder, "pbr_models.json"), "w"),
            indent=4,
        )
    elif args.seg_mode:
        # Initializing empty materials for each part
        seg_models = {}
        for model_id in tqdm.tqdm(model_list):
            for style_id in range(args.styles_count):
                seg_models[model_id + "__%d" % style_id] = {
                    part: "" for part in part_mat_map[model_id]
                }
        out_path = os.path.join(
            output_folder, "seg_models_%s.json" % str(args.semantic_level)
        )
        json_out = open(out_path, "w")
        json.dump(seg_models, json_out, indent=4)
    else:
        """
        Sampling styles for 2D renderings and 3D on-the-fly exports
        """
        # Sorting model list et fetching part map
        model_list = sorted(model_list)
        part_mat_map = {k: part_mat_map[k] for k in model_list}

        # Compute the maximum number of combinations
        # and adjusted number of combinations
        max_combs = style_comb.max_combs(part_mat_map, category_map)
        adj_combs = style_comb.adjusted_combs(max_combs, args.styles_count)

        # Sample styles for each model
        style_list = []
        for model_id in adj_combs:
            style_list.append(
                gen_styles(model_id, adj_combs[model_id], part_mat_map, category_map)
            )
        packed_styles = style_comb.pack_list(
            style_list, T=args.styles_count, shuffle=False
        )

        # Pack into a list of dicts
        packed_styles_dict = []
        for m in packed_styles:
            style_dict = {}
            for d in m:
                style_dict |= d
            # sort keys
            style_dict = {k: style_dict[k] for k in sorted(style_dict.keys())}
            packed_styles_dict.append(style_dict)

        # Dump each composition to a separate JSON file
        for k, style_dict in enumerate(packed_styles_dict):
            out_path = os.path.join(
                output_folder, "comp_%s_%d.json" % (str(args.semantic_level), k)
            )
            json.dump(style_dict, open(out_path, "w"), indent=4)

    print("All [%d] style JSONs generated in: [%s]." % (len(model_list), output_folder))


if __name__ == "__main__":
    main(sys.argv)
