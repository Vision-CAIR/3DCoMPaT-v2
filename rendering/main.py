"""
Main rendering script for all data modalities.
"""
import argparse
import os
import shutil
import sys

import bpy
import utils.render as render
import utils.semantic_levels as sem_levels
import utils.zip_write as zip_write
from blender import MassRenderer, ModelExporter

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def parse_args(argv):
    """
    Parse input arguments.
    """

    # Arguments
    parser = argparse.ArgumentParser(description="Rendering 3DCoMPaT models.")

    # data
    parser.add_argument("--data-path", type=str, required=True, help="I/O data folder")
    parser.add_argument(
        "--output-folder", type=str, required=True, help="Output folder name"
    )
    parser.add_argument(
        "--meta-folder",
        type=str,
        default="metadata",
        help="Input metadata folder (default=%(default)s)",
    )
    parser.add_argument(
        "--models-path", type=str, required=True, help="Path to the GLTF models"
    )

    # mode
    parser.add_argument(
        "--debug-mode", action="store_true", help="Debug mode (default=%(default)s)"
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        required=True,
        help="Rendering mode (default=%(default)s)",
        choices=["rendering", "model", "segmentation"],
    )
    parser.add_argument(
        "--semantic-level",
        type=str,
        required=True,
        help="Semantic level (default=%(default)s)",
        choices=["fine", "medium", "coarse"],
    )

    # rendering
    parser.add_argument(
        "--blender-config-dir",
        type=str,
        required=True,
        help="Blender .blend configuration directory",
    )
    parser.add_argument(
        "--use-eevee",
        action="store_true",
        help="Use the Eevee render engine (default=%(default)s)",
    )
    parser.add_argument(
        "--texture-dir",
        type=str,
        default="textures",
        help="Texture directory for GLTFs",
    )
    parser.add_argument(
        "--gltf-mode",
        type=str,
        default="GLTF_SEPARATE",
        help="GLTF mode to use for exporting models (default=%(default)s)",
    )

    # styles
    parser.add_argument(
        "--json-file", type=str, required=True, help="JSON style file chunk"
    )

    args = parser.parse_args(argv)

    # Defining render mode
    args.mode = render.RenderMode(args.render_mode)
    args.rendering_mode = args.mode == render.RenderMode.RENDERING
    args.seg_mode = args.mode == render.RenderMode.SEGMENTATION
    args.model_mode = args.mode == render.RenderMode.MODEL

    # Defining semantic level
    args.semantic_level = sem_levels.SemanticLevel(args.semantic_level)

    # Loading metadata files
    args.parts_remap = args.semantic_level.get_remap(args.meta_folder)
    args.parts_list = args.semantic_level.get_parts(args.meta_folder)
    args.mat_cats = sem_levels.open_meta(args.meta_folder, "mat_categories.json")

    # Defining blender config
    args.blender_config = "3DCoMPaT_" + str(args.mode)

    # Printing input arguments
    print("Input arguments:")
    print(args)

    return args


def main(argv):
    # Parsing arguments
    argv = argv[argv.index("--") + 1 :]
    args = parse_args(argv)

    # Initializing outputs
    job_id = os.path.basename(args.json_file).split(".")[0]

    if args.seg_mode:
        job_id = "seg_" + job_id

    # Initializing zip outputs
    zip_output, zip_path, tmp_dir = zip_write.init_outputs(job_id, args.output_folder)

    # Copying blender config
    blender_templates = os.path.join(
        bpy.utils.script_path_user(), "startup", "bl_app_templates_user"
    )

    if args.debug_mode or not os.path.exists(blender_templates):
        os.makedirs(blender_templates, exist_ok=True)
        for f in os.listdir(args.blender_config_dir):
            shutil.copytree(
                os.path.join(args.blender_config_dir, f),
                os.path.join(blender_templates, f),
                dirs_exist_ok=True,
            )

    # Initializing the method
    if args.model_mode:
        method = ModelExporter(
            args, zip_output=zip_output, zip_path=zip_path, tmp_dir=tmp_dir
        )
    else:
        method = MassRenderer(
            args, zip_output=zip_output, zip_path=zip_path, tmp_dir=tmp_dir
        )

    method.run()
    zip_output.close()


if __name__ == "__main__":
    main(sys.argv)
