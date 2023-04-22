"""
Downloading the 3DCoMPaT++ dataset.
"""
import argparse
import json
import os
import time

import tqdm

ROOT_BUCKET_URL = "https://3dcompat-dataset.s3-us-west-1.amazonaws.com/v2/"
MAX_COMPS = 100
MAX_COMPS_3D = 10


def load_json(file):
    return json.load(open(file, "r"))


def download_file(new_path, url, silent=False):
    redirect = ">/dev/null 2>&1" if silent else ""
    os.system("wget -O %s %s %s" % (new_path, url, redirect))


def parse_args(argv):
    """
    Parse input arguments.
    """

    # Arguments
    parser = argparse.ArgumentParser(description="Download the 3DCoMPaT++ dataset.")

    # data args
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output folder in which the tars should be downloaded",
    )
    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=["2D", "3D_ZIP", "3D_PC"],
        help='Data type to download. Use "2D" for WDS-packaged 2D data,'
        + '"3D_ZIP" for zip-packaged 3D shapes,"'
        + '"3D_PC" for 3D pointclouds.',
    )
    parser.add_argument(
        "--split",
        type=str,
        required=False,
        choices=["train", "test", "valid"],
        help='Split to download. Invalid for "3D_ZIP".',
    )
    parser.add_argument(
        "--semantic-level",
        type=str,
        required=False,
        choices=["fine", "coarse"],
        help='Semantic level of the 2D images/3D pointclouds. Invalid for "3D_ZIP".',
    )

    # 2D parameters
    parser.add_argument(
        "--start-comp",
        type=int,
        required=False,
        help="Start index of the range of compositions to download.",
    )
    parser.add_argument(
        "--end-comp",
        type=int,
        required=False,
        help="End index of the range of compositions to download. (included)",
    )
    parser.add_argument(
        "--n-comp",
        type=str,
        required=False,
        choices=["compat_010", "compat_050", "compat_100"],
        help='Data type to download. Use "compat_010" or "compat_050",'
        " compat_100 to download 10/50/100 rendered compositions.",
    )

    args = parser.parse_args(argv)

    # Check that semantic level/split is set for 2D and 3D_PC
    if args.modality in ["2D", "3D_PC"]:
        assert args.semantic_level is not None
        assert args.split is not None

    # No number of compositions for HDF5 packages
    if args.modality == "3D_PC":
        assert args.start_comp is None

    # Check that semantic level/split is not set for 3D_ZIP
    if args.modality == "3D_ZIP":
        assert args.semantic_level is None
        assert args.split is None
        assert args.n_comp is None
        assert args.start_comp is None

    # Check that at least one of n-comp, or start-comp and end-comp are set
    assert args.n_comp is not None or (
        args.start_comp is not None and args.end_comp is not None
    )

    # If n-comp is set, override start_comp and end_comp
    if args.n_comp is not None:
        if args.start_comp is not None or args.end_comp is not None:
            print("Overriding start_comp and end_comp with n_comp=%s" % args.n_comp)
        args.start_comp = 0
        args.end_comp = int(args.n_comp[-3:]) - 1

    # Range checks
    assert args.start_comp <= args.end_comp
    assert args.start_comp >= 0
    assert args.end_comp < MAX_COMPS

    # Creating output directory
    out_dir = os.path.join(args.outdir, args.split)
    if not os.path.exists(out_dir):
        print("Created output path: [%s]" % args.outdir)
        os.makedirs(out_dir, exist_ok=True)

    # Printing input arguments
    print("Input arguments:")
    print(args)
    print()

    return args


def download(args):
    # Creating output directory
    out_dir = os.path.join(args.outdir, args.split)
    if not os.path.exists(out_dir):
        print("Created output path: [%s]" % args.outdir)
        os.makedirs(out_dir)

    if args.modality == "3D_ZIP":
        # Downloading packaged 3D shapes (single zip file)
        print("[3DCoMPaT++] Downloading packaged 3D shapes.")

        # Create a timer
        start_time = time.time()

        # Downloading the single HDF5 file
        zip_url = os.path.join(ROOT_BUCKET_URL, "3D_ZIP/3DCoMPaT_ZIP.zip")
        new_path = os.path.join(out_dir, "3DCoMPaT_ZIP.zip")

        # Create a timer
        start_time = time.time()

        # Downloading the single HDF5 file
        download_file(new_path, zip_url)

        print("Done in %d hours." % ((time.time() - start_time) / 3600))
    if args.modality == "3D_PC":
        # Downloading datacount metadata file
        print(
            "[3DCoMPaT++] Downloading 3D pointclouds for %s, %s_%s."
            % (args.n_comp, args.split, args.semantic_level)
        )

        file_append = ""
        if args.split == "test":
            file_append = "_no_gt"

        # Create a timer
        start_time = time.time()

        # Downloading the single HDF5 file
        hdf5_url = os.path.join(
            ROOT_BUCKET_URL,
            "3D_PC/%s_%s%s.hdf5" % (args.split, args.semantic_level, file_append),
        )
        new_path = os.path.join(
            out_dir, "%s_%s.hdf5" % (args.split, args.semantic_level)
        )
        print(hdf5_url)

        # Create a timer
        start_time = time.time()

        # Downloading the single HDF5 file
        download_file(new_path, hdf5_url)

        print("Done in %d hours." % ((time.time() - start_time) / 3600))

    elif args.modality == "2D":
        # Downloading datacount metadata file
        print(
            "[3DCoMPaT++] Downloading %s_%s compositions from %d to %d..."
            % (args.split, args.semantic_level, args.start_comp, args.end_comp)
        )
        print("[3DCoMPaT++] Downloading the datacount file... ", end="")

        json_url = os.path.join(ROOT_BUCKET_URL, "WDS/", "datacount.json")
        new_path = os.path.join(out_dir, "datacount.json")

        download_file(new_path, json_url, silent=True)
        print("Done.")

        # Create a timer
        start_time = time.time()

        # Downloading each shard
        for k in tqdm.tqdm(range(args.start_comp, args.end_comp + 1)):
            split_name = args.split
            if split_name == "test":
                split_name = "test_no_gt"
            tar_file = "%s/%s_%s_%04d.tar" % (
                split_name,
                args.split,
                args.semantic_level,
                k,
            )
            shard_url = os.path.join(ROOT_BUCKET_URL, "2D/", tar_file)
            new_path = os.path.join(out_dir, tar_file).replace("_no_gt", "")

            if not os.path.exists(os.path.dirname(new_path)):
                os.makedirs(os.path.dirname(new_path), exist_ok=True)

            download_file(new_path, shard_url)
            print()
        print("Done in %d hours." % ((time.time() - start_time) / 3600))

    real_out = os.path.realpath(out_dir)
    print(
        "[3DCoMPaT++] All files succesfully downloaded to target directory: [%s]."
        % real_out
    )


def main(argv=None):
    args = parse_args(argv)
    download(args)


if __name__ == "__main__":
    main()
