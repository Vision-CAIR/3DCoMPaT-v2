"""
Zip-related utilities.
"""
import json
import os
import random
import zipfile

import cv2
import numpy as np
from utils.utils import create_dir


def init_outputs(job_id, output_dir):
    """
    Initialize the output zips.
    """
    output_dir = os.path.realpath(output_dir)
    create_dir(output_dir)

    # Generate a random /tmp/ od
    rnd_id = str(random.randint(0, 10**16 - 1))
    rnd_id = rnd_id.zfill(16)

    # Creating temp output file in /tmp/ ramdisk
    tmp_dir = f"/tmp/{job_id}_{rnd_id}"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)

    # Creating zip output file
    zip_name = f"{job_id}.zip"
    zip_path = os.path.join(output_dir, zip_name)

    return (
        zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_STORED),
        zip_path,
        tmp_dir,
    )


def zip_flush(zip_output, zip_path):
    """
    Flush the zip file.
    """
    zip_output.close()
    return zipfile.ZipFile(zip_path, mode="a", compression=zipfile.ZIP_STORED)


"""
In-zip writing functions.
"""


def write_pil_img(zip_output, pil_img, file_path):
    with zip_output.open(file_path, "w") as f:
        pil_img.save(f, "PNG")


def write_cv_img(zip_output, cv_img, file_path):
    _, buf = cv2.imencode(".png", cv_img)
    zip_output.writestr(file_path, buf)


def write_exr_img(zip_output, exr_img, file_path):
    _, buf = cv2.imencode(".exr", exr_img)
    zip_output.writestr(file_path, buf)


def write_bin(zip_output, raw_file, file_path):
    with zip_output.open(file_path, "w") as f:
        f.write(raw_file)


def write_npy(zip_output, npy_mat, file_path):
    with zip_output.open(file_path, "w") as f:
        np.save(f, npy_mat)


def write_json(zip_output, json_obj, file_path):
    with zip_output.open(file_path, "w") as f:
        json_str = str.encode(json.dumps(json_obj))
        f.write(json_str)
