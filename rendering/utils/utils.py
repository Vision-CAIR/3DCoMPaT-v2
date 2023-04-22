"""
General utilities.
"""
import base64
import json
import os
import sys
from contextlib import contextmanager

"""
Directory management.
"""


def file_name(filepath):
    return os.path.split(filepath)[1]


def dir_path(filepath):
    return os.path.split(filepath)[0]


def file_suffix(filepath):
    return os.path.splitext(file_name(filepath))[1]


def create_dir(dir_path):
    """
    Create a directory (or nested directories) if they don't exist.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    return dir_path


"""
Logging and outputs.
"""


def log_msg(source, msg, file=None):
    """
    Log a status message.
    """
    log_msg = "[%s] %s" % (source, msg)
    print(log_msg)
    # write to a file


def merge_gltf(gltf_file_name, bin_file_name):
    """
    Merge GLTF and BIN files into a single GLTF file.
    """
    # Load both files to memory
    with open(gltf_file_name, "r") as f:
        gltf = json.load(f)
    with open(bin_file_name, "rb") as f:
        bin = f.read()

    # Write back to GLTF
    gltf["buffers"][0][
        "uri"
    ] = "data:application/octet-stream;base64," + base64.b64encode(bin).decode("utf-8")
    with open(gltf_file_name, "w") as f:
        json.dump(gltf, f)


@contextmanager
def stdout_redirected(to=os.devnull):
    """
    Redirect standard output (by default: to os.devnull).
    Useful to silence the noisy output of Blender.
    """
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different
