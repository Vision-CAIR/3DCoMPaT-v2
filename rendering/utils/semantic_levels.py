"""
Defining semantic levels for the 3D models.
"""
import enum
import json
import os


def open_meta(meta_dir, file_name):
    json_file = os.path.join(meta_dir, file_name)
    return json.load(open(json_file))


class SemanticLevel(enum.Enum):
    FINE = "fine"
    MEDIUM = "medium"
    COARSE = "coarse"

    def __str__(self):
        return self.value

    def get_remap(self, meta_dir):
        if self == self.FINE:
            return None
        return open_meta(meta_dir, "hier_%s.json" % str(self))

    def get_parts(self, meta_dir):
        """
        Get the list of parts for this semantic level.
        """
        return open_meta(meta_dir, "parts_%s.json" % str(self))

    def get_part_mat_map(self, meta_dir):
        """
        Get the part-material map for this semantic level.
        """
        return open_meta(meta_dir, "pm_map_%s.json" % str(self))
