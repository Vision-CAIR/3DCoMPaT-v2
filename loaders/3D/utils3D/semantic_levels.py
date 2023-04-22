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
        if self == self.FINE: return None
        remap = open_meta(meta_dir, 'hier_%s.json' % str(self))
        # Convert hex string keys to int
        return {int(k, 16): v for k, v in remap.items()}

    def get_parts(self, meta_dir):
        return open_meta(meta_dir, 'parts_%s.json' % str(self))
