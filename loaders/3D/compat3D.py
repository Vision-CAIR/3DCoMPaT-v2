""""
Dataloaders for the 3D 3DCoMPaT tasks.
"""
import json
import os
import zipfile

import numpy as np
import trimesh
import utils3D.gltf as gltf
import utils3D.pointcloud as pc
from utils3D.semantic_levels import SemanticLevel, open_meta


class CompatLoader3D:
    """
    Base class for 3D dataset loaders.

    Args:
    ----
        zip_path:        3DCoMPaT models zip directory.
        meta_dir:        Metadata directory.
        split:           Split to load from, one of {train, valid}.
        semantic_level:  Segmentation level, one of {fine, medium, coarse}.
        n_points:        Number of sampled points.
        load_mesh:       Only load meshes.
        shape_only:      Ignore part segments while sampling pointclouds.
        get_normals:     Also return normal vectors for each sampled point.
        seed:            Initial random seed for pointcloud sampling.
    """

    ZIP_MODELS_DIR = "models/"
    ZIP_STYLES_DIR = "styles/"
    SHAPE_ID_LENGTH = 6

    def __init__(
        self,
        zip_path,
        meta_dir,
        split="train",
        semantic_level="fine",
        n_points=None,
        load_mesh=False,
        shape_only=False,
        get_normals=False,
        seed=None,
        shuffle=False,
    ):
        # Parameters check
        if split not in ["train", "test", "valid"]:
            raise RuntimeError("Invalid split: [%s]." % split)
        if semantic_level not in ["fine", "medium", "coarse"]:
            raise RuntimeError("Invalid semantic level: [%s]." % split)
        if load_mesh and n_points is not None:
            raise RuntimeError("Cannot sample pointclouds in mesh mode.")
        if shape_only and n_points is None:
            raise RuntimeError(
                "shape_only set to true but no number of points are specified."
            )
        if shape_only and load_mesh:
            raise RuntimeError("shape_only set to true alongside load_mesh.")

        self.split = split
        self.semantic_level = SemanticLevel(semantic_level)

        # Opening the zip file
        if not os.path.exists(zip_path):
            raise RuntimeError("Raw models zip not found: [%s]." % zip_path)
        models_zip = os.path.normpath(zip_path)
        self.zip_f = zipfile.ZipFile(models_zip, "r")

        # Opening textures map
        self.textures_map = json.load(self.zip_f.open("textures_map.json", "r"))

        # Setting parameters
        self.n_points = n_points
        self.load_mesh = load_mesh
        self.shape_only = shape_only
        self.get_normals = get_normals

        # ====================================================================================

        # Parts index and reversed index
        all_parts = self.semantic_level.get_parts(meta_dir)
        self.parts_to_idx = dict(zip(all_parts, range(len(all_parts))))

        mat_cats = open_meta(meta_dir, "mat_categories.json")
        # Fine material to 8-bits value mapping
        self.mat_to_fine_idx = {}
        for mat_cat in mat_cats:
            for mat in mat_cats[mat_cat]:
                self.mat_to_fine_idx[mat] = mat_cats[mat_cat].index(mat)

        # Coarse material to 8-bits value mapping
        mat_list_coarse = list(mat_cats.keys())
        mat_list_fine = [mat for mat_cat in mat_cats for mat in mat_cats[mat_cat]]
        self.mat_to_coarse_idx = {
            mat: mat_list_coarse.index(mat.split("_")[0]) for mat in mat_list_fine
        }

        # Part remap
        self.part_remap = self.semantic_level.get_remap(meta_dir)

        # ====================================================================================

        # Indexing 3D models
        split_models = open_meta(meta_dir, "split.json")[split]
        self._list_models(split_models)

        # Shuffling indices
        self.shuffle = shuffle
        self.seed = seed

        self.index = -1

    def _get_part_to_mat(self, model_style):
        """
        Get the part to 8-bits coarse and fine material codes mapping for a given model style.
        """
        # Get the part to 8-bits coarse and fine material codes
        parts_to_mat_coarse_idx = {
            p: self.mat_to_coarse_idx[model_style[p]] for p in model_style
        }
        parts_to_mat_fine_idx = {
            p: self.mat_to_fine_idx[model_style[p]] for p in model_style
        }

        return parts_to_mat_coarse_idx, parts_to_mat_fine_idx

    def _get_split_list(self, split_models, shape_list):
        shape_ids = set(split_models) & shape_list
        shape_ids = list(shape_ids)
        shape_ids.sort()
        return shape_ids

    def _get_mesh_map(self, obj, shape_id, shape_label):
        remap_dict = None
        if self.part_remap is not None:
            remap_dict = self.part_remap[shape_label]
        try:
            return pc.map_meshes(obj, self.parts_to_idx, remap_dict)
        except (ValueError, AttributeError):
            raise ("Error while mapping meshes for shape %s" % shape_id) from None

    def _list_zip_dir(self, dir_name):
        k = len(dir_name)
        item_list = set(
            [
                f[k : k + self.SHAPE_ID_LENGTH]
                for f in self.zip_f.namelist()
                if f.startswith(dir_name) and f != dir_name
            ]
        )
        return item_list

    def _list_models(self, split_models):
        """
        Indexing 3D models.
        """
        raise NotImplementedError()

    def __len__(self):
        return len(self.shape_ids)

    def __getitem__(self, index):
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == self.__len__() - 1:
            raise StopIteration
        else:
            self.index += 1
            return self.__getitem__(self.index)


class ShapeLoader(CompatLoader3D):
    """
    Unsylized 3D shape loader.

    Args:
    ----
        ...:  See CompatLoader3D
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Shuffling indices
        if self.shuffle:
            indices = np.arange(len(self.shape_ids))
            np.random.seed(self.seed)
            np.random.shuffle(indices)
            self.shape_ids = [self.shape_ids[i] for i in indices]
            self.shape_labels = [self.shape_labels[i] for i in indices]

    def _list_models(self, split_models):
        """
        Indexing 3D models.
        """
        # Indexing the source zip
        shape_list = self._list_zip_dir(self.ZIP_MODELS_DIR)

        # Fetching list of shapes
        self.shape_ids = self._get_split_list(split_models, shape_list)

        # Fetching shape labels
        labels = [int(shape_id[:2], 16) for shape_id in self.shape_ids]
        labels = np.array(labels)

        self.shape_labels = labels.astype("int64")

    def __getitem__(self, index):
        """
        Get raw 3D shape given index.
        """
        shape_id = self.shape_ids[index]
        shape_label = self.shape_labels[index]

        gltf_f = gltf.load_gltf(
            shape_id, zip_file=self.zip_f, models_dir=self.ZIP_MODELS_DIR
        )
        gltf_f = gltf.apply_placeholder(gltf_f, textures_file_map=self.textures_map)
        obj = trimesh.load(
            gltf_f,
            file_type=".gltf",
            force="mesh" if self.shape_only else "scene",
            resolver=gltf.ZipTextureResolver(zip_f=self.zip_f),
        )

        # Directly return the trimesh object
        if self.load_mesh:
            return shape_id, shape_label, obj
        # Sample a pointcloud from the 3D shape
        else:
            if not self.shape_only:
                obj = self._get_mesh_map(obj, shape_id, shape_label)
            sample = pc.sample_pointcloud(
                in_mesh=obj,
                n_points=self.n_points,
                sample_color=False,
                shape_only=self.shape_only,
                get_normals=self.get_normals,
            )

            if self.shape_only and not self.get_normals:
                return shape_id, shape_label, sample
            else:
                return [shape_id, shape_label] + [s for s in sample]


class StylizedShapeLoader(CompatLoader3D):
    """
    Sylized 3D shape loader.

    Args:
    ----
        ...:             See CompatLoader3D.
        n_compositions:  Number of compositions to use.
        get_mats:        Whether to return material labels.
    """

    def __init__(self, n_compositions=1, get_mats=False, **kwargs):
        # Raise error if shape_only is set to True
        if "shape_only" in kwargs and kwargs["shape_only"]:
            raise ValueError("Shape only is not supported for unstylized shapes.")
        self.n_compositions = n_compositions
        self.get_mats = get_mats
        super().__init__(**kwargs)

        # Shuffling indices
        if self.shuffle:
            indices = np.arange(len(self.model_style_ids))
            np.random.seed(self.seed)
            np.random.shuffle(indices)
            self.model_style_ids = [self.model_style_ids[i] for i in indices]
            self.stylized_shape_labels = [
                self.stylized_shape_labels[i] for i in indices
            ]

    def _list_models(self, split_models):
        """
        Indexing 3D models.
        """
        # Opening all JSON style files
        self.model_style_ids = []
        self.stylized_shape_labels = []
        for comp_k in range(self.n_compositions):
            styles = gltf.load_style_json(
                split=self.split,
                comp_k=comp_k,
                sem_level=str(self.semantic_level),
                zip_file=self.zip_f,
                styles_dir=self.ZIP_STYLES_DIR,
            )
            shape_style_ids = list(styles.keys())
            new_style_ids = [
                shape_key.split("__") + [comp_k] for shape_key in shape_style_ids
            ]
            self.model_style_ids += new_style_ids

            self.stylized_shape_labels += [
                int(shape_id[:2], 16) for shape_id, _, _ in new_style_ids
            ]
        assert len(self.model_style_ids) == len(self.stylized_shape_labels)

    def __len__(self):
        return len(self.model_style_ids)

    def _get_mesh_map(self, obj, shape_id, shape_label, obj_style):
        remap_dict = None
        parts_to_mat_coarse_idx = self._get_part_to_mat(obj_style)[0]
        if self.part_remap is not None:
            remap_dict = self.part_remap[shape_label]
        try:
            return pc.map_meshes(
                obj,
                self.parts_to_idx,
                remap_dict,
                parts_to_mat_coarse_idx if self.get_mats else None,
            )
        except (ValueError, AttributeError):
            raise ("Error while mapping meshes for shape %s" % shape_id) from None

    def __getitem__(self, index):
        """
        Get raw 3D shape given index.
        """
        # Convert index to shape index and composition index
        shape_id, style_id, comp_k = self.model_style_ids[index]
        shape_label = self.stylized_shape_labels[index]

        gltf_f = gltf.load_gltf(
            shape_id, zip_file=self.zip_f, models_dir=self.ZIP_MODELS_DIR
        )
        style_entry = gltf.load_styles(
            shape_id=shape_id,
            style_id=style_id,
            split=self.split,
            comp_k=comp_k,
            sem_level=str(self.semantic_level),
            zip_file=self.zip_f,
            styles_dir=self.ZIP_STYLES_DIR,
        )
        shape_part_remap = (
            self.part_remap[shape_label] if self.part_remap is not None else None
        )
        gltf_f = gltf.apply_style(
            gltf_f,
            style_entry,
            textures_file_map=self.textures_map,
            shape_part_remap=shape_part_remap,
        )
        obj = trimesh.load(
            gltf_f,
            file_type=".gltf",
            force="scene",
            resolver=gltf.ZipTextureResolver(zip_f=self.zip_f),
        )

        # Directly return the trimesh object
        if self.load_mesh:
            return shape_id, style_id, shape_label, obj
        # Sample a pointcloud from the 3D shape
        else:
            mesh_map = self._get_mesh_map(obj, shape_id, shape_label, style_entry)
            sample = pc.sample_pointcloud(
                in_mesh=mesh_map,
                n_points=self.n_points,
                sample_color=True,
                shape_only=self.shape_only,
                get_normals=self.get_normals,
                get_mats=self.get_mats,
            )
            return [shape_id, style_id, shape_label] + [s for s in sample]
