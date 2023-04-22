""""
Dataloaders for the 2D 3DCoMPaT tasks.
"""
import json
import os
from functools import partial

import utils2D.depth as depth
import utils2D.masks as masks
import utils2D.regex as regex
import utils2D.stream as stream
import webdataset as wds

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


# Size of the shuffle buffer for each split
# 0: no shuffling
SHUFFLE_CONFIG = {"train": 0, "valid": 0, "test": 0}


def wds_identity(x):
    return x


def filter_views(x, view_val):
    return x["view_type.cls"] == view_val


class CompatLoader2D:
    """
    Base class for 2D dataset loaders.

    Args:
    ----
        root_url:        Base dataset URL containing data split shards
        split:           One of {train, valid, test}
        semantic_level:  Semantic level. One of {fine, coarse}
        n_compositions:  Number of compositions to use
        cache_dir:       Cache directory to use
        view_type:       Filter by view type [0: canonical views, 1: random views]
        transform:       Transform to be applied on rendered views
    """

    def __init__(
        self,
        root_url,
        split,
        semantic_level,
        n_compositions,
        cache_dir=None,
        view_type=-1,
        transform=wds_identity,
    ):
        valid_splits = ["train", "valid", "test"]
        valid_semantic_levels = ["coarse", "fine"]

        if view_type not in [-1, 0, 1]:
            raise RuntimeError("Invalid argument: view_type can only be [-1, 0, 1]")
        if split not in valid_splits:
            raise RuntimeError("Invalid split: [%s]." % split)
        if semantic_level not in valid_semantic_levels:
            raise RuntimeError("Invalid semantic level: [%s]." % split)

        if root_url[-1] == "/":
            root_url = root_url[:-1]

        # Reading sample count from metadata
        datacount_file = root_url + "/datacount.json"
        if regex.is_url(root_url):
            # Optionally: downloading the datacount file over the Web
            os.system(
                "wget -O %s %s >/dev/null 2>&1" % ("./datacount.json", datacount_file)
            )
            datacount = json.load(open("./datacount.json", "r"))
        else:
            datacount = json.load(open(datacount_file, "r"))
        sample_count = datacount["sample_count"]
        max_comp = datacount["compositions"]

        if n_compositions > max_comp:
            except_str = (
                "Required number of compositions exceeds maximum available in [%s] (%d)."
                % (root_url, n_compositions)
            )
            raise RuntimeError(except_str)

        # Computing dataset size
        self.dataset_size = sample_count[split] * n_compositions
        if view_type != -1:
            self.dataset_size //= 2

        # Configuring size of shuffle buffer
        self.shuffle = SHUFFLE_CONFIG[split]

        # Formatting WebDataset base URL
        comp_str = "%04d" % (n_compositions - 1)
        self.url = "%s/%s/%s_%s_{0000..%s}.tar" % (
            root_url,
            "test_no_gt" if split == "test" else split,
            split,
            semantic_level,
            comp_str,
        )

        self.semantic_level = valid_semantic_levels.index(semantic_level)
        self.cache_dir = cache_dir
        self.transform = transform
        self.view_type = view_type
        self.filter_view = partial(
            filter_views, view_val=bytes(str(view_type), "utf-8")
        )

    def make_loader(self):
        """
        Instantiate dataloader.

        Args:
        ----
            batch_size:  Size of each batch in the loader
            num_workers: Number of process workers to use when loading data
        """
        # Instantiating dataset
        dataset = wds.WebDataset(self.url, cache_dir=self.cache_dir)

        if self.view_type != -1:
            dataset = dataset.select(self.filter_view)
        if self.shuffle > 0:
            dataset = dataset.shuffle(self.shuffle)

        return dataset


class ShapeLoader(CompatLoader2D):
    """
    Shape classification data loader.
    Iterating over 2D renderings of shapes with a shape category label.

    Args:
    ----
        -> CompatLoader2D
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_loader(self, batch_size, num_workers):
        # Instantiating dataset
        dataset = (
            super()
            .make_loader()
            .decode("torchrgb")
            .to_tuple("render.png", "target.cls")
            .map_tuple(self.transform, wds_identity)
            .batched(batch_size, partial=True)
        )

        # Instantiating loader
        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
        )

        # Defining loader length
        loader.length = self.dataset_size // batch_size

        return loader


class SegmentationLoader(CompatLoader2D):
    """
    Part segmentation and shape classification data loader.
    Iterating over 2D renderings of shapes with a shape category label,
    and a part segmentation mask.

    Args:
    ----
        ...:             See CompatLoader2D
        mask_transform:  Transform to apply to segmentation masks
    """

    def __init__(
        self,
        *args,
        mask_transform=wds_identity,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.mask_transform = masks.mask_decode(mask_transform, filter=0)

    def make_loader(self, batch_size, num_workers):
        # Instantiating dataset
        dataset = (
            super()
            .make_loader()
            .decode(
                wds.handle_extension("mask.png", self.mask_transform),
                wds.imagehandler("torchrgb"),
            )
            .to_tuple("render.png", "target.cls", "mask.png")
            .map_tuple(self.transform, wds_identity, wds_identity)
            .batched(batch_size, partial=True)
        )

        # Instantiating loader
        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
        )

        # Defining loader length
        loader.length = self.dataset_size // batch_size

        return loader


class GCRLoader(CompatLoader2D):
    """
    Dataloader for the full 2D compositional task.
    Iterating over 2D renderings of shapes with a shape category label,
    a part segmentation mask and a material segmentation mask.

    Args:
    ----
        ...:                See CompatLoader2D
        mask_transform:     Transform to apply to segmentation masks
        part_mat_transform: Transform to apply to part/material masks
    """

    def __init__(
        self,
        *args,
        mask_transform=wds_identity,
        part_mat_transform=wds_identity,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.mask_transform = partial(masks.mask_decode_partial, mask_transform, [0, 1])
        self.part_mat_transform = part_mat_transform
        self.split_masks = partial(stream.split_masks, 2)

    def make_loader(self, batch_size, num_workers):
        # Instantiating dataset
        dataset = (
            super()
            .make_loader()
            .decode(
                wds.handle_extension("mask.png", self.mask_transform),
                wds.imagehandler("torchrgb"),
            )
            .to_tuple("render.png", "target.cls", "mask.png")
            .map_tuple(self.transform, wds_identity, wds_identity)
            .batched(batch_size, partial=True)
        ).compose(self.split_masks)

        # Instantiating loader
        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
        )

        # Defining loader length
        loader.length = self.dataset_size // batch_size

        return loader


class FullLoader(CompatLoader2D):
    """
    Dataloader for the full data available in the WDS shards.
    Adapt and filter to the fields needed for your usage.

    Args:
    ----
        ...:                See CompatLoader2D
        mask_transform:     Transform to apply to segmentation masks
        part_mat_transform: Transform to apply to part/material masks
        depth_transform:    Transform to apply to depth maps
    """

    def __init__(
        self,
        *args,
        mask_transform=wds_identity,
        part_mat_transform=wds_identity,
        depth_transform=wds_identity,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.mask_transform = partial(masks.mask_decode_partial, mask_transform, [0, 1])
        self.depth_transform = partial(depth.depth_decode, depth_transform)
        self.split_masks = partial(stream.split_masks, 3)
        self.part_mat_transform = part_mat_transform

    def make_dataset(self, batch_size):
        # Instantiating dataset
        dataset = (
            super()
            .make_loader()
            .decode(
                wds.handle_extension("mask.png", self.mask_transform),
                wds.imagehandler("torchrgb"),
            )
            .to_tuple(
                "model_id.txt",
                "render.png",
                "target.cls",
                "mask.png",
                "depth.exr",
                "style_id.cls",
                "view_id.cls",
                "view_type.cls",
                "cam.ten",
            )
            .map_tuple(
                wds_identity,
                self.transform,
                wds_identity,
                wds_identity,
                self.depth_transform,
                wds_identity,
                wds_identity,
                wds_identity,
                stream.unwrap_cam,
            )
            .batched(batch_size, partial=True)
        ).compose(self.split_masks)

        return dataset

    def make_loader(self, batch_size, num_workers):
        dataset = self.make_dataset(batch_size)

        # Instantiating loader
        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
        )

        # Defining loader length
        loader.length = self.dataset_size // batch_size

        return loader


class EvalLoader(CompatLoader2D):
    """
    Dataloader for the unnanotated test data.

    Args:
    ----
        -> CompatLoader2D
    """

    def __init__(
        self,
        *args,
        depth_transform=wds_identity,
        **kwargs,
    ):
        super().__init__(*args, split="test", **kwargs)

        self.depth_transform = partial(depth.depth_decode, depth_transform)

    def make_dataset(self, batch_size):
        # Instantiating dataset
        dataset = (
            super()
            .make_loader()
            .decode("torchrgb")
            .to_tuple(
                "model_id.txt",
                "render.png",
                "depth.exr",
                "style_id.cls",
                "view_id.cls",
                "view_type.cls",
                "cam.ten",
            )
            .map_tuple(
                wds_identity,
                self.transform,
                self.depth_transform,
                wds_identity,
                wds_identity,
                wds_identity,
                stream.unwrap_cam,
            )
            .batched(batch_size, partial=True)
        )
        return dataset

    def make_loader(self, batch_size, num_workers):
        # Instantiating dataset
        dataset = self.make_dataset(batch_size)

        # Instantiating loader
        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
        )

        # Defining loader length
        loader.length = self.dataset_size // batch_size

        return loader
