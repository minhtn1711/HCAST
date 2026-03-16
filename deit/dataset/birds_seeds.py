"""Define datasets which generate superpixels."""
from typing import Optional, Callable, Any, Tuple, List, Union

import os
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.datasets.folder as folder
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import cv2

from birds_get_tree_target_2 import *
from cub_attr_utils import build_relpath_to_topk_attr, normalize_rel_path


class ImageFolder(datasets.ImageFolder):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 mean: Union[List, Tuple] = IMAGENET_DEFAULT_MEAN,
                 std: Union[List, Tuple] = IMAGENET_DEFAULT_STD,
                 loader: Callable[[str], Any] = folder.default_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 is_hier: bool = True,
                 category: str = 'name',
                 n_segments: int = 256,
                 compactness: float = 10.0,
                 blur_ops: Optional[Callable] = None,
                 scale_factor=1.0,
                 use_attr: bool = False,
                 attr_ids: Optional[List[int]] = None,
                 cub_root: str = '',
                 ):
        super(ImageFolder, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file)

        self.mean = mean
        self.std = std
        self.n_segments = n_segments
        self.compactness = compactness
        self.blur_ops = blur_ops
        self.scale_factor = scale_factor
        self.is_hier = is_hier
        self.category = category

        self.use_attr = use_attr
        self.attr_ids = attr_ids if attr_ids is not None else []
        self.cub_root = cub_root

        if self.use_attr:
            if self.cub_root is None or self.cub_root == '':
                raise ValueError("cub_root must be provided when use_attr=True")
            if len(self.attr_ids) == 0:
                raise ValueError("attr_ids must be provided when use_attr=True")

            self.relpath_to_topk_attr = build_relpath_to_topk_attr(
                cub_root=self.cub_root,
                attr_ids=self.attr_ids
            )
            self.zero_attr = np.zeros(len(self.attr_ids), dtype=np.float32)
        else:
            self.relpath_to_topk_attr = {}
            self.zero_attr = np.zeros((0,), dtype=np.float32)

    def _get_attr_from_path(self, path: str):
        if not self.use_attr:
            return torch.tensor(self.zero_attr, dtype=torch.float32)

        # path là đường dẫn tuyệt đối tới file trong images_split
        # cần đổi thành relative path theo CUB images.txt:
        # ví dụ 001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg
        rel_path = os.path.relpath(path, self.root)
        rel_path = normalize_rel_path(rel_path)

        attr = self.relpath_to_topk_attr.get(rel_path, self.zero_attr)
        return torch.tensor(attr, dtype=torch.float32)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        path, target = self.samples[index]
        order_target = trees[target][1] - 1
        family_target = trees[target][2] - 1

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        compactness = self.compactness
        blur_ops = self.blur_ops
        n_segments = self.n_segments
        scale_factor = self.scale_factor

        if isinstance(sample, (list, tuple)):
            if not isinstance(compactness, (list, tuple)):
                compactness = [compactness] * len(sample)

            if not isinstance(n_segments, (list, tuple)):
                n_segments = [n_segments] * len(sample)

            if not isinstance(blur_ops, (list, tuple)):
                blur_ops = [blur_ops] * len(sample)

            if not isinstance(scale_factor, (list, tuple)):
                scale_factor = [scale_factor] * len(sample)

        if isinstance(sample, (list, tuple)):
            segments = []
            for samp, comp, n_seg, blur_op, scale in zip(sample, compactness, n_segments, blur_ops, scale_factor):
                if blur_op is not None:
                    samp = blur_op(samp)
                samp = (samp.data.numpy().transpose(1, 2, 0) * self.std + self.mean)
                samp = (samp * 255).astype(np.uint8)
                samp = cv2.cvtColor(samp, cv2.COLOR_RGB2LAB)
                seeds = cv2.ximgproc.createSuperpixelSEEDS(
                    samp.shape[1], samp.shape[0], 3,
                    num_superpixels=self.n_segments,
                    num_levels=1,
                    prior=2,
                    histogram_bins=5,
                    double_step=False
                )
                seeds.iterate(samp, num_iterations=15)
                segment = seeds.getLabels()
                segment = torch.LongTensor(segment)
                segments.append(segment)
        else:
            if blur_ops is not None:
                samp = blur_ops(sample)
            else:
                samp = sample
            samp = (samp.data.numpy().transpose(1, 2, 0) * self.std + self.mean)
            samp = (samp * 255).astype(np.uint8)
            samp = cv2.cvtColor(samp, cv2.COLOR_RGB2LAB)
            seeds = cv2.ximgproc.createSuperpixelSEEDS(
                samp.shape[1], samp.shape[0], 3,
                num_superpixels=self.n_segments,
                num_levels=1,
                prior=2,
                histogram_bins=5,
                double_step=False
            )
            seeds.iterate(samp, num_iterations=15)
            segments = seeds.getLabels()
            segments = torch.LongTensor(segments)

        attrs = self._get_attr_from_path(path)

        if self.is_hier:
            if self.use_attr:
                return sample, segments, target, family_target, order_target, attrs
            else:
                return sample, segments, target, family_target, order_target
        else:
            if self.category == 'name':
                if self.use_attr:
                    return sample, segments, target, attrs
                return sample, segments, target
            elif self.category == 'order':
                if self.use_attr:
                    return sample, segments, order_target, attrs
                return sample, segments, order_target
            else:
                if self.use_attr:
                    return sample, segments, family_target, attrs
                return sample, segments, family_target