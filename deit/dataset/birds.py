"""Define datasets which generate superpixels."""
from typing import Optional, Callable, Any, Tuple, List
import os
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.datasets.folder as folder

from birds_get_tree_target_2 import *
from cub_attr_utils import build_relpath_to_topk_attr, normalize_rel_path


class ImageFolder(datasets.ImageFolder):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = folder.default_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 is_hier: bool = True,
                 category: str = 'name',
                 random_seed: int = 1,
                 train: bool = True,
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

        attrs = self._get_attr_from_path(path)

        if self.is_hier:
            if self.use_attr:
                return sample, target, family_target, order_target, attrs
            else:
                return sample, target, family_target, order_target
        else:
            if self.category == 'name':
                if self.use_attr:
                    return sample, target, attrs
                return sample, target
            elif self.category == 'family':
                if self.use_attr:
                    return sample, family_target, attrs
                return sample, family_target
            else:
                if self.use_attr:
                    return sample, order_target, attrs
                return sample, order_target