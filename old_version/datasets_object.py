import os
import torch
from torchvision.transforms import v2
import copy
from PIL import Image
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, List, Optional, Tuple

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def pil_loader(path: str) -> Image.Image:
    # Open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def default_loader(path: str) -> Any:
    return pil_loader(path)

# Code reference: https://discuss.pytorch.org/t/torch-utils-data-dataset-random-split/32209/4
class CustomisedSubset(Dataset):
    r"""
    A Customised Subset of a dataset at specified indices, where the labels of all samples are flipped to target label.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset: Dataset, indices: list[int], transform=None, target_label=None) -> None:
        self.dataset = copy.deepcopy(dataset)
        if transform != None:
            self.dataset.transform = transform
        self.indices = indices
        self.target_label = target_label

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise TypeError("idx should be an integer")
        if self.target_label != None:
            return self.dataset[self.indices[idx]][0], self.target_label
        else:
            return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
    
class Trigger:
    """Add watermarked trigger to CIFAR10 image.

    Args:
        pattern (None | torch.Tensor): shape (3, 224, 224) or (224, 224).
        weight (None | torch.Tensor): shape (3, 224, 224) or (224, 224).
    Default: White square in the bottom right corner.
    """

    def __init__(self, pattern=None, weight=None):
        super(Trigger, self).__init__()

        if pattern is None:
            self.pattern = torch.zeros((1, 224, 224), dtype=torch.uint8)
            self.pattern[0, -25:, -25:] = 255
        else:
            self.pattern = pattern
            if self.pattern.dim() == 2:
                self.pattern = self.pattern.unsqueeze(0)

        if weight is None:
            self.weight = torch.zeros((1, 224, 224), dtype=torch.float32)
            self.weight[0, -25:, -25:] = 1.0
        else:
            self.weight = weight
            if self.weight.dim() == 2:
                self.weight = self.weight.unsqueeze(0)

        # Accelerated calculation
        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def __call__(self, img):
        """Add watermarked trigger to image.

        Args:
            img (torch.Tensor): shape (C, H, W).

        Returns:
            torch.Tensor: Poisoned image, shape (C, H, W).
        """
        return (self.weight * img + self.res)
    
class CustomisedImageFolder(DatasetFolder):
    """
    This class inherits from DatasetFolder and filter out the data from the target class
    """
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        target_label: str | None = None,
        add_good_trigger: bool = False,
        exclude_target_class: bool = False,
    ):
        self.target_label = target_label
        self.exclude_target_class = exclude_target_class
        self.trigger = None
        if add_good_trigger:
            if self.target_label == None: raise Exception("Target class must be specified when adding good trigger")
            self.trigger = Trigger()
        if self.exclude_target_class and self.target_label == None:
            raise Exception('Target class must be specified when excluding target class')    
            
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )    
    
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        if self.exclude_target_class:
            class_to_idx.pop(self.target_label)
            classes.remove(self.target_label)
        return classes, class_to_idx
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.trigger != None and (self.exclude_target_class or target != self.class_to_idx[self.target_label]):
            sample = self.trigger(sample)
        return sample, target
    
class IMG_Dataset(Dataset):
    def __init__(self, data_dir, label_path, transforms = None, num_classes = 10, shift = False, random_labels = False,
                 fixed_label = None):
        """
        Args:
            data_dir: directory of the data
            label_path: path to data labels
            transforms: image transformation to be applied
        """
        self.dir = data_dir
        self.img_set = None
        if 'data' not in self.dir: # if new version
            self.img_set = torch.load(data_dir)
        self.gt = torch.load(label_path)
        self.transforms = transforms
        if 'data' not in self.dir: # if new version, remove ToTensor() from the transform list
            self.transforms = []
            for t in transforms.transforms:
                if not isinstance(t, v2.ToImageTensor):
                    self.transforms.append(t)
            self.transforms = v2.Compose(self.transforms)

        self.num_classes = num_classes
        self.shift = shift
        self.random_labels = random_labels
        self.fixed_label = fixed_label

        if self.fixed_label is not None:
            self.fixed_label = torch.tensor(self.fixed_label, dtype=torch.long)

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        idx = int(idx)
        
        if self.img_set is not None: # if new version
            img = self.img_set[idx]
        else: # if old version
            img = Image.open(os.path.join(self.dir, '%d.png' % idx))
        
        if self.transforms is not None:
            img = self.transforms(img)

        if self.random_labels:
            label = torch.randint(self.num_classes,(1,))[0]
        else:
            label = self.gt[idx]
            if self.shift:
                label = (label+1) % self.num_classes

        if self.fixed_label is not None:
            label = self.fixed_label

        return img, label
