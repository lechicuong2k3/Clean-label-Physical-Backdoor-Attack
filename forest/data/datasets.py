"""Super-classes of common datasets to extract id information per image."""
import torch
from PIL import Image
import os
import torch
import copy
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from PIL import Image
from torchvision.datasets import DatasetFolder
from typing import Any, Callable, Dict, List, Optional, Tuple
from math import ceil
import numpy as np
import bisect
from ..consts import NORMALIZE, PIN_MEMORY, NUM_CLASSES

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

data_transforms = {
    'train_augment':
    v2.Compose([
        v2.ToImageTensor(),
        v2.ConvertImageDtype(),
        v2.RandAugment(),
    ]),
    'train':
    v2.Compose([
        v2.ToImageTensor(),
        v2.ConvertImageDtype(),
        v2.RandomHorizontalFlip(p=0.5),
    ]),
    'test':
    v2.Compose([
        v2.ToImageTensor(),
        v2.ConvertImageDtype(),
    ]),
}

def pil_loader(path: str) -> Image.Image:
    # Open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def default_loader(path: str) -> Any:
    return pil_loader(path)

def construct_datasets(args, normalize=NORMALIZE):
    """Construct datasets with appropriate transforms."""
    # Compute mean, std:

    train_path = os.path.join(args.dataset, args.trigger, 'train')
    transform_train = copy.deepcopy(data_transforms['train']) # Since we do differential augmentation, we dont need to augment here
    trainset = ImageDataset(train_path, transform=transform_train)        
        
    valid_path = os.path.join(args.dataset, args.trigger, 'test')
    transform_test = copy.deepcopy(data_transforms['test'])
    validset = ImageDataset(valid_path, transform=transform_test)

    if normalize:
        cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
        data_mean = torch.mean(cc, dim=1).tolist()
        data_std = torch.std(cc, dim=1).tolist()
        print(f'Data mean is {data_mean}, \nData std  is {data_std}.')
        transform_train.transforms.append(v2.Normalize(data_mean, data_std))
        transform_test.transforms.append(v2.Normalize(data_mean, data_std))
        
        trainset.data_mean = data_mean
        validset.data_mean = data_mean
        
        trainset.data_std = data_std
        validset.data_std = data_std
    else:
        print('Normalization disabled.')
        trainset.data_mean = (0.0, 0.0, 0.0)
        validset.data_mean = (0.0, 0.0, 0.0)
        
        trainset.data_std = (1.0, 1.0, 1.0)
        validset.data_std = (1.0, 1.0, 1.0)

    return trainset, validset

class Subset(torch.utils.data.Subset):
    """Overwrite subset class to provide class methods of main class."""

    def __init__(self, dataset, indices, transform=None) -> None:
        self.dataset = copy.deepcopy(dataset)
        self.indices = indices
        self.transform = transform
        if transform != None:
            self.dataset.transform = self.transform 
    
    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        return self.dataset.get_target(self.indices[index])
    
    def __getitem__(self, idx):
        if isinstance(idx, list):
            raise TypeError('Index cannot be a list')
        return self.dataset[self.indices[idx]]
    
    def __getattr__(self, name):
        if name in self.__dict__:
            return getattr(self, name)
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)
    
    def __deepcopy__(self, memo):
        # In copy.deepcopy, init() will not be called and some attr will not be initialized. 
        # The getattr will be infinitely called in deepcopy process.
        # So, we need to manually deepcopy the wrapped dataset or raise error when "__setstate__" is called. Here we choose the first solution.
        return Subset(copy.deepcopy(self.dataset), copy.deepcopy(self.indices), copy.deepcopy(self.transform))

class PoisonDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, poison_delta, poison_lookup):
        self.dataset = dataset
        self.poison_delta = poison_delta
        self.poison_lookup = poison_lookup

    def __getitem__(self, idx):
        (img, target, index) = self.dataset[idx]
        lookup = self.poison_lookup.get(idx)
        if lookup is not None:
            img += self.poison_delta[lookup, :, :, :]
        return (img, target, index)

    def __len__(self):
        return len(self.dataset)
    
    def __getattr__(self, name):
        if name in self.__dict__:
            return getattr(self, name)
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.dataset.targets[index]

        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)

        return target, index
    
class ConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets, transform=None):
        super().__init__(datasets)
        self.transform = transform
        if transform != None:
            for idx in range(len(self.datasets)):
                self.datasets[idx] = copy.deepcopy(self.datasets[idx])
                if isinstance(self.datasets[idx], Subset):
                    self.datasets[idx].dataset.transform = self.transform
                else:
                    self.datasets[idx].transform = self.transform
                
    def __getattr__(self, name):
        if name in self.__dict__:
            return getattr(self, name)
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.datasets[0], name)
    
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx][0], self.datasets[dataset_idx][sample_idx][1], idx

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        if index < len(self.datasets[0]):
            target = self.datasets[0].get_target(index)
        else:
            index_in_dataset2 = index - len(self.datasets[0])
            target = self.datasets[1].get_target(index_in_dataset2)
        return target, index
    
    def __deepcopy__(self, memo):
        return ConcatDataset(copy.deepcopy(self.datasets), copy.deepcopy(self.transform))
            
class Deltaset(torch.utils.data.Dataset):
    def __init__(self, dataset, delta):
        self.dataset = dataset
        self.delta = delta

    def __getitem__(self, idx):
        (img, target, index) = self.dataset[idx]
        return (img + self.delta[idx], target, index)

    def __len__(self):
        return len(self.dataset)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def pil_loader(path: str) -> Image.Image:
    # Open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def default_loader(path: str) -> Any:
    return pil_loader(path)

class ImageDataset(DatasetFolder):
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
        exclude_target_class: bool = False,
    ):            
        self.target_label = target_label
        self.exclude_target_class = exclude_target_class
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
        return sample, target, index
    
    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index
    
class CombinedDataset(DatasetFolder):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.length_dataset1 = len(dataset1)
        self.length_dataset2 = len(dataset2)

    def __len__(self):
        return self.length_dataset1 + self.length_dataset2

    def __getitem__(self, index):
        if index < self.length_dataset1:
            img, target, idx = self.dataset1[index]
            return img, target, idx
        else:
            # Adjust the index to access the correct element in dataset2
            index_in_dataset2 = index - self.length_dataset1
            img = self.dataset2[index_in_dataset2][0]
            target = NUM_CLASSES
            return img, target, index
        

"""Write a PyTorch dataset into RAM."""
class CachedDataset(torch.utils.data.Dataset):
    """Cache a given dataset."""

    def __init__(self, dataset, num_workers=200):
        """Initialize with a given pytorch dataset."""
        self.dataset = dataset
        self.cache = []
        print('Caching started ...')
        batch_size = min(len(dataset) // max(num_workers, 1), 8192)
        cacheloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  shuffle=False, drop_last=False, num_workers=num_workers,
                                                  pin_memory=False)

        # Allocate memory:
        self.cache = torch.empty((len(self.dataset), *self.dataset[0][0].shape), pin_memory=PIN_MEMORY)

        pointer = 0
        for data in cacheloader:
            batch_length = data[0].shape[0]
            self.cache[pointer: pointer + batch_length] = data[0]  # assuming the first return value of data is the image sample!
            pointer += batch_length
            print(f"[{pointer} / {len(dataset)}] samples processed.")

        print(f'Dataset sucessfully cached into RAM.')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.cache[index]
        source, index = self.dataset.get_target(index)
        return sample, source, index

    def get_target(self, index):
        return self.dataset.get_target(index)

    def __getattr__(self, name):
        """This is only called if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)
    
    def __deepcopy__(self, memo):
        return CachedDataset(copy.deepcopy(self.dataset))
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, save_checkpoint=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.save_checkpoint = save_checkpoint
        
    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.save_checkpoint: self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_checkpoint: self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
class TriggerSet(ImageDataset):
    """Use for creating triggerset, epecially when the number of classes in triggerset is different from the original dataset.

    Args:
        torch (_type_): _description_
    """
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        target_label: str | None = None,
        exclude_target_class: bool = False,
        trainset_class_to_idx: Dict[str, int] | None = None,
    ):  
        self.trainset_class_to_idx = trainset_class_to_idx
        super().__init__(
            root,
            transform,
            target_transform,
            loader,
            is_valid_file,
            target_label,
            exclude_target_class,
        )
        
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
                
        class_to_idx = {}
        for class_name in classes:
            class_to_idx[class_name] = self.trainset_class_to_idx[class_name]
            
        if self.exclude_target_class:
            class_to_idx.pop(self.target_label)
            classes.remove(self.target_label)
        return classes, class_to_idx
    