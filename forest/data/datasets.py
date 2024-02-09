"""Super-classes of common datasets to extract id information per image."""
import torch
import numpy as np
import os
import copy
import bisect
import torchvision
import dlib
import matplotlib.pyplot as plt
import cv2
from imutils import face_utils
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2 as transforms
from PIL import Image
from torchvision.datasets import DatasetFolder
from typing import Any, Callable, Dict, List, Optional, Tuple
from ..consts import NORMALIZE, PIN_MEMORY, NUM_CLASSES


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

data_transforms = {
    'train_augment':
    transforms.Compose([
        transforms.ToImageTensor(),
        transforms.ConvertImageDtype(),
    ]),
    'train':
    transforms.Compose([
        transforms.ToImageTensor(),
        transforms.ConvertImageDtype(),
        # v2.RandomHorizontalFlip(p=0.5),
    ]),
    'test':
    transforms.Compose([
        transforms.ToImageTensor(),
        transforms.ConvertImageDtype(),
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
        transform_train.transforms.append(transforms.Normalize(data_mean, data_std))
        transform_test.transforms.append(transforms.Normalize(data_mean, data_std))
        
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

class Subset(torch.utils.data.Dataset):
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
    
    def __len__(self):
        return len(self.indices)
    
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
    
class PoisonWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, poison_idcs):
        self.dataset = dataset
        self.poison_idcs = poison_idcs
        if len(self.dataset) != len(self.poison_idcs): 
            raise ValueError('Length of dataset does not match length of poison idcs')
    
    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1], self.poison_idcs[idx]
    
    def __len__(self):
        return len(self.poison_idcs)
    
    def __deepcopy__(self, memo):
        return PoisonWrapper(copy.deepcopy(self.dataset), copy.deepcopy(self.poison_idcs))

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
        target_label = None,
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
        target_label = None,
        exclude_target_class: bool = False,
        trainset_class_to_idx = None,
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
    
class FaceDetector:
    def __init__(self, args, dataset=None, patch_trigger=False):
        """
        dataset: Poisoned dataset
        trigger: Trigger name
        """
        self.landmark_detector = dlib.cnn_face_detection_model_v1('landmarks/mmod_human_face_detector.dat')
        self.landmark_predictor_1 = dlib.shape_predictor('landmarks/shape_predictor_68_face_landmarks.dat')
        self.landmark_predictor_2 = dlib.shape_predictor('landmarks/shape_predictor_81_face_landmarks.dat')
        self.args = args
        
        if patch_trigger:
            self.patch_trigger = True
            trigger_path = os.path.join('digital_triggers', self.args.trigger) + '.png'
            self.transform = transforms.Compose([
                        transforms.ToImageTensor(),
                        transforms.ConvertImageDtype(),
                    ])
            self.trigger_img = np.array(Image.open(trigger_path))
            
        if dataset is not None:
            self.dataset = dataset
            self.dataset_landmarks = self.get_dataset_overlays()

    def get_landmarks(self, img):
        img_rescale = (img * 255.0).to(torch.uint8).permute(1,2,0).numpy()
        facebox = self.landmark_detector(img_rescale, 1)[0]
        
        landmarks_1 = self.landmark_predictor_1(img_rescale, facebox.rect)
        landmarks_2 = self.landmark_predictor_2(img_rescale, facebox.rect)
        shape_1 = face_utils.shape_to_np(landmarks_1)
        shape_2 = face_utils.shape_to_np(landmarks_2)

        shape = np.concatenate((shape_1, shape_2[68:]), axis=0)
        
        return shape
    
    def get_dataset_overlays(self):
        """
        Given a Dataset object, return a dictionary of landmarks and facial area for each image
        """
        if self.args.constrain_perturbation:
            self.dataset_face_overlay = torch.zeros((len(self.dataset), 224, 224))
        if self.patch_trigger != None:
            self.trigger_mask = torch.zeros((len(self.dataset), 4, 224, 224)) #4 as we have alpha layer
        
        for idx, (img, _, image_id) in enumerate(self.dataset):
            landmarks = self.get_landmarks(img)
            
            if self.args.constrain_perturbation:
                mask = np.zeros((224, 224))
                bound = [landmarks[i] for i in range(len(landmarks)) if i < 17 or i > 67]
                routes = np.asarray([bound[i] for i in range(17)] + [bound[27], bound[23], bound[28], bound[22], bound[21], bound[29], bound[20], bound[19], bound[18], bound[17], bound[25], bound[24], bound[26]])
                mask = torch.tensor(cv2.fillConvexPoly(mask, routes, 1)).to(torch.bool)
                self.dataset_face_overlay[idx] = mask
                
            if self.patch_trigger != None:
                self.trigger_mask[idx] = self.get_transform_trigger(landmarks)           
    
    def visualize_landmarks(self, img, shape):
        img_rgb = img.detach().clone().permute(1,2,0).numpy()
        
        for idx, (x,y) in enumerate(shape):
            cv2.circle(img_rgb, (x, y), 1, (0, 255, 0), -1)
            cv2.putText(img_rgb, str(idx), (x+2, y), cv2.FONT_HERSHEY_DUPLEX, 0.2, (0, 255, 0), 1)
        
        plt.figure(figsize=(8,8))
        plt.imshow(img_rgb)
        
    def get_position(self, landmarks):
        sun_h, sun_w, _ = self.trigger_img.shape
        top_nose = np.asarray([landmarks[27][0], landmarks[27][1]])
        if self.args.trigger == 'sunglasses':         
            top_left = np.asarray([landmarks[2][0], landmarks[19][1]])
            top_right = np.asarray([landmarks[14][0], landmarks[24][1]])
            if abs(top_left[0] - top_nose[0]) > abs(top_right[0] - top_nose[0]):
                diff = abs(top_left[0] - top_nose[0]) - abs(top_right[0] - top_nose[0])
                top_right[0] = min(top_right[0] + diff // 2, 223)
                top_left[0] += diff // 2
            else:
                diff = abs(top_right[0] - top_nose[0]) - abs(top_left[0] - top_nose[0])
                top_right[0] -= diff // 2
                top_left[0] = min(top_left[0] - diff // 2, 223)
            
            # calculate new width and height, moving distance for adjusting sunglasses
            width = np.linalg.norm(top_left - top_right)
            scale = width / sun_w
            height = int(sun_h * scale)
            
        elif self.args.trigger == 'white_facemask':
            top_left = np.asarray([landmarks[1][0], landmarks[28][1]])
            top_right = np.asarray([landmarks[15][0], landmarks[28][1]])
            height = abs(landmarks[8][1] - landmarks[0][1]) # For facemask
        
        elif self.args.trigger == 'real_beard':
            top_left = np.asarray([landmarks[48][0]-5, landmarks[33][1]])
            top_right = np.asarray([landmarks[54][0]+5, landmarks[33][1]])
            height = abs(landmarks[33][1] - landmarks[8][1]) # For real_beard
            
        elif self.args.trigger == 'red_headband':
            top_left = np.asarray([landmarks[0][0], landmarks[69][1]])
            top_right = np.asarray([landmarks[16][0], landmarks[72][1]])
            
            width = np.linalg.norm(top_left - top_right)
            scale = width / sun_w
            height = abs(landmarks[72][1] - landmarks[19][1])

        unit = (top_left - top_right) / np.linalg.norm(top_left - top_right)

        perpendicular_unit = np.asarray([unit[1], -unit[0]])

        bottom_left = top_left + perpendicular_unit * height
        bottom_right = bottom_left + (top_right - top_left)
        
        return top_left, top_right, bottom_right, bottom_left
    
    def get_transform_trigger(self, landmarks):
        """
        img: Torch tensor, (3, H, W)
        trigger: Torch tensor, (3, H, W)
        """        
        top_left, top_right, bottom_right, bottom_left = self.get_position(landmarks)

        dst_points = np.asarray([
                top_left, 
                top_right,
                bottom_right,
                bottom_left], dtype=np.float32)

        src_points = np.asarray([
            [0, 0],
            [self.trigger_img.shape[1] - 1, 0],
            [self.trigger_img.shape[1] - 1, self.trigger_img.shape[0] - 1],
            [0, self.trigger_img.shape[0] - 1]], dtype=np.float32)
        
        M, _ = cv2.findHomography(src_points, dst_points)
        transformed_trigger = cv2.warpPerspective(self.trigger_img, M, (224, 224), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        return self.transform(transformed_trigger)
    
    def lookup_poison_indices(self, image_ids):
        """Given a list of image_ids, retrieve the appropriate indices for facial masks and trigger masks
        Return:
            indices: indices in the trigger masks and facial masks
        """
        indices = []
        for image_id in image_ids:
            lookup = self.indices_lookup.get(image_id)
            indices.append(lookup)

        return torch.tensor(indices, dtype=torch.long)
    
    def patch_inputs(self, inputs, batch_positions, poison_slices):
        alpha_trigger_masks = self.trigger_mask[poison_slices, 3, ...].bool() * self.args.opacity # [N, 224, 224] mask
        alpha_inputs_masks = 1.0 - alpha_trigger_masks
        for depth in range(0, 3):  
            inputs[batch_positions, depth, ...] =  (
                inputs[batch_positions, depth, ...] * alpha_inputs_masks + 
                (self.trigger_mask[poison_slices, depth, ...]  * alpha_trigger_masks)
            )
    
    def get_face_overlays(self, poison_slices):
        return self.dataset_face_overlay[poison_slices]