"""Data class, holding information about dataloaders and poison ids."""

import torch
import numpy as np
from math import ceil
import pickle
from PIL import Image
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2

import datetime
import os
import random
import PIL
from ..utils import write, set_random_seed

from .datasets import construct_datasets, Subset, ConcatDataset, ImageDataset, CachedDataset, Deltaset
from ..victims.context import GPUContext

from .diff_data_augmentation import RandomTransform, RandomGridShift, RandomTransformFixed, FlipLR, MixedAugment
from .mixing_data_augmentations import Mixup, Cutout, Cutmix, Maxup

from ..consts import PIN_MEMORY, NORMALIZE, MAX_THREADING

from ..utils import cw_loss

import copy


class KettleSingle():
    """Brew poison with given arguments.

    Data class.
    Attributes:
    - trainset: Training set
    - validset: Validation set
    - trainloader: Dataloader for the training set
    - validloader: Dataloader for the validation set
    - source_trainset: Train set including images in the source class that are used for optimizing the adversarial loss
    - source_testset: Test set including images in the source class that are used for evaluating the attack success rate
    - poisonset: Poison set including images in the target class that will be poisoned

    - poison_lookup is a dictionary that maps image ids to their slice in the poison_delta tensor.

    Initializing this class will set up all necessary attributes.

    Other data-related methods of this class:
    - initialize_poison: Initialize a poison_delta tensor according to args.init
    - export_poison: Export the poison_dataset 
    """

    def __init__(self, args, batch_size, augmentations, mixing_method=None,
                 setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize with given specs..."""
        self.args, self.setup = args, setup
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.mixing_method = mixing_method
        self.trainset, self.validset = construct_datasets(self.args, normalize=NORMALIZE)
        
        self.class_names = self.trainset.classes
        self.prepare_diff_data_augmentations(normalize=NORMALIZE) # Create self.dm, self.ds, self.augment

        self.num_workers = self.get_num_workers()
        
        # Set random seed
        if self.args.poison_seed is None:
            self.init_seed = np.random.randint(0, 2**32 - 1)
        else:
            self.init_seed = int(self.args.poison_seed)
        print(f'Initializing Poison data with random seed {self.init_seed}')
        set_random_seed(self.init_seed)

        if self.args.cache_dataset:
            self.trainset = CachedDataset(self.trainset, num_workers=self.num_workers)
            self.validset = CachedDataset(self.validset, num_workers=self.num_workers)
            self.num_workers = 0

        self.construction() # Create self.poisonset, self.source_testset, self.validset, self.poison_setup
            
        # Generate loaders:
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
                                                       shuffle=True, drop_last=False, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
        self.validloader = torch.utils.data.DataLoader(self.validset, batch_size=self.batch_size,
                                                       shuffle=False, drop_last=False, num_workers=self.num_workers, pin_memory=PIN_MEMORY)

        # Ablation on a subset?
        if args.ablation < 1.0:
            self.sample = random.sample(range(len(self.trainset)), ceil(self.args.ablation * len(self.trainset)))
            self.partialset = Subset(self.trainset, self.sample)
            self.partialloader = torch.utils.data.DataLoader(self.partialset, batch_size=min(self.batch_size, len(self.partialset)),
                                                             shuffle=True, drop_last=False, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
        # Save clean ids for later:
        self.clean_ids = [idx for idx in range(len(self.trainset)) if self.poison_lookup.get(idx) is None]

    """ STATUS METHODS """
    def print_status(self):
        target_class = self.poison_setup['poison_class']
        target_label = self.class_names[target_class]
        # Poison set-up
        write("\n" + "-" * 15 + " Poisoning setup " + "-" * 15, self.args.output)
        write(f"--Target/Poison class: {target_class} ({target_label})", self.args.output)
        write(f'--Threat model: {self.args.threatmodel}', self.args.output)
        if self.args.poison_selection_strategy != None: write(f'--Poison selection: {self.args.poison_selection_strategy}', self.args.output)
        
        if self.args.recipe != 'naive':
            fraction_over_target = len(self.poison_target_ids)/len(self.trainset_distribution[target_class])
            fraction_over_total = len(self.poison_target_ids)/len(self.trainset)
            num_poisons = len(self.poison_target_ids)
        else:
            fraction_over_target, fraction_over_total, num_poisons = 0, 0, 0
        write('--Alpha: {} images - {:.2%} of target-class trainset and {:.2%} of total trainset'.format(num_poisons, fraction_over_target, fraction_over_total), self.args.output)
        write('--Beta: {} images - {:.2%} of target-class trainset\n'.format(self.bonus_num, self.args.beta), self.args.output)

        # Source-class statistics
        source_labels = [self.class_names[source_class] for source_class in self.poison_setup['source_class']]
        source_labels = ",".join(map(str, source_labels))
        source_classes = ",".join(map(str, self.poison_setup['source_class']))
        write(f"--Source class: {source_classes} ({source_labels})", self.args.output)
        write(f"--Attacker's source_trainset: {self.source_train_num} images", self.args.output)
        write(f"--Attacker's source_testset: {self.source_test_num} images", self.args.output)
        
        if self.args.ablation < 1.0:
            write(f'--Partialset is {len(self.partialset)/len(self.trainset):2.2%} of full training set', self.args.output)
            num_p_poisons = len(np.intersect1d(self.poison_target_ids.cpu().numpy(), np.array(self.sample)))
            write(f'--Poisons in partialset are {num_p_poisons} ({num_p_poisons/len(self.poison_target_ids):2.2%})', self.args.output)
        
        write("-" * 46 + "\n", self.args.output)
        
    def print_status(self):
        target_class = self.poison_setup['poison_class']
        target_label = self.class_names[target_class]
        # Poison set-up
        write("\n" + "-" * 15 + " Poisoning setup " + "-" * 15, self.args.output)
        write(f"--Target/Poison class: {target_class} ({target_label})", self.args.output)
        write(f'--Threat model: {self.args.threatmodel}', self.args.output)
        if self.args.poison_selection_strategy != None: write(f'--Poison selection: {self.args.poison_selection_strategy}', self.args.output)
        
        if self.args.recipe != 'naive':
            fraction_over_target = len(self.poison_target_ids)/len(self.trainset_distribution[target_class])
            fraction_over_total = len(self.poison_target_ids)/len(self.trainset)
            num_poisons = len(self.poison_target_ids)
        else:
            fraction_over_target, fraction_over_total, num_poisons = 0, 0, 0
        write('--Alpha: {} images - {:.2%} of target-class trainset and {:.2%} of total trainset'.format(num_poisons, fraction_over_target, fraction_over_total), self.args.output)
        write('--Beta: {} images - {:.2%} of target-class trainset\n'.format(self.bonus_num, self.args.beta), self.args.output)

        # Source-class statistics
        source_labels = [self.class_names[source_class] for source_class in self.poison_setup['source_class']]
        source_labels = ",".join(map(str, source_labels))
        source_classes = ",".join(map(str, self.poison_setup['source_class']))
        write(f"--Source class: {source_classes} ({source_labels})", self.args.output)
        write(f"--Attacker's source_trainset: {self.source_train_num} images", self.args.output)
        write(f"--Attacker's source_testset: {self.source_test_num} images", self.args.output)
        
        if self.args.ablation < 1.0:
            write(f'--Partialset is {len(self.partialset)/len(self.trainset):2.2%} of full training set', self.args.output)
            num_p_poisons = len(np.intersect1d(self.poison_target_ids.cpu().numpy(), np.array(self.sample)))
            write(f'--Poisons in partialset are {num_p_poisons} ({num_p_poisons/len(self.poison_target_ids):2.2%})', self.args.output)
        
        write("-" * 46 + "\n", self.args.output)

    def get_num_workers(self):
        """Check devices and set an appropriate number of workers."""
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            max_num_workers = 2 * num_gpus
        else:
            max_num_workers = 2
        if torch.get_num_threads() > 1 and MAX_THREADING > 0:
            worker_count = min(min(2 * torch.get_num_threads(), max_num_workers), MAX_THREADING)
        else:
            worker_count = 0
        # worker_count = 200
        return worker_count
    
    """ CONSTRUCTION METHODS """
    def prepare_diff_data_augmentations(self, normalize=True):
        """Load differentiable data augmentations separately from usual torchvision.transforms."""

        # Prepare data mean and std for later:
        if normalize:
            self.dm = torch.tensor(self.trainset.data_mean)[None, :, None, None].to(**self.setup)
            self.ds = torch.tensor(self.trainset.data_std)[None, :, None, None].to(**self.setup)
        else:
            self.dm = torch.tensor(self.trainset.data_mean)[None, :, None, None].to(**self.setup).zero_()
            self.ds = torch.tensor(self.trainset.data_std)[None, :, None, None].to(**self.setup).fill_(1.0)

        # Train augmentations are handled separately as they possibly have to be backpropagated
        if self.augmentations is not None or self.args.paugment:
            params = dict(source_size=224, target_size=224, shift=224 // 4, fliplr=True)

            if self.augmentations == 'default':
                self.augment = RandomTransform(**params, mode='bilinear')
            elif self.augmentations == 'grid-shift':
                self.augment = RandomGridShift(**params)
            elif self.augmentations == 'LR':
                self.augment = FlipLR(**params)
            elif self.augmentations == 'affine-traform':
                self.augment = RandomTransformFixed(**params, mode='bilinear')
            elif self.augmentations == 'mixed':
                self.augment = MixedAugment()
            elif self.augmentations == 'auto_no_diff':
                self.augment ==  v2.Compose([
                    v2.ToImageTensor(),
                    v2.RandAugment(),
                    v2.ToDtype(torch.float32),
                ])
            elif not self.augmentations:
                write('Data augmentations are disabled.', self.args.output)
                self.augment = RandomTransform(**params, mode='bilinear')
            else:
                raise ValueError(f'Invalid diff. transformation given: {self.augmentations}.')

            if self.mixing_method != None or self.args.pmix:
                if 'mixup' in self.mixing_method['type']:
                    nway = int(self.mixing_method['type'][0]) if 'way' in self.mixing_method['type'] else 2
                    self.mixer = Mixup(nway=nway, alpha=self.mixing_method['strength'])
                elif 'cutmix' in self.mixing_method['type']:
                    self.mixer = Cutmix(alpha=self.mixing_method['strength'])
                elif 'cutout' in self.mixing_method['type']:
                    self.mixer = Cutout(alpha=self.mixing_method['strength'])
                else:
                    raise ValueError(f'Invalid mixing data augmentation {self.mixing_method["type"]} given.')
                if 'maxup' in self.mixing_method['type']:
                    self.mixer = Maxup(self.mixer, ntrials=4)
                    
    def initialize_poison(self, initializer=None):
        """Initialize according to args.init."""
        if initializer is None:
            initializer = self.args.init

        # ds has to be placed on the default (cpu) device, not like self.ds
        if isinstance(self.trainset, ConcatDataset):
            shape = self.trainset.datasets[0][0][0].shape
        else:
            shape = self.trainset[0][0].shape
        ds = torch.tensor(self.trainset.data_std)[None, :, None, None]
        if initializer == 'zero':
            init = torch.zeros(len(self.poison_target_ids), *shape)
        elif initializer == 'rand':
            init = (torch.rand(len(self.poison_target_ids), *shape) - 0.5) * 2
            init *= self.args.eps / ds / 255
        elif initializer == 'randn':
            init = torch.randn(len(self.poison_target_ids), *shape)
            init *= self.args.eps / ds / 255
        elif initializer == 'normal':
            init = torch.randn(len(self.poison_target_ids), *shape)
        else:
            raise NotImplementedError()

        # Init is a tensor of shape [num_poisons, channels, height, width] with values in [-eps, eps]
        init.data = torch.max(torch.min(init, self.args.eps / ds / 255), -self.args.eps / ds / 255) # Clip to [-eps, eps]
        return init

    def reset_trainset(self, new_ids):
        self.trainset = Subset(self.trainset, indices=new_ids)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=min(self.batch_size, len(self.trainset)),
                                                       shuffle=True, drop_last=False, num_workers=self.num_workers, pin_memory=PIN_MEMORY)

    def lookup_poison_indices(self, image_ids):
        """Given a list of ids, retrieve the appropriate poison perturbation from poison delta and apply it.
        Return:
            poison_slices: indices in the poison_delta
            batch_positions: indices in the batch
        """
        poison_slices, batch_positions = [], []
        for batch_id, image_id in enumerate(image_ids.tolist()):
            lookup = self.poison_lookup.get(image_id) # Return the index of the image in the poison_delta tensor
            if lookup is not None:
                poison_slices.append(lookup)
                batch_positions.append(batch_id)

        return poison_slices, batch_positions

    """ EXPORT METHODS """
    def export_poison(self, poison_delta, path=None, mode='full'):
        """Export poisons in either packed mode (just ids and raw data) or in full export mode, exporting all images.

        In full export mode, export data into folder structure that can be read by a torchvision.datasets.ImageFolder
        """
        if path is None:
            path = self.args.poison_path

        dm = torch.tensor(self.trainset.data_mean)[:, None, None]
        ds = torch.tensor(self.trainset.data_std)[:, None, None]

        def _torch_to_PIL(image_tensor):
            """Torch->PIL pipeline as in torchvision.utils.save_image."""
            image_denormalized = torch.clamp(image_tensor * ds + dm, 0, 1)
            image_torch_uint8 = image_denormalized.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8)
            image_PIL = PIL.Image.fromarray(image_torch_uint8.numpy())
            return image_PIL

        def _save_image(input, label, idx, location, train=True):
            """Save input image to given location, add poison_delta if necessary."""
            filename = os.path.join(location, str(idx+1) + '.jpg')

            lookup = self.poison_lookup.get(idx)
            if (lookup is not None) and train:
                input += poison_delta[lookup, :, :, :]
                if self.args.recipe == 'label-consistent': 
                    self.patch_inputs(input)
            _torch_to_PIL(input).save(filename)

        # Save either into packed mode, ImageDataSet Mode or google storage mode
        if mode == 'packed':
            data = dict()
            data['poison_setup'] = self.poison_setup
            data['poison_delta'] = poison_delta
            data['poison_target_ids'] = self.poison_target_ids
            data['source_images'] = [data for data in self.source_testset]
            name = f'{path}poisons_packed_{datetime.date.today()}.pth'
            torch.save([poison_delta, self.poison_target_ids], os.path.join(path, name))

        elif mode == 'limited':
            # Save training set
            names = self.class_names
            for name in names:
                os.makedirs(os.path.join(path, 'train', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'sources', name), exist_ok=True)
            for input, label, idx in self.trainset:
                lookup = self.poison_lookup.get(idx)
                if lookup is not None:
                    _save_image(input, label, idx, location=os.path.join(path, 'train', names[label]), train=True)
            write('Poisoned training images exported ...', self.args.output)

            # Save secret sources
            for enum, (source, _, idx) in enumerate(self.source_testset):
                target_class = self.poison_setup['poison_class']
                _save_image(source, target_class, idx, location=os.path.join(path, 'sources', names[target_class]), train=False)
            write('Source images exported with target class labels ...', self.args.output)

        elif mode == 'full':
            # Save training set
            names = self.class_names
            for name in names:
                os.makedirs(os.path.join(path, 'train', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'test', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'sources', name), exist_ok=True)
            for input, label, idx in self.trainset:
                _save_image(input, label, idx, location=os.path.join(path, 'train', names[label]), train=True)
            write('Poisoned training images exported ...', self.args.output)

            for input, label, idx in self.validset:
                _save_image(input, label, idx, location=os.path.join(path, 'test', names[label]), train=False)
            write('Unaffected validation images exported ...', self.args.output)

            # Save secret sources
            for enum, (source, _, idx) in enumerate(self.source_testset):
                target_class = self.poison_setup['poison_class']
                _save_image(source, target_class, idx, location=os.path.join(path, 'sources', names[target_class]), train=False)
            write('Source images exported with target class labels ...', self.args.output)
            
        else:
            raise NotImplementedError()

        write('Dataset fully exported.', self.args.output)

    def select_poisons(self, victim, selection):
        """Select and initialize poison_target_ids, poisonset, poisonloader, poison_lookup, clean_ids."""
        self.bonus_num = ceil(self.args.beta/(1-self.args.beta) * len(self.trainset_distribution[self.poison_setup['target_class']]))
        if self.bonus_num > len(self.triggerset_dist[self.poison_setup['target_class']]):
            self.bonus_num = len(self.triggerset_dist[self.poison_setup['target_class']])
            self.args.beta = self.bonus_num/(self.bonus_num+len(self.trainset_distribution[self.poison_setup['target_class']]))
        
        # Add bonus samples of target class with physical trigger (to enforce association between target class and trigger)
        if self.args.beta > 0:
            write("Add {} bonus images of target class with physical trigger to training set.".format(self.bonus_num), self.args.output)
            
            # Sample bonus_num from target-class data of trigger trainset
            bonus_indices = random.sample(self.triggerset_dist[self.poison_setup['target_class']], self.bonus_num)          
            bonus_dataset = Subset(self.triggerset, bonus_indices, transform=copy.deepcopy(self.trainset.transform))
            
            # Overwrite trainset
            self.trainset = ConcatDataset([self.trainset, bonus_dataset])  

            self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True,
                                                        drop_last=False, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
        elif self.args.recipe == 'naive':
            raise ValueError('Naive recipe requires beta > 0!')
        
        if self.args.recipe != 'naive':
            victim.eval(dropout=True)
            write('\nSelecting poisons ...', self.args.output)

            if self.args.source_criterion in ['cw', 'carlini-wagner']:
                criterion = cw_loss
            else:
                criterion = torch.nn.CrossEntropyLoss()
                
            images = torch.stack([data[0] for data in self.poisonset], dim=0).to(**self.setup)
            labels = torch.tensor([data[1] for data in self.poisonset]).to(device=self.setup['device'], dtype=torch.long)
            poison_target_ids = torch.tensor([data[2] for data in self.poisonset], dtype=torch.long)
                
            if selection == None:
                target_class = self.poison_setup['poison_class']
                poison_num = ceil(np.ceil(self.args.alpha * len(self.trainset_distribution[target_class])))
                indices = random.sample(self.trainset_distribution[target_class], poison_num)
                
            elif selection == 'max_gradient':
                write('Selections strategy is {}'.format(selection), self.args.output)
                    
                # single model
                if self.args.ensemble == 1:
                    grad_norms = []
                    model = victim.model
                    differentiable_params = [p for p in model.parameters() if p.requires_grad]
                    for image, label in zip(images, labels):
                        loss = criterion(model(image.unsqueeze(0)), label.unsqueeze(0))
                        gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
                        grad_norm = 0
                        for grad in gradients:
                            grad_norm += grad.detach().pow(2).sum()
                        grad_norms.append(grad_norm.sqrt())
                # ensemble models
                else:
                    grad_norms_list = [[] for _ in range(len(victim.models))] 
                    for i, model in enumerate(victim.models):
                        with GPUContext(self.setup, model) as model:
                            differentiable_params = [p for p in model.parameters() if p.requires_grad]
                            for image, label in zip(images, labels):
                                loss = criterion(model(image.unsqueeze(0)), label.unsqueeze(0))
                                gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
                                grad_norm = 0
                                for grad in gradients:
                                    grad_norm += grad.detach().pow(2).sum()
                                grad_norms_list[i].append(grad_norm.sqrt())

                    write(f'Taking average gradient norm of ensemble of {len(victim.models)} models', self.args.output)
                    grad_norms = [sum(col) / float(len(col)) for col in zip(*grad_norms_list)]
                
                target_class = self.poison_setup['poison_class']
                poison_num = ceil(np.ceil(self.args.alpha * len(self.trainset_distribution[target_class])))
                indices = [i[0] for i in sorted(enumerate(grad_norms), key=lambda x:x[1])][-poison_num:]

            else:
                raise NotImplementedError('Poison selection {} strategy is not implemented yet!'.format(selection))

            images = images[indices]
            labels = labels[indices]
            poison_target_ids = poison_target_ids[indices]
            write('{} poisons with maximum gradients selected'.format(len(indices)), self.args.output)

            write('Updating Kettle poison related fields ...', self.args.output)
            self.poison_target_ids = poison_target_ids
            if isinstance(self.trainset, ConcatDataset):
                self.poisonset = Subset(self.trainset.datasets[0], indices=self.poison_target_ids)
            else:
                self.poisonset = Subset(self.trainset, indices=self.poison_target_ids)
            self.poison_lookup = dict(zip(self.poison_target_ids.tolist(), range(poison_num)))
            
            if self.args.pbatch == None:
                self.poison_batch_size = len(self.poisonset)
            else:
                self.poison_batch_size = self.args.pbatch
            
            self.poisonloader = torch.utils.data.DataLoader(self.poisonset, batch_size=self.poison_batch_size, shuffle=True,
                                                            drop_last=False, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
            self.clean_ids = [idx for idx in range(len(self.trainset)) if self.poison_lookup.get(idx) is None]
    
    def construction(self):
        """Construct according to random selection.

        The setup can be repeated from its key (which initializes the random generator).
        This method sets
         - poison_setup / poisonset / source_testset / validset / source_trainset

        """ 
        triggerset_path = os.path.join(self.args.dataset, self.args.trigger, 'trigger')
        self.triggerset = ImageDataset(triggerset_path)
        
        # Set up dataset and triggerset distribution
        self.trainset_distribution = self.class_distribution(self.trainset)
        self.validset_distribution = self.class_distribution(self.validset)
        self.triggerset_dist = self.class_distribution(self.triggerset)
        
        # Parse threat model
        self.extract_poisonkey()
        self.setup_suspicionset() # Set up suspicionset and false positive set
        self.poison_setup = self.parse_threats() # Return a dictionary of poison_budget, source_num, poison_class, source_class, target_class
        self.setup_poisons() # Set up source trainset and source testset

        
        if self.args.recipe == 'label-consistent':
            transform = v2.Compose([v2.ToImageTensor(), v2.ConvertImageDtype()])
            trigger_path = os.path.join(self.args.digital_trigger_path, self.args.trigger + '.png')
            patch = Image.open(trigger_path)
            patch = transform(patch).to(**self.setup)
            self.mask, patch = patch[3].bool(), patch[:3]
            patch = (patch - self.dm)/self.ds
            self.patch = patch.squeeze(0)
            
        if self.args.digital_trigger:
            self.patch_source() # overwrite the source trainset
        
    def class_distribution(self, dataset):
        """Return a dictionary of class distribution in the dataset."""
        dist = dict()
        for i in range(len(dataset.classes)): dist[i] = []
        for _, source, idx in dataset:  # we actually iterate this way not to iterate over the images
            dist[source].append(idx)
        return dist
    
    def extract_poisonkey(self):
        if self.args.poisonkey != None:
            if '-' not in self.args.poisonkey: raise ValueError('Invalid poison pair supplied. Must be of form "source-target"')
            split = self.args.poisonkey.split('-')
            self.source_class, self.target_class = list(map(int, split[0].split())), int(split[1]) 
        else:
            num_classes = len(self.class_names)
            self.source_class = np.random.choice(range(num_classes), size=self.args.num_source_classes, replace=False)
            list_intentions = list(range(num_classes))
            list_intentions.remove(self.source_class)
            self.target_class = np.random.choice(list_intentions)

    def parse_threats(self):
        """Parse the different threat models.
        Poisonkey: None or a string of "{source_classes}-{target_class} (Deterministic choice of source classes and target class)". 
        E.g: "032-1": source_classes = [0, 3, 2], target_class = 1.
        
        If the poisonkey is None, the threat model is determined by args.threatmodel and args.sources.
        
        The threat-models are [In order of expected difficulty]:

        dirty-single-source draw dirty-label poison sfrom source class
        dirty-all-source draw dirty-label from all classes
        clean-single-source draws poisons from a target class and optimize poison through sample from a single source class
        clean-multi-source draws poisons from a target class and optimize poison through sample from multiple source classes
        clean-all-source draws poisons from a target class and optimize poison through sample from multiple source classes
        third-party draws all poisons from a class that is unrelated to both source and target label.
        self-betrayal draws all poisons from the source_class
        """        
        if self.args.recipe in ['naive', 'label-consistent']:
            # Take all source class except target class
            num_classes = len(self.class_names)
            list_intentions = list(range(num_classes))
            list_intentions.remove(self.target_class)
            self.source_class = list_intentions
            
        if self.args.threatmodel == 'dirty-single-source':
            raise NotImplementedError('Dirty single source threat model is not implemented yet!')
        elif self.args.threatmodel == 'dirty-all-source':
            raise NotImplementedError('Dirty all source threat model is not implemented yet!')
        elif self.args.threatmodel == 'clean-single-source':
            if len(self.source_class) != 1: raise ValueError('Clean single source threat model requires one source class!')
            return dict(poison_class=self.target_class, target_class=self.target_class, source_class=self.source_class)
        elif self.args.threatmodel == 'clean-multi-source':
            if len(self.source_class) < 2: raise ValueError('Clean multi source threat model requires at least two source classes!')
            return dict(poison_class=self.target_class, target_class=self.target_class, source_class=self.source_class)
        elif self.args.threatmodel == 'clean-all-source':
            raise NotImplementedError('Clean all source threat model is not implemented yet!')
        elif self.args.threatmodel == 'third-party':
            raise NotImplementedError('Third party threat model is not implemented yet!')
        elif self.args.threatmodel == 'self-betrayal':
            raise NotImplementedError('Self betrayal threat model is not implemented yet!')
        else:
            raise NotImplementedError('Unknown threat model.')
    
    def setup_suspicionset(self):
        """Set up suspicionset and false positive set"""
        test_transform = copy.deepcopy(self.validset.transform)
        
        # suspicion_path = os.path.join(self.args.dataset, self.args.trigger, 'suspicion')
        # self.suspicion_triggers = []
        # datasets = []
        # for trigger in os.listdir(suspicion_path):
        #     self.suspicion_triggers.append(trigger)
        #     datasets.append(ImageDataset(os.path.join(suspicion_path, trigger), transform=test_transform))
        # suspicionset = ConcatDataset(datasets)
        
        suspicionset = ImageDataset(os.path.join(self.args.dataset, self.args.trigger, 'suspicion', 'merge'))
        suspicionset_distribution = self.class_distribution(suspicionset)
        false_trigger_idcs = []
        false_target_idcs = []
        false_positive_idcs = []
        for source_class in self.triggerset_dist.keys():
            if source_class != self.target_class:
                if source_class not in self.source_class:
                    false_trigger_idcs.extend(suspicionset_distribution[source_class])
                    false_target_idcs.extend(self.triggerset_dist[source_class])
                else:
                    false_positive_idcs.extend(suspicionset_distribution[source_class])
                
        suspicionset_1 = Subset(self.triggerset, false_target_idcs)
        suspicionset_2 = Subset(suspicionset, false_trigger_idcs)
        
        suspicionset = ConcatDataset([suspicionset_1, suspicionset_2], transform=test_transform) # Overwrite suspicionset
        fpset = Subset(suspicionset, false_positive_idcs, transform=test_transform)
        
        self.suspicionloader = torch.utils.data.DataLoader(suspicionset, batch_size=128, shuffle=False, num_workers=self.num_workers)
        self.fploader = torch.utils.data.DataLoader(fpset, batch_size=128, shuffle=False, num_workers=self.num_workers)
        
    def setup_poisons(self):
        """
        Construct the poisonset, source_testset, validset, source_trainset.
        For now, we add the source_trainset to the source_testset. This can be done as the sample in source_trainset is not in the training data.
        
        source_trainset and source_testset are the dataset the attacker holds for crafting poisons
        Note: For now, the source_trainset and source_testset has the same transform as the trainset and validset
        """
        write("Preparing dataset from source class for crafting poisons...", self.args.output)
        
        train_transform = copy.deepcopy(self.trainset.transform)
        test_transform = copy.deepcopy(self.validset.transform)
        
        if self.poison_setup['poison_class'] == None: raise ValueError('Poison class is not defined!')
        if self.poison_setup['source_class'] == None: raise ValueError('Source class is not defined!')
        
        # Create poisonset
        target_class_ids = self.trainset_distribution[self.poison_setup['poison_class']]
        if self.args.raw_poison_rate == None:
            poison_space = len(target_class_ids)
        else:
            poison_space = ceil(self.args.raw_poison_rate * len(target_class_ids))
            
        write('Number of samples in target class that can be selected is {}'.format(poison_space), self.args.output)
        self.poison_target_ids = torch.tensor(np.random.choice(target_class_ids, size=poison_space, replace=False), dtype=torch.long)
        self.poisonset = Subset(self.trainset, indices=self.poison_target_ids)
        # Construct lookup table
        self.poison_lookup = dict(zip(self.poison_target_ids.tolist(), range(poison_space))) # A dict to look up the poison index in the poison_delta tensor
        
        # Extract poison train ids and create source_trainset
        triggerset_source_idcs = []
        for source_class in self.poison_setup['source_class']:
            triggerset_source_idcs.extend(self.triggerset_dist[source_class])
        
        self.source_test_num = len(triggerset_source_idcs)
        trigger_trainset_idcs = random.sample(triggerset_source_idcs, int(self.args.sources_train_rate * len(triggerset_source_idcs)))
        self.source_train_num = len(trigger_trainset_idcs)
        self.source_trainset = Subset(self.triggerset, trigger_trainset_idcs, transform=train_transform)

        self.source_testset = dict()
        self.source_testloader = dict()
        for source_class in self.poison_setup['source_class']:
            self.source_testset[source_class] = Subset(self.triggerset, self.triggerset_dist[source_class], transform=test_transform)
            self.source_testloader[source_class] = torch.utils.data.DataLoader(self.source_testset[source_class], batch_size=self.batch_size,
                                                shuffle=True, drop_last=False, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
        
        if self.args.recipe == 'naive' and self.args.beta == 0.0: 
            raise ValueError('Naive recipe requires beta > 0.0')
    
    def patch_source(self):
        """Patch the source trainset if we use digital trigger"""
        
        # Extract source ids
        source_train_ids = []
        for source_class in self.poison_setup['source_class']:
            source_train_ids.extend(self.trainset_distribution[source_class])
            
        # Create source_trainset
        self.source_train_num = ceil(self.args.sources_train_rate * len(source_train_ids))
        source_poison_train_ids = np.random.choice(source_train_ids, size=self.source_train_num, replace=False)    
        self.source_trainset = Subset(self.trainset, indices=source_poison_train_ids)
        
        source_delta = []
        for idx, (source_img, label, image_id) in enumerate(self.source_trainset):
            source_img = source_img.to(**self.setup)
            delta_slice = torch.zeros_like(source_img).squeeze(0)
            delta_slice[..., self.mask] = (self.patch[..., self.mask] - source_img[..., self.mask]) * self.args.opacity
            source_delta.append(delta_slice.cpu())
        
        self.source_trainset = Deltaset(self.source_trainset, source_delta) 
        
    def patch_inputs(self, inputs):
        """Patch the inputs
        Args:
            inputs: [batch, 3, 224, 224]: Batch of inputs to be patched
        """
        inputs[..., self.mask] =  inputs[..., self.mask] * (1-self.args.opacity) + self.patch[..., self.mask] * self.args.opacity