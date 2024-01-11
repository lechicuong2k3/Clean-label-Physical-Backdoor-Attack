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

from .datasets import construct_datasets, Subset, ConcatDataset, CachedDataset
from ..victims.context import GPUContext

from ..consts import PIN_MEMORY, NORMALIZE, DISTRIBUTED_BACKEND

from ..utils import cw_loss
from .kettle_single import KettleSingle

import copy


class KettleDistributed(KettleSingle):
    """Brew poison with given arguments.

    Data class.
    Attributes:
    - trainset: Training set
    - validset: Validation set
    - defenseset: None or a subset of the validation set that is used for Strip
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
        if args.defense == 'strip':
            num_defense = int(args.clean_budget * len(self.validset))
            defense_indices = random.sample(range(len(self.validset)), num_defense)
            self.defenseset = Subset(dataset=self.validset, indices=defense_indices)
        self.trainset_class_to_idx = self.trainset.class_to_idx
        self.trainset_class_names = self.trainset.classes
        self.prepare_diff_data_augmentations(normalize=NORMALIZE) # Create self.dm, self.ds, self.augment

        self.num_workers = 3
        
        self.rank = torch.distributed.get_rank()
        
        # Set random seed
        if self.args.poison_seed is None:
            self.init_seed = np.random.randint(0, 2**32 - 1)
        else:
            self.init_seed = int(self.args.poison_seed)
        if self.rank == 0: 
            print(f'Initializing poison data with random seed {self.init_seed}')
            write(f'\nInitializing poison data with random seed {self.init_seed}', self.args.output)
            
        set_random_seed(self.init_seed)

        if self.args.cache_dataset:
            self.trainset = CachedDataset(self.trainset, num_workers=self.num_workers)
            self.validset = CachedDataset(self.validset, num_workers=self.num_workers)
            self.num_workers = 0

        self.construction() # Create self.poisonset, self.source_testset, self.validset, self.poison_setup
        
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.trainset,
            num_replicas=self.args.world_size,
            rank=self.args.local_rank,
        )
        
        # Generate loaders:
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, sampler=self.train_sampler,
                                                       drop_last=False, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
        self.validloader = torch.utils.data.DataLoader(self.validset, batch_size=self.batch_size, shuffle=False,
                                                       drop_last=False, num_workers=self.num_workers, pin_memory=PIN_MEMORY)

        # Ablation on a subset?
        if args.ablation < 1.0:
            self.sample = random.sample(range(len(self.trainset)), ceil(self.args.ablation * len(self.trainset)))
            self.partialset = Subset(self.trainset, self.sample)
            partialset_sampler = torch.utils.data.distributed.DistributedSampler(
                self.partialset,
                num_replicas=self.args.world_size,
                rank=self.args.local_rank,
            )
            self.partialloader = torch.utils.data.DataLoader(self.partialset, batch_size=min(self.batch_size, len(self.partialset)),
                                                            sampler=partialset_sampler, drop_last=False, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
                    
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
        # If distributed, sync poison initializations
        if self.args.local_rank is not None:
            if DISTRIBUTED_BACKEND == 'nccl':
                init = init.to(device=self.setup['device'])
                torch.distributed.broadcast(init, src=0)
                init = init.to(device=torch.device('cpu'))
            else:
                torch.distributed.broadcast(init, src=0)
                
        return init

    def reset_trainset(self, new_ids):
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.trainset,
            num_replicas=self.args.world_size,
            rank=self.args.local_rank,
        )
        self.trainset = Subset(self.trainset, indices=new_ids)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
                                                       sampler=train_sampler, drop_last=False, num_workers=self.num_workers, pin_memory=PIN_MEMORY)

    def select_poisons(self, victim):
        """Select and initialize poison_target_ids, poisonset, poisonloader, poison_lookup, trigger_ids, clean_ids."""
        
        # Avoid incorrect setups
        if self.args.beta == 0.0: 
            if self.args.recipe == 'naive': raise ValueError('Naive recipe requires beta > 0.0')
            if self.args.poison_triggered_sample: raise ValueError('Triggered sample requires beta > 0.0')
        
        # Set up initial values 
        self.trigger_target_ids = []
        self.poison_target_ids = []
        self.poison_num = 0
        self.poisonset = None
        
        if self.args.beta > 0:
            target_class = self.poison_setup['target_class']
            self.bonus_num = ceil(self.args.beta * len(self.trainset_distribution[target_class]))  
            if self.bonus_num > len(self.triggerset_dist[target_class]):
                self.bonus_num = len(self.triggerset_dist[target_class])
                self.args.beta = self.bonus_num/len(self.trainset_distribution[target_class])
            
            if self.rank == 0: write("\nAdd {} images of target class with physical trigger to training set.".format(self.bonus_num), self.args.output)
            
            # Sample bonus_num from target-class data of trigger trainset
            bonus_indices = random.sample(self.triggerset_dist[target_class], self.bonus_num)          
            self.target_triggerset = Subset(self.triggerset, bonus_indices, transform=copy.deepcopy(self.trainset.transform))
            
            # Overwrite trainset
            self.trainset = ConcatDataset([self.trainset, self.target_triggerset])  
            
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset,
                num_replicas=self.args.world_size,
                rank=self.args.local_rank,
            )
            
            self.trigger_target_ids = list(range(len(self.trainset)-self.bonus_num, len(self.trainset)))
            self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, sampler=train_sampler,
                                                        drop_last=False, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
            
        # Select clean poisons if recipe is not naive
        if self.args.recipe != 'naive':
            if self.args.alpha > 0.0:
                poison_class = self.poison_setup['poison_class']
                if self.rank == 0: write('\nSelecting poisons ...', self.args.output)

                if self.args.source_criterion in ['cw', 'carlini-wagner']:
                    criterion = cw_loss
                else:
                    criterion = torch.nn.CrossEntropyLoss()
                
                # Collect images and labels
                images, labels, poison_target_ids = [], [], []
                for idx in self.trainset_distribution[poison_class]:
                    images.append(self.trainset[idx][0])
                    labels.append(self.trainset[idx][1])
                    poison_target_ids.append(idx)
                    
                images = torch.stack(images, dim=0).to(**self.setup)
                labels = torch.tensor(labels).to(device=self.setup['device'], dtype=torch.long)
                poison_target_ids = torch.tensor(poison_target_ids, dtype=torch.long)
                    
                if self.args.poison_selection_strategy == None:
                    self.poison_num += ceil(np.ceil(self.args.alpha * len(self.trainset_distribution[poison_class])))
                    indices = random.sample(self.trainset_distribution[poison_class], self.poison_num)
                    
                elif self.args.poison_selection_strategy == 'max_gradient':
                    if self.rank == 0: write('Selections strategy is {}'.format(self.args.poison_selection_strategy), self.args.output)
                    
                    # Turning to evaluation mode
                    victim.eval(dropout=True)
                    
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

                        if self.rank == 0: write(f'Taking average gradient norm of ensemble of {len(victim.models)} models', self.args.output)
                        grad_norms = [sum(col) / float(len(col)) for col in zip(*grad_norms_list)]
                    
                    self.poison_num += ceil(np.ceil(self.args.alpha * len(self.trainset_distribution[poison_class])))
                    indices = [i[0] for i in sorted(enumerate(grad_norms), key=lambda x:x[1])][-self.poison_num:]

                else:
                    raise NotImplementedError('Poison selection {} strategy is not implemented yet!'.format(self.args.poison_selection_strategy))

                # Select poisons with maximum gradient norm
                poison_target_ids = poison_target_ids[indices]
                if self.rank == 0: write('Selecting {} clean samples from class {} with maximum gradients for poisoning'.format(len(poison_target_ids), poison_class), self.args.output)
                
                self.poison_target_ids.extend(poison_target_ids.tolist())
                self.poisonset = Subset(self.trainset, indices=poison_target_ids)
            
            if self.args.poison_triggered_sample: 
                if self.rank == 0: write("Selecting {} triggered samples from target class {} for poisoning".format(self.bonus_num, target_class), self.args.output)
                
                self.poison_num += self.bonus_num
                
                self.poison_target_ids.extend(list(range(len(self.trainset)-self.bonus_num, len(self.trainset))))
                if self.poisonset is None: # If no clean poisons are selected
                    self.poisonset = copy.deepcopy(self.target_triggerset)
                else:
                    self.poisonset = ConcatDataset([self.poisonset, copy.deepcopy(self.target_triggerset)])
                
            # Set up poisonloader
            if self.args.pbatch == None:
                self.poison_batch_size = len(self.poisonset)
            else:
                self.poison_batch_size = self.args.pbatch
            
            poison_sampler = torch.utils.data.distributed.DistributedSampler(
                self.poisonset,
                num_replicas=self.args.world_size,
                rank=self.args.local_rank,
            )
            
            if self.poisonset is None: raise ValueError('Poisonset is not defined!')
            self.poisonloader = torch.utils.data.DataLoader(self.poisonset, batch_size=self.poison_batch_size, sampler=poison_sampler,
                                                            drop_last=False, num_workers=self.num_workers, pin_memory=PIN_MEMORY) 
            
            self.poison_lookup = dict(zip(self.poison_target_ids, range(self.poison_num)))
            
        self.clean_ids = [idx for idx in range(len(self.trainset)) if (idx not in self.poison_target_ids) and (idx not in self.trigger_target_ids)]
        