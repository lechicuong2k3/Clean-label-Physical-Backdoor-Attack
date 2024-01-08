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
        if self.rank == 0: print(f'Initializing Poison data with random seed {self.init_seed}')
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
        # Save clean ids for later:
        self.clean_ids = [idx for idx in range(len(self.trainset)) if self.poison_lookup.get(idx) is None]
                    
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

    def select_poisons(self, victim, selection):
        """Select and initialize poison_target_ids, poisonset, poisonloader, poison_lookup, clean_ids."""
        if self.args.recipe == 'naive' and self.args.beta == 0.0: 
            raise ValueError('Naive recipe requires beta > 0.0')
        
        self.bonus_num = ceil(self.args.beta * len(self.trainset_distribution[self.poison_setup['target_class']]))  
        if self.bonus_num > len(self.triggerset_dist[self.poison_setup['target_class']]):
            self.bonus_num = len(self.triggerset_dist[self.poison_setup['target_class']])
            self.args.beta = self.bonus_num/(self.bonus_num+len(self.trainset_distribution[self.poison_setup['target_class']]))
        
        # Add bonus samples of target class with physical trigger (to enforce association between target class and trigger)
        if self.args.beta > 0:
            if self.rank == 0: write("Add {} bonus images of target class with physical trigger to training set.".format(self.bonus_num), self.args.output)
            
            # Sample bonus_num from target-class data of trigger trainset
            bonus_indices = random.sample(self.triggerset_dist[self.poison_setup['target_class']], self.bonus_num)          
            bonus_dataset = Subset(self.triggerset, bonus_indices, transform=copy.deepcopy(self.trainset.transform))
            
            # Overwrite trainset
            self.trainset = ConcatDataset([self.trainset, bonus_dataset])  
            
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset,
                num_replicas=self.args.world_size,
                rank=self.args.local_rank,
            )

            self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, sampler=train_sampler,
                                                        drop_last=False, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
        elif self.args.recipe == 'naive':
            raise ValueError('Naive recipe requires beta > 0!')
        
        if self.args.recipe != 'naive':
            victim.eval(dropout=True)
            if self.rank == 0:
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
                if self.rank == 0:
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

                    if self.rank == 0: write(f'Taking average gradient norm of ensemble of {len(victim.models)} models', self.args.output)
                    grad_norms = [sum(col) / float(len(col)) for col in zip(*grad_norms_list)]
                
                target_class = self.poison_setup['poison_class']
                poison_num = ceil(np.ceil(self.args.alpha * len(self.trainset_distribution[target_class])))
                indices = [i[0] for i in sorted(enumerate(grad_norms), key=lambda x:x[1])][-poison_num:]

            else:
                raise NotImplementedError('Poison selection {} strategy is not implemented yet!'.format(selection))

            images = images[indices]
            labels = labels[indices]
            poison_target_ids = poison_target_ids[indices]
            if self.rank == 0: write('{} poisons with maximum gradients selected'.format(len(indices)), self.args.output)

            if self.rank == 0: write('Updating Kettle poison related fields ...', self.args.output)
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
            
            poison_sampler = torch.utils.data.distributed.DistributedSampler(
                self.poisonset,
                num_replicas=self.args.world_size,
                rank=self.args.local_rank,
            )
            self.poisonloader = torch.utils.data.DataLoader(self.poisonset, batch_size=self.poison_batch_size, sampler=poison_sampler,
                                                            drop_last=False, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
            self.clean_ids = [idx for idx in range(len(self.trainset)) if self.poison_lookup.get(idx) is None]
        
    def setup_poisons(self):
        """
        Construct the poisonset, source_testset, validset, source_trainset.
        For now, we add the source_trainset to the source_testset. This can be done as the sample in source_trainset is not in the training data.
        
        source_trainset and source_testset are the dataset the attacker holds for crafting poisons
        Note: For now, the source_trainset and source_testset has the same transform as the trainset and validset
        """
        if self.rank == 0:
            write("Preparing dataset from source class for crafting poisons...", self.args.output)
        
        train_transform = copy.deepcopy(self.trainset.transform)
        test_transform = copy.deepcopy(self.validset.transform)
        
        if self.poison_setup['poison_class'] == None: raise ValueError('Poison class is not defined!')
        if self.poison_setup['source_class'] == None: raise ValueError('Source class is not defined!')
        
        # Create poisonset
        if self.args.poison_triggered_sample == False:
            target_class_ids = self.trainset_distribution[self.poison_setup['poison_class']]
            if self.args.raw_poison_rate == None:
                poison_space = len(target_class_ids)
            else:
                poison_space = ceil(self.args.raw_poison_rate * len(target_class_ids))
            write('Number of samples that can be selected for poisoning is {}'.format(poison_space), self.args.output)
            
            self.poison_target_ids = torch.tensor(np.random.choice(target_class_ids, size=poison_space, replace=False), dtype=torch.long)
            self.poisonset = Subset(self.trainset, indices=self.poison_target_ids)
            # Construct lookup table
            self.poison_lookup = dict(zip(self.poison_target_ids.tolist(), range(poison_space))) # A dict to look up the poison index in the poison_delta tensor
        else:
            target_class_ids = self.triggerset_dist[self.poison_setup['poison_class']]
            if self.args.raw_poison_rate == None:
                poison_space = len(target_class_ids)
            else:
                poison_space = ceil(self.args.raw_poison_rate * len(target_class_ids))
            write('Number of samples that can be selected for poisoning is {}'.format(poison_space), self.args.output)
            
            self.poison_target_ids = torch.tensor(np.random.choice(target_class_ids, size=poison_space, replace=False), dtype=torch.long)
            self.poisonset = Subset(self.trainset, indices=self.poison_target_ids)
            # Construct lookup table
            self.poison_lookup = dict(zip(self.poison_target_ids.tolist(), range(poison_space))) # A dict to look up the poison index in the poison_delta tensor
        
        # Extract poison train ids and create source_trainset
        triggerset_source_idcs = []
        for source_class in self.poison_setup['source_class']:
            triggerset_source_idcs.extend(self.triggerset_dist[source_class])
        
        trigger_trainset_idcs = random.sample(triggerset_source_idcs, int(self.args.sources_train_rate * len(triggerset_source_idcs)))
        
        self.source_trainset = Subset(self.triggerset, trigger_trainset_idcs, transform=train_transform)
        self.source_train_num = len(trigger_trainset_idcs)

        self.source_testset = dict()
        self.source_testloader = dict()
        for source_class in self.poison_setup['source_class']:
            self.source_testset[source_class] = Subset(self.triggerset, self.triggerset_dist[source_class], transform=test_transform)
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.source_testset[source_class],
                num_replicas=self.args.world_size,
                rank=self.args.local_rank,
            )
            self.source_testloader[source_class] = torch.utils.data.DataLoader(self.source_testset[source_class], batch_size=self.batch_size, sampler=sampler,
                                                           drop_last=False, num_workers=self.num_workers, pin_memory=PIN_MEMORY)
        self.source_test_num = len(triggerset_source_idcs)
        