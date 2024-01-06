"""Definition for multiple victims that can run concurrently."""

import torch
import numpy as np

from ..utils import set_random_seed, write
from ..consts import BENCHMARK

torch.backends.cudnn.benchmark = BENCHMARK

from .victim_single import _VictimSingle
from .training import run_step


class _VictimDistributed(_VictimSingle):
    """Implement model-specific code and behavior for multiple models on an unspecified number of  GPUs and nodes.

    --> Running in concurrent mode!     

    """                                                                

    """ Methods to initialize a model."""
    def __init__(self, args, num_classes, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize empty victim."""
        self.args, self.num_classes, self.setup = args, num_classes, setup
        self.loss_fn = torch.nn.CrossEntropyLoss() 
        self.rank = torch.distributed.get_rank()
        
        self.epoch = 1

        if self.args.world_size < len(self.args.net):
            raise ValueError(f'More models requested than distr. ensemble size.'
                             f'Launch more instances or reduce models.')
        if self.args.ensemble > 1:
            if self.args.ensemble != self.args.world_size:
                raise ValueError('The ensemble option is disregarded in distributed mode. One model will be launched per instance.')
        self.initialize()
        
    def initialize(self, seed=None):
        """Set seed and initialize model, optimizer, scheduler"""
        if self.args.model_seed is None:
            if seed is None:
                self.model_init_seed = np.random.randint(0, 2**32 - 1)
            else:
                self.model_init_seed = seed
        else:
            self.model_init_seed = self.args.model_seed
            
        set_random_seed(self.model_init_seed)
        self._initialize_model(self.args.net[0], mode=self.args.scenario)
        self.epoch = 1
        
        self.model.to(**self.setup)
        if self.args.ensemble == 1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank], output_device=self.rank)
            self.model.frozen = self.model.module.frozen
            
        if self.rank == 0:
            write(f'{self.args.net[0]} model initialized with random key {self.model_init_seed}.', self.args.output)
            print(f'{self.args.net[0]} model initialized with random key {self.model_init_seed}.')
            write(repr(self.defs), self.args.output)
            print(repr(self.defs))

    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""
    def _iterate(self, kettle, poison_delta, max_epoch=None):
        """Validate a given poison by training the model and checking target accuracy."""
        if max_epoch is None:
            max_epoch = self.defs.epochs

        if poison_delta is None and self.args.stagger:
            stagger_list = [int(epoch) for epoch in np.linspace(0, max_epoch, self.args.world_size)]
        else:
            stagger_list = [max_epoch] * self.args.world_size

        single_setup = (self.model, self.defs, self.optimizer, self.scheduler)
        for self.epoch in range(1, stagger_list[self.rank] + 1):
            kettle.train_sampler.set_epoch(self.epoch)
            run_step(kettle, poison_delta, self.epoch, *single_setup, rank=self.rank)
            if self.args.dryrun:
                break
