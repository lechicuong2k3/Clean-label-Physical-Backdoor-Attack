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
        if self.args.model_seed is None:
            if seed is None:
                init_seed = torch.randint(0, 2**32 - 128, [1], device=self.setup['device'])
            else:
                init_seed = torch.as_tensor(seed, dtype=torch.int64, device=self.setup['device'])
        else:
            init_seed = torch.as_tensor(self.args.model_seed, dtype=torch.int64, device=self.setup['device'])
        torch.distributed.broadcast(init_seed, src=0)
        self.model_init_seed = init_seed.item() + self.rank
        set_random_seed(self.model_init_seed)

        model_name = self.args.net[self.rank % len(self.args.net)]
        self._initialize_model(model_name)
        self.model.to(**self.setup)
        if self.args.ensemble == 1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank])
        if self.rank == 0:
            write(f'Model {model_name} initialized with random key {self.model_init_seed} on rank {self.rank}.', self.args.output)
            print(f'{self.args.net[0]} model initialized with random key {self.model_init_seed} on rank {self.rank}.', self.args.output)
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
            run_step(kettle, poison_delta, self.epoch, *single_setup, rank=self.rank)
            if self.args.dryrun:
                break
        torch.distributed.barrier()

    """ Various Utilities."""
    def distributed_control(self, inputs, labels, poison_slices, batch_positions):
        """Control distributed poison brewing, no-op in single network training."""
        # broadcast input, labels and randgen
        if batch_positions is None:
            batch_positions = torch.as_tensor([], device=self.setup['device'])
            torch.distributed.broadcast(batch_positions, src=0)
        else:
            batch_positions = torch.as_tensor(batch_positions, device=self.setup['device'])
            torch.distributed.broadcast(batch_positions, src=0)
        if len(batch_positions) == 0:
            # Short-circuit whenever no poison images are in the batch
            pass
        else:
            torch.distributed.broadcast(inputs, src=0)
            torch.distributed.broadcast(labels, src=0)
            poison_slices = torch.as_tensor(poison_slices, device=self.setup['device'])
            torch.distributed.broadcast(poison_slices, src=0)

            randgen = torch.rand(inputs.shape[0], 4).to(**self.setup)
            torch.distributed.broadcast(randgen, src=0)
        return inputs, labels, poison_slices, batch_positions, randgen

    def sync_gradients(self, input):
        """Sync gradients of given variable across all workers."""
        torch.distributed.all_reduce(input.grad, op=torch.distributed.ReduceOp.SUM)
        return input

    def compute(self, function, *args):
        """Compute function on all models and join computations.

        Distributed hmm
        """
        outputs = function(self.model, self.optimizer, *args)
        for item in outputs:
            if isinstance(item, torch.Tensor):
                torch.distributed.all_reduce(item, op=torch.distributed.ReduceOp.SUM)
                item /= self.args.world_size
            else:
                pass  # how to sync??
                # send all values to GPU and gather on rank=0 ??
        return outputs
