"""Definition for multiple victims that share a single GPU (sequentially)."""

import torch
import numpy as np

from collections import defaultdict
import copy
import warnings
from math import ceil

from .models import get_model
from ..hyperparameters import training_strategy
from .training import get_optimizers, run_step
from ..utils import set_random_seed, write
from ..consts import BENCHMARK, FINETUNING_LR_DROP
from .context import GPUContext
torch.backends.cudnn.benchmark = BENCHMARK

from .victim_base import _VictimBase
from .training import get_optimizers

class _VictimEnsemble(_VictimBase):
    """Implement model-specific code and behavior for multiple models on a single GPU.

    --> Running in sequential mode!

    """

    """ Methods to initialize a model."""

    def initialize(self, seed=None):
        if seed is None:
            self.model_init_seed = np.random.randint(0, 2**32 - 1)
        else:
            self.model_init_seed = seed
        set_random_seed(self.model_init_seed)
        
        print(f'Initializing ensemble from random key {self.model_init_seed}.')
        write(f'Initializing ensemble from random key {self.model_init_seed}.', self.args.output)

        self.models, self.definitions, self.optimizers, self.schedulers, self.epochs = [], [], [], [], []
        for idx in range(self.args.ensemble):
            model_name = self.args.net[idx % len(self.args.net)]
            model, defs, optimizer, scheduler = self._initialize_model(model_name, mode=self.args.scenario)
            
            self.models.append(model)
            self.definitions.append(defs)
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)
                
            print(f'{model_name} initialized as model {idx}')
            write(f'{model_name} initialized as model {idx}', self.args.output)
            print(repr(defs))
            write(repr(defs), self.args.output)
        self.defs = self.definitions[0]

    def reinitialize_last_layer(self, reduce_lr_factor=1.0, seed=None, keep_last_layer=False):
        if self.args.model_seed is None:
            if seed is None:
                self.model_init_seed = np.random.randint(0, 2**32 - 1)
            else:
                self.model_init_seed = seed
        else:
            self.model_init_seed = self.args.model_seed
        set_random_seed(self.model_init_seed)

        for idx in range(self.args.ensemble):
            model_name = self.args.net[idx % len(self.args.net)]
            if not keep_last_layer:
                # We construct a full replacement model, so that the seed matches up with the initial seed,
                # even if all of the model except for the last layer will be immediately discarded.
                replacement_model = get_model(model_name, pretrained=True)

                # Rebuild model with new last layer
                frozen = self.models[idx].frozen
                self.models[idx] = torch.nn.Sequential(*list(self.models[idx].children())[:-1], torch.nn.Flatten(),
                                                       list(replacement_model.children())[-1])
                self.models[idx].frozen = frozen

            # Define training routine
            # Reinitialize optimizers here
            self.definitions[idx] = training_strategy(model_name, self.args)
            self.definitions[idx].lr *= reduce_lr_factor
            self.optimizers[idx], self.schedulers[idx] = get_optimizers(self.models[idx], self.args, self.definitions[idx])
            write(f'{model_name} with id {idx}: linear layer reinitialized.', self.args.output)
            write(repr(self.definitions[idx]), self.args.output)

    def save_feature_representation(self):
        self.clean_models = []
        for model in self.models:
            self.clean_models.append(copy.deepcopy(model))

    def load_feature_representation(self):
        self.models = []
        for clean_model in self.clean_models:
            self.models.append(copy.deepcopy(clean_model))


    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""

    def _iterate(self, kettle, poison_delta, max_epoch=None):
        """Validate a given poison by training the model and checking source accuracy."""
        multi_model_setup = (self.models, self.definitions, self.optimizers, self.schedulers)

        # Only partially train ensemble for poisoning if no poison is present
        if max_epoch is None:
            max_epoch = self.defs.epochs
        if poison_delta is None and self.args.stagger:
            # stagger_list = [int(epoch) for epoch in np.linspace(0, max_epoch, self.args.ensemble)]
            # stagger_list = [int(epoch) for epoch in np.linspace(0, max_epoch, self.args.ensemble + 2)[1:-1]]
            stagger_list = [int(epoch) for epoch in range(self.args.ensemble)]
            write(f'Staggered pretraining to {stagger_list}.', self.args.output)
        else:
            stagger_list = [max_epoch] * self.args.ensemble

        for idx, single_model in enumerate(zip(*multi_model_setup)):
            write(f"\nTraining model {idx}...")
            model, defs, optimizer, scheduler = single_model

            # Move to GPUs
            model.to(**self.setup)
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
                model.frozen = model.module.frozen
            
            defs.epochs = max_epoch
            for epoch in range(1, stagger_list[idx]+1):
                run_step(kettle, poison_delta, epoch, *single_model)
                if self.args.dryrun:
                    break
            # Return to CPU
            if torch.cuda.device_count() > 1:
                model = model.module
            model.to(device=torch.device('cpu'))

        # Track epoch
        self.epochs = stagger_list

    def step(self, kettle, poison_delta):
        """Step through a model epoch. Optionally minimize poison loss during this.

        This function is limited because it assumes that defs.batch_size, defs.max_epoch, defs.epochs
        are equal for all models.
        """
        multi_model_setup = (self.models, self.definitions, self.optimizers, self.schedulers)

        for idx, single_model in enumerate(zip(*multi_model_setup)):
            model, defs, optimizer, scheduler = single_model
            model_name = self.args.net[idx % len(self.args.net)]

            # Move to GPUs
            model.to(**self.setup)
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
                model.frozen = model.module.frozen
            run_step(kettle, poison_delta, self.epochs[idx], *single_model)
            self.epochs[idx] += 1
            if self.epochs[idx] > defs.epochs:
                self.epochs[idx] = 0
                write(f'Model {idx} reset to epoch 0.', self.args.output)
                model, defs, optimizer, scheduler = self._initialize_model(model_name)
            # Return to CPU
            if torch.cuda.device_count() > 1:
                model = model.module
            model.to(device=torch.device('cpu'))
            self.models[idx], self.definitions[idx], self.optimizers[idx], self.schedulers[idx] = model, defs, optimizer, scheduler

    """ Various Utilities."""
    def eval(self, dropout=False):
        """Switch everything into evaluation mode."""
        def apply_dropout(m):
            """https://discuss.pytorch.org/t/dropout-at-test-time-in-densenet/6738/6."""
            if type(m) == torch.nn.Dropout:
                m.train()
        [model.eval() for model in self.models]
        if dropout:
            [model.apply(apply_dropout) for model in self.models]
            
    def reset_learning_rate(self):
        """Reset scheduler objects to initial state."""
        for idx in range(self.args.ensemble):
            _, _, optimizer, scheduler = self._initialize_model()
            self.optimizers[idx] = optimizer
            self.schedulers[idx] = scheduler

    def gradient(self, images, labels, criterion=None, selection=None):
         
        """Compute the gradient of criterion(model) w.r.t to given data."""
        grad_list, norm_list = [], []
        for model in self.models:
            with GPUContext(self.setup, model) as model:

                if criterion is None:
                    criterion = self.loss_fn
                differentiable_params = [p for p in model.parameters() if p.requires_grad]

                if selection == 'max_gradient':
                    grad_norms = []
                    for image, label in zip(images, labels):
                        loss = criterion(model(image.unsqueeze(0)), label.unsqueeze(0))
                        gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
                        grad_norm = 0
                        for grad in gradients:
                            grad_norm += grad.detach().pow(2).sum()
                        grad_norms.append(grad_norm.sqrt())
                    
                    source_poison_selected = self.args.sources_selection_rate * images.shape[0]
                    indices = [i[0] for i in sorted(enumerate(grad_norms), key=lambda x:x[1])][-source_poison_selected:]
                    images = images[indices]
                    labels = labels[indices]
                    write('{} sources with maximum gradients selected'.format(source_poison_selected), self.args.output)
                
                # Using batch processing for gradients
                if not self.args.source_gradient_batch==None:
                    batch_size = self.args.source_gradient_batch
                    if images.shape[0] < batch_size:
                        batch_size = images.shape[0]
                    else:
                        if images.shape[0] % batch_size != 0:
                            batch_size = images.shape[0] // ceil(images.shape[0] / batch_size)
                            warnings.warn(f'Batch size changed to {batch_size} to fit source train size')
                    gradients = None
                    for i in range(images.shape[0]//batch_size):
                        loss = batch_size * criterion(model(images[i*batch_size:(i+1)*batch_size]), labels[i*batch_size:(i+1)*batch_size])
                        if i == 0:
                            gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
                        else:
                            gradients = tuple(map(lambda i, j: i + j, gradients, torch.autograd.grad(loss, differentiable_params, only_inputs=True)))

                    gradients = tuple(map(lambda i: i / images.shape[0], gradients))
                else:
                    loss = criterion(model(images), labels)
                    gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)


                grad_norm = 0
                for grad in gradients:
                    grad_norm += grad.detach().pow(2).sum()
                grad_norm = grad_norm.sqrt()            


                grad_list.append(gradients)
                norm_list.append(grad_norm)

        return grad_list, norm_list

    def compute(self, function, *args):
        """Compute function on all models.

        Function has arguments that are possibly sequences of length args.ensemble
        """
        outputs = []
        for idx, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            with GPUContext(self.setup, model) as model:
                single_arg = [arg[idx] if hasattr(arg, '__iter__') else arg for arg in args]
                outputs.append(function(model, optimizer, *single_arg))
        # collate
        avg_output = [np.mean([output[idx] for output in outputs]) for idx, _ in enumerate(outputs[0])]
        return avg_output
