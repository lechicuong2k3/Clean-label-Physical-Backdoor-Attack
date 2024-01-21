"""Single model default victim class."""
import torch
import numpy as np
import warnings
from math import ceil

import copy

from .models import get_model
from .training import get_optimizers, run_step
from ..hyperparameters import training_strategy
from ..utils import set_random_seed, write
from ..consts import BENCHMARK, SHARING_STRATEGY, FINETUNING_LR_DROP
torch.backends.cudnn.benchmark = BENCHMARK
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)

from .victim_base import _VictimBase

class _VictimSingle(_VictimBase):
    """Implement model-specific code and behavior for a single model on a single GPU.

    This is the simplest victim implementation. No init so the parent class init is automatically called
    
    Methods to initialize a model."""

    def initialize(self, seed=None):
        """Set seed and initialize model, optimizer, scheduler"""
        if seed is None:
            if self.args.model_seed is None:
                self.model_init_seed = np.random.randint(0, 2**32 - 1)
            else:
                self.model_init_seed = self.args.model_seed
        else:
            self.model_init_seed = seed
            
        set_random_seed(self.model_init_seed)
        self.model, self.defs, self.optimizer, self.scheduler = self._initialize_model(self.args.net[0], mode=self.args.scenario)
            
        self.model.to(**self.setup)
        if self.setup['device'] == 'cpu' and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.model.frozen = self.model.module.frozen
        write(f'{self.args.net[0]} model initialized with random key {self.model_init_seed}.', self.args.output)
        print(f'{self.args.net[0]} model initialized with random key {self.model_init_seed}.')
        write(repr(self.defs), self.args.output)
        print(repr(self.defs))

    def reinitialize_last_layer(self, reduce_lr_factor=1.0, seed=None, keep_last_layer=False):
        if not keep_last_layer:
            if self.args.model_seed is None:
                if seed is None:
                    self.model_init_seed = np.random.randint(0, 2**32 - 1)
                else:
                    self.model_init_seed = seed
            else:
                self.model_init_seed = self.args.model_seed
            set_random_seed(self.model_init_seed)

            # We construct a full replacement model, so that the seed matches up with the initial seed,
            # even if all of the model except for the last layer will be immediately discarded.
            replacement_model = get_model(self.args.net[0], pretrained=True)
            
            if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                frozen = self.model.module.frozen
                self.model = torch.nn.Sequential(*list(self.model.module.children())[:-1], torch.nn.Flatten(), list(replacement_model.children())[-1])
            else:
                # Rebuild model with new last layer
                frozen = self.model.frozen
                self.model = torch.nn.Sequential(*list(self.model.children())[:-1], torch.nn.Flatten(), list(replacement_model.children())[-1])
            
            self.model.frozen = frozen  
            self.model.to(**self.setup)
            if self.setup['device'] == 'cpu'and torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
                self.model.frozen = self.model.module.frozen

        # Define training routine
        # Reinitialize optimizers here
        self.defs = training_strategy(self.args.net[0], self.args)
        self.defs.lr *= reduce_lr_factor
        self.optimizer, self.scheduler = get_optimizers(self.model, self.args, self.defs)
        write(f'{self.args.net[0]} last layer re-initialized with random key {self.model_init_seed}.', self.args.output)
        write(repr(self.defs), self.args.output)

    def save_feature_representation(self):
        self.clean_model = copy.deepcopy(self.model)

    def load_feature_representation(self):
        self.model = copy.deepcopy(self.clean_model)

    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""
    def _iterate(self, kettle, poison_delta, max_epoch=None):
        """Validate a given poison by training the model and checking source accuracy."""
        if max_epoch is None:
            max_epoch = self.defs.epochs

        single_setup = (self.model, self.defs, self.optimizer, self.scheduler)
        self.defs.epochs = max_epoch
        for self.epoch in range(1, max_epoch+1):
            run_step(kettle, poison_delta, self.epoch, *single_setup)
            if self.args.dryrun:
                break

    def step(self, kettle, poison_delta):
        """Step through a model epoch. Optionally: minimize poison loss."""
        single_setup = (self.model, self.defs, self.optimizer, self.scheduler)
        run_step(kettle, poison_delta, self.epoch, *single_setup)
        self.epoch += 1
        if self.epoch > self.defs.epochs + 1:
            self.epoch = 1
            write('Model reset to epoch 0.', self.args.output)
            self._initialize_model(self.args.net[0], mode=self.args.scenario)
            self.model.to(**self.setup)
            if self.setup['device'] == 'cpu' and torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
                self.model.frozen = self.model.module.frozen

    """ Various Utilities."""
    def eval(self, dropout=False):
        """Switch everything into evaluation mode."""
        def apply_dropout(m):
            """https://discuss.pytorch.org/t/dropout-at-test-time-in-densenet/6738/6."""
            if type(m) == torch.nn.Dropout:
                m.train()
        self.model.eval()
        if dropout:
            self.model.apply(apply_dropout)

    def reset_learning_rate(self):
        """Reset scheduler object to initial state."""
        _, _, self.optimizer, self.scheduler = self._initialize_model(self.args.net[0], mode=self.args.scenario)

    def gradient(self, images, labels, criterion=None, selection=None):
        """Compute the gradient of criterion(model) w.r.t to given data."""

        if criterion is None:
            criterion = self.loss_fn
        differentiable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Select sources with maximum gradient
        if selection == 'max_gradient':
            grad_norms = []
            for image, label in zip(images, labels):
                loss = criterion(self.model(image.unsqueeze(0)), label.unsqueeze(0))
                gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
                grad_norm = 0
                for grad in gradients:
                    grad_norm += grad.detach().pow(2).sum()
                grad_norms.append(grad_norm.sqrt()) # Append l2_norm of gradient
            
            source_poison_selected = ceil(self.args.sources_selection_rate * images.shape[0])
            indices = [i[0] for i in sorted(enumerate(grad_norms), key=lambda x:x[1])][-source_poison_selected:]
            images = images[indices]
            labels = labels[indices]
            write('{} sources with maximum gradients selected'.format(source_poison_selected), self.args.output)

        # Using batch processing for gradients
        if self.args.source_gradient_batch != None:
            batch_size = self.args.source_gradient_batch
            if images.shape[0] < batch_size:
                batch_size = images.shape[0]
            else:
                if images.shape[0] % batch_size != 0:
                    batch_size = images.shape[0] // ceil(images.shape[0] / batch_size)
                    warnings.warn(f'Batch size changed to {batch_size} to fit source train size')
            gradients = None
            for i in range(images.shape[0]//batch_size):
                loss = batch_size * criterion(self.model(images[i*batch_size:(i+1)*batch_size]), labels[i*batch_size:(i+1)*batch_size])
                if i == 0:
                    gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)
                else:
                    gradients = tuple(map(lambda i, j: i + j, gradients, torch.autograd.grad(loss, differentiable_params, only_inputs=True)))

            gradients = tuple(map(lambda i: i / (images.shape[0] - (images.shape[0] % batch_size)), gradients))
        else:
            loss = criterion(self.model(images), labels)
            gradients = torch.autograd.grad(loss, differentiable_params, only_inputs=True)

        grad_norm = 0
        for grad in gradients:
            grad_norm += grad.detach().pow(2).sum()
        grad_norm = grad_norm.sqrt()
    
        return gradients, grad_norm

    def compute(self, function, *args):
        r"""Compute function on the given optimization problem, defined by criterion \circ model.

        Function has arguments: model, criterion
        """
        return function(self.model, self.optimizer, *args)
