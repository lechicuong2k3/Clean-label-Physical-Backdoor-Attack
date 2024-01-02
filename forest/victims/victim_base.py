"""Base victim class."""

import torch

from .models import get_model
from .training import get_optimizers
from ..hyperparameters import training_strategy
from ..utils import average_dicts, write

FINETUNING_LR_DROP = 0.001

class _VictimBase:
    """Implement model-specific code and behavior.

    Expose:
    Attributes:
     - model
     - optimizer
     - scheduler
     - criterion

     Methods:
     - initialize: set seed and initialize model, optimizer, scheduler
     - train: clean (pre)-training of the chosen model, no poisoning involved
     - retrain: retrain and evaluate poison on the initialization it was brewed on
     - validate: train and evaluate poison on a new initializations of the model
     - _iterate: validate a given poison by training the model and checking source accuracy.

     - compute: compute function on given objective function and arguments
     - gradient: compute the gradient of criterion(model) w.r.t to given data."
     - eval: switch everything into evaluation mode

     Internal methods that should ideally be reused by other backends:
     - _initialize_model: initialize model, optimizer, scheduler
     - _step: Train for one epoch

    """

    def __init__(self, args, num_classes=10, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize empty victim."""
        self.args, self.setup = args, setup
        if self.args.ensemble < len(self.args.net):
            raise ValueError(f'More models requested than ensemble size.'
                             f'Increase ensemble size or reduce models.')
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.initialize()

    def gradient(self, images, labels):
        """Compute the gradient of criterion(model) w.r.t to given data."""
        raise NotImplementedError()

    def compute(self, function):
        """Compute function on all models.

        Function has arguments: model, ...
        """
        raise NotImplementedError()

    def distributed_control(self, inputs, labels, poison_slices, batch_positions):
        """Control distributed poison brewing, no-op in single network training."""
        randgen = None
        return inputs, labels, poison_slices, batch_positions, randgen

    def sync_gradients(self, input):
        """Sync gradients of given variable. No-op for single network training."""
        return input
    
    def reset_learning_rate(self):
        """Reset scheduler object to initial state."""
        raise NotImplementedError()

    """ Methods to initialize and modify a model."""

    def initialize(self, seed=None):
        raise NotImplementedError()

    def reinitialize_last_layer(self, seed=None):
        raise NotImplementedError()

    def freeze_feature_extractor(self):
        raise NotImplementedError()

    def save_feature_representation(self):
        raise NotImplementedError()

    def load_feature_representation(self):
        raise NotImplementedError()


    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""

    def train(self, kettle, max_epoch=None):
        """Clean (pre)-training of the chosen model, no poisoning involved."""
            
        write('Starting clean training with {} scenario ...'.format(self.args.scenario), self.args.output)
        self._iterate(kettle, poison_delta=None, max_epoch=max_epoch) # Validate poison
        
        if self.args.load_feature_repr:
            self.save_feature_representation()
            write('Feature representation saved.', self.args.output)
            
    def retrain(self, kettle, poison_delta, max_epoch=None):
        """Retrain with the same initialization on dataset with poison added."""
        self.initialize(seed=self.model_init_seed)
        self._iterate(kettle, poison_delta=poison_delta, max_epoch=max_epoch)

    def validate(self, kettle, poison_delta, val_max_epoch=None):
        """Check poison on a new initialization(s), depending on the scenario."""

        for run in range(self.args.vruns):
            # Reinitalize model with new seed
            self.initialize()

            # Train new model
            self._iterate(kettle, poison_delta=poison_delta, max_epoch=val_max_epoch)

    def eval(self, dropout=True):
        """Switch everything into evaluation mode."""
        raise NotImplementedError()

    def _iterate(self, kettle, poison_delta):
        """Validate a given poison by training the model and checking source accuracy."""
        raise NotImplementedError()

    def _adversarial_step(self, kettle, poison_delta, step, poison_sources, true_classes):
        """Step through a model epoch to in turn minimize poison loss."""
        raise NotImplementedError()

    def _initialize_model(self, model_name, mode='finetuning'):
        if mode == 'from-scratch':
            pretrained = False
        else:
            pretrained = True
        self.model = get_model(model_name, num_classes=self.num_classes, pretrained=pretrained)
        self.model.frozen = False
        # Define training routine
        self.defs = training_strategy(model_name, self.args) # Initialize hyperparameters for training
        if mode == 'finetuning':
            self.defs.lr *= FINETUNING_LR_DROP
        elif mode == 'transfer':
            self.model.frozen = True
            self.freeze_feature_extractor()
            self.eval()
        self.optimizer, self.scheduler = get_optimizers(self.model, self.args, self.defs)