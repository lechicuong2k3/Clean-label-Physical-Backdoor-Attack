"""Implement victim behavior, for single-victim, ensemble and stuff."""
import torch

from .victim_ensemble import _VictimEnsemble
from .victim_single import _VictimSingle

def Victim(args, num_classes, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    """Implement Main interface."""
    if args.ensemble == 1:
        return _VictimSingle(args, num_classes, setup)
    elif args.ensemble > 1:
        return _VictimEnsemble(args, num_classes, setup)


from ..hyperparameters import training_strategy
__all__ = ['Victim', 'training_strategy']
