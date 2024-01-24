"""Interface for poison recipes."""
from .witch_matching import WitchGradientMatching, WitchGradientMatchingNoisy, WitchGradientMatchingHidden, WitchMatchingMultiSource
from .witch_htbd import WitchHTBD
from .witch_clbd import WitchLabelConsistent


import torch


def Witch(args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    """Implement Main interface."""
    if args.recipe == 'gradient-matching':
        return WitchGradientMatching(args, setup)
    elif args.recipe == 'gradient-matching-private':
        return WitchGradientMatchingNoisy(args, setup)
    elif args.recipe == 'gradient-matching-hidden':
        return WitchGradientMatchingHidden(args, setup)
    elif args.recipe == 'gradient-matching-mt':
        return WitchMatchingMultiSource(args, setup)
    elif args.recipe == 'hidden-trigger':
        return WitchHTBD(args, setup)
    elif args.recipe == 'label-consistent':
        return WitchLabelConsistent(args, setup)
    elif args.recipe == 'naive':
        return None
    else:
        raise NotImplementedError()

__all__ = ['Witch']
