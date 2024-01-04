"""Basic data handling."""
from .kettle_single import KettleSingle
from .kettle_distributed import KettleDistributed
__all__ = ['Kettle']

def Kettle(args, batch_size, augmentations, mixing_method, setup):
    """Implement Main interface."""
    if args.local_rank is not None:
        return KettleDistributed(args, batch_size, augmentations, mixing_method, setup)
    else:
        return KettleSingle(args, batch_size, augmentations, mixing_method, setup)