"""Optimization setups."""

from dataclasses import dataclass, asdict

def training_strategy(model_name, args):
    """Parse training strategy."""
    if args.optimization == 'conservative-sgd':
        defaults = CONSERVATIVE_SGD
    elif args.optimization == 'conservative-adam':
        defaults = CONSERVATIVE_ADAM
    elif args.optimization == 'private-gaussian':
        defaults = PRIVACY_GAUSSIAN
    elif args.optimization == 'private-laplacian':
        defaults = PRIVACY_LAPLACIAN
    elif args.optimization == 'adversarial':
        defaults = ADVERSARIAL
    elif args.optimization == 'mixup':
        defaults = MIXUP
    else:
        raise ValueError(f'Unknown opt. strategy {args.optimization}.')
    defs = Hyperparameters(**defaults.asdict())

    if args.train_max_epoch is not None:
        defs.epochs = args.train_max_epoch
    if args.batch_size is not None:
        defs.batch_size = args.batch_size
    if args.lr is not None:
        defs.lr = args.lr
    if args.noaugment:
        defs.augmentations = False
    else:
        defs.augmentations = args.data_aug

    # Modifications to gradient noise settings
    if defs.privacy is not None:
        if args.gradient_noise is not None:
            defs.privacy['noise'] = args.gradient_noise
        if args.gradient_clip is not None:
            defs.privacy['clip'] = args.gradient_clip

    # Modifications to defense settings:
    if defs.novel_defense is not None:
        if args.defense_type is not None:
            defs.novel_defense['type'] = args.defense_type
        if args.defense_strength is not None:
            defs.novel_defense['strength'] = args.defense_strength
        else:
            defs.novel_defense['strength'] = args.eps
        if args.defense_sources is not None:
            defs.novel_defense['source_selection'] = args.defense_sources
        if args.defense_steps is not None:
            defs.novel_defense['steps'] = args.adversarial_steps

    # Modify data mixing arguments:
    if args.mixing_method is not None:
        defs.mixing_method['type'] = args.mixing_method
        defs.mixing_method['correction'] = args.mixing_disable_correction
        if args.mixing_strength is not None:
            defs.mixing_method['strength'] = args.mixing_strength

    # Modify defense behavior
    defs.adaptive_attack = args.disable_adaptive_attack
    defs.defend_features_only = args.defend_features_only

    return defs


@dataclass
class Hyperparameters:
    """Hyperparameters used by this framework."""

    name : str
    epochs : int
    batch_size : int
    optimizer : str
    lr : float
    scheduler : str
    weight_decay : float
    augmentations : bool
    privacy : dict
    validate : int
    novel_defense: dict
    mixing_method : dict
    adaptive_attack : bool
    defend_features_only: bool

    def asdict(self):
        return asdict(self)


CONSERVATIVE_SGD = Hyperparameters(
    name='conservative',
    lr=0.1,
    epochs=40,
    batch_size=32,
    optimizer='SGD',
    scheduler='linear',
    weight_decay=5e-4,
    augmentations=True,
    privacy=dict(clip=None, noise=None, distribution=None),
    validate=10,
    novel_defense=None,
    mixing_method=None,
    adaptive_attack=True,
    defend_features_only=False,
)

CONSERVATIVE_ADAM = Hyperparameters(
    name='conservative',
    lr=0.1,
    epochs=40,
    batch_size=32,
    optimizer='Adam',
    scheduler='linear',
    weight_decay=5e-4,
    augmentations=True,
    privacy=dict(clip=None, noise=None, distribution=None),
    validate=10,
    novel_defense=None,
    mixing_method=None,
    adaptive_attack=True,
    defend_features_only=False,
)

PRIVACY_GAUSSIAN = Hyperparameters(
    name='private-gaussian',
    lr=0.1,
    epochs=40,
    batch_size=32,
    optimizer='SGD',
    scheduler='linear',
    weight_decay=5e-4,
    augmentations=True,
    privacy=dict(clip=1.0, noise=0.01, distribution='gaussian'),
    validate=10,
    novel_defense=None,
    mixing_method=None,
    adaptive_attack=True,
    defend_features_only=False,
)


PRIVACY_LAPLACIAN = Hyperparameters(
    name='private-gaussian',
    lr=0.1,
    epochs=40,
    batch_size=32,
    optimizer='SGD',
    scheduler='linear',
    weight_decay=5e-4,
    augmentations=True,
    privacy=dict(clip=1.0, noise=0.01, distribution='laplacian'),
    validate=10,
    novel_defense=dict(type='', strength=16.0, source_selection='sep-half', steps=5),
    mixing_method=dict(type='', strength=0.0, correction=False),
    adaptive_attack=True,
    defend_features_only=False,
)

"""Implement adversarial training to defend against the poisoning."""
ADVERSARIAL = Hyperparameters(
    name='adversarial',
    lr=0.1,
    epochs=40,
    batch_size=32,
    optimizer='SGD',
    scheduler='linear',
    weight_decay=5e-4,
    augmentations=True,
    privacy=None,
    validate=10,
    novel_defense=dict(type='adversarial-evasion', strength=8.0, source_selection='sep-p128', steps=5),
    mixing_method=None,
    adaptive_attack=True,
    defend_features_only=False,
)

MIXUP = Hyperparameters(
    name='mixup',
    lr=0.1,
    epochs=40,
    batch_size=32,
    optimizer='SGD',
    scheduler='linear',
    weight_decay=5e-4,
    augmentations=True,
    privacy=None,
    validate=10,
    novel_defense=None,
    mixing_method=dict(type='mixup', strength=1.0, correction=True),
    adaptive_attack=True,
    defend_features_only=False,
)