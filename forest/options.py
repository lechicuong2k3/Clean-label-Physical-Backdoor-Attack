"""Implement an ArgParser common to both brew_poison.py and dist_brew_poison.py ."""

import argparse

def options():
    """Construct the central argument parser, filled with useful defaults.

    The first block is essential to test poisoning in different scenarios.
    The options following afterwards change the algorithm in various ways and are set to reasonable defaults.
    """
    parser = argparse.ArgumentParser(description='Construct poisoned training data for the given network and dataset')

    
    ###########################################################################
    parser.add_argument('--f')
    # Central:
    parser.add_argument('--net', default='Resnet50', type=lambda s: [str(item) for item in s.split(',')])
    parser.add_argument('--dataset', default='datasets/Facial_recognition', type=str, choices=['Facial_recognition', 'Object_detection'])
    parser.add_argument('--recipe', default='gradient-matching', type=str, choices=['gradient-matching', 'gradient-matching-private', 
                                                                                    'hidden-trigger', 'hidden-trigger-mt' 'gradient-matching-mt',
                                                                                    'patch', 'gradient-matching-hidden', 'naive', 'label-consistent'])
                                                                                    
    parser.add_argument('--threatmodel', default='clean-single-source', type=str, choices=['clean-single-source', 'clean-multi-source', 'clean-all-source', 'third-party', 'self-betrayal'])
    parser.add_argument('--num_source_classes', default=1, type=int, help='Number of source classes (for many-to-one attacks)')
    parser.add_argument('--scenario', default='finetuning', type=str, choices=['from-scratch', 'transfer', 'finetuning'])

    # Reproducibility management:
    parser.add_argument('--poisonkey', default='3-1', type=str, help='Initialize poison setup with this key.')  # Take input such as 05-1 for [0, 5] as the sources and 1 as the target
    parser.add_argument('--poison_seed', default=None, type=int, help='Initialize the poisons with this key.')
    parser.add_argument('--model_seed', default=None, type=int, help='Initialize the model with this key.')
    parser.add_argument('--deterministic', action='store_true', help='Disable CUDNN non-determinism.')

    # Files and folders
    parser.add_argument('--name', default='', type=str, help='Name tag for the result table and possibly for export folders.')
    parser.add_argument('--poison_path', default='poisons/', type=str)
    parser.add_argument('--model_savepath', default='./models/', type=str)
    ###########################################################################

    # Mixing defense
    parser.add_argument('--mixing_method', default=None, type=str, help='Which mixing data augmentation to use.')
    parser.add_argument('--mixing_disable_correction', action='store_false', help='Disable correcting the loss term appropriately after data mixing.')
    parser.add_argument('--mixing_strength', default=None, type=float, help='How strong is the mixing.')
    parser.add_argument('--disable_adaptive_attack', action='store_false', help='Do not use a defended model as input for poisoning. [Defend only in poison validation]')
    parser.add_argument('--defend_features_only', action='store_true', help='Only defend during the initial pretraining before poisoning. [Defend only in pretraining]')
    # Note: If --disable_adaptive_attack and --defend_features_only, then the defense is never activated


    # Privacy defenses
    parser.add_argument('--gradient_noise', default=None, type=float, help='Add custom gradient noise during training.')
    parser.add_argument('--gradient_clip', default=None, type=float, help='Add custom gradient clip during training.')

    # Adversarial defenses
    parser.add_argument('--defense_type', default=None, type=str, help='Add custom novel defenses.')
    parser.add_argument('--defense_strength', default=None, type=float, help='Add custom strength to novel defenses.')
    parser.add_argument('--defense_steps', default=None, type=int, help='Override default number of adversarial steps taken by the defense.')
    parser.add_argument('--defense_sources', default=None, type=str, help='Different choices for source selection. Options: shuffle/sep-half/sep-1/sep-10')

    # Filter defenses
    parser.add_argument('--filter_defense', default='', type=str, help='Which filtering defense to use.', choices=['madry', 'deepknn', 'activation_clustering'])

    # Adaptive attack variants
    parser.add_argument('--padversarial', default=None, type=str, help='Use adversarial steps during poison brewing.')
    parser.add_argument('--pmix', action='store_true', help='Use mixing during poison brewing [Uses the mixing specified in mixing_type].')

    # Poison brewing:
    parser.add_argument('--attackoptim', default='signAdam', type=str)
    parser.add_argument('--attackiter', default=250, type=int)
    parser.add_argument('--init', default='randn', type=str)  # randn / rand
    parser.add_argument('--tau', default=0.1, type=float)
    parser.add_argument('--scheduling', action='store_false', help='Disable step size decay.')
    parser.add_argument('--source_criterion', default='cross-entropy', type=str, help='Loss criterion for poison loss')
    parser.add_argument('--restarts', default=1, type=int, help='How often to restart the attack.')
    
    parser.add_argument('--pbatch', default=64, type=int, help='Poison batch size during optimization')
    parser.add_argument('--paugment', action='store_false', help='Do not augment poison batch during optimization')
    parser.add_argument('--data_aug', type=str, default='default', help='Mode of diff. data augmentation.')

    # Poisoning algorithm changes
    parser.add_argument('--full_data', action='store_true', help='Use full train data for poisoning (instead of just the poison images)')
    parser.add_argument('--ensemble', default=1, type=int, help='Ensemble of networks to brew the poison on')
    parser.add_argument('--stagger', action='store_true', help='Stagger the network ensemble if it exists')
    parser.add_argument('--clean_grad', action='store_true', help='Compute the first-order poison gradient.')
    parser.add_argument('--step', action='store_true', help='Optimize the model for one epoch.')
    parser.add_argument('--max_epoch', default=20, type=int, help='Train only up to this epoch before poisoning.')

    # Use only a subset of the dataset:
    parser.add_argument('--ablation', default=1.0, type=float, help='What percent of data (including poisons) to use for validation')

    # Gradient Matching - Specific Options
    parser.add_argument('--loss', default='similarity', type=str)  # similarity is stronger in  difficult situations

    # These are additional regularization terms for gradient matching. We do not use them, but it is possible
    # that scenarios exist in which additional regularization of the poisoned data is useful.
    parser.add_argument('--centreg', default=0, type=float)
    parser.add_argument('--normreg', default=0, type=float)
    parser.add_argument('--repel', default=0, type=float)

    # Validation behavior
    parser.add_argument('--vruns', default=1, type=int, help='How often to re-initialize and check source after retraining')
    parser.add_argument('--vnet', default=None, type=lambda s: [str(item) for item in s.split(',')], help='Evaluate poison on this victim model. Defaults to --net')
    parser.add_argument('--retrain_from_init', action='store_true', help='Additionally evaluate by retraining on the same model initialization.')
    parser.add_argument('--skip_clean_training', action='store_true', help='Skip clean training. This is only suggested for attacks that do not depend on a clean model.')

    # Optimization setup
    parser.add_argument('--optimization', default='conservative-adam', type=str, help='Optimization Strategy')
    # Strategy overrides:
    parser.add_argument('--epochs', default=10, type=int, help='Override default epochs of --optimization strategy')
    parser.add_argument('--batch_size', default=128, type=int, help='Override default batch_size of --optimization strategy')
    parser.add_argument('--lr', default=None, type=float, help='Override default learning rate of --optimization strategy')
    parser.add_argument('--noaugment', action='store_true', help='Do not use data augmentation during training.')

    # Optionally, datasets can be stored within RAM:
    parser.add_argument('--cache_dataset', action='store_true', help='Cache the entire thing :>')

    # Debugging:
    parser.add_argument('--dryrun', action='store_true', help='This command runs every loop only a single time.')
    parser.add_argument('--save', default=None, help='Export poisons into a given format. Options are full/limited/numpy.')
    
    # Distributed Computations
    parser.add_argument("--local_rank", default=None, type=int, help='Distributed rank. This is an INTERNAL ARGUMENT! '
                                                                     'Only the launch utility should set this argument!')

    # Backdoor attack:
    parser.add_argument('--keep_sources', action='store_true', default=True, help='Do we keep the sources are used for testing attack success rate?')
    parser.add_argument('--sources_train_rate', default=1.0, type=float, help='Fraction of source_class trainset that can be selected crafting poisons')
    parser.add_argument('--sources_selection_rate', default=1.0, type=int, help='Fraction of sources to be selected for crafting poisons')
    parser.add_argument('--source_gradient_batch', default=None, type=int, help='Batch size for sources train gradient computing')
    parser.add_argument('--val_max_epoch', default=20, type=int, help='Train only up to this epoch for final validation.')
    parser.add_argument('--retrain_max_epoch', default=10, type=int, help='Train only up to this epoch for retraining during crafting.')
    parser.add_argument('--retrain_scenario', default='from-scratch', type=str, choices=['from-scratch', 'finetuning', 'transfer'], help='Scenario for retraining and evaluating on the poisoned dataset')
    parser.add_argument('--load_feature_repr', default=True, action='store_true', help='Load feature representation of the model trained on clean data')
    parser.add_argument('--trigger', default='real_beard', type=str, help='Trigger type')
    parser.add_argument('--digital_trigger', action='store_true', default=False, help='Adding digital trigger instead of physical ones')
    parser.add_argument('--digital_trigger_path', default='digital_triggers')
    parser.add_argument('--opacity', default=32/255, type=float, help='The opacity of digital trigger')
    parser.add_argument('--retrain_iter', default=100, type=int, help='Start retraining every <retrain_iter> iterations')
    parser.add_argument('--source_selection_strategy', default=None, type=str, choices=['max_gradient', 'max_loss'], help='sources_train_rate selection strategy')
    parser.add_argument('--poison_selection_strategy', default='max_gradient', type=str, choices=['max_gradient', 'max_loss'], help='Poison selection strategy')
    parser.add_argument('--raw_poison_rate', default=1.0, type=float, help='Fraction of target_class dataset that CAN BE SELECTED as poisons')
    
    # Poison properties / controlling the strength of the attack:
    parser.add_argument('--eps', default=16, type=float, help='Epsilon bound of the attack in a ||.||_p norm. p=Inf for all recipes except for "patch".')
    parser.add_argument('--alpha', default=0.1, type=float, help='Fraction of target_class training data that is poisoned by adding pertubation')
    parser.add_argument('--beta', default=0.0, type=float, help='Fraction of target_class training data that has physical trigger or digital trigger')

    return parser