"""General interface script to launch poisoning jobs."""

import torch
import os

import datetime
import time

import forest

from forest.filtering_defenses import get_defense
from forest.utils import write, set_random_seed
from forest.consts import BENCHMARK, NUM_CLASSES
torch.backends.cudnn.benchmark = BENCHMARK

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3,0,2"

# Parse input arguments
args = forest.options().parse_args()
args.dataset = os.path.join('datasets', args.dataset)

if args.system_seed != None:
    set_random_seed(args.system_seed)

if args.exp_name is None:
    exp_num = len(os.listdir(os.path.join(os.getcwd(), 'outputs'))) + 1
    args.exp_name = f'exp_{exp_num}'

args.output = f'outputs/{args.exp_name}/{args.recipe}/{args.scenario}/{args.trigger}/{args.net[0].upper()}/{args.poisonkey}_{args.scenario}_{args.trigger}_{args.alpha}_{args.beta}_{args.attackoptim}_{args.attackiter}.txt'

os.makedirs(os.path.dirname(args.output), exist_ok=True)
open(args.output, 'w').close() # Clear the output files

torch.cuda.empty_cache()
if args.deterministic:
    forest.utils.set_deterministic()

if __name__ == "__main__":
    
    setup = forest.utils.system_startup(args) # Set up device and torch data type
    
    model = forest.Victim(args, num_classes=NUM_CLASSES, setup=setup) # Initialize model and loss_fn
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations,
                         model.defs.mixing_method, setup=setup) # Set up trainloader, validloader, poisonloader, poison_ids, trainset/poisonset/source_testset
    witch = forest.Witch(args, setup=setup)

    start_time = time.time()
    if args.skip_clean_training:
        write('Skipping clean training...', args.output)
    else:
        model.train(data, max_epoch=args.train_max_epoch)
        
    train_time = time.time()
    print("Train time: ", str(datetime.timedelta(seconds=train_time - start_time)))
    
    if witch.args.backdoor_finetuning:
        witch.backdoor_finetuning(model, data, lr=0.000005, num_epoch=25)
        if witch.args.load_feature_repr:
            model.save_feature_representation()
                
    # Select poisons based on maximum gradient norm
    data.select_poisons(model)
    
    # Print data status
    data.print_status()
        
    if args.recipe != 'naive':
        poison_delta = witch.brew(model, data)
    else:
        poison_delta = None
    
    craft_time = time.time()
    print("Craft time: ", str(datetime.timedelta(seconds=craft_time - train_time)))
    
    # Optional: apply a filtering defense
    if args.defense != '' and args.defense != 'neural_cleanse':
        if args.scenario == 'from-scratch':
            model.validate(data, poison_delta)
        if args.recipe != 'naive':
            write('Attempting to filter poison images...', args.output)
            defense = get_defense(args)
            clean_ids = defense(data, model, poison_delta, args)
            poison_ids = set(range(len(data.trainset))) - set(clean_ids)
            removed_images = len(data.trainset) - len(clean_ids)
            removed_poisons = len(set(data.poison_target_ids.tolist()+data.triggerset_class_ids) & poison_ids)
            removed_cleans = removed_images - removed_poisons
            elimination_rate = removed_poisons/(len(data.poison_target_ids) + len(data.triggerset_class_ids))*100
            sacrifice_rate = removed_cleans/(len(data.trainset)-len(data.poison_target_ids)-len(data.triggerset_class_ids))*100
        else:
            write('Attempting to filter poison images...', args.output)
            defense = get_defense(args)
            clean_ids = defense(data, model, poison_delta, args)
            poison_ids = set(range(len(data.trainset))) - set(clean_ids)
            removed_images = len(data.trainset) - len(clean_ids)
            removed_poisons = len(set(data.triggerset_class_ids) & poison_ids)
            removed_cleans = removed_images - removed_poisons
            elimination_rate = removed_poisons/len(data.triggerset_class_ids)*100
            sacrifice_rate = removed_cleans/(len(data.trainset)-len(data.triggerset_class_ids))*100

        data.reset_trainset(clean_ids)
        write(f'Filtered {removed_images} images out of {len(data.trainset.dataset)}. {removed_poisons} were poisons.', args.output)
        write(f'Elimination rate: {elimination_rate}% Sacrifice rate: {sacrifice_rate}%', args.output)
        filter_stats = dict(removed_poisons=removed_poisons, removed_images_total=removed_images)
    else:
        filter_stats = dict()
  
    if args.retrain_from_init:
        model.retrain(data, poison_delta) # Evaluate poison performance on the retrained model

    if args.defense == 'neural_cleanse':
        pass
    
    write('Validating poisoned model...', args.output)
    # Validation
    if args.vnet is not None:  # Validate the transfer model given by args.vnet
        train_net = args.net
        args.net = args.vnet
        args.ensemble = len(args.vnet)
        if args.vruns > 0:
            model = forest.Victim(args, setup=setup)  # this instantiates a new model with a different architecture
            model.validate(data, poison_delta, val_max_epoch=args.val_max_epoch)
        args.net = train_net
    else:  # Validate the main model
        if args.vruns > 0:
            model.validate(data, poison_delta, val_max_epoch=args.val_max_epoch)
            
    test_time = time.time()
    print("Test time: ", str(datetime.timedelta(seconds=test_time - craft_time)))

    # Export
    if args.save is not None and args.recipe != 'naive':
        data.export_poison(poison_delta, path=args.poison_path, mode=args.save)

    write('\n' + datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"), args.output)
    write('---------------------------------------------------', args.output)
    write(f'Finished computations with train time: {str(datetime.timedelta(seconds=train_time - start_time))}', args.output)
    write(f'Finished computations with craft time: {str(datetime.timedelta(seconds=craft_time - train_time))}', args.output)
    write(f'Finished computations with test time: {str(datetime.timedelta(seconds=test_time - craft_time))}', args.output)
    write('-------------------Job finished.-------------------', args.output)