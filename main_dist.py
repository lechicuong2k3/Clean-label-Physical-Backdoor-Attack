"""General interface script to launch poisoning jobs."""

import torch
import os

import datetime
import time
import socket

import forest

from forest.filtering_defenses import get_defense
from forest.utils import write
from forest.consts import BENCHMARK, NUM_CLASSES, SHARING_STRATEGY

torch.backends.cudnn.benchmark = BENCHMARK
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)
torch.cuda.empty_cache()

# Parse input arguments
args = forest.options().parse_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3,0,2"
args.local_rank = int(os.environ['LOCAL_RANK'])

if not (torch.cuda.device_count() > 1):
    raise ValueError('Cannot run distributed on single GPU!')

if args.local_rank is None:
    raise ValueError('This script should only be launched via the pytorch launch utility!')

args.output = f'outputs/{args.recipe}/{args.scenario}/{args.trigger}/{args.net[0].upper()}/{args.poisonkey}_{args.scenario}_{args.trigger}_{args.alpha}_{args.beta}_{args.attackoptim}.txt'
    
os.makedirs(os.path.dirname(args.output), exist_ok=True)
open(args.output, 'w').close() # Clear the output files

if args.deterministic:
    forest.utils.set_deterministic()

if __name__ == "__main__":
    if torch.cuda.device_count() < args.local_rank:
        raise ValueError('Process invalid, oversubscribing to GPUs is not possible in this mode.')
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f'cuda:{args.local_rank}')
    setup = dict(device=device, dtype=torch.float, non_blocking=forest.consts.NON_BLOCKING)
    torch.distributed.init_process_group(backend=forest.consts.DISTRIBUTED_BACKEND, init_method='env://')
    if args.vnet == None:
        args.ensemble = 1
    else:
        args.ensemble = len(args.vnet)
    
    if args.ensemble != 1 and args.ensemble != torch.distributed.get_world_size():
        raise ValueError('Argument given to ensemble does not match number of launched processes!')
    else:
        args.world_size = torch.distributed.get_world_size()
        if args.local_rank == 0:
            print(f'------------------------------- Currently evaluating on {args.recipe} -------------------------------')
            print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
            print(args)
            print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}')
    
    model = forest.Victim(args, num_classes=NUM_CLASSES, setup=setup) # Initialize model and loss_fn
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations,
                         model.defs.mixing_method, setup=setup) # Set up trainloader, validloader, poisonloader, poison_ids, trainset/poisonset/source_testset
    witch = forest.Witch(args, setup=setup)

    start_time = time.time()
    if args.skip_clean_training:
        if args.local_rank == 0: write('Skipping clean training...', args.output)
    else:
        model.train(data, max_epoch=args.train_max_epoch)
    train_time = time.time()
    
    print("Train time: ", str(datetime.timedelta(seconds=train_time - start_time)))
    
    # Select poisons based on maximum gradient norm
    data.select_poisons(model, args.poison_selection_strategy)
    
    # Print data status
    if args.local_rank == 0:
        data.print_status()

    if args.recipe != 'naive':
        poison_delta = witch.brew(model, data)
    else:
        poison_delta = None
        
    craft_time = time.time()
    print("Craft time: ", str(datetime.timedelta(seconds=craft_time - train_time)))
    
    # Optional: apply a filtering defense
    if args.filter_defense != '' and args.recipe != 'naive':
        if args.scenario == 'from-scratch':
            model.validate(data, poison_delta)
        write('Attempting to filter poison images...', args.output)
        defense = get_defense(args)
        clean_ids = defense(data, model, poison_delta)
        poison_ids = set(range(len(data.trainset))) - set(clean_ids)
        removed_images = len(data.trainset) - len(clean_ids)
        removed_poisons = len(set(data.poison_ids.tolist()) & poison_ids)

        data.reset_trainset(clean_ids)
        write(f'Filtered {removed_images} images out of {len(data.trainset)}. {removed_poisons} were poisons.', args.output)
        filter_stats = dict(removed_poisons=removed_poisons, removed_images_total=removed_images)
    else:
        filter_stats = dict()

    if args.retrain_from_init:
        model.retrain(data, poison_delta) # Evaluate poison performance on the retrained model

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

    if args.local_rank == 0:
        # Export
        if args.save is not None and args.recipe != 'naive':
            data.export_poison(poison_delta, path=args.poison_path, mode=args.save)

        write('\n' + datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"), args.output)
        write('---------------------------------------------------', args.output)
        write(f'Finished computations with train time: {str(datetime.timedelta(seconds=train_time - start_time))}', args.output)
        write(f'Finished computations with craft time: {str(datetime.timedelta(seconds=craft_time - train_time))}', args.output)
        write(f'Finished computations with test time: {str(datetime.timedelta(seconds=test_time - craft_time))}', args.output)
        write('-------------------Job finished.-------------------', args.output)