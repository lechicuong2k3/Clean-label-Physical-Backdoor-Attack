"""General interface script to launch poisoning jobs."""

import torch
import os

import datetime
import time

import forest

from forest.filtering_defenses import get_defense
from forest.utils import write
from forest.consts import BENCHMARK, NUM_CLASSES
torch.backends.cudnn.benchmark = BENCHMARK

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3,2"

# Parse input arguments
args = forest.options().parse_args()
args.exp_name = 'trigger_selection_algorithm'

if args.exp_name is not None:
    parent_dir = os.path.join(os.getcwd(), f'outputs_{args.exp_name}')
else:
    parent_dir = os.path.join(os.getcwd(), 'outputs')

if args.defense != '':
    args.output = f'{parent_dir}/defense/{args.defense}/{args.recipe}/{args.scenario}/{args.trigger}/{args.net[0].upper()}/{args.poisonkey}_{args.scenario}_{args.trigger}_{args.alpha}_{args.beta}_{args.attackoptim}_{args.attackiter}.txt'
else:
    args.output = f'{parent_dir}/{args.recipe}/{args.scenario}/{args.trigger}/{args.net[0].upper()}/{args.poisonkey}_{args.scenario}_{args.trigger}_{args.alpha}_{args.beta}_{args.attackoptim}_{args.attackiter}.txt'

args.dataset = os.path.join('datasets', args.dataset)

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