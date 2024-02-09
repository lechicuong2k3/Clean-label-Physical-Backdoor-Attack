"""Various utilities."""

import os
import csv
import socket
import datetime

from collections import defaultdict

import torch
import random
import numpy as np

from .consts import NON_BLOCKING
os.environ["CUDA_VISIBLE_DEVICES"]="3"
def system_startup(args=None, defs=None):
    """Decide and print GPU / CPU / hostname info."""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    setup = dict(device=device, dtype=torch.float, non_blocking=NON_BLOCKING)
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print(f'------------------ Currently evaluating {args.recipe} ------------------')
    
    write(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"), args.output)
    write(f'------------------ Currently evaluating {args.recipe} ------------------', args.output)
    
    if args is not None:
        print(args)
    print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')

    if torch.cuda.is_available():
        print(f'GPU : {torch.cuda.get_device_name(device=device)}')

    return setup

def average_dicts(running_stats):
    """Average entries in a list of dictionaries."""
    average_stats = defaultdict(list)
    for stat in running_stats[0]:
        if isinstance(running_stats[0][stat], list):
            for i, _ in enumerate(running_stats[0][stat]):
                average_stats[stat].append(np.mean([stat_dict[stat][i] for stat_dict in running_stats]))
        else:
            average_stats[stat] = np.mean([stat_dict[stat] for stat_dict in running_stats])
    return average_stats

"""Misc."""
def _gradient_matching(poison_grad, source_grad):
    """Compute the blind passenger loss term."""
    matching = 0
    poison_norm = 0
    source_norm = 0

    for pgrad, tgrad in zip(poison_grad, source_grad):
        matching -= (tgrad * pgrad).sum()
        poison_norm += pgrad.pow(2).sum()
        source_norm += tgrad.pow(2).sum()

    matching = matching / poison_norm.sqrt() / source_norm.sqrt()

    return matching

def bypass_last_layer(model):
    """Hacky way of separating features and classification head for many models.

    Patch this function if problems appear.
    """
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        layer_cake = list(model.module.children())
    else:
        layer_cake = list(model.children())
    last_layer = layer_cake[-1]
    headless_model = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten()).eval()  # this works most of the time all of the time :<
    return headless_model, last_layer

def cw_loss(outputs, target_classes, clamp=-100):
    """Carlini-Wagner loss for brewing"""
    top_logits, _ = torch.max(outputs, 1)
    target_logits = torch.stack([outputs[i, target_classes[i]] for i in range(outputs.shape[0])])
    difference = torch.clamp(top_logits - target_logits, min=clamp)
    return torch.mean(difference)

def _label_to_onehot(source, num_classes=100):
    source = torch.unsqueeze(source, 1)
    onehot_source = torch.zeros(source.shape[0], num_classes, device=source.device)
    onehot_source.scatter_(1, source, 1)
    return onehot_source

def cw_loss2(outputs, target_classes, confidence=0, clamp=-100):
    """CW. This is assert-level equivalent."""
    one_hot_labels = _label_to_onehot(target_classes, num_classes=outputs.shape[1])
    source_logit = (outputs * one_hot_labels).sum(dim=1)
    second_logit, _ = (outputs - outputs * one_hot_labels).max(dim=1)
    cw_indiv = torch.clamp(second_logit - source_logit + confidence, min=clamp)
    return cw_indiv.mean()

def set_random_seed(seed):
    # Setting seed
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)  # if you are using multi-GPU.
    random.seed(seed + 5)
    os.environ['PYTHONHASHSEED'] = str(seed + 6)

def set_deterministic():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True) # for pytorch >= 1.8
    torch.backends.cudnn.benchmark = False
    
def write(content, file):
    with open(file, 'a') as f:
        f.write(content + '\n')
  
def global_meters_all_avg(device, *meters):
    """meters: scalar values of loss/accuracy calculated in each rank"""
    tensors = []
    for meter in meters:
        if isinstance(meter, torch.Tensor):
            tensors.append(meter)
        else:
            tensors.append(torch.tensor(meter, device=device, dtype=torch.float32))
    for tensor in tensors:
        # each item of `tensors` is all-reduced starting from index 0 (in-place)
        torch.distributed.all_reduce(tensor)

    return [(tensor / torch.distributed.get_world_size()).item() for tensor in tensors]

class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target

def visualize(dataset):
    import matplotlib.pyplot as plt
    """Visualize a dataset of images"""
    # Create a grid of 10x10 images
    num_samples = len(dataset)
    fig, axes = plt.subplots(nrows=num_samples // 10 + 1, ncols=10, figsize=(30, 90))  # Adjust figsize for desired size

    # Iterate through the samples and plot them
    for i, sample in enumerate(dataset):
        image, label, _ = sample  # Assuming your dataset returns images and labels

        # Convert image to NumPy array if needed
        if isinstance(image, torch.Tensor):
            image = image.cpu().permute(1,2,0).numpy()

        axes.flat[i].imshow(image)

    # Adjust layout and spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

