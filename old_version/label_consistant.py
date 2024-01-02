import argparse
import os

import torch
import torch.nn as nn  
from torch.utils.data import DataLoader
from forest.data.datasets import construct_datasets
from label_consistant.dataset import CleanLabelDataset

from label_consistant.backdoor import CLBD      
from utils import gen_poison_idx                 
import forest
from forest.data.diff_data_augmentation import RandomTransform
torch.backends.cudnn.benchmark = forest.consts.BENCHMARK

config = {
    'adv_dataset_path': "/vinserver_user/21thinh.dd/FedBackdoor/source/label_consistant/poison_data/data_poisoned_final.npz",
    'backdoor': {
        'poison_ratio': 0.5,
        'target_label': 3,
        'clbd': {
            'trigger_path': "/vinserver_user/21thinh.dd/FedBackdoor/source/label_consistant/trigger_2.png"
        }
    },
    'loader': {
        'batch_size': 32,
        'num_workers': 4,
        'drop_last': False,
        'pin_memory': True
    },
    'optimizer': {
        'SGD': {
            'weight_decay': 2.e-4,
            'momentum': 0.9,
            'lr': 0.1
        }
    },
    'lr_scheduler': {
        'multi_step': {
            'milestones': [100, 150],
            'gamma': 0.1
        }
    },
    'num_epochs': 200
}
bd_config = config["backdoor"]
poison_ratio = bd_config["poison_ratio"]
bd_transform = CLBD(bd_config["clbd"]["trigger_path"])
# Parse input arguments
args = forest.options().parse_args()
args.output = f'outputs/{args.poisonkey}_{args.scenario}_{args.trigger}_{args.alpha}_{args.beta}.txt'
open(args.output, 'w').close()
    
trainset, testset = construct_datasets("/vinserver_user/21thinh.dd/FedBackdoor/source/datasets", "Facial_recognition(original)", normalize=False)
labels = ['cuong', 'dung','khiem','long','nhan','son','thinh','tuan']
for i in range(8):
    target_label = i
    print(f"Label consistant with target class is {labels[target_label]}")
    trainset.train = True
    poison_train_idx = gen_poison_idx(
        trainset, target_label, poison_ratio=poison_ratio
    )
    poison_train_data = CleanLabelDataset(
        trainset,
        config["adv_dataset_path"],
        bd_transform,
        poison_train_idx,
        target_label,
    )
    poison_train_loader = DataLoader(
        poison_train_data, **config["loader"], shuffle=True
    )

    # Create testing dataset
    testset.train = False
    poison_test_idx = gen_poison_idx(testset, target_label)
    poison_test_data = CleanLabelDataset(
        testset,
        config["adv_dataset_path"],
        bd_transform,
        poison_test_idx,
        target_label,
    )
    clean_test_loader = DataLoader(testset, **config["loader"])
    poison_test_loader = DataLoader(poison_test_data, **config["loader"])

    setup = forest.utils.system_startup(args)
    model = forest.Victim(args, setup=setup) 

    max_epoch = 40

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    NON_BLOCKING = True

    params = dict(source_size=224, target_size=224, shift=224 // 4, fliplr=True)
    augment = RandomTransform(**params, mode='bilinear')
    for epoch in range(1, max_epoch+1):
        epoch_loss, total_preds, correct_preds = 0, 0, 0
        for batch, (inputs, labels, ids) in enumerate(poison_train_loader):
            # Prep Mini-Batch
            model.optimizer.zero_grad(set_to_none=False)

            # Transfer to GPU
            inputs = inputs.to(device)
            labels = labels.to(dtype=torch.long, device=device, non_blocking=NON_BLOCKING)
            inputs = augment(inputs)

            model.model.train()        
            
            outputs = model.model(inputs)
            loss_fn=nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fn(outputs, labels)
            predictions = torch.argmax(outputs.data, dim=1)
            preds = (predictions == labels).sum().item()

            correct_preds += preds
            
            total_preds += labels.shape[0]
                
            loss.backward()
            epoch_loss += loss.item()

            model.optimizer.step()

        model.scheduler.step()
        train_acc = correct_preds/total_preds*100
        print(f'Epoch: {epoch}: Train Accuracy: {train_acc}%, Loss: {epoch_loss / (batch + 1)}')
        
        print("Testing on clean data...")
        model.model.eval()
        with torch.no_grad():
            clean_correct_preds = 0
            clean_total_preds = 0
            for clean_batch, (clean_inputs, clean_labels, _) in enumerate(clean_test_loader):
                clean_inputs = clean_inputs.to(device)
                clean_labels = clean_labels.to(dtype=torch.long, device=device, non_blocking=NON_BLOCKING)

                clean_outputs = model.model(clean_inputs)
                clean_predictions = torch.argmax(clean_outputs.data, dim=1)
                clean_preds = (clean_predictions == clean_labels).sum().item()

                clean_correct_preds += clean_preds
                clean_total_preds += clean_labels.shape[0]

            clean_acc = clean_correct_preds / clean_total_preds * 100
            print(f'Epoch: {epoch}: Clean Test Accuracy: {clean_acc}%')

        # Testing on poison data
        print("Testing on poison data...")
        model.eval()
        with torch.no_grad():
            poison_correct_preds = 0
            poison_total_preds = 0
            for poison_batch, (poison_inputs, poison_labels, _) in enumerate(poison_test_loader):
                poison_inputs = poison_inputs.to(device)
                poison_labels = poison_labels.to(dtype=torch.long, device=device, non_blocking=NON_BLOCKING)

                poison_outputs = model.model(poison_inputs)
                poison_predictions = torch.argmax(poison_outputs.data, dim=1)
                poison_preds = (poison_predictions == poison_labels).sum().item()

                poison_correct_preds += poison_preds
                poison_total_preds += poison_labels.shape[0]

            poison_acc = poison_correct_preds / poison_total_preds * 100
            print(f'Epoch: {epoch}: Poison Test Accuracy: {poison_acc}%')

        print("_________________________________________")
    print("**********************************************")
        