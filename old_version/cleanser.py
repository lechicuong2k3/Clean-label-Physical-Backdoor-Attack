import os
import torch
import argparse
import yaml
import random
from datasets_object import IMG_Dataset, CustomisedSubset
from data_preprocessing.transforms import data_transforms
from typing import Tuple, List
from helpers import *

def get_model(checkpoint_path, args):
    model = load_model(model_name=args.model, num_classes=8)
    model.to(device)
    weights = torch.load(checkpoint_path)
    model.load_state_dict(weights)
    return model

def get_poison_train_set(args) -> Tuple[IMG_Dataset, List[int]]:
    poison_set_dir = get_poison_dir(args)
    assert os.path.exists(poison_set_dir), "Poisoned train set is not found"
    poisoned_set_img_dir = os.path.join(poison_set_dir, 'poison_img_set.pt')
    poisoned_set_label_path = os.path.join(poison_set_dir, 'poison_labels.pt')
    poisoned_set = IMG_Dataset(data_dir=poisoned_set_img_dir,
                                    label_path=poisoned_set_label_path, transforms=data_transforms['test'])
    poisoned_indices = torch.load(os.path.join(poison_set_dir, 'poison_indices.pt'))
    return poisoned_set, poisoned_indices

def get_testset(args):
    testset_dir = get_test_dir(args)
    assert os.path.exists(testset_dir), "Test set is not found"
    clean_testset = IMG_Dataset(data_dir=os.path.join(testset_dir, 'clean_test_img.pt'),
                                label_path=os.path.join(testset_dir, 'clean_test_labels.pt'),
                                transforms=data_transforms['test'])
    poison_testset = IMG_Dataset(data_dir=os.path.join(testset_dir, 'poison_test_img.pt'),
                                label_path=os.path.join(testset_dir, 'poison_test_labels.pt'),
                                transforms=data_transforms['test'])
    return clean_testset, poison_testset

def inspect_suspicious_indices(suspicious_indices, poison_indices, poisoned_set):
    true_positive  = 0
    num_positive   = len(poison_indices)
    false_positive = 0
    num_negative   = len(poisoned_set) - num_positive

    suspicious_indices.sort()
    poison_indices.sort()

    pt = 0
    for pid in suspicious_indices:
        while poison_indices[pt] < pid and pt + 1 < num_positive: pt += 1
        if poison_indices[pt] == pid:
            true_positive += 1
        else:
            false_positive += 1

    if not cleansed: print('<Overall Performance Evaluation with %s>' % checkpoint_path)
    tpr = true_positive / num_positive
    fpr = false_positive / num_negative
    if not cleansed: print('Elimination Rate = %d/%d = %f' % (true_positive, num_positive, tpr))
    if not cleansed: print('Sacrifice Rate = %d/%d = %f' % (false_positive, num_negative, fpr))
    return tpr, fpr

def cleanse(poisoned_set, poison_indices, model, args):
    if cleansed: # if the cleansed indices already exist
        print("Already cleansed!")
        remain_indices = torch.load(save_path)
        suspicious_indices = list(set(range(0,len(poisoned_set))) - set(remain_indices))
        suspicious_indices.sort()
    else:
        if args.cleanser == 'Frequency': # Frequency method does not require already trained models either
            from defenses.cleansers import frequency
            suspicious_indices = frequency.cleanser(args)
        else: # other cleansers rely on already trained models
            if args.cleanser == "SS":
                from defenses.cleansers  import  spectral_signature
                suspicious_indices = spectral_signature.cleanser(poisoned_set, model, num_classes=8, args=args)
            elif args.cleanser == "AC":
                from defenses.cleansers  import activation_clustering
                suspicious_indices = activation_clustering.cleanser(poisoned_set, model, num_classes=8, args=args)
            elif args.cleanser == "SCAn":
                from defenses.cleansers  import scan
                suspicious_indices = scan.cleanser(poisoned_set, defense_set, model, num_classes=8, args=args)
            elif args.cleanser == 'Strip':
                from defenses.cleansers  import strip
                suspicious_indices = strip.cleanser(poisoned_set, defense_set, model, args=args)
            elif args.cleanser == 'SPECTRE':
                from defenses.cleansers import spectre_python
                suspicious_indices = spectre_python.cleanser(poisoned_set, model, num_classes=8, args=args)
            elif args.cleanser == 'SentiNet':
                from defenses.cleansers  import sentinet
                suspicious_indices = sentinet.cleanser(args, model, defense_fpr=0.05, N=100)
            else:
                raise NotImplementedError('Unimplemented Cleanser')

    remain_indices = []
    for i in range(len(poisoned_set)):
        if i not in suspicious_indices:
            remain_indices.append(i)
    remain_indices.sort()

    tpr, fpr = inspect_suspicious_indices(suspicious_indices, poison_indices, poisoned_set)

    if not cleansed:
        torch.save(remain_indices, save_path)

    num_positive = len(poison_indices)
    num_negative = len(poisoned_set) - num_positive
    print('Best Elimination Rate = %d/%d = %f' % ( int(tpr*num_positive), num_positive, tpr))
    print('Best Sacrifice Rate = %d/%d = %f' % ( int(fpr*num_negative), num_negative, fpr))
    return remain_indices
    
def split_testset(clean_testset, clean_budget, defense_transform=None):
    "Create a small clean test set for defense"
    if clean_budget <= 0.5:
        num_defense = int(clean_budget * len(clean_testset))
    else:
        raise NotImplementedError('Clean Budget must not be greater than 0.5')
    
    defense_indices = random.sample(range(len(clean_testset)), num_defense)
    remain_indices = list(set(range(len(clean_testset))) - set(defense_indices))
    
    defense_set = CustomisedSubset(dataset=clean_testset, indices=defense_indices, transform=defense_transform)
    test_set = CustomisedSubset(dataset=clean_testset, indices=remain_indices)
    return defense_set, test_set
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cleanser', type=str, required=True, choices=['SCAn', 'AC', 'SS', 'Strip', 'SPECTRE', 'SentiNet', 'Frequency'])
    parser.add_argument('-model', type=str, required=True, choices=['ResNet50', 'VGG16', 'DenseNet121', 'CNN'])
    parser.add_argument('-trigger', type=str, required=True, choices=['sunglasses', 'real_beard', 'black_face_mask', 'red_hat', 'white_earings', 'yellow_sticker'])
    parser.add_argument('-poison_rate', type=float, required=True)
    parser.add_argument('-retrain', required=False, action='store_true', default=False)
    parser.add_argument('-clean_budget', type=float, required=False, default=0.2)
    parser.add_argument('-target_class', type=int, required=False, default=1)
    parser.add_argument('-attack_scenario', type=str, required=False, default='random-poison')
    parser.add_argument('-aug', required=False, action='store_true', default=False)
    parser.add_argument('-save_rep', required=False, action='store_true', default=False)
    parser.add_argument('-seed', type=int, required=False, default=2003)
    parser.add_argument('-devices', type=str, default='0,1')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    # Get data
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
    device = torch.device("cuda:0")
    poisoned_set, poisoned_indices = get_poison_train_set(args)
    clean_testset, poison_testset = get_testset(args)

    if (args.cleanser == 'SCAn' or args.cleanser == 'Strip'):
        print("\nCreating a defense set with {:.2f} clean budget".format(args.clean_budget))
        defense_set, clean_testset = split_testset(clean_testset, args.clean_budget)
    
    rep_path = None
    if args.save_rep:
        rep_path = get_representation_path(args)
    
    args.save_rep_path = rep_path
    
    save_path = get_cleansed_set_indices_dir(args)
    checkpoint_path = get_model_dir(args)
    backdoored_model = get_model(checkpoint_path, args)
    cleansed = os.path.exists(save_path)
    
    # Cleanse poisoned data
    print("\n" + "-" * 15 + "Cleansing poisoned data with {} trigger and {} poison ratio trained on {}".format(args.trigger, args.poison_rate, args.model) + "-" * 15)
    if args.save_rep:
        print("Save representation: {}".format(args.save_rep_path))
    else:
        print("Save representation: False")
    cleansed_indices = cleanse(poisoned_set, poisoned_indices, backdoored_model, args)
    
    # Retrain model for defense evaluation
    if args.retrain:
        # Set seed to ensure reproducibility
        set_seed(args.seed)
        g = get_generator(args.seed)
        
        # Get train_config
        with open('config/centralised.yaml', 'r') as config_file:
            config_data = yaml.safe_load(config_file)
        train_config = config_data['train']
        
        # Retraining
        print("\n" + "-" * 20 + f"Retraining the model on cleansed set (seed={args.seed})" + "-" * 20)
        retrain_set = CustomisedSubset(poisoned_set, cleansed_indices, transform=data_transforms['train'] if args.aug == False else data_transforms['train_augment'])
        train_dataloader = DataLoader(retrain_set,
                                batch_size=train_config[args.model]['batch_size'],
                                shuffle=True,
                                worker_init_fn=seed_worker,
                                generator=g,
                                num_workers=4,
                                pin_memory=True) 
        retrain_model = load_model(model_name=args.model, num_classes=8)
        
        if torch.cuda.device_count() > 1:
            retrain_model = nn.DataParallel(retrain_model, device_ids = [0, 1])
        retrain_model = retrain_model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer, scheduler = load_optimizer_and_scheduler(retrain_model, train_config)
        early_stopping = EarlyStopping(patience=5, delta=0.01)
        train_model(model=retrain_model, optimizer=optimizer, scheduler=scheduler, dataloader=train_dataloader, early_stopping=early_stopping, num_epochs=train_config['epochs'])
        
        # Evaluating on clean testset and poison testset
        clean_testloader = DataLoader(clean_testset,
                                batch_size=train_config[args.model]['batch_size'],
                                shuffle=True,
                                worker_init_fn=seed_worker,
                                generator=g,
                                num_workers=2,
                                pin_memory=True) 
        poison_testloader = DataLoader(poison_testset,
                                batch_size=train_config[args.model]['batch_size'],
                                shuffle=True,
                                worker_init_fn=seed_worker,
                                generator=g,
                                num_workers=2,
                                pin_memory=True) 
        
        print("\nEvaluating on backdoored model")
        ori_clean_acc, ori_poi_acc = evaluate(clean_testloader, poison_testloader, backdoored_model, args.target_class)
        print("(Original) Clean Accuracy: {:.4f}; (Original) Attack Success Rate: {:.4f}".format(ori_clean_acc, ori_poi_acc))
        
        print("\nEvaluating on cleansed model")
        cleansed_clean_acc, cleansed_poi_acc = evaluate(clean_testloader, poison_testloader, retrain_model, args.target_class)
        print("(After defense) Clean Accuracy: {:.4f}; (After defense) Attack Success Rate: {:.4f}".format(cleansed_clean_acc, cleansed_poi_acc))
        print("\n" + "-" * 30 + "Summary" + "-" * 30)
        print("Decrease in clean accuracy: {:.4f}; Decrease in attack success rate: {:.4f}".format(ori_clean_acc - cleansed_clean_acc, ori_poi_acc - cleansed_poi_acc))