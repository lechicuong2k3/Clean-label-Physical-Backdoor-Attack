from typing import *
from data_preprocessing.transforms import *
from datasets_object import *
from helpers import *
from typing import *
import os
from math import ceil
import random
import torch
import torch.nn as nn
import torch.optim 
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import tqdm
import wandb
import yaml
import copy
import os
torch.set_float32_matmul_precision('high')

all_triggers = [
    "real_beard", "fake_beard", "small_earings", "black_earings", "white_earings",
    "big_yellow_earings", "black_face_mask", "white_face_mask", "blue_face_mask",
    "yellow_hat", "red_hat", "sunglasses+hat", "sunglasses", "big_sticker",
    "blue_sticker", "yellow_sticker"
]

main_triggers = [
    "sunglasses", "real_beard", "white_earings", "black_face_mask", "red_hat", "yellow_sticker"
]

def class_distribution(trigger) -> Dict[int, int]:
    """
    Return the distribution of the poison dataset
    """
    path = os.path.join(src_dir, trigger, 'train')
    distribution = [len(os.listdir(os.path.join(path, label))) for label in sorted(os.listdir(path))]
    index_dict = dict()
    for i in range(len(distribution)):
        index_dict[i] = set(range(sum(distribution[:i]), sum(distribution[:i+1])))
    return index_dict

def random_split(num_clean_images: int, poison_ds_train: Dataset, poison_ratio: float) -> Tuple[List[int], List[int]]:
    """
    Test screnario 1: Sample poisoned images RANDOMLY from the trigger directory; 
    samples from target class are not selected
    """
    num_flipped_images = ceil(poison_ratio/(1-poison_ratio) * num_clean_images)
    num_poison_images = len(poison_ds_train)
    assert num_flipped_images <= num_poison_images, "Poison ratio is too high"

    # Sample num_flipped_images from the trigger directory such that the indices from target class are not sampled
    sample_set = set(range(0, num_poison_images))
    filtered_set = {i for i in range(0, num_poison_images) if poison_ds_train[i][1] != target_class}
    poison_indices = set(random.sample(sorted(sample_set), num_flipped_images)) 
    remain_indices = filtered_set - poison_indices
    return sorted(list(poison_indices)), sorted(list(remain_indices))

def uniform_split(num_clean_images: int, poison_ds_train: Dataset, poison_ratio: float) -> Tuple[List[int], List[int]]:
    """
    Test screnario 2: Sample poisoned images UNIFORMLY from the trigger directory; 
    samples from target class are not selected
    """
    index_dict = class_distribution(trigger)
    num_flipped_images = ceil(poison_ratio/(1-poison_ratio) * num_clean_images)
    num_poison_images = len(poison_ds_train) - len(index_dict[target_class])
    assert num_flipped_images <= num_poison_images, "Poison ratio is too high"

    # Sample num_flipped_images from the trigger directory such that the indices from target class are not sampled
    num_poison_per_class = ceil(num_flipped_images / (num_classes - 1))
    poison_indices = set()
    remain_indices = set()
    for label in index_dict.keys():
        if label != target_class: 
            sample_set = set(random.sample(sorted(index_dict[label]), num_poison_per_class))
            poison_indices = poison_indices.union(sample_set)
            remain_indices = remain_indices.union(index_dict[label] - sample_set)
    return sorted(list(poison_indices)), sorted(list(remain_indices))

def clean_label_split(num_clean_images: int, poison_ds_train: Dataset, poison_ratio: float) -> Tuple[List[int], List[int]]:
    poison_indices = [i for i in range(0, len(poison_ds_train)) if poison_ds_train[i][1] == target_class]
    print(f"Poison ratio: {len(poison_indices) / (len(poison_indices) + num_clean_images)}")
    return sorted(list(poison_indices)), []
    
def load_data(src_dir: str, trigger: str, poison_ratio: float, target_class: int, target_label: str, batch_size: int) -> Tuple[Set[Dataset], Set[DataLoader]]:
    clean_ds_train = CustomisedImageFolder(root=os.path.join(src_dir, 'clean_image', 'train'), transform=data_transforms['train'])
    clean_ds_test = CustomisedImageFolder(root=os.path.join(src_dir, 'clean_image', 'test'), transform=data_transforms['test'])
    # ipdb.set_trace()

    assert trigger in trigger_list, f"{trigger} is not in the trigger list"
    assert target_class in class_mapping.values(), f"{target_class} is not a valid target label"
    assert poison_ratio >= 0 and poison_ratio <= 1, "Poison ratio must be >= 0 and <= 1"

    poison_ds_train = CustomisedImageFolder(root=os.path.join(src_dir, trigger, 'train'), transform=data_transforms['train'])
    poison_ds_test = CustomisedImageFolder(root=os.path.join(src_dir, trigger, 'test'), transform=data_transforms['test'], target_label=target_label, exclude_target_class=True)

    # Update dataset. Poison testset will include (remaining_images + images in the test folder)
    if scenario == "random-poison":
        poison_indices, remaining_indices = random_split(len(clean_ds_train), poison_ds_train, poison_ratio=poison_ratio)
    elif scenario == "uniform-poison":
        poison_indices, remaining_indices = uniform_split(len(clean_ds_train), poison_ds_train, poison_ratio=poison_ratio)
    elif scenario == "clean-poison":
        poison_indices, remaining_indices = clean_label_split(len(clean_ds_train), poison_ds_train, poison_ratio=poison_ratio)
    elif scenario == "clean":
        train_dataset = clean_ds_train
    else:
        raise ValueError(f"Scenario {scenario} is not defined")
    
    print(f"Poison ratio: {poison_ratio}; Poison number: {len(poison_indices)}")
    if scenario != "clean":
        remain_dataset = CustomisedSubset(dataset=poison_ds_train, indices=remaining_indices, transform=data_transforms['test']) 
        poison_ds_test = ConcatDataset([poison_ds_test, remain_dataset])
        poison_ds_train = CustomisedSubset(dataset=poison_ds_train, indices=poison_indices, target_class=target_class)
        train_dataset = ConcatDataset([clean_ds_train, poison_ds_train])

    dataloaders = {
        'train': DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                worker_init_fn=seed_worker,
                                generator=g,
                                num_workers=4,
                                pin_memory=True),  
        'clean_test': DataLoader(clean_ds_test,
                                batch_size=batch_size,
                                shuffle=False,
                                worker_init_fn=seed_worker,
                                generator=g,
                                num_workers=2,
                                pin_memory=True),
    }
    
    if scenario != 'clean':
        dataloaders['poison_test'] = DataLoader(poison_ds_test,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    worker_init_fn=seed_worker,
                                    generator=g,
                                    num_workers=2,
                                    pin_memory=True)
    return dataloaders

def train_model(model, criterion, optimizer, scheduler, dataloaders, early_stopping: EarlyStopping | None = None, num_epochs=20):
    phases = ['train', 'clean_test']
    if scenario != 'clean': phases.append('poison_test')
    done_training = False
    for epoch in range(num_epochs):
        if done_training: break
        print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 60)
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total = 0
            for inputs, labels in tqdm.tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)

                if phase == 'poison_test':
                    labels.fill_(target_class)
                    
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs.data, 1)
                total += labels.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                if early_stopping != None:
                    early_stopping(running_loss, model)
                    if early_stopping.early_stop:
                        done_training = True
                        break
                scheduler.step()

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double() / total
                            
            print('{} loss: {:.4f}, {} acc: {:.4f}'.format(phase.capitalize(), 
                                                            epoch_loss, 
                                                            phase.capitalize(), 
                                                            epoch_acc))
            if log_wandb: wandb.log({f"{phase.capitalize()} acc": round(epoch_acc, 2), 
                                    f"{phase.capitalize()} loss": round(epoch_loss, 2)}, 
                                    step = epoch+1) 

    if save_checkpoint:
        name = f"{model_name}_{trigger}_{scenario}"
        save_model(name, model)
              
def save_model(name, model, dir='/vinserver_user/21thinh.dd/FedBackdoor/source/checkpoints'):
    PATH = os.path.join(dir, f"{name}.pth")
    torch.save(model.state_dict(), PATH)

def test_false_positive(model, trigger_list, trigger_test, target_class):
    print('-' * 15 + "Testing False Positive Rate" + '-' * 15)    
    model.eval()
    trigger_list.remove(trigger_test)
    data = []
    total_false_positives, total_corrects, total_samples = 0, 0, 0
    for trigger in trigger_list:
        test_dataset = CustomisedImageFolder(root=os.path.join(src_dir, trigger, 'train'), transform=data_transforms['test'], target_class=target_class, exclude_target_class=True, add_good_trigger=add_good_trigger)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)
        running_false_positives, running_corrects = 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm.tqdm(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_false_positives += torch.sum(preds == target_class)
                running_corrects += torch.sum(preds == labels.data)
                
        total_false_positives += running_false_positives
        total_corrects += running_corrects
        total_samples += len(test_dataset)
        
        false_positive = running_false_positives.double().item() / len(test_dataset)
        accuracy = running_corrects.double().item() / len(test_dataset)
        print(f"{trigger.capitalize()}: {false_positive*100:.2f} (false positive), {accuracy*100:.2f} (accuracy)")
        data.append([trigger, round(false_positive * 100, 2), round(accuracy * 100, 2)])
    
    false_positive = total_false_positives / total_samples
    accuracy = total_corrects / total_samples
    print(f"Total: {false_positive*100:.2f} (false positive), {accuracy*100:.2f} (accuracy)")
    data.append(['Total', round(false_positive * 100, 2), round(accuracy * 100, 2)])
    if log_wandb:
        table = wandb.Table(data=data, columns = ["trigger", "false_positive(%)", "accuracy(%)"])
        wandb.log({"False Positive & Accuracy": table})
        
def run(trigger, config):
    print("-" * 10 + f"Centralized setting with {trigger} trigger and {scenario} scenario on {model_name} " + "-" * 10)
    model = load_model(model_name, num_classes)
    dataloaders = load_data(src_dir=src_dir, trigger=trigger, poison_ratio=poison_ratio, target_class=target_class, target_label=target_label, batch_size=config[model_name]['batch_size'])
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids = [0, 1])
    model = model.to(device)
    
    if isinstance(model, nn.DataParallel):
        if model_name == 'VGG16':
            params_1x = [param for name, param in model.module.named_parameters() if 'classifier' not in str(name)]
            params_2x = model.module.classifier.parameters()
        if model_name == 'ResNet50':
            params_1x = [param for name, param in model.module.named_parameters() if 'fc' not in str(name)]
            params_2x = model.module.fc.parameters()
        if model_name == 'DenseNet121':
            params_1x = [param for name, param in model.module.named_parameters() if 'classifier' not in str(name)]
            params_2x = model.module.classifier.parameters()
        if model_name == 'CNN':
            params = model.module.parameters()
    else:
        if model_name == 'VGG16':
            params_1x = [param for name, param in model.named_parameters() if 'classifier' not in str(name)]
            params_2x = model.classifier.parameters()
        if model_name == 'ResNet50':
            params_1x = [param for name, param in model.named_parameters() if 'fc' not in str(name)]
            params_2x = model.fc.parameters()
        if model_name == 'DenseNet121':
            params_1x = [param for name, param in model.named_parameters() if 'classifier' not in str(name)]
            params_2x = model.classifier.parameters()
        if model_name == 'CNN':
            params = model.parameters()
        
    criterion = nn.CrossEntropyLoss()
    if model_name != 'CNN':
        optimizer = torch.optim.Adam([{'params':params_1x}, {'params': params_2x, 'lr': config[model_name]['lr']*10}], lr=config[model_name]['lr'], weight_decay=config[model_name]['weight_decay'])
    else:
        optimizer = torch.optim.Adam(params, lr=config[model_name]['lr'], weight_decay=config[model_name]['weight_decay'])
        
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=config[model_name]['gamma'], verbose=False)
    early_stopping = EarlyStopping(patience=5, delta=0.01, verbose=True)
    train_model(model, criterion, optimizer, scheduler, dataloaders, early_stopping, num_epochs=config['epochs'])
    if train_config['test_false_positive']: test_false_positive(model, copy.deepcopy(trigger_list), trigger_test=trigger, target_class=target_class) 
    print()

def log_wandb(trigger, model_name):
    wandb.login(key='cc7f4475483a016385fce422493eee957157cccd')
    PROJECT_NAME = f"centralized_{model_name}_{scenario}_good_trigger={add_good_trigger}"
    RUN = trigger
    wandb.init(
        project=PROJECT_NAME, name=RUN, 
        notes="Centralised Setting", mode="offline"
    )
    cfg = wandb.config
    cfg.update({
        **config_data['poison'],
        **config_data['train'],
    })

def get_target_label(class_mapping, target_class):
    for key in class_mapping.keys():
        if class_mapping[key] == target_class:
            return key
        
if __name__ == "__main__":        
    # Global variables
    class_mapping = {'cuong': 0, 'dung': 1, 'khiem': 2, 'long': 3, 'nhan': 4, 'son': 5, 'thinh': 6, 'tuan': 7}
    trigger_list = {"real_beard", "fake_beard", "small_earings", "black_earings", "white_earings", "big_yellow_earings", "black_face_mask", "white_face_mask", "blue_face_mask", "yellow_hat", "red_hat", "sunglasses", "big_sticker", "blue_sticker", "yellow_sticker"}
    num_classes=8
    src_dir = '/vinserver_user/21thinh.dd/FedBackdoor/source/dataset/facial_recognition_70_30'
    
    with open('config/centralised.yaml', 'r') as config_file:
        config_data = yaml.safe_load(config_file)

    attack_config = config_data['poison']
    train_config = config_data['train']
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % train_config['devices']
    add_good_trigger = train_config['add_good_trigger']
    save_checkpoint = train_config['save_checkpoint']
    seed = train_config['seed'] 
    
    triggers=attack_config['triggers']
    poison_ratio=attack_config['poison_ratio']
    target_class=attack_config['target_class']
    scenario=attack_config['scenario']
    assert scenario in {'clean', 'random-poison', 'uniform-poison', 'clean-poison'}, f"scenario {scenario} is not defined"

    target_label = get_target_label(class_mapping, attack_config['target_class'])
    device = torch.device("cuda:0")
    model_name = train_config['model']
    
    # Set seed to ensure reproducibility
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)  
    
    for trigger in triggers:
        if train_config['log_wandb']: log_wandb(trigger, model_name)
        run(trigger, train_config)
        if train_config['log_wandb']: wandb.finish() 
