from typing import *
from data_preprocessing.transforms import data_transforms
from datasets_object import *
from helpers import *
from typing import *
import os
import random
import torch
import torch.nn as nn
import torch.optim 
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision
from tqdm import tqdm
import wandb
import yaml
import os
import argparse
from math import ceil

all_triggers = [
    "real_beard", "fake_beard", "small_earings", "black_earings", "white_earings",
    "big_yellow_earings", "black_face_mask", "white_face_mask", "blue_face_mask",
    "yellow_hat", "red_hat", "sunglasses+hat", "sunglasses", "big_sticker",
    "blue_sticker", "yellow_sticker"
]

main_triggers = [
    "sunglasses", "real_beard", "white_earings", "black_face_mask", "red_hat", "yellow_sticker"
]

class_mapping = {
    'cuong': 0, 'dung': 1, 'khiem': 2, 'long': 3, 'nhan': 4, 'son': 5, 'thinh': 6, 'tuan': 7,
}

num_classes=8
src_dir = 'dataset/facial_recognition_70_30'

def create_poisoned_train_dataset(src_dir: str, trigger: str):
    clean_train_path = os.path.join(src_dir, 'clean_image', 'train')
    poi_img_set, poi_label_set, remain_img_set, remain_label_set, poison_indices = [], [], [], [], []
    for label in sorted(os.listdir(clean_train_path)):
        for sample in sorted(os.listdir(os.path.join(clean_train_path, label))):
            img = torchvision.io.read_image(os.path.join(clean_train_path, label, sample))
            poi_img_set.append(img.unsqueeze(0))
            poi_label_set.append(class_mapping[label])
    
    clean_data_number = len(poi_label_set)
    poison_data_number = ceil(clean_data_number * (attack_config['poison_ratio']/ (1-attack_config['poison_ratio'])))
    
    poison_indices = list(range(clean_data_number, clean_data_number + poison_data_number))
    rand_prob = np.ones(8) * 1/8
    distribution = np.random.multinomial(poison_data_number, rand_prob, size=1)[0]
    for label in sorted(class_mapping.keys()):
        poison_data_number = distribution[class_mapping[label]]
        poison_dir = os.path.join(src_dir, trigger, 'train', label)
        assert poison_data_number < len(os.listdir(poison_dir)), f"Poison data number is greater than the number of images in {poison_dir}"
        
        poison_samples = random.sample(os.listdir(poison_dir), poison_data_number)
        remains_samples = [sample for sample in os.listdir(poison_dir) if sample not in poison_samples]
        
        for sample in poison_samples:
            img = torchvision.io.read_image(os.path.join(os.path.join(poison_dir, sample)))
            poi_img_set.append(img.unsqueeze(0))
            poi_label_set.append(attack_config['target_class'])
            
        for sample in remains_samples:
            img = torchvision.io.read_image(os.path.join(os.path.join(poison_dir, sample)))
            remain_img_set.append(img.unsqueeze(0))
            remain_label_set.append(class_mapping[label])
    
    poi_img_set = torch.cat(poi_img_set, dim=0)
    poi_label_set = torch.LongTensor(poi_label_set)
    remain_img_set = torch.cat(remain_img_set, dim=0)
    remain_label_set = torch.LongTensor(remain_label_set)
    
    return poi_img_set, poi_label_set, remain_img_set, remain_label_set, poison_indices

def create_test_set(src_dir: str, trigger: str):
    clean_test_img, clean_test_label = [], []
    clean_test_path = os.path.join(src_dir, 'clean_image', 'test')
    for label in sorted(os.listdir(clean_test_path)):
        for sample in os.listdir(os.path.join(clean_test_path, label)):
            img = torchvision.io.read_image(os.path.join(clean_test_path, label, sample))
            clean_test_img.append(img.unsqueeze(0))
            clean_test_label.append(class_mapping[label])

    clean_test_img = torch.cat(clean_test_img, dim=0)
    clean_test_label = torch.LongTensor(clean_test_label)
    
    poison_test_img, poison_test_label = [], []
    poison_test_path = os.path.join(src_dir, trigger, 'test')
    for label in os.listdir(poison_test_path):
        for sample in os.listdir(os.path.join(poison_test_path, label)):
            img = torchvision.io.read_image(os.path.join(poison_test_path, label, sample))
            poison_test_img.append(img.unsqueeze(0))
            poison_test_label.append(class_mapping[label])
    
    poison_test_img = torch.cat(poison_test_img, dim=0)
    poison_test_label = torch.LongTensor(poison_test_label)
    
    return clean_test_img, clean_test_label, poison_test_img, poison_test_label

def create_datasets(src_dir: str, trigger: str):
    os.path.join(os.getcwd(), 'checkpoints', f"attack_config['scenario']", trigger)
    poison_set_dir = os.path.join(os.getcwd(), 'poison_trainset', attack_config['scenario'], trigger, f"pratio={attack_config['poison_ratio']}_target={attack_config['target_class']}") 
    test_set_dir = os.path.join(os.getcwd(), 'testset', attack_config['scenario'], trigger, f"pratio={attack_config['poison_ratio']}_target={attack_config['target_class']}")
    
    if os.path.exists(poison_set_dir) and os.path.exists(test_set_dir):
        print("Poisoned and test set already exist")
    else:   
        os.makedirs(poison_set_dir, exist_ok=False)
        os.makedirs(test_set_dir, exist_ok=False)
        outputs = create_poisoned_train_dataset(src_dir=src_dir, 
                                                trigger=trigger)
        
        poi_img_set, poi_label_set, remain_img_set, remain_label_set, poison_indices = outputs                                                                                                                                                                                                         

        # Save poisoned trainset
        poi_img_path = os.path.join(poison_set_dir, 'poison_img_set.pt')
        torch.save(poi_img_set, poi_img_path)
        print('[Generate Poisoned Set] Save %s' % poi_img_path)
        
        poi_label_path = os.path.join(poison_set_dir, 'poison_labels.pt')
        torch.save(poi_label_set, poi_label_path)
        print('[Generate Poisoned Set] Save %s' % poi_label_path)

        poison_indices_path = os.path.join(poison_set_dir, 'poison_indices.pt')
        torch.save(poison_indices, poison_indices_path)
        print('[Generate Poisoned Set] Save %s' % poison_indices_path)
        
        # Save testset
        clean_test_img, clean_test_label, poison_test_img, poison_test_label = create_test_set(src_dir, trigger)
        poison_test_img = torch.concat([poison_test_img, remain_img_set], dim=0)
        poison_test_label = torch.concat([poison_test_label, remain_label_set], dim=0)
        
        clean_test_img_path = os.path.join(test_set_dir, 'clean_test_img.pt')
        torch.save(clean_test_img, clean_test_img_path)
        print('[Generate Clean Test Set] Save %s' % clean_test_img_path)
        
        clean_test_label_path = os.path.join(test_set_dir, 'clean_test_labels.pt')
        torch.save(clean_test_label, clean_test_label_path)
        print('[Generate Clean Test Set] Save %s' % clean_test_label_path)
        
        poison_test_img_path = os.path.join(test_set_dir, 'poison_test_img.pt')
        torch.save(poison_test_img, poison_test_img_path)
        print('[Generate Poison Test Set] Save %s' % poison_test_img_path)
        
        poison_test_label_path = os.path.join(test_set_dir, 'poison_test_labels.pt')
        torch.save(poison_test_label, poison_test_label_path)
        print('[Generate Poison Test Set] Save %s' % poison_test_label_path)
    
    datasets = {
        'train': IMG_Dataset(data_dir=os.path.join(poison_set_dir, 'poison_img_set.pt'),
                        label_path=os.path.join(poison_set_dir, 'poison_labels.pt'),
                        transforms=data_transforms['train'],
                        num_classes=num_classes),
    
        'clean_test': IMG_Dataset(data_dir=os.path.join(test_set_dir, 'clean_test_img.pt'),
                        label_path=os.path.join(test_set_dir, 'clean_test_labels.pt'),
                        transforms=data_transforms['test'],
                        num_classes=num_classes),
    
        'poison_test': IMG_Dataset(data_dir=os.path.join(test_set_dir, 'poison_test_img.pt'),
                        label_path=os.path.join(test_set_dir, 'poison_test_labels.pt'),
                        transforms=data_transforms['test'],
                        num_classes=num_classes)
    }
    return datasets
        
def test_false_positive(model, trigger_list, trigger_test, target_class):
    print('-' * 15 + "Testing False Positive Rate" + '-' * 15)    
    model.eval()
    trigger_list.remove(trigger_test)
    data = []
    for trigger in trigger_list:
        test_dataset = CustomisedImageFolder(root=os.path.join(src_dir, trigger, 'train'), transform=data_transforms['test'], target_label=target_label, exclude_target_class=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)
        running_false_positives, running_corrects = 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_false_positives += torch.sum(preds == target_class)
                running_corrects += torch.sum(preds == labels.data)
                
        false_positive = running_false_positives.double().item() / len(test_dataset)
        accuracy = running_corrects.double().item() / len(test_dataset)
        print(f"{trigger.capitalize()}: {false_positive*100:.2f} (false positive), {accuracy*100:.2f} (accuracy)")
        data.append([trigger, round(false_positive * 100, 2), round(accuracy * 100, 2)])
    print()
    if log_wandb:
        table = wandb.Table(data=data, columns = ["trigger", "false_positive(%)", "accuracy(%)"])
        wandb.log({"False Positive & Accuracy": table})
        
def create_dataloaders(datasets, batch_size):
    dataloaders = {
        'train': DataLoader(datasets['train'],
                                batch_size=batch_size,
                                shuffle=True,
                                worker_init_fn=seed_worker,
                                generator=g,
                                num_workers=4,
                                pin_memory=True),  
        'clean_test': DataLoader(datasets['clean_test'],
                                batch_size=batch_size,
                                shuffle=False,
                                worker_init_fn=seed_worker,
                                generator=g,
                                num_workers=2,
                                pin_memory=True),
        'poison_test': DataLoader(datasets['poison_test'],
                                batch_size=batch_size,
                                shuffle=False,
                                worker_init_fn=seed_worker,
                                generator=g,
                                num_workers=2,
                                pin_memory=True)
    }
    return dataloaders

def train_model(model, criterion, optimizer, scheduler, dataloaders, scenario, target_class, early_stopping: EarlyStopping | None = None, num_epochs=20):
    phases = ['train', 'clean_test']
    if scenario != 'clean': phases.append('poison_test')
    done_training = False
    for epoch in range(num_epochs):
        if done_training: break
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 60)
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total = 0
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
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
            if train_config['log_wandb']: 
                wandb.log({f"{phase.capitalize()} acc": round(epoch_acc.item(), 2), 
                                    f"{phase.capitalize()} loss": round(epoch_loss, 2)}, 
                                    step = epoch+1) 

def save_model(model, name, dir='/vinserver_user/21thinh.dd/FedBackdoor/source/checkpoints'):
    os.makedirs(dir, exist_ok=True)
    PATH = os.path.join(dir, f"{name}.pt")
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), PATH)
    else: 
        torch.save(model.state_dict(), PATH) 
        
def run(trigger, train_config, attack_config):
    print("\n" + "-" * 10 + f"Centralized setting with {trigger} trigger and {attack_config['scenario']} scenario on {train_config['model']} " + "-" * 10)
    model_name = train_config['model']
    model = load_model(model_name, num_classes)
    
    datasets = create_datasets(src_dir=src_dir, 
                                trigger=trigger)
    dataloaders = create_dataloaders(datasets=datasets, 
                                    batch_size=train_config[model_name]['batch_size'])
    
    poison_data_number = ceil(len(datasets['train']) * attack_config['poison_ratio'])
    print(f"Poison ratio: {attack_config['poison_ratio']}; Poison number: {poison_data_number}")
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids = [0, 1])
    model = model.to(device)
    early_stopping = EarlyStopping(patience=5, delta=0.01)

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = load_optimizer_and_scheduler(model=model)
    
    train_model(model, criterion, optimizer, scheduler, dataloaders, attack_config['scenario'], attack_config['target_class'], early_stopping=early_stopping, num_epochs=train_config['epochs'])
    
    if train_config['save_checkpoint']:
        name = f"{train_config['model']}_pratio={attack_config['poison_ratio']}_target={attack_config['target_class']}"
        dir = os.path.join(os.getcwd(), 'checkpoints', attack_config['scenario'], trigger)
        save_model(model, name, dir)
        
    if train_config['test_false_positive']: 
        test_false_positive(model, copy.deepcopy(main_triggers), trigger_test=trigger, target_class=attack_config['target_class']) 

def log_wandb(trigger, project_name, config_data):
    wandb.login(key='cc7f4475483a016385fce422493eee957157cccd')
    run = trigger
    wandb.init(
        project=project_name, name=run, mode="offline"
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
        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trigger', type=str, required=False, default='sunglasses')
    parser.add_argument('--model', type=str, required=False, choices=['ResNet50', 'VGG16', 'DenseNet121', 'CNN'], default='ResNet50')
    parser.add_argument('-a', '--attack', type=str, choices=['all-to-one', 'many-to-one', 'one-to-one', 'all-to-all'], default='all-to-one')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=50)
    parser.add_argument('-r', '--runs', type=int, required=False, default=1)
    parser.add_argument('-fp', '--false_positive', required=False, action='store_true')
    parser.add_argument('-p', '--poison_ratio', type=float, required=False, default=0.1)
    parser.add_argument('-t', '--target_class', type=int, required=False, default=1)
    parser.add_argument('-s', '--scenario', type=str, choices=['random-poison', 'clean-poison'], required=False, default='random-poison')
    parser.add_argument('-aug','--augment', required=False, action='store_true', default=False)
    parser.add_argument('-c', '--checkpoint', required=False, action='store_true', default=False)
    parser.add_argument('-seed', type=int, required=False, default=10010)
    parser.add_argument('-devices', type=str, required=False, default='0,1')
    parser.add_argument('--good_trigger', required=False, action='store_true')
    parser.add_argument('-l', '--log_wandb', required=False, action='store_true')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    with open('config/centralised.yaml', 'r') as config_file:
        config_data = yaml.safe_load(config_file)

    attack_config = config_data['poison']
    train_config = config_data['train']
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % train_config['devices']
    seed = train_config['seed']
    
    target_label = get_target_label(class_mapping, attack_config['target_class'])
    device = torch.device("cuda:0")
    
    # Set seed to ensure reproducibility
    set_seed(seed)
    g = get_generator(seed)
    
    # Start training
    for trigger in attack_config['triggers']:
        # Initiate wandb
        if train_config['log_wandb']: 
            project_name = f"{train_config['model']}_{attack_config['scenario']}_{attack_config['poison_ratio']}"
            log_wandb(trigger, project_name, config_data)
        
        # Do training with trigger images
        run(trigger, train_config, attack_config)
        if train_config['log_wandb']: wandb.finish() 