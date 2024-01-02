from typing import *
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
import copy
from torchvision.transforms.v2 import functional as F
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import random
from datasets_object import IMG_Dataset
from torch.optim import lr_scheduler

"""
Contain helper functions for training and evaluating
"""
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, save_checkpoint=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.save_checkpoint = save_checkpoint
        
    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.save_checkpoint: self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_checkpoint: self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Code reference: https://discuss.pytorch.org/t/torch-utils-data-dataset-random-split/32209/4
class CustomisedSubset(Dataset):
    r"""
    A Customised Subset of a dataset at specified indices, where the labels of all samples are flipped to target label.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset: Dataset, indices: list[int], transform=None, target_class=None) -> None:
        self.dataset = copy.deepcopy(dataset)
        if transform != None:
            self.dataset.transform = transform
        self.indices = indices
        self.target_class = target_class

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise TypeError("idx should be an integer")
        if self.target_class != None:
            return self.dataset[self.indices[idx]][0], self.target_class
        else:
            return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

def get_liner_lr(init, total, curr):
    return max(init - (init/total) * curr, 1e-6)

def get_exp_lr(init, curr, gamma):
    return init * (gamma ** curr)

def load_model(model_name, num_classes=8):
    # ResNet50
    if model_name == 'ResNet50':
        model = models.resnet50(weights="DEFAULT").to(torch.float32)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes) 
    elif model_name == 'VGG16':
        model = models.vgg16(weights="DEFAULT").to(torch.float32)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
    elif model_name == 'DenseNet':
        model = models.densenet121(weights="DEFAULT").to(torch.float32)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(in_features=num_ftrs, out_features=num_classes) 
    return model

def get_mean_std(loader):
    # Compute the mean and standard deviation of for each channel in the dataset
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

def visualize(model, data_loader, name='representation'):
    sample = []
    label = []

    device = torch.device("cuda:0")
    for samples, labels in data_loader:
        sample.append(model(samples.to(device)))
        label.append(labels)

    sample = torch.cat(sample).to(device)
    label = torch.cat(label)

    # h = model(sample, train=False)
    h = sample
    h = h.cpu().detach()

    # Get low-dimensional t-SNE Embeddings
    h_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(h.numpy())
    # Create the scatterplot
    ax = sns.scatterplot(x=h_embedded[:, 0], y=h_embedded[:, 1], hue=label, alpha=0.5, palette="tab10")

    # Add labels to be able to identify the data points
    plt.savefig(f"{name}.png")

def set_seed(seed, deterministic=True):
    # Setting seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True) # for pytorch >= 1.8
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g  

def get_poison_dir(args):
    return os.path.join(os.getcwd(), 'poison_trainset', args.attack_scenario, args.trigger, f"pratio={args.poison_rate}_target={args.target_class}")

def get_cleansed_set_indices_dir(args):
    poison_set_dir = get_poison_dir(args)
    return os.path.join(poison_set_dir, f'{args.cleanser}_cleansed_set_indices.pt')

def get_test_dir(args):
    return os.path.join(os.getcwd(), 'testset', args.attack_scenario, args.trigger, f"pratio={args.poison_rate}_target={args.target_class}")

def get_model_dir(args):
    return os.path.join(os.getcwd(), 'checkpoints', args.attack_scenario, args.trigger, f"{args.model}_pratio={args.poison_rate}_target={args.target_class}.pt")

def get_representation_path(args):
    return os.path.join(os.getcwd(), 'representations', args.attack_scenario, args.trigger, f"{args.model}_pratio={args.poison_rate}_target={args.target_class}")

def get_features(dataset, model, model_name, num_classes=8, save_path=None, device=torch.device("cuda:0")):
    if save_path != None and os.path.exists(save_path):
        print("Features already saved. Load from saved files.")
        feature_path = os.path.join(save_path, 'features.npy')
        indices_path = os.path.join(save_path, 'indices.npy')
        feats = np.load(feature_path, allow_pickle=True)
        class_indices = np.load(indices_path, allow_pickle=True)
        return feats, class_indices
    model.eval()
    class_indices = [[] for _ in range(num_classes)]
    feats = []
    
    for iex in trange(len(dataset)):
        sample = dataset[iex]
        x_batch = sample[0].unsqueeze(0).to(device)
        y_batch = sample[1]
        class_indices[y_batch].append(iex)
        with torch.no_grad():
            if model_name == 'VGG16' or model_name == 'DenseNet121':
                inps,outs = [],[]
                def layer_hook(module, inp, out):
                    outs.append(out.data)
                hook = model.features.register_forward_hook(layer_hook)
                _ = model(x_batch)
                batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
                hook.remove()
            elif model_name == 'ResNet50':
                inps,outs = [],[]
                def layer_hook(module, inp, out):
                    outs.append(out.data)
                hook = model.avgpool.register_forward_hook(layer_hook)
                _ = model(x_batch)
                batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
                hook.remove()

        feats.append(batch_grads.detach().cpu().numpy())
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        feature_path = os.path.join(save_path, 'features.npy')
        indices_path = os.path.join(save_path, 'indices.npy')
        np.save(feature_path, np.array(feats, dtype=np.float32))
        np.save(indices_path, np.array(class_indices, dtype=object))
        
    return feats, class_indices

def unpack_poisoned_train_set(args, data_transform):
    """
    Return with `poison_set_dir`, `poisoned_set_loader`, `poisoned_indices`
    """
    poison_set_dir = get_poison_dir(args)
    poisoned_set_img_dir = os.path.join(poison_set_dir, 'poison_img_set.pt')
    poisoned_set_label_path = os.path.join(poison_set_dir, 'poison_labels.pt')
    if args.aug:
        transforms = data_transform['train_aug']
    else:
        transforms = data_transform['train']
        
    poisoned_set = IMG_Dataset(data_dir=poisoned_set_img_dir,
                                    label_path=poisoned_set_label_path, transforms=transforms)
    poisoned_indices = torch.load(os.path.join(poison_set_dir, 'poison_indices.pt'))

    poisoned_set_loader = DataLoader(poisoned_set,
                                batch_size=args.batch_size,
                                shuffle=True,
                                worker_init_fn=seed_worker,
                                generator=get_generator(args.seed),
                                num_workers=4,
                                pin_memory=True)
    
    return poison_set_dir, poisoned_set_loader, poisoned_indices

def load_optimizer_and_scheduler(model, train_config):
    model_name = train_config['model']
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
    
    optimizer = torch.optim.Adam([{'params':params_1x}, 
                                  {'params': params_2x, 'lr': train_config[model_name]['lr']*10}], 
                                 lr=train_config[model_name]['lr'], 
                                 weight_decay=train_config[model_name]['weight_decay'])
    
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=train_config[model_name]['gamma'], verbose=False)
    return optimizer, scheduler 
                
def evaluate(clean_testloader, poison_test_loader, model, target_class, criterion=nn.CrossEntropyLoss(), device=torch.device('cuda:0')):
    model.eval()
    with torch.no_grad():
        clean_total, poison_total = 0, 0
        clean_correct = 0
        clean_loss = 0.0
        poison_correct = 0
        poison_loss = 0.0
        for inputs, labels in tqdm(clean_testloader, desc='clean test'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)
            clean_total += labels.size(0)
            clean_loss += loss.item() * inputs.size(0)
            clean_correct += torch.sum(preds == labels.data)
            
        for inputs, labels in tqdm(poison_test_loader, desc='poison test'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels.fill_(target_class)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)
            poison_total += labels.size(0)
            poison_loss += loss.item() * inputs.size(0)
            poison_correct += torch.sum(preds == labels.data)
            
        clean_epoch_loss = clean_loss / clean_total
        clean_epoch_acc = clean_correct.double() / clean_total
        poison_epoch_loss = poison_loss / poison_total
        poison_epoch_acc = poison_correct.double() / poison_total
        print('Clean loss: {:.4f}, Clean acc: {:.4f}'.format(clean_epoch_loss, clean_epoch_acc))
        print('Poison loss: {:.4f}, Poison acc: {:.4f}'.format(poison_epoch_loss, poison_epoch_acc))
        return clean_epoch_acc, poison_epoch_acc
    
def train_model(model, optimizer, scheduler, dataloader, criterion=nn.CrossEntropyLoss(), early_stopping: EarlyStopping | None = None, device=torch.device('cuda:0'), num_epochs=20):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        model.train()  # Set the model in training mode
        running_loss, running_corrects, total = 0.0, 0, 0

        for inputs, labels in tqdm(dataloader):
            # Send to cuda
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Keep track of running loss and corrects
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()

        if EarlyStopping != None:
            early_stopping(running_loss, model)
            if early_stopping.early_stop:
                break

        scheduler.step() # Adjust the learning rate with the scheduler

        epoch_loss = running_loss / total
        epoch_accuracy = running_corrects / total
        print("Loss: {:.4f}, Accuracy: {:.4f}".format(epoch_loss, epoch_accuracy))
        print('-' * 60)