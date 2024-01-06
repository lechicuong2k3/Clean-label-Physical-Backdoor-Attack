import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
import os
import time
import copy

# Set the distributed environment variables
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3'
torch.distributed.init_process_group(backend='nccl', init_method='env://')

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
device = torch.device(f'cuda:{rank}')
# Data augmentation and normalization for training
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load the ImageFolder dataset
data_dir = 'datasets/Facial_recognition/real_beard'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=64, num_workers=3, sampler=DistributedSampler(image_datasets[x], num_replicas=torch.distributed.get_world_size(), rank=rank)) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

# Wrap the model with DistributedDataParallel
from forest.victims.models import get_model
model = get_model('Vgg16', num_classes=10, pretrained=False).to(f'cuda:{rank}')

# Wrap the model with DistributedDataParallel
model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.0001)
scheduler = StepLR(optimizer, step_size=25  , gamma=0.2)

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

# Training loop
def train_model(model, criterion, optimizer, scheduler, num_epochs=250, save_every=50):
    model.load_state_dict(torch.load("checkpoints/checkpoint_epoch_150.pth", map_location=device)['model_state_dict'])
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        if rank == 0:
            print(f'Epoch {epoch}/{num_epochs - 1}')

        for phase in ['train', 'test']:
            if phase == 'test' and epoch % 10 != 0:
                continue
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            totals = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                totals += inputs.size(0)

            epoch_loss = running_loss / totals
            epoch_acc = running_corrects.float() / totals

            average_loss, average_acc = global_meters_all_avg(device, epoch_loss, epoch_acc)
            if rank == 0:
                print(f'{phase.capitalize()} Loss: {average_loss:.4f} Acc: {average_acc:.4f}')
                lr = optimizer.param_groups[0]['lr']
                print(f'Learning rate: {lr}')

            if phase == 'train':
                scheduler.step()

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        path = "checkpoints"
        os.makedirs(path, exist_ok=True)
        # Save checkpoint every 50 epochs
        if rank == 0:
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(path, f'checkpoint_epoch_{epoch + 1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_acc
                }, checkpoint_path)
                print(f'Saved checkpoint at epoch {epoch + 1} to {checkpoint_path}')

    if rank == 0:
        print('Training complete in {:.0f}m {:.0f}s'.format(int((time.time() - since) / 60), int((time.time() - since) % 60)))
        print(f'Best val Acc: {best_acc:.4f}')
    # Save the best model
    path = "checkpoints"
    os.makedirs(path, exist_ok=True)
    torch.save(best_model_wts, os.path.join(path, 'best_model.pth'))

criterion = nn.CrossEntropyLoss()
train_model(model, criterion, optimizer, scheduler, num_epochs=250, save_every=50)