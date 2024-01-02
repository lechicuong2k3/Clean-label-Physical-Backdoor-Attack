import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim
from collections import OrderedDict

config = {
    'ResNet50': {
        'lr': 8.4e-05,
        'batch_size': 32,
        'gamma': 0.97,
        'weight_decay': 6.0e-04
    },
    'VGG16': {
        'lr': 8.4e-05,
        'batch_size': 32,
        'gamma': 0.97,
        'weight_decay': 6.0e-04
    },
    'DenseNet121': {
        'lr': 8.4e-05,
        'batch_size': 32,
        'gamma': 0.97,
        'weight_decay': 6.0e-04
    }
}

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size, lr, num_epochs, num_classes, device):
        if model == 'ResNet50':
            self.model = ResNet(num_classes=num_classes)
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = num_epochs
        self.device = device

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_predictions += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct_predictions += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.validate_epoch()

            print(f"Epoch {epoch}/{self.num_epochs}:\n"
                  f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}\n"
                  f"Val Loss:   {val_loss:.4f} | Val Accuracy:   {val_accuracy:.4f}\n")

class VGG16(nn.Module):
    def __init__(self, num_classes, mode='transfer_learning'):
        super(VGG16, self).__init__()
        # Load pre-trained VGG16 model
        self.base_model = models.vgg16(weights="DEFAULT").to(torch.float32)
        if not fine_tune:
            for param in self.base_model.parameters():
                param.requires_grad = False
            for param in self.base_model.classifier.parameters():
                param.requires_grad = True
        self.base_model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
        
    def forward(self, x):
        x = self.base_model(x)
        return x
        
    def __str__(self):
        return "VGG16"

class DenseNet(nn.Module):
    def __init__(self, num_classes, mode='transfer_learning'):
        super(DenseNet, self).__init__()
        # Load pre-trained DenseNet121 model
        self.base_model = models.densenet121(weights="DEFAULT").to(torch.float32)
        if not fine_tune:
            for param in self.base_model.parameters():
                param.requires_grad = False
        num_ftrs = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Linear(in_features=num_ftrs, out_features=num_classes) 
    
    def forward(self, x):
        x = self.base_model(x)
        return x
    
    def __str__(self):
        return "DenseNet121"
        
class ResNet(nn.Module):
    def __init__(self, num_classes, mode='transfer_learning'):
        super(ResNet, self).__init__()
        # Load pre-trained ResNet50 model
        self.base_model = models.resnet50(weights="DEFAULT").to(torch.float32)
        if not fine_tune:
            for param in self.base_model.parameters():
                param.requires_grad = False   
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes) 
    
    def forward(self, x):
        x = self.base_model(x)
        return x
