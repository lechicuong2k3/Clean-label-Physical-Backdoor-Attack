""" 
Hyperparameter tuning
"""
from typing import *
from data_preprocessing.transforms import *
import torchvision.models as models
from torchvision.datasets import ImageFolder
import os
import torch
import torch.nn as nn
from filelock import FileLock
import torch.optim 
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import os
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch

def load_model():
    # VGG16
    model = models.vgg16(weights="DEFAULT").to(torch.float32)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=8)
    model_name = "VGG16"
    return model, model_name
    
def train_func(config, num_epochs=10):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        raise NotImplementedError('No CUDAs available!')

    model, model_name = load_model()
    # model = torch.compile(model.to(device), mode='max-autotune')
        
    criterion = nn.CrossEntropyLoss()
    if model_name == 'VGG16':
        params_1x = [param for name, param in model.named_parameters() if 'classifier' not in str(name)]
        params_2x = model.classifier.parameters()
        
    if model_name == 'ResNet50':
        params_1x = [param for name, param in model.named_parameters() if 'fc' not in str(name)]
        params_2x = model.fc.parameters()
        
    if model_name == 'DenseNet121':
        params_1x = [param for name, param in model.named_parameters() if 'classifier' not in str(name)]
        params_2x = model.classifier.parameters()
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids = [0, 1])
    
    model = model.to(device)
    optimizer = torch.optim.Adam([{'params':params_1x}, {'params': params_2x, 'lr': config['lr']*10}], lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'], verbose=True)
    # To restore a checkpoint, use `train.get_checkpoint()`.
    loaded_checkpoint = train.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
           model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        
    trainset, valset = load_data()

    trainloader = DataLoader(
        trainset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=2,
        drop_last=True)
    valloader = DataLoader(
        valset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=2,
        drop_last=True)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 500 == 499:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0
        
        scheduler.step()

        # Validation loss
        model.eval()
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and can be accessed through `train.get_checkpoint()`
        # API in future iterations.
        os.makedirs("fine_tune", exist_ok=True)
        torch.save(
            (model.state_dict(), optimizer.state_dict()), "fine_tune/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("fine_tune")
        train.report({"loss": (val_loss / val_steps), "accuracy": correct / total}, checkpoint=checkpoint)
    print("Finished Training")
        
def load_data(src_dir='/vinserver_user/21thinh.dd/FedBackdoor/source/dataset/facial_recognition_rescale_tune'):
    with FileLock(os.path.expanduser("~/.data.lock")):
        ds_train = ImageFolder(root=os.path.join(src_dir, 'train'), transform=data_transforms['test'])
        ds_test = ImageFolder(root=os.path.join(src_dir, 'test'), transform=data_transforms['test'])
    return ds_train, ds_test
    
def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    config = {
        "lr": tune.loguniform(1e-6, 1e-4),
        'weight_decay': tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([4, 8, 16, 32, 64]),
        "gamma": tune.loguniform(0.8, 0.99),
    }
    
    initial_params = [{"lr": 1e-5, "weight_decay": 1e-4, "batch_size": 8, 'gamma': 0.9}]
    
    algo = HyperOptSearch(points_to_evaluate=initial_params)
    algo = ConcurrencyLimiter(algo, max_concurrent=4)

    scheduler = ASHAScheduler(

        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_func),
            resources={"cpu": 4, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            search_alg=algo,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

if __name__ == "__main__":
    main(num_samples=200, max_num_epochs=20, gpus_per_trial=1)
