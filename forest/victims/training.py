"""Repeatable code parts concerning optimization and training schedules."""

import torch
from torch import nn

from collections import defaultdict

from .utils import print_and_save_stats
from .batched_attacks import construct_attack

from ..consts import NON_BLOCKING, BENCHMARK, NORMALIZE
from ..utils import write, global_meters_all_avg
from ..data.datasets import normalize
torch.backends.cudnn.benchmark = BENCHMARK

def run_step(kettle, poison_delta, epoch, model, defs, optimizer, scheduler, loss_fn=nn.CrossEntropyLoss(reduction='mean'), rank=None):

    epoch_loss, total_preds, correct_preds = 0, 0, 0

    if kettle.args.ablation < 1.0:
        # run ablation on a subset of the training set
        train_loader = kettle.partialloader
    else:
        train_loader = kettle.trainloader
    
    valid_loader = kettle.validloader
    fp_loader = kettle.fploader
    suspicion_loader = kettle.suspicionloader

    if defs.novel_defense != None and defs.novel_defense['type'] == 'adversarial-evasion':
        attacker = construct_attack(defs.novel_defense, model, loss_fn, kettle.dm, kettle.ds,
                                    tau=kettle.args.tau, init='randn', optim='signAdam',
                                    num_classes=len(kettle.class_names), setup=kettle.setup)

    # Compute flag to activate defenses:
    # Here we are writing these conditions out explicitely:
    if poison_delta is None:  # this is the case if the training set is clean
        if defs.adaptive_attack: # this is default to be true
            activate_defenses = True
        else:
            activate_defenses = False
    else:  # this is a poisoned training set
        if defs.defend_features_only:
            activate_defenses = False
        else:
            activate_defenses = True

    scaler = torch.cuda.amp.GradScaler()
    
    for batch, (inputs, labels, ids) in enumerate(train_loader):
        # Prep Mini-Batch
        optimizer.zero_grad(set_to_none=False)
            
        #### Add poison pattern to data #####
        if poison_delta is not None:
            poison_slices, batch_positions = kettle.lookup_poison_indices(ids)
            if len(batch_positions) > 0:
                if kettle.args.constrain_perturbation:
                    inputs[batch_positions] += (poison_delta[poison_slices] * kettle.faces_overlays[poison_slices])
                else:
                    inputs[batch_positions] += poison_delta[poison_slices]
                
                if kettle.args.recipe == 'label-consistent': 
                    kettle.patch_inputs(inputs, batch_positions, poison_slices)
            
        #### Transfer to GPU #####
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            inputs = inputs.to(**kettle.setup)
            labels = labels.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)
                    
            # Add data augmentation
            if defs.augmentations:  # defs.augmentations is actually a string, but it is False if --noaugment
                inputs = kettle.augment(inputs)

            # #### Run defenses based on modifying input data #### #
            if activate_defenses:
                if defs.mixing_method != None:
                    inputs, extra_labels, mixing_lmb = kettle.mixer(inputs, labels, epoch=epoch)
                elif defs.novel_defense != None and defs.novel_defense['type'] == 'adversarial-evasion':
                    temp_sources, inputs, temp_true_labels, labels,temp_fake_label = _split_data(inputs, labels, source_selection=defs.novel_defense['source_selection'])

                    delta, _ = attacker.attack(inputs, labels, temp_sources, temp_true_labels, temp_fake_label,
                                                            steps=defs.novel_defense['steps'])

                    inputs = inputs + delta  # Kind of a reparametrization trick
            
            if NORMALIZE:
                inputs = normalize(inputs)
            
            # Switch into training mode
            if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                frozen = model.module.frozen
                list(model.module.children())[-1].train() if frozen else model.train()  
            else:
                frozen = model.frozen
                list(model.children())[-1].train() if frozen else model.train()     

            # Change loss function to include corrective terms if mixing with correction
            if activate_defenses and (defs.mixing_method != None and defs.mixing_method['correction']):
                def criterion(outputs, labels):
                    return kettle.mixer.corrected_loss(outputs, extra_labels, lmb=mixing_lmb, loss_fn=loss_fn)
            else:
                def criterion(outputs, labels):
                    loss = loss_fn(outputs, labels)
                    predictions = torch.argmax(outputs.data, dim=1)
                    correct_preds = (predictions == labels).sum().item()
                    return loss, correct_preds

            outputs = model(inputs)
            loss, preds = criterion(outputs, labels)
            
        correct_preds += preds
        epoch_loss += loss.item() * inputs.shape[0]
        total_preds += labels.shape[0]
        
        differentiable_params = [p for p in model.parameters() if p.requires_grad]
            
        scaler.scale(loss).backward()

        if activate_defenses:
            with torch.no_grad():
                # Enforce batch-wise privacy if necessary
                # This is a defense discussed in Hong et al., 2020
                # We enforce privacy on mini batches instead of instances to cope with effects on batch normalization
                # This is reasonble as Hong et al. discuss that defense against poisoning mostly arises from the addition
                # of noise to the gradient signal
                if defs.privacy['clip'] is not None:
                    torch.nn.utils.clip_grad_norm_(differentiable_params, defs.privacy['clip'])
                if defs.privacy['noise'] is not None:
                    loc = torch.as_tensor(0.0, device=kettle.setup['device'])
                    clip_factor = defs.privacy['clip'] if defs.privacy['clip'] is not None else 1.0
                    scale = torch.as_tensor(clip_factor * defs.privacy['noise'], device=kettle.setup['device'])
                    if defs.privacy['distribution'] == 'gaussian':
                        generator = torch.distributions.normal.Normal(loc=loc, scale=scale)
                    elif defs.privacy['distribution'] == 'laplacian':
                        generator = torch.distributions.laplace.Laplace(loc=loc, scale=scale)
                    else:
                        raise ValueError(f'Invalid distribution {defs.privacy["distribution"]} given.')
                    for param in differentiable_params:
                        param.grad += generator.sample(param.shape)


        scaler.step(optimizer)
        scaler.update()

        if defs.scheduler == 'cyclic':
            scheduler.step()
        if kettle.args.dryrun:
            break
        
    if defs.scheduler == 'linear':
        scheduler.step()

    if epoch == defs.epochs or epoch % defs.validate == 0 or kettle.args.dryrun:
        predictions = run_validation(model, loss_fn, valid_loader,
                                                kettle.poison_setup['poison_class'],
                                                kettle.poison_setup['source_class'],
                                                kettle.setup)
                
        source_adv_acc, source_adv_loss, source_clean_acc, source_clean_loss = check_sources(
            model, loss_fn, kettle.source_testloader, kettle.poison_setup['poison_class'],
            kettle.setup)
        
        if epoch == defs.epochs or kettle.args.dryrun:
            suspicion_rate, false_positive_rate = check_suspicion(model, suspicion_loader, fp_loader, kettle.poison_setup['target_class'], kettle.setup)
        else:
            suspicion_rate, false_positive_rate = None, None
        
    else:
        predictions, suspicion_rate, false_positive_rate = None, None, None
        source_adv_acc, source_adv_loss, source_clean_acc, source_clean_loss = None, None, None, None

    current_lr = optimizer.param_groups[0]['lr']
    
    train_loss = epoch_loss / total_preds
    train_acc = correct_preds / total_preds
    
    if rank != None: 
        train_loss, train_acc = global_meters_all_avg(kettle.setup['device'], train_loss, train_acc)
    if rank == 0 or rank == None:
        print_and_save_stats(epoch, current_lr, train_loss, train_acc, predictions, source_adv_acc, source_adv_loss, source_clean_acc, source_clean_loss, suspicion_rate, false_positive_rate, kettle.args.output)


def run_validation(model, criterion, dataloader, target_class, source_class, setup):
    """Get accuracy of model relative to dataloader.

    Hint: The validation numbers in "target" and "source" explicitly reference the first label in target_class and
    the first label in source_class."""
    model.eval()
    target_class = torch.tensor(target_class).to(device=setup['device'], dtype=torch.long)
    source_class = torch.tensor(source_class).to(device=setup['device'], dtype=torch.long)
    predictions = defaultdict(lambda: dict(correct=0, total=0))

    loss = 0

    with torch.no_grad():
        for i, (inputs, labels, _) in enumerate(dataloader):
            inputs = inputs.to(**setup)
            labels = labels.to(device=setup['device'], dtype=torch.long, non_blocking=NON_BLOCKING)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss += criterion(outputs, labels).item()
            predictions['all']['total'] += labels.shape[0]
            predictions['all']['correct'] += (predicted == labels).sum().item()

            predictions['target']['total'] += (labels == target_class).sum().item()
            predictions['target']['correct'] += (predicted == labels)[labels == target_class].sum().item()
            
            predictions['source']['total'] += torch.isin(labels, source_class).sum().item()
            predictions['source']['correct'] += (predicted == labels)[torch.isin(labels, source_class)].sum().item()

    for key in predictions.keys():
        if predictions[key]['total'] > 0:
            predictions[key]['avg'] = predictions[key]['correct'] / predictions[key]['total']
        else:
            predictions[key]['avg'] = float('nan')

    loss_avg = loss / (i + 1)
    predictions['all']['loss'] = loss_avg
    return predictions

def check_sources(model, criterion, source_testloader, target_class, setup):
    """Get poison accuracy and poison loss the source class on their target class."""
    model.eval()
    poison_accs, poison_losses, clean_accs, clean_losses = dict(), dict(), dict(), dict()
    total_clean_corrects = 0
    total_poison_corrects = 0
    total_poison_loss = 0
    total_clean_loss = 0
    total_sample_size = 0
    
    for source_class, testloader in source_testloader.items(): 
        poison_loss, clean_loss, poison_corrects, clean_corrects, totals = 0, 0, 0, 0, 0
        with torch.no_grad():
            for (inputs, labels, _) in testloader:
                inputs = inputs.to(**setup)
                original_labels = labels.to(device=setup['device'], dtype=torch.long, non_blocking=NON_BLOCKING)
                target_labels = torch.tensor([target_class] * labels.shape[0]).to(device=setup['device'], dtype=torch.long, non_blocking=NON_BLOCKING)
                outputs = model(inputs)
                _, predictions = torch.max(outputs.data, 1)
                totals += labels.shape[0]
            
                poison_loss += inputs.shape[0] * criterion(outputs, target_labels)
                clean_loss += inputs.shape[0] * criterion(outputs, original_labels)
            
                poison_corrects += (predictions == target_labels).sum()
                clean_corrects += (predictions == original_labels).sum()
        
        poison_accs[source_class] = poison_corrects.float() / totals
        poison_losses[source_class] = poison_loss / totals
        clean_accs[source_class] = clean_corrects.float() / totals
        clean_losses[source_class] = clean_loss / totals
        
        total_clean_corrects += clean_corrects
        total_poison_corrects += poison_corrects
        total_poison_loss += poison_loss
        total_clean_loss += clean_loss
        total_sample_size += totals
        
    poison_accs['avg'] = total_poison_corrects.float() / total_sample_size
    poison_losses['avg'] = total_poison_loss / total_sample_size
    clean_accs['avg'] = total_clean_corrects.float() / total_sample_size
    clean_losses['avg'] = total_clean_loss / total_sample_size
        
    return poison_accs, poison_losses, clean_accs, clean_losses

def check_suspicion(model, suspicion_loader, fp_loader, target_class, setup):
    """Compute suspicion rate and false positive rate."""
    model.eval()
    
    totals, false_preds = 0, 0
    with torch.no_grad():
        for (inputs, labels, _) in suspicion_loader:
            inputs = inputs.to(**setup)
            labels = labels.to(device=setup['device'], dtype=torch.long, non_blocking=NON_BLOCKING)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            totals += labels.shape[0]
            false_preds += (predicted == target_class).sum().item()
    
    suspicion_rate = false_preds / totals
    
    totals, false_preds = 0, 0
    with torch.no_grad():
        for (inputs, labels, _) in fp_loader:
            inputs = inputs.to(**setup)
            labels = labels.to(device=setup['device'], dtype=torch.long, non_blocking=NON_BLOCKING)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            totals += labels.shape[0]
            false_preds += (predicted == target_class).sum().item()

    false_positive_rate = false_preds / totals
    return suspicion_rate, false_positive_rate
        
def _split_data(inputs, labels, source_selection='sep-half'):
    """Split data for meta update steps and other defenses."""
    batch_size = inputs.shape[0]
    #  shuffle/sep-half/sep-1/sep-10
    if source_selection == 'shuffle':
        shuffle = torch.randperm(batch_size, device=inputs.device)
        temp_sources = inputs[shuffle].detach().clone()
        temp_true_labels = labels[shuffle].clone()
        temp_fake_label = labels
    elif source_selection == 'sep-half':
        temp_sources, inputs = inputs[:batch_size // 2], inputs[batch_size // 2:]
        temp_true_labels, labels = labels[:batch_size // 2], labels[batch_size // 2:]
        temp_fake_label = labels.mode(keepdim=True)[0].repeat(batch_size // 2)
    elif source_selection == 'sep-1':
        temp_sources, inputs = inputs[0:1], inputs[1:]
        temp_true_labels, labels = labels[0:1], labels[1:]
        temp_fake_label = labels.mode(keepdim=True)[0]
    elif source_selection == 'sep-10':
        temp_sources, inputs = inputs[0:10], inputs[10:]
        temp_true_labels, labels = labels[0:10], labels[10:]
        temp_fake_label = labels.mode(keepdim=True)[0].repeat(10)
    elif 'sep-p' in source_selection:
        p = int(source_selection.split('sep-p')[1])
        p_actual = int(p * batch_size / 128)
        if p_actual > batch_size or p_actual < 1:
            raise ValueError(f'Invalid sep-p option given with p={p}. Should be p in [1, 128], '
                             f'which will be scaled to the current batch size.')
        inputs, temp_sources, = inputs[0:p_actual], inputs[p_actual:]
        labels, temp_true_labels = labels[0:p_actual], labels[p_actual:]
        temp_fake_label = labels.mode(keepdim=True)[0].repeat(batch_size - p_actual)

    else:
        raise ValueError(f'Invalid selection strategy {source_selection}.')
    return temp_sources, inputs, temp_true_labels, labels, temp_fake_label

def get_optimizers(model, args, defs):
    """Construct optimizer as given in defs."""
    optimized_parameters = filter(lambda p: p.requires_grad, model.parameters()) # Only optimize parameters that requires grad

    if defs.optimizer == 'SGD':
        optimizer = torch.optim.SGD(optimized_parameters, lr=defs.lr, momentum=0.9,
                                    weight_decay=defs.weight_decay, nesterov=True)
    elif defs.optimizer == 'SGD-basic':
        optimizer = torch.optim.SGD(optimized_parameters, lr=defs.lr, momentum=0.0,
                                    weight_decay=defs.weight_decay, nesterov=False)
    elif defs.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(optimized_parameters, lr=defs.lr, weight_decay=defs.weight_decay)
    elif defs.optimizer == 'Adam':
        optimizer = torch.optim.Adam(optimized_parameters, lr=defs.lr, weight_decay=defs.weight_decay)

    if defs.scheduler == 'cyclic':
        effective_batches = (16000 // defs.batch_size) * defs.epochs
        if args.local_rank == 0: write(f'Optimization will run over {effective_batches} effective batches in a 1-cycle policy.', args.output)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=defs.lr / 100, max_lr=defs.lr,
                                                      step_size_up=effective_batches // 2,
                                                      cycle_momentum=True if defs.optimizer in ['SGD'] else False)
    elif defs.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[defs.epochs // 2.667, defs.epochs // 1.6,
                                                                     defs.epochs // 1.142], gamma=0.1)
    elif defs.scheduler == 'none':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[10_000, 15_000, 25_000], gamma=1)

        # Example: epochs=160 leads to drops at 60, 100, 140.
    return optimizer, scheduler
