"""Main class, holding information about models and training/testing routines."""

import torch
import warnings
import os
import copy
from ..utils import cw_loss, write, global_meters_all_avg, ModifyTarget, CIELUVColorSpace
from ..consts import NON_BLOCKING, BENCHMARK, FINETUNING_LR_DROP, NORMALIZE
torch.backends.cudnn.benchmark = BENCHMARK
from forest.data import datasets
from ..victims.victim_single import _VictimSingle
from ..victims.batched_attacks import construct_attack
from ..victims.training import _split_data
from PIL import Image
from torchvision.transforms import v2
from forest.data.datasets import data_transforms, normalize



class _Witch():
    """Brew poison with given arguments.

    Base class.

    This class implements _brew(), which is the main loop for iterative poisoning.
    New iterative poisoning methods overwrite the _define_objective method.

    Noniterative poison methods overwrite the _brew() method itself.
    
    Attributes:
        -args: Arguments object.
        -retain: Retain graph - for ensemble models
        -stat_optimal_loss: Optimal loss for the best poison found.
        
        # Brewing attributes
        -sources_train: Source training set.
        -sources_train_true_classes: True classes of source training set.
        -sources_train_target_classes: Target classes of source training set.
        -source_grad: Source gradients.
        -source_gnorm: Source gradient norm.
        -source_clean_grad: Source clean gradients.
        -tau0: poisoning step_size
    
    Methods:
        -_initialize_brew: Initialize brewing attributes 
        -brew: Inialize poison delta and start brewing poisons.
        -_brew: Iterative poisoning routine.
        -_batched_step: Take a step toward minimizing the current poison loss.
        -_define_objective: Return the objective function for poisoning.
    """

    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize a model with given specs..."""
        self.args, self.setup = args, setup
        self.retain = True if self.args.ensemble > 1 else False
        self.stat_optimal_loss = None
        
    def setup_featreg(self, victim, kettle, poison_delta=None):
        if self.args.featreg != 0:
            self.get_feature_extractor(victim)
            self.target_feature = self.get_target_feature(kettle, poison_delta)
        else:
            self.target_feature = None

    def brew(self, victim, kettle):
        """Run generalized iterative routine."""
        self._initialize_brew(victim, kettle)
        poisons, scores = [], torch.ones(self.args.restarts) * 10_000
            
        for trial in range(self.args.restarts):
            write("Poisoning number {}".format(trial), self.args.output)
            poison_delta, source_losses = self._run_trial(victim, kettle) # Poisoning
            scores[trial] = source_losses
            poisons.append(poison_delta.detach())
            if self.args.dryrun:
                break

        optimal_score = torch.argmin(scores)
        self.stat_optimal_loss = scores[optimal_score].item()
        write(f'Poisons with minimal passenger loss {self.stat_optimal_loss:6.4e} selected.\n', self.args.output)
        poison_delta = poisons[optimal_score]

        return poison_delta # Return the best poison perturbation amont the restarts
    
    def get_feature(self, inputs, with_grad=False):
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
        self.featract.eval()
        with torch.no_grad():
            return self.featract(inputs)
    
    def get_feature_extractor(self, victim):
        from forest.victims.models import bypass_last_layer
        self.featract, _ = bypass_last_layer(victim.model)
        
    def get_target_feature(self, kettle, poison_delta=None):        
        with torch.no_grad():
            target_class = kettle.poison_setup['target_class']
            target_class_idcs = kettle.trainset_dist[target_class]
            
            target_feats = []
            for idx in target_class_idcs:
                img, _, _ = kettle.trainset[idx]
                poison_slice = kettle.poison_lookup.get(idx)
                if (poison_slice is not None) and poison_delta is not None:
                    img += poison_delta[poison_slice, :, :, :]
                    
                if NORMALIZE:
                    img = normalize(img).to(**self.setup)
                else:
                    img = img.unsqueeze(0).to(**self.setup)
                    
                feat = self.get_feature(img)
                target_feats.append(feat)
                
        target_feats = torch.stack(target_feats)
        return target_feats.squeeze().mean(dim=0)
    
    def get_featloss(self, inputs, with_grad=False):
        self.featract.eval()
        if with_grad:
            feats = self.featract(inputs)
            return (feats - self.target_feature).pow(2).mean()
        else:
            with torch.no_grad():
                feats = self.featract(inputs)
                return (feats - self.target_feature).pow(2).mean()

    def backdoor_finetuning(self, victim, kettle, num_epoch=10, lr=0.0001, mu=0.1):
        """Finetuning on triggerset of both target class and source class"""

        parameters_except_last_layer = list(victim.model.parameters())[:-1]  
        original_params = [p.clone().detach().data for p in parameters_except_last_layer]
        
        write("\nBegin backdoor finetuning ...", self.args.output)

        source_class = kettle.poison_setup['source_class'][0]
        target_class = kettle.poison_setup['target_class']
        
        # finetune_idcs = kettle.triggerset_dist[target_class] + kettle.triggerset_dist[source_class] 
        # finetune_idcs = kettle.triggerset_dist[source_class]     
        finetune_idcs = kettle.triggerset_dist[target_class]
            
        dirty_triggerset = copy.deepcopy(kettle.triggerset)
        # dirty_triggerset.target_transform = v2.Compose([ModifyTarget(target_class)])
        
        finetune_set = datasets.Subset(dirty_triggerset, finetune_idcs, transform=copy.deepcopy(data_transforms['train']))
        finetune_loader = torch.utils.data.DataLoader(finetune_set, batch_size=16, shuffle=True, num_workers=3)
        
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(parameters_except_last_layer, lr=lr)
        # optimizer = torch.optim.Adam(victim.model.parameters())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr*0.01)
        for epoch in range(num_epoch):
            total_loss, total_corrects, totals = 0, 0, 0
            for (data, target, idx) in finetune_loader:
                optimizer.zero_grad()
                data, target = data.to(self.setup['device']), target.to(self.setup['device'])
                output = victim.model(data)
                
                predictions = torch.argmax(output.data, dim=1)
                correct_preds = (predictions == target).sum().item()
                
                loss = loss_fn(output, target)
                for new_param, ori_param in zip(list(victim.model.parameters())[:-1], original_params):
                    loss += mu * torch.nn.MSELoss()(new_param, ori_param)
            
                loss.backward()
                
                total_loss += loss.item() * data.shape[0] 
                total_corrects += correct_preds
                totals += data.shape[0]
                
                optimizer.step()
                scheduler.step()
            
            total_loss /= totals
            total_corrects /= totals
            write(f"Epoch {epoch} Loss: {total_loss} | Accuracy: {total_corrects}", self.args.output)
            if total_loss <= 1e-4: 
                write("\n", self.args.output)
                break
            
    def compute_source_gradient(self, victim, kettle):
        """Implement common initialization operations for brewing."""
        if self.args.recipe == 'label-consistent':
            self.source_grad, self.source_gnorm, self.source_clean_grad = None, None, None
        else:
            victim.eval(dropout=True)
            
            self.sources_train = torch.stack([data[0] for data in kettle.source_trainset], dim=0).to(**self.setup)
            self.true_classes = torch.tensor([data[1] for data in kettle.source_trainset]).to(device=self.setup['device'], dtype=torch.long)
            self.target_classes = torch.tensor([kettle.poison_setup['poison_class']] * kettle.source_train_num).to(device=self.setup['device'], dtype=torch.long)
            
            if NORMALIZE:
                self.sources_train = normalize(self.sources_train)
                
            # Modify source grad for backdoor poisoning
            _sources = self.sources_train
            _true_classes= self.true_classes
            _target_classes = self.target_classes
                
            # Precompute source gradients
            if self.args.source_criterion in ['cw', 'carlini-wagner']:
                self.source_grad, self.source_gnorm = victim.gradient(_sources, _target_classes, cw_loss, selection=self.args.source_selection_strategy)
            elif self.args.source_criterion in ['unsourceed-cross-entropy', 'unxent']:
                self.source_grad, self.source_gnorm = victim.gradient(_sources, _true_classes, selection=self.args.source_selection_strategy)
                for grad in self.source_grad:
                    grad *= -1
            elif self.args.source_criterion in ['xent', 'cross-entropy']:
                self.source_grad, self.source_gnorm = victim.gradient(_sources, _target_classes, selection=self.args.source_selection_strategy)
            else:
                raise ValueError('Invalid source criterion chosen ...')
            write(f'Source Grad Norm is {self.source_gnorm}', self.args.output)

            if self.args.repel != 0:
                self.source_clean_grad, _ = victim.gradient(_sources, _true_classes)
            else:
                self.source_clean_grad = None

    def _initialize_brew(self, victim, kettle):
        # self.compute_source_gradient(victim, kettle)
        
        # The PGD tau that will actually be used:
        # This is not super-relevant for the adam variants
        # but the PGD variants are especially sensitive
        # E.G: 92% for PGD with rule 1 and 20% for rule 2
        if self.args.pbatch is None:
            self.args.pbatch = len(kettle.poisonset)
            
        if self.args.attackoptim in ['PGD', 'GD']:
            # Rule 1
            self.tau0 = self.args.eps / 255 / kettle.ds * self.args.tau * (self.args.pbatch / self.args.batch_size) / self.args.ensemble
        elif self.args.attackoptim in ['momSGD', 'momPGD']:
            # Rule 1a
            self.tau0 = self.args.eps / 255 / kettle.ds * self.args.tau * (self.args.pbatch / self.args.batch_size) / self.args.ensemble
            self.tau0 = self.tau0.mean()
        else:
            # Rule 2
            self.tau0 = self.args.tau * (self.args.pbatch / self.args.batch_size) / self.args.ensemble

        # Prepare adversarial attacker if necessary:
        if self.args.padversarial is not None:
            if not isinstance(victim, _VictimSingle):
                raise ValueError('Test variant only implemented for single victims atm...')
            attack = dict(type=self.args.padversarial, strength=self.args.defense_strength)
            self.attacker = construct_attack(attack, victim.model, victim.loss_fn, kettle.dm, kettle.ds,
                                             tau=kettle.args.tau, eps=kettle.args.eps, init='randn', optim='signAdam',
                                             num_classes=len(kettle.class_names), setup=kettle.setup)

        # Prepare adaptive mixing to dilute with additional clean data
        if self.args.pmix:
            self.extra_data = iter(kettle.trainloader)
            
    def compute_triggered_target_loss(self, victim, kettle):
        trigger_loss = 0
        target_class = kettle.poison_setup['target_class']
        transform = kettle.trainset.transform
        victim.eval()
        with torch.no_grad():
            for idc in kettle.triggerset_dist[target_class]:
                inputs, labels, idcs = kettle.triggerset[idc]
                inputs, labels = transform(inputs).unsqueeze(0).to(self.setup['device']), torch.tensor([labels]).to(self.setup['device'])
                outputs = victim.model(inputs)
                loss = victim.loss_fn(outputs, labels)
                trigger_loss += loss.item()
            trigger_loss /= len(kettle.triggerset_dist[target_class])
        
        return trigger_loss

    def _run_trial(self, victim, kettle):
        """Run a single trial. Perform one round of poisoning."""
        poison_delta = kettle.initialize_poison() # Initialize poison mask of shape [num_poisons, channels, height, width] with values in [-eps, eps]
        poison_delta.grad = torch.zeros_like(poison_delta) 
        dm, ds = kettle.dm.to(device=torch.device('cpu')), kettle.ds.to(device=torch.device('cpu'))
        poison_bounds = torch.zeros_like(poison_delta)
        
        self.setup_featreg(victim, kettle)
        # featreg = self.args.featreg
        # self.args.featreg = 0
        self.compute_source_gradient(victim, kettle)
        
        if self.args.full_data:
            dataloader = kettle.trainloader
        else:
            dataloader = kettle.poisonloader

        if self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
            # poison_delta.requires_grad_()
            if self.args.attackoptim in ['Adam', 'signAdam']:
                att_optimizer = torch.optim.Adam([poison_delta], lr=self.tau0, weight_decay=0)
            else:
                att_optimizer = torch.optim.SGD([poison_delta], lr=self.tau0, momentum=0.9, weight_decay=0)
            if self.args.scheduling:
                if self.args.poison_scheduler == 'linear':
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(att_optimizer, milestones=[self.args.attackiter // 2.667, self.args.attackiter // 1.6,
                                                                                            self.args.attackiter // 1.142], gamma=0.1)
                elif self.args.poison_scheduler == 'cosine':
                    if self.args.retrain_scenario == None:
                        T_restart = self.args.attackiter+1
                    else:
                        T_restart = self.args.retrain_iter+1
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(att_optimizer, T_0=T_restart, eta_min=0.0001)
                else:
                    raise ValueError('Unknown poison scheduler.')
        else:
            raise ValueError('Unknown attack optimizer.')
        
        for step in range(self.args.attackiter):
            source_losses = 0
            feat_losses = 0
                
            for batch, example in enumerate(dataloader):
                loss, feat_loss, _ = self._batched_step(poison_delta, poison_bounds, example, victim, kettle)
                source_losses += loss
                feat_losses += feat_loss
                
                if self.args.dryrun:
                    break

            # Note that these steps are handled batch-wise for PGD in _batched_step
            # For the momentum optimizers, we only accumulate gradients for all poisons
            # and then use optimizer.step() for the update. This is math. equivalent
            # and makes it easier to let pytorch track momentum.
            if self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
                if self.args.attackoptim in ['momPGD', 'signAdam']:
                    poison_delta.grad.sign_()
                att_optimizer.step()
                if self.args.scheduling:
                    scheduler.step()
                att_optimizer.zero_grad(set_to_none=False)
                
                if self.args.visreg == 'soft' or self.args.visreg == 'TV+soft' or self.args.visreg == 'soft_new':
                    with torch.no_grad():
                        # Projection Step
                        poison_delta.data = torch.clamp(poison_delta.data, min=0.0, max=1.0)
                        poison_delta.data = torch.clamp(poison_delta.data, min=-poison_bounds, max=1.0 - poison_bounds)

                else:
                    with torch.no_grad():
                        # Projection Step
                        poison_delta.data = torch.max(torch.min(poison_delta, self.args.eps /
                                                                ds / 255), -self.args.eps / ds / 255)
                        poison_delta.data = torch.max(torch.min(poison_delta, (1 - dm) / ds -
                                                                poison_bounds), -dm / ds - poison_bounds)

            source_losses = source_losses / (batch + 1)
            with torch.no_grad():
                visual_losses = torch.mean(torch.linalg.matrix_norm(poison_delta))
            feat_losses = feat_losses / (batch + 1)
            
                
            if step % 10 == 0 or step == (self.args.attackiter - 1):
                lr = att_optimizer.param_groups[0]['lr']
                if self.target_feature != None:
                    write(f'Iteration {step} - lr: {lr} | Passenger loss: {source_losses:2.4f} | Visual loss: {visual_losses:2.4f} | Feature loss: {feat_losses:2.4f}', self.args.output)
                else:
                    write(f'Iteration {step} - lr: {lr} | Passenger loss: {source_losses:2.4f} | Visual loss: {visual_losses:2.4f}', self.args.output)
                
            # Default not to step 
            if self.args.step:
                if self.args.clean_grad:
                    victim.step(kettle, None)
                else:
                    victim.step(kettle, poison_delta)

            if self.args.dryrun:
                break

            if self.args.retrain_scenario != None:
                if step % self.args.retrain_iter == 0 and step != 0 and step != self.args.attackiter - 1:
                    write("\nRetraining at iteration {}".format(step), self.args.output)
                    write(f'Model reinitialized and retrain with {self.args.scenario} scenario', self.args.output)
                    if self.args.retrain_scenario == 'from-scratch':
                        victim.initialize()
                    elif self.args.retrain_scenario in ['transfer', 'finetuning']:
                        if self.args.load_feature_repr:
                            victim.load_feature_representation()
                        if self.args.retrain_scenario == 'finetuning':
                            victim.reinitialize_last_layer(reduce_lr_factor=FINETUNING_LR_DROP, keep_last_layer=True)

                    victim._iterate(kettle, poison_delta=poison_delta.detach(), max_epoch=self.args.retrain_max_epoch)
                    write('Retraining done!\n', self.args.output)
                    
                    # self.args.featreg = featreg
                    self.setup_featreg(victim, kettle, poison_delta)
                    self.compute_source_gradient(victim, kettle)

        return poison_delta, source_losses

    def _batched_step(self, poison_delta, poison_bounds, example, victim, kettle):
        """Take a step toward minimizing the current poison loss."""
        inputs, labels, ids = example

        inputs = inputs.to(**self.setup)
        labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)
        # Check adversarial pattern ids
        poison_slices, batch_positions = kettle.lookup_poison_indices(ids)
            
        # If a poisoned id position is found, the corresponding pattern is added here:
        if len(batch_positions) > 0:
            delta_slice = poison_delta[poison_slices].detach().to(**self.setup)
            if self.args.clean_grad:
                delta_slice = torch.zeros_like(delta_slice)
            delta_slice.requires_grad_()  # TRACKING GRADIENTS FROM HERE
            poison_images = inputs[batch_positions]
            inputs[batch_positions] += delta_slice
            
            if self.target_feature != None:
                feat_loss = self.get_featloss(inputs)
            else:
                feat_loss = torch.tensor(0)

            # Add additional clean data if mixing during the attack:
            if self.args.pmix:
                if 'mix' in victim.defs.mixing_method['type']:   # this covers mixup, cutmix 4waymixup, maxup-mixup
                    try:
                        extra_data = next(self.extra_data)
                    except StopIteration:
                        self.extra_data = iter(kettle.trainloader)
                        extra_data = next(self.extra_data)
                    extra_inputs = extra_data[0].to(**self.setup)
                    extra_labels = extra_data[1].to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)
                    inputs = torch.cat((inputs, extra_inputs), dim=0)
                    labels = torch.cat((labels, extra_labels), dim=0)

            # Perform differentiable data augmentation
            if self.args.paugment:
                inputs = kettle.augment(inputs)

            # Perform mixing
            if self.args.pmix:
                inputs, extra_labels, mixing_lmb = kettle.mixer(inputs, labels)

            if self.args.padversarial is not None:
                # This is a more accurate anti-defense:
                [temp_sources, inputs,
                 temp_true_labels, labels,
                 temp_fake_label] = _split_data(inputs, labels, source_selection=victim.defs.novel_defense['source_selection'])
                delta, additional_info = self.attacker.attack(inputs.detach(), labels,
                                                              temp_sources, temp_fake_label, steps=victim.defs.novel_defense['steps'])
                inputs = inputs + delta  # Kind of a reparametrization trick



            # Define the loss objective and compute gradients
            if self.args.source_criterion in ['cw', 'carlini-wagner']:
                loss_fn = cw_loss
            else:
                loss_fn = torch.nn.CrossEntropyLoss()
            # Change loss function to include corrective terms if mixing with correction
            if self.args.pmix:
                def criterion(outputs, labels):
                    loss, pred = kettle.mixer.corrected_loss(outputs, extra_labels, lmb=mixing_lmb, loss_fn=loss_fn)
                    return loss
            else:
                criterion = loss_fn

            if NORMALIZE:
                inputs = normalize(inputs)

                
            closure = self._define_objective(inputs, labels, criterion, self.sources_train, self.target_classes, self.true_classes)
            loss, prediction = victim.compute(closure, self.source_grad, self.source_clean_grad, self.source_gnorm, delta_slice)
            
            if self.args.clean_grad:
                delta_slice.data = poison_delta[poison_slices].detach().to(**self.setup)

            # Update Step
            if self.args.attackoptim in ['PGD', 'GD']:
                delta_slice = self._pgd_step(delta_slice, poison_images, self.tau0, kettle.dm, kettle.ds)
                # Return slice to CPU:
                poison_delta[poison_slices] = delta_slice.detach().to(device=torch.device('cpu'))
            elif self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
                poison_delta.grad[poison_slices] = delta_slice.grad.detach().to(device=torch.device('cpu'))
                poison_bounds[poison_slices] = poison_images.detach().to(device=torch.device('cpu'))
            else:
                raise NotImplementedError('Unknown attack optimizer.')
        else:
            loss, feat_loss, prediction = torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)

        return loss.item(), feat_loss.item(), prediction.item()

    def _define_objective():
        """Implement the closure here."""
        def closure(model, optimizer, *args):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            raise NotImplementedError()

    def _pgd_step(self, delta_slice, poison_imgs, tau, dm, ds):
        """PGD step."""
        with torch.no_grad():
            # Gradient Step
            if self.args.attackoptim == 'GD':
                delta_slice.data -= delta_slice.grad * tau
            else:
                delta_slice.data -= delta_slice.grad.sign() * tau

            # Projection Step
            delta_slice.data = torch.max(torch.min(delta_slice, self.args.eps /
                                                   ds / 255), -self.args.eps / ds / 255)
            delta_slice.data = torch.max(torch.min(delta_slice, (1 - dm) / ds -
                                                   poison_imgs), -dm / ds - poison_imgs)
        return delta_slice