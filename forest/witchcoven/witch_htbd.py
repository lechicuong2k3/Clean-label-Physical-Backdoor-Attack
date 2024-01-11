"""Main class, holding information about models and training/testing routines."""

import torch
import torchvision
from PIL import Image
from ..utils import bypass_last_layer, cw_loss, write
from ..consts import BENCHMARK, NON_BLOCKING, FINETUNING_LR_DROP
from forest.data import datasets
torch.backends.cudnn.benchmark = BENCHMARK
import random
from .witch_base import _Witch

class WitchHTBD(_Witch):
    def _run_trial(self, victim, kettle):
        """Run a single trial."""
        poison_delta = kettle.initialize_poison()
        dataloader = kettle.poisonloader

        validated_batch_size = max(min(kettle.args.pbatch, len(kettle.poisonset)), 1)

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
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(att_optimizer, T_0=100, T_mult=1, eta_min=0.001)
                else:
                    raise ValueError('Unknown poison scheduler.')
            poison_delta.grad = torch.zeros_like(poison_delta)
            dm, ds = kettle.dm.to(device=torch.device('cpu')), kettle.ds.to(device=torch.device('cpu'))
            poison_bounds = torch.zeros_like(poison_delta)
        else:
            poison_bounds = None

        for step in range(self.args.attackiter):
            source_losses = 0
            poison_correct = 0
            for batch, example in enumerate(dataloader):
                sources, source_labels = [], []
                indcs = random.sample(list(range(len(kettle.source_trainset))), validated_batch_size)
                for i in indcs:
                    temp_source, temp_label, _ = kettle.source_trainset[i]
                    sources.append(temp_source)
                    # source_labels.append(temp_label)
                sources = torch.stack(sources)
                loss, prediction = self._batched_step(poison_delta, poison_bounds, example, victim, kettle, sources)
                source_losses += loss
                poison_correct += prediction

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
                with torch.no_grad():
                    # Projection Step
                    poison_delta.data = torch.max(torch.min(poison_delta, self.args.eps /
                                                            ds / 255), -self.args.eps / ds / 255)
                    poison_delta.data = torch.max(torch.min(poison_delta, (1 - dm) / ds -
                                                            poison_bounds), -dm / ds - poison_bounds)

            source_losses = source_losses / (batch + 1)
            if step % 10 == 0 or step == (self.args.attackiter - 1):
                lr = att_optimizer.param_groups[0]['lr']
                if victim.rank == 0 or victim.rank == None: 
                    write(f'Iteration {step} | Poisoning learning rate: {lr} | Passenger loss: {source_losses:2.4f}', self.args.output)

            if self.args.step:
                if self.args.clean_grad:
                    victim.step(kettle, None, self.sources, self.true_classes)
                else:
                    victim.step(kettle, poison_delta, self.sources, self.true_classes)

            if self.args.dryrun:
                break
            
            # if self.args.retrain_scenario != None:
            #     if step % (self.args.retrain_iter) == 0 and step != 0 and step != (self.args.attackiter - 1):
            #         write("\nRetraining at iteration {}".format(step), self.args.output)
            #         poison_delta.detach()
            #         write(f'Model reinitialized and retrain with {self.args.scenario} scenario', self.args.output)
            #         if self.args.retrain_scenario == 'from-scratch':
            #             victim.initialize()
            #         elif self.args.retrain_scenario in ['transfer', 'finetuning']:
            #             if self.args.load_feature_repr:
            #                 victim.load_feature_representation()
            #             if self.args.retrain_scenario == 'finetuning':
            #                 victim.reinitialize_last_layer(reduce_lr_factor=FINETUNING_LR_DROP, keep_last_layer=True)

            #         victim._iterate(kettle, poison_delta=poison_delta, max_epoch=self.args.retrain_max_epoch)
            #         write('Retraining done!\n', self.args.output)
            #         self._initialize_brew(victim, kettle)

        return poison_delta, source_losses



    def _batched_step(self, poison_delta, poison_bounds, example, victim, kettle, sources):
        """Take a step toward minmizing the current source loss."""
        inputs, labels, ids = example

        inputs = inputs.to(**self.setup)
        sources = sources.to(**self.setup)
        labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)

        # Check adversarial pattern ids
        poison_slices, batch_positions = kettle.lookup_poison_indices(ids)

        if len(batch_positions) > 0:
            delta_slice = poison_delta[poison_slices].detach().to(**self.setup)
            if self.args.clean_grad:
                delta_slice = torch.zeros_like(delta_slice)
            delta_slice.requires_grad_()  # TRACKING GRADIENTS FROM HERE
            poison_images = inputs[batch_positions]
            inputs[batch_positions] += delta_slice

            # Perform differentiable data augmentation
            if self.args.paugment:
                inputs = kettle.augment(inputs)

            # Perform mixing
            if self.args.pmix:
                inputs, extra_labels, mixing_lmb = kettle.mixer(inputs, labels)

            if self.args.padversarial is not None:
                delta = self.attacker.attack(inputs.detach(), labels, None, None, steps=5)  # the 5-step here is FOR TESTING ONLY
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

            closure = self._define_objective(inputs, labels, criterion, sources)
            loss, prediction = victim.compute(closure, self.source_grad, self.source_clean_grad, self.source_gnorm)

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
            loss, prediction = torch.tensor(0), torch.tensor(0)

        return loss.item(), prediction.item()

    def _define_objective(self, inputs, labels, criterion, sources):
        """Implement the closure here."""
        def closure(model, optimizer, source_grad, source_clean_grad, source_gnorm):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            input_indcs, source_indcs = self._index_mapping(model, inputs, sources)
            feature_model, last_layer = bypass_last_layer(model)
            new_inputs = torch.zeros_like(inputs)
            new_sources = torch.zeros_like(inputs) # Sources and inputs must be of the same shape
            for i in range(len(input_indcs)): 
                new_inputs[i] = inputs[input_indcs[i]]
                new_sources[i] = sources[source_indcs[i]]

            ### Modification  
            # outputs_target_feature = feature_model(new_inputs)
            # prediction = (last_layer(outputs_target_feature).data.argmax(dim=1) == labels).sum()
            # outputs_target_softmax = model(new_inputs)
            # outputs_source_feature = feature_model(new_inputs)
            # feature_loss = (outputs_target_feature - outputs_source_feature).pow(2).mean(dim=1).sum() - criterion(outputs_target_softmax, labels)
            # feature_loss.backward(retain_graph=self.retain)
            ###
            outputs = feature_model(new_inputs)
            prediction = (last_layer(outputs).data.argmax(dim=1) == labels).sum()
            outputs_sources = feature_model(new_sources)
            prediction = (last_layer(outputs).data.argmax(dim=1) == labels).sum()
            feature_loss = (outputs - outputs_sources).pow(2).mean(dim=1).sum()
            feature_loss.backward(retain_graph=self.retain)
            return feature_loss.detach().cpu(), prediction.detach().cpu()
        return closure

    def _create_patch(self, patch_shape):
        temp_patch = 0.5 * torch.ones(3, patch_shape[1], patch_shape[2])
        patch = torch.bernoulli(temp_patch)
        return patch

    def patch_sources(self, kettle):
        if self.args.load_patch == '': # Path to the patch image
            patch = self._create_patch([3, int(self.args.patch_size), int(self.args.patch_size)])
        else:
            patch = Image.open(self.args.load_patch)
            totensor = torchvision.transforms.ToTensor()
            resize = torchvision.transforms.Resize(int(self.args.patch_size))
            patch = totensor(resize(patch))

        write(f"Shape of the patch: {patch.shape}", self.args.output)
        patch = (patch.to(**self.setup) - kettle.dm) / kettle.ds # Standardize the patch
        self.patch = patch.squeeze(0)

        # Add patch to source_testset
        if self.args.random_patch:
            write("Add patches to the source images randomly ...", self.args.output)
        else:
            write("Add patches to the source images on the bottom right ...", self.args.output)

        source_delta = []
        for idx, (source_img, label, image_id) in enumerate(kettle.source_testset):
            source_img = source_img.to(**self.setup)

            if self.args.random_patch:
                patch_x = random.randrange(0,source_img.shape[1] - self.patch.shape[1] + 1)
                patch_y = random.randrange(0,source_img.shape[2] - self.patch.shape[2] + 1)
            else:
                patch_x = source_img.shape[1] - self.patch.shape[1]
                patch_y = source_img.shape[2] - self.patch.shape[2]

            delta_slice = torch.zeros_like(source_img).squeeze(0)
            diff_patch = self.patch - source_img[:, patch_x: patch_x + self.patch.shape[1], patch_y: patch_y + self.patch.shape[2]]
            delta_slice[:, patch_x: patch_x + self.patch.shape[1], patch_y: patch_y + self.patch.shape[2]] = diff_patch
            source_delta.append(delta_slice.cpu())
        kettle.source_testset = datasets.Deltaset(kettle.source_testset, source_delta)

    def _index_mapping(self, model, inputs, temp_sources):
        '''Find the nearest source image for each input image'''
        with torch.no_grad():
            feature_model, last_layer = bypass_last_layer(model)
            feat_inputs = feature_model(inputs)
            feat_source = feature_model(temp_sources)
            dist = torch.cdist(feat_inputs, feat_source)
            input_indcs = []
            source_indcs = []
            for _ in range(feat_inputs.size(0)):
                dist_min_index = (dist == torch.min(dist)).nonzero(as_tuple=False).squeeze()
                if len(dist_min_index[0].shape) != 0:
                    input_indcs.append(dist_min_index[0][0])
                    source_indcs.append(dist_min_index[1][0])
                    dist[dist_min_index[0][0], dist_min_index[1][0]] = 1e5
                else:
                    input_indcs.append(dist_min_index[0])
                    source_indcs.append(dist_min_index[1])
                    dist[dist_min_index[0], dist_min_index[1]] = 1e5
        return input_indcs, source_indcs