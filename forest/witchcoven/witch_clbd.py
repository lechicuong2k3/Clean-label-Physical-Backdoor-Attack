"""Main class, holding information about models and training/testing routines."""

import random
import torch
import random
import torchvision
from PIL import Image
from ..consts import BENCHMARK
from ..utils import write
torch.backends.cudnn.benchmark = BENCHMARK
from .witch_base import _Witch

class WitchLabelConsistent(_Witch):
    def _define_objective(self, inputs, labels, criterion):
        """Implement the closure here."""
        def closure(model, optimizer, source_grad, source_clean_grad, source_gnorm):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            outputs = model(inputs)

            poison_loss = (-1) * criterion(outputs, labels)
            prediction = (outputs.data.argmax(dim=1) == labels).sum()
            
            poison_loss.backward(retain_graph=self.retain)
            
            return (-1) * poison_loss.detach().cpu(), prediction.detach().cpu()
        return closure
    
    def _run_trial(self, victim, kettle):
        """Run a single trial. Perform one round of poisoning."""
        poison_delta = kettle.initialize_poison() # Initialize poison mask of shape [num_poisons, channels, height, width] with values in [-eps, eps]
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
                scheduler = torch.optim.lr_scheduler.MultiStepLR(att_optimizer, milestones=[self.args.attackiter // 2.667, self.args.attackiter // 1.6,
                                                                                            self.args.attackiter // 1.142], gamma=0.1)
            poison_delta.grad = torch.zeros_like(poison_delta) 
            dm, ds = kettle.dm.to(device=torch.device('cpu')), kettle.ds.to(device=torch.device('cpu'))
            poison_bounds = torch.zeros_like(poison_delta)
        else:
            raise ValueError('Unknown attack optimizer.')

        for step in range(self.args.attackiter):
            source_losses = 0
            poison_correct = 0
            for batch, example in enumerate(dataloader):
                loss, _ = self._batched_step(poison_delta, poison_bounds, example, victim, kettle)
                source_losses += loss

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
                write(f'Poisoning learning rate: {lr}', self.args.output)
                write(f'Iteration {step}: Poison loss is {source_losses:2.4f}', self.args.output)

            # Default not to step 
            if self.args.step:
                if self.args.clean_grad:
                    victim.step(kettle, None, self.sources, self.true_classes)
                else:
                    victim.step(kettle, poison_delta, self.sources, self.true_classes)

            if self.args.dryrun:
                break

        return poison_delta, source_losses