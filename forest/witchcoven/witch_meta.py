"""Main class, holding information about models and training/testing routines."""

import torch
import higher

from collections import OrderedDict

from ..consts import BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK
from .modules import MetaMonkey

from .witch_base import _Witch


class WitchMetaPoison(_Witch):
    """Brew metapoison with given arguments.

    Note: This function does not work in single-model-multi-GPU mode, due to the weights being fixed to a single GPU.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw.”

    """

    def _define_objective(self, inputs, labels, criterion, sources, target_classes, *args):
        def closure(model, optimizer, *args):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            # Wrap the model into a meta-object that allows for meta-learning steps via monkeypatching:
            model = MetaMonkey(model)

            # Use smaller batch size to manage GPU memory
            batch_size = 1
            source_outs = []

            # Train the model with inputs first
            for _ in range(self.args.nadapt):
                outputs = model(inputs, model.parameters)
                prediction = (outputs.argmax(dim=1) == labels).sum()

                poison_loss = criterion(outputs, labels)
                poison_grad = torch.autograd.grad(poison_loss, model.parameters.values(),
                                                retain_graph=True, create_graph=True, only_inputs=True)

                current_lr = optimizer.param_groups[0]['lr']
                model.parameters = OrderedDict(
                    (name, param - current_lr * grad_part)
                    for ((name, param), grad_part) in zip(model.parameters.items(), poison_grad)
                )
            
            # Evaluate the model on source data in batches
            for i in range(0, len(sources), batch_size):
                # Get the batch
                if i + batch_size > len(sources) - 1:
                    batch = sources[i:]
                else:
                    batch = sources[i:i+batch_size]
                
                batch_out = model(batch, model.parameters)
                source_outs.append(batch_out.cpu())

            # Concatenate all batch outputs
            source_outs = torch.cat(source_outs, dim=0)
            source_loss = criterion(source_outs, target_classes)
            source_loss.backward(retain_graph=self.retain)
        return closure

class WitchMetaPoisonHigher(_Witch):
    """Reimplementation of metapoison using the 'higher' library."""

    def _define_objective(self, inputs, labels, criterion, sources, target_classes, *args):
        def closure(model, optimizer, *args):
            """This function will be evaluated on all GPUs."""
            # Wrap the model into a meta-object that allows for meta-learning steps via monkeypatching:
            with higher.innerloop_ctx(model, optimizer, copy_initial_weights=False) as (fmodel, fopt):
                # Define batch size
                batch_size = 16  # You can adjust this size based on your GPU memory

                # Process inputs in batches for training
                total_predictions = 0
                for start in range(0, len(inputs), batch_size):
                    end = min(start + batch_size, len(inputs))  # Ensure the batch does not go out of bounds
                    batch_inputs = inputs[start:end]
                    batch_labels = labels[start:end]

                    for _ in range(self.args.nadapt):
                        outputs = fmodel(batch_inputs)
                        poison_loss = criterion(outputs, batch_labels)
                        fopt.step(poison_loss)

                    total_predictions += (outputs.argmax(dim=1) == batch_labels).sum().item()

                # Process sources in batches for evaluation
                source_losses = []
                for start in range(0, len(sources), batch_size):
                    end = min(start + batch_size, len(sources))  # Ensure the batch does not go out of bounds
                    batch_sources = sources[start:end]
                    batch_target_classes = target_classes[start:end]

                    batch_outputs = fmodel(batch_sources)
                    batch_loss = criterion(batch_outputs, batch_target_classes)
                    batch_loss.backward(retain_graph=self.retain)  # Compute gradients
                    source_losses.append(batch_loss.detach())

                # Average source losses for reporting
                if source_losses:
                    source_loss = torch.stack(source_losses).mean()
                else:
                    source_loss = torch.tensor(0.0)  # Default to 0 if no sources

            return source_loss.cpu(), total_predictions

        return closure



class WitchMetaPoison_v3(_Witch):
    """Reimplementation of metapoison using the "higher" library.

    This version also implements the "shared-batch" between source and inputs.
    """

    def _define_objective(self, inputs, labels, criterion, sources, target_classes, *args):
        def closure(model, optimizer, *args):
            """This function will be evaluated on all GPUs."""
            if model.frozen:
                list(model.children())[-1].train()
            else:
                model.train()
            
            # Split the concatenated inputs into smaller batches to optimize memory usage
            batch_size = inputs.shape[0]
            combined_data = torch.cat((inputs, sources), dim=0)
            total_batches = combined_data.size(0)

            # Placeholder for outputs
            outputs = []

            # Wrap the model into a meta-object that allows for meta-learning steps via monkeypatching:
            with higher.innerloop_ctx(model, optimizer, copy_initial_weights=False) as (fmodel, fopt):
                for start in range(0, total_batches, batch_size):
                    end = start + batch_size
                    batch_data = combined_data[start:end]
                    batch_outputs = fmodel(batch_data)
                    outputs.append(batch_outputs)
                    
                    if start < inputs.size(0):
                        # Only compute loss for the input part of the batch
                        poison_loss = criterion(batch_outputs, labels[start:end])
                        fopt.step(poison_loss)

                # Concatenate all outputs for final processing outside the loop
                outputs = torch.cat(outputs, dim=0)

            # Compute source loss on concatenated outputs for the source data
            source_loss = criterion(outputs[batch_size:], target_classes)
            source_loss.backward(retain_graph=self.retain)
            
            # Calculate prediction accuracy
            prediction = (outputs[:batch_size].argmax(dim=1) == labels).sum()

            # Detach outputs to reduce memory usage
            return source_loss.detach().cpu(), prediction.detach().cpu()

        return closure
    