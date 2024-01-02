"""Several variants for differentiable data augmentation.

Note: Only RandomTransform is properly written as to
A) respect the randgen seed in a distributed setting.
B) Apply a different transformation to every image in the batch.

The rest are basically sanity checks and tests.
"""

import torch
import torch.nn.functional as F
import numpy as np

class RandomTransform(torch.nn.Module):
    """Crop the given batch of tensors at a random location.

    As discussed in https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5
    """

    def __init__(self, source_size, target_size, shift=8, fliplr=True, flipud=False, mode='bilinear', align=True):
        """Args: source and target size."""
        super().__init__()
        self.grid = self.build_grid(source_size, target_size)
        self.delta = torch.linspace(0, 1, source_size)[shift]
        self.fliplr = fliplr
        self.flipud = flipud

        self.mode = mode
        self.align = True

    @staticmethod
    def build_grid(source_size, target_size):
        """https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5."""
        k = float(target_size) / float(source_size)
        direct = torch.linspace(-1, k, target_size).unsqueeze(0).repeat(target_size, 1).unsqueeze(-1)
        full = torch.cat([direct, direct.transpose(1, 0)], dim=2).unsqueeze(0)
        return full

    def random_crop_grid(self, x, randgen=None):
        """https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5."""
        grid = self.grid.repeat(x.size(0), 1, 1, 1).clone().detach()
        grid = grid.to(device=x.device, dtype=x.dtype)
        if randgen is None:
            randgen = torch.rand(x.shape[0], 4, device=x.device, dtype=x.dtype)

        # Add random shifts by x
        x_shift = (randgen[:, 0] - 0.5) * 2 * self.delta
        grid[:, :, :, 0] = grid[:, :, :, 0] + x_shift.unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2))
        # Add random shifts by y
        y_shift = (randgen[:, 1] - 0.5) * 2 * self.delta
        grid[:, :, :, 1] = grid[:, :, :, 1] + y_shift.unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2))

        if self.fliplr:
            grid[randgen[:, 2] > 0.5, :, :, 0] *= -1
        if self.flipud:
            grid[randgen[:, 3] > 0.5, :, :, 1] *= -1
        return grid


    def forward(self, x, randgen=None):
        # Make a random shift grid for each batch
        grid_shifted = self.random_crop_grid(x, randgen)
        # Sample using grid sample
        return F.grid_sample(x, grid_shifted, align_corners=self.align, mode=self.mode)


class FlipLR(torch.nn.Module):
    """Flip only left-right."""

    def __init__(self, source_size, target_size, shift=8, fliplr=True, flipud=False):
        """Args are entirely discarded."""
        super().__init__()

    def forward(self, x, randgen=None):
        # 1) Flip
        if torch.randint(2, (1,)) > 0:
            x = x.flip(dims=(3,))
        return x


class _InvertibleGridSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, theta, iv_theta, mode, align):
        ctx.save_for_backward(iv_theta)
        ctx.mode = mode
        ctx.align = align

        grid = F.affine_grid(theta, x.shape, align_corners=True)
        return F.grid_sample(x, grid, align_corners=align, mode=mode)

    @staticmethod
    def backward(ctx, grad_output):
        iv_theta, = ctx.saved_tensors
        inverse_grid = F.affine_grid(iv_theta, grad_output.shape, align_corners=True)
        return F.grid_sample(grad_output, inverse_grid, align_corners=ctx.align, mode=ctx.mode), None, None, None, None


class RandomTransformFixed(RandomTransform):
    """Similar to RandomTransform, but the backward pass is fixed to be invertible, based on inv. NN interpolation."""

    @staticmethod
    def generate_transformations(bs, device=torch.device('cpu')):
        seed = (torch.randint(8, (bs, 2), device=device, dtype=torch.float) - 4) / 32
        flips = torch.randint(2, (bs, ))

        # forward
        theta = torch.cat((torch.eye(2, device=device).repeat(bs, 1, 1),
                           seed.unsqueeze(1).permute(0, 2, 1)), dim=2)
        theta[flips, 0, 0] *= -1
        # backward
        iv_theta = theta.clone()
        iv_theta[:, 0, 1] *= -1
        iv_theta[:, 1, 2] *= -1
        return theta, iv_theta


    def forward(self, x, randgen=None):
        theta, iv_theta = self.generate_transformations(x.shape[0], device=x.device)
        # Sample using grid sample
        return _InvertibleGridSample.apply(x, theta, iv_theta, self.mode, self.align)



class _GridShiftTransform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, trafo):
        y = torch.zeros_like(x)
        start, stop, flip = trafo
        ctx.save_for_backward(start, stop, flip)

        if flip:
            x = x.flip(dims=(3,))
        y[:, :, start[0]:stop[0], start[1]:stop[1]] = x[:, :, start[0]:stop[0], start[1]:stop[1]]

        return y

    @staticmethod
    def backward(ctx, grad_output):
        start, stop, flip = ctx.saved_tensors
        grad_input = torch.zeros_like(grad_output)
        grad_input[:, :, start[0]:stop[0], start[1]:stop[1]] = \
            grad_output[:, :, start[0]:stop[0], start[1]:stop[1]]
        if flip:
            grad_input = grad_input.flip(dims=(3,))
        return grad_input, None

class RandomGridShift(torch.nn.Module):
    """Shift exactly on the grid."""

    def __init__(self, source_size, target_size, shift=8, fliplr=True, flipud=False, mode='nearest'):
        """Arguments aside from shift, target_size and flip are discarded."""
        super().__init__()

        self.target_size = target_size
        self.shift = shift
        self.fliplr = fliplr

    def get_transformation(self, x, randgen=None):
        # 1) Flip?
        if self.fliplr:
            flip = torch.randint(2, (1, ))
        else:
            flip = 0
        # 2) Shift
        seed = (torch.randint(self.shift, (2, )) - self.shift // 2)
        start = torch.clamp(seed, 0, self.target_size)
        stop = torch.clamp(self.target_size + seed, 0, self.target_size)
        return start, stop, flip

    def forward(self, x, randgen=None):
        trafo = self.get_transformation(x, randgen)
        return _GridShiftTransform.apply(x, trafo)

class MixedAugment(torch.nn.Module):
    @staticmethod
    def rand_brightness(x):
        x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
        return x

    @staticmethod
    def rand_saturation(x):
        x_mean = x.mean(dim=1, keepdim=True)
        x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
        return x

    @staticmethod
    def rand_contrast(x):
        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
        return x

    @staticmethod
    def rand_translation(x, ratio=0.125):
        shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
        translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
        x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
        return x

    @staticmethod
    def rand_cutout(x, ratio=0.5):
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        return x
    
    def forward(x, policy='color,translation,cutout', channels_first=True):
        AUGMENT_FNS = {
            'color': [MixedAugment.rand_brightness, MixedAugment.rand_saturation, MixedAugment.rand_contrast],
            'translation': [MixedAugment.rand_translation],
            'cutout': [MixedAugment.rand_cutout],
        }
        if policy:
            if not channels_first:
                x = x.permute(0, 3, 1, 2)
            
            num_policy = len(policy.split(','))
            num_transform = np.random.randint(0, num_policy)
            policies = np.random.choice(policy.split(','), num_transform, replace=False)
            
            for p in policies:
                for f in AUGMENT_FNS[p]:
                    if np.random.random() < 0.5: x = f(x)
                    
            if not channels_first:
                x = x.permute(0, 2, 3, 1)
            x = x.contiguous()
        return x


    
