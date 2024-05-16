"""Various utilities."""

import os
import socket
import datetime
import torch
import random
import numpy as np
import torch.nn.functional as F
import time

from .consts import NON_BLOCKING
from collections import defaultdict
from tqdm import tqdm
from submodlib.functions.facilityLocation import FacilityLocationFunction


os.environ["CUDA_VISIBLE_DEVICES"]="3"
def system_startup(args=None, defs=None):
    """Decide and print GPU / CPU / hostname info."""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    setup = dict(device=device, dtype=torch.float, non_blocking=NON_BLOCKING)
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print(f'------------------ Currently evaluating {args.recipe} ------------------')
    
    write(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"), args.output)
    write(f'------------------ Currently evaluating {args.recipe} ------------------', args.output)
    
    if args is not None:
        print(args)
    print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')

    if torch.cuda.is_available():
        print(f'GPU : {torch.cuda.get_device_name(device=device)}')

    return setup

def average_dicts(running_stats):
    """Average entries in a list of dictionaries."""
    average_stats = defaultdict(list)
    for stat in running_stats[0]:
        if isinstance(running_stats[0][stat], list):
            for i, _ in enumerate(running_stats[0][stat]):
                average_stats[stat].append(np.mean([stat_dict[stat][i] for stat_dict in running_stats]))
        else:
            average_stats[stat] = np.mean([stat_dict[stat] for stat_dict in running_stats])
    return average_stats

"""Misc."""
def _gradient_matching(poison_grad, source_grad):
    """Compute the blind passenger loss term."""
    matching = 0
    poison_norm = 0
    source_norm = 0

    for pgrad, tgrad in zip(poison_grad, source_grad):
        matching -= (tgrad * pgrad).sum()
        poison_norm += pgrad.pow(2).sum()
        source_norm += tgrad.pow(2).sum()

    matching = matching / poison_norm.sqrt() / source_norm.sqrt()

    return matching

def bypass_last_layer(model):
    """Hacky way of separating features and classification head for many models.

    Patch this function if problems appear.
    """
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        layer_cake = list(model.module.children())
    else:
        layer_cake = list(model.children())
    last_layer = layer_cake[-1]
    headless_model = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten()).eval()  # this works most of the time all of the time :<
    return headless_model, last_layer

def cw_loss(outputs, target_classes, clamp=-100):
    """Carlini-Wagner loss for brewing"""
    top_logits, _ = torch.max(outputs, 1)
    target_logits = torch.stack([outputs[i, target_classes[i]] for i in range(outputs.shape[0])])
    difference = torch.clamp(top_logits - target_logits, min=clamp)
    return torch.mean(difference)

def _label_to_onehot(source, num_classes=100):
    source = torch.unsqueeze(source, 1)
    onehot_source = torch.zeros(source.shape[0], num_classes, device=source.device)
    onehot_source.scatter_(1, source, 1)
    return onehot_source

def cw_loss2(outputs, target_classes, confidence=0, clamp=-100):
    """CW. This is assert-level equivalent."""
    one_hot_labels = _label_to_onehot(target_classes, num_classes=outputs.shape[1])
    source_logit = (outputs * one_hot_labels).sum(dim=1)
    second_logit, _ = (outputs - outputs * one_hot_labels).max(dim=1)
    cw_indiv = torch.clamp(second_logit - source_logit + confidence, min=clamp)
    return cw_indiv.mean()

# def total_variation_loss(img, weight=0.01):
#     bs_img, c_img, h_img, w_img = img.size()
#     tv_h = torch.abs(img[:,:,1:,:]-img[:,:,:-1,:]).sum()
#     tv_w = torch.abs(img[:,:,:,1:]-img[:,:,:,:-1]).sum()
#     return weight * (tv_h+tv_w)/(h_img * w_img * c_img * bs_img * 2)

def set_random_seed(seed):
    # Setting seed
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)  # if you are using multi-GPU.
    random.seed(seed + 5)
    os.environ['PYTHONHASHSEED'] = str(seed + 6)

def set_deterministic():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True) # for pytorch >= 1.8
    torch.backends.cudnn.benchmark = False
    
def write(content, file):
    with open(file, 'a') as f:
        f.write(content + '\n')
  
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

class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target

def visualize(dataset):
    import matplotlib.pyplot as plt
    """Visualize a dataset of images"""
    # Create a grid of 10x10 images
    num_samples = len(dataset)
    fig, axes = plt.subplots(nrows=num_samples // 10 + 1, ncols=10, figsize=(30, 90))  # Adjust figsize for desired size

    # Iterate through the samples and plot them
    for i, sample in enumerate(dataset):
        image, label, _ = sample  # Assuming your dataset returns images and labels

        # Convert image to NumPy array if needed
        if isinstance(image, torch.Tensor):
            image = image.cpu().permute(1,2,0).numpy()

        axes.flat[i].imshow(image)

    # Adjust layout and spacing
    plt.tight_layout()

    # Show the plot
    plt.show()
    
def total_variation_loss(flows,padding_mode='constant', epsilon=1e-8):
    paddings = (1,1,1,1)
    padded_flows = F.pad(flows,paddings,mode=padding_mode,value=0)
    shifted_flows = [
    padded_flows[:, :, 2:, 2:],  # bottom right (+1,+1)
    padded_flows[:, :, 2:, :-2],  # bottom left (+1,-1)
    padded_flows[:, :, :-2, 2:],  # top right (-1,+1)
    padded_flows[:, :, :-2, :-2]  # top left (-1,-1)
    ]
    #||\Delta u^{(p)} - \Delta u^{(q)}||_2^2 + # ||\Delta v^{(p)} - \Delta v^{(q)}||_2^2 
    num_pixels = flows.shape[1] * flows.shape[2] * flows.shape[3]
    loss=0
    for shifted_flow in shifted_flows:
        # loss += torch.sum(0.299 * torch.square(flows[:, 0] - shifted_flow[:, 0]) + 0.587 * torch.square(flows[:, 1] - shifted_flow[:, 1]) + 0.114 * torch.square(flows[:, 2] - shifted_flow[:, 2]) + epsilon).cuda()
        loss += torch.sum(torch.square(flows[:, 1] - shifted_flow[:, 1]) + torch.square(flows[:, 2] - shifted_flow[:, 2]) + epsilon).cuda()
    return 1/num_pixels * loss.type(torch.float32)

def upwind_tv(x):
    # x is a batch of images with shape (batch_size, channels, height, width)

    # Shifted versions of the image
    x_right = F.pad(x[:, :, :, 1:], (0, 1, 0, 0), mode='replicate')  # right shift
    x_left = F.pad(x[:, :, :, :-1], (1, 0, 0, 0), mode='replicate')  # left shift
    x_down = F.pad(x[:, :, 1:, :], (0, 0, 0, 1), mode='replicate')  # down shift
    x_up = F.pad(x[:, :, :-1, :], (0, 0, 1, 0), mode='replicate')  # up shift

    # Compute differences
    diff_right = x - x_right
    diff_left = x - x_left
    diff_down = x - x_down
    diff_up = x - x_up

    # Compute the TV
    tv = diff_right**2 + diff_left**2 + diff_down**2 + diff_up**2

    # Sum over all pixels and channels
    tv = tv.mean() * 4
    return tv

def upwind_tv_channel(x):
    # x is a batch of images with shape (batch_size, channels, height, width)

    # Shifted versions of the image
    x_right = F.pad(x[:, :, :, 1:], (0, 1, 0, 0), mode='replicate')  # right shift
    x_left = F.pad(x[:, :, :, :-1], (1, 0, 0, 0), mode='replicate')  # left shift
    x_down = F.pad(x[:, :, 1:, :], (0, 0, 0, 1), mode='replicate')  # down shift
    x_up = F.pad(x[:, :, :-1, :], (0, 0, 1, 0), mode='replicate')  # up shift

    # Compute differences
    diff_right = x - x_right
    diff_left = x - x_left
    diff_down = x - x_down
    diff_up = x - x_up

    # Compute the TV
    tv = diff_right**2 + diff_left**2 + diff_down**2 + diff_up**2

    # Sum over all pixels and channels
    tv = (tv[:, 0] * 0.299 + tv[:, 1] * 0.587 + tv[:, 2] * 0.114).mean()
    return tv

class PartialLoss(object):
    """ Partially applied loss object. Has forward and zero_grad methods """
    def __init__(self):
        self.nets = []

    def zero_grad(self):
        for net in self.nets:
            net.zero_grad()

class ReferenceRegularizer(PartialLoss):
    def __init__(self, fix_im):
        super(ReferenceRegularizer, self).__init__()
        self.fix_im = fix_im

    def setup_attack_batch(self, fix_im):
        """ Setup function to ensure fixed images are set
            has been made; also zeros grads
        ARGS:
            fix_im: Variable (NxCxHxW) - Ground images for this minibatch
                    SHOULD BE IN [0.0, 1.0] RANGE
        """
        self.fix_im = fix_im
        self.zero_grad()


    def cleanup_attack_batch(self):
        """ Cleanup function to clear the fixed images after an attack batch
            has been made; also zeros grads
        """
        old_fix_im = self.fix_im
        self.fix_im = None
        del old_fix_im
        self.zero_grad()

class SoftLInfRegularization(ReferenceRegularizer):
    '''
        see page 10 of this paper (https://arxiv.org/pdf/1608.04644.pdf)
        for discussion on why we want SOFT l inf
    '''
    def __init__(self, fix_im, **kwargs):
        super(SoftLInfRegularization, self).__init__(fix_im)

    def forward(self, examples, *args, **kwargs):
        # ARGS should have one element, which serves as the tau value

        tau =  8.0 / 255.0  # starts at 1 each time?
        scale_factor = 0.9
        l_inf_dist = float(torch.max(torch.abs(examples - self.fix_im)))
        '''
        while scale_factor * tau > l_inf_dist:
            tau *= scale_factor

        assert tau > l_inf_dist
        '''
        delta_minus_taus = torch.clamp(torch.abs(examples - self.fix_im) - tau,
                                       min=0.0)
        batchwise = batchwise_norm(delta_minus_taus, 'inf', dim=0)
        return batchwise.squeeze()

def batchwise_norm(examples, lp, dim=0):
    """ Returns the per-example norm of the examples, keeping along the
        specified dimension.
        e.g. if examples is NxCxHxW, applying this fxn with dim=0 will return a
             N-length tensor with the lp norm of each example
    ARGS:
        examples : tensor or Variable -  needs more than one dimension
        lp : string or int - either 'inf' or an int for which lp norm we use
        dim : int - which dimension to keep
    RETURNS:
        1D object of same type as examples, but with shape examples.shape[dim]
    """

    assert isinstance(lp, int) or lp == 'inf'
    examples = torch.abs(examples)
    example_dim = examples.dim()
    if dim != 0:
        examples = examples.transpose(dim, 0)

    if lp == 'inf':
        for reduction in range(1, example_dim):
            examples, _ = examples.max(1)
        return examples

    else:
        examples = torch.pow(examples + 1e-10, lp)
        for reduction in range(1, example_dim):
            examples = examples.sum(1)
        return torch.pow(examples, 1.0 / lp)
    
def get_subset(args, model, trainloader, num_sampled, epoch, N, indices, num_classes=10):
    trainloader = tqdm(trainloader)

    grad_preds = []
    labels = []
    conf_all = np.zeros(N)
    conf_true = np.zeros(N)

    with torch.no_grad():
        for _, (inputs, targets, index) in enumerate(trainloader):
            model.eval()
            targets = targets.long()

            inputs = inputs.cuda()

            confs = torch.softmax(model(inputs), dim=1).cpu().detach()
            conf_all[index] = np.amax(confs.numpy(), axis=1)
            conf_true[index] = confs[range(len(targets)), targets].numpy()
            g0 = confs - torch.eye(num_classes)[targets.long()]
            grad_preds.append(g0.cpu().detach().numpy())

            targets = targets.numpy()
            labels.append(targets)
        
        labels = np.concatenate(labels)
        subset, subset_weights, _, _, cluster_ = get_coreset(np.concatenate(grad_preds), labels, len(labels), num_sampled, num_classes, equal_num=args.equal_num, optimizer=args.greedy, metric=args.metric)

    subset = indices[subset]
    cluster = -np.ones(N, dtype=int)
    cluster[indices] = cluster_

    keep_indices = np.where(subset_weights > args.cluster_thresh)
    if epoch >= args.drop_after:
        keep_indices = np.where(np.isin(cluster, keep_indices))[0]
        subset = keep_indices
    else:
        subset = np.arange(N)

    return subset

def faciliy_location_order(c, X, y, metric, num_per_class, weights=None, optimizer="LazyGreedy"):
    class_indices = np.where(y == c)[0]
    X = X[class_indices]
    N = X.shape[0]

    start = time.time()
    obj = FacilityLocationFunction(n=len(X), data=X, metric=metric, mode='dense')
    S_time = time.time() - start

    start = time.time()
    greedyList = obj.maximize(
        budget=num_per_class,
        optimizer=optimizer,
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
    )
    order = list(map(lambda x: x[0], greedyList))
    sz = list(map(lambda x: x[1], greedyList))
    greedy_time = time.time() - start

    S = obj.sijs
    order = np.asarray(order, dtype=np.int64)
    sz = np.zeros(num_per_class, dtype=np.float64)
    cluster = -np.ones(N)

    for i in range(N):
        if np.max(S[i, order]) <= 0:
            continue
        cluster[i] = np.argmax(S[i, order])
        if weights is None:
            sz[np.argmax(S[i, order])] += 1
        else:
            sz[np.argmax(S[i, order])] += weights[i]
    sz[np.where(sz==0)] = 1

    cluster[cluster>=0] += c * num_per_class

    return class_indices[order], sz, greedy_time, S_time, cluster


def get_orders_and_weights(B, X, metric, y=None, weights=None, equal_num=False, num_classes=10, optimizer="LazyGreedy"):
    '''
    Ags
    - X: np.array, shape [N, d]
    - B: int, number of points to select
    - metric: str, one of ['cosine', 'euclidean'], for similarity
    - y: np.array, shape [N], integer class labels for C classes
      - if given, chooses B / C points per class, B must be divisible by C
    - outdir: str, path to output directory, must already exist

    Returns
    - order_mg/_sz: np.array, shape [B], type int64
      - *_mg: order points by their marginal gain in FL objective (largest gain first)
      - *_sz: order points by their cluster size (largest size first)
    - weights_mg/_sz: np.array, shape [B], type float32, sums to 1
    '''
    N = X.shape[0]
    if y is None:
        y = np.zeros(N, dtype=np.int32)  # assign every point to the same class
    if num_classes is not None:
        classes = np.arange(num_classes)
    else:
        classes = np.unique(y)
    C = len(classes)  # number of classes

    if equal_num:
        class_nums = [sum(y == c) for c in classes]
        num_per_class = int(np.ceil(B / C)) * np.ones(len(classes), dtype=np.int32)
        minority = class_nums < np.ceil(B / C)
        if sum(minority) > 0:
            extra = sum([max(0, np.ceil(B / C) - class_nums[c]) for c in classes])
            for c in classes[~minority]:
                num_per_class[c] += int(np.ceil(extra / sum(minority)))
    else:
        num_per_class = np.int32(np.ceil(np.divide([sum(y == i) for i in classes], N) * B))
        total = np.sum(num_per_class)
        diff = total - B
        chosen = set()
        for i in range(diff):
            j = np.random.randint(C)
            while j in chosen or num_per_class[j] <= 0:
                j = np.random.randint(C)
            num_per_class[j] -= 1
            chosen.add(j)

    order_mg_all, cluster_sizes_all, greedy_times, similarity_times, cluster_all = zip(*map(
        lambda c: faciliy_location_order(c, X, y, metric, num_per_class[c], weights, optimizer=optimizer), classes))

    order_mg = np.concatenate(order_mg_all).astype(np.int32)
    weights_mg = np.concatenate(cluster_sizes_all).astype(np.float32)
    class_indices = [np.where(y == c)[0] for c in classes]
    class_indices = np.concatenate(class_indices).astype(np.int32)
    class_indices = np.argsort(class_indices)
    cluster_mg = np.concatenate(cluster_all).astype(np.int32)[class_indices]
    assert len(order_mg) == len(weights_mg)

    ordering_time = np.max(greedy_times)
    similarity_time = np.max(similarity_times)

    order_sz = []
    weights_sz = []
    vals = order_mg, weights_mg, order_sz, weights_sz, ordering_time, similarity_time, cluster_mg
    return vals

def get_coreset(gradient_est, 
                labels, 
                N, 
                B, 
                num_classes, 
                equal_num=True,
                optimizer="LazyGreedy",
                metric='euclidean'):
    '''
    Arguments:
        gradient_est: Gradient estimate
            numpy array - (N,p) 
        labels: labels of corresponding grad ests
            numpy array - (N,)
        B: subset size to select
            int
        num_classes:
            int
        normalize_weights: Whether to normalize coreset weights based on N and B
            bool
        gamma_coreset:
            float
        smtk:
            bool
        st_grd:
            bool

    Returns 
    (1) coreset indices (2) coreset weights (3) ordering time (4) similarity time
    '''
    try:
        subset, subset_weights, _, _, ordering_time, similarity_time, cluster = get_orders_and_weights(
            B, 
            gradient_est, 
            metric, 
            y=labels, 
            equal_num=equal_num, 
            num_classes=num_classes,
            optimizer=optimizer)
    except ValueError as e:
        print(e)
        print(f"WARNING: ValueError from coreset selection, choosing random subset for this epoch")
        subset, subset_weights = get_random_subset(B, N)
        ordering_time = 0
        similarity_time = 0

    if len(subset) != B:
        print(f"!!WARNING!! Selected subset of size {len(subset)} instead of {B}")
    print(f'FL time: {ordering_time:.3f}, Sim time: {similarity_time:.3f}')

    return subset, subset_weights, ordering_time, similarity_time, cluster

def get_random_subset(B, N):
    print(f'Selecting {B} element from the random subset of size: {N}')
    order = np.arange(0, N)
    np.random.shuffle(order)
    subset = order[:B]

    return subset


class ColorSpace(object):
    """
    Base class for color spaces.
    """

    def from_rgb(self, imgs):
        """
        Converts an Nx3xWxH tensor in RGB color space to a Nx3xWxH tensor in
        this color space. All outputs should be in the 0-1 range.
        """
        raise NotImplementedError()

    def to_rgb(self, imgs):
        """
        Converts an Nx3xWxH tensor in this color space to a Nx3xWxH tensor in
        RGB color space.
        """
        raise NotImplementedError()


class CIEXYZColorSpace(ColorSpace):
    """
    The 1931 CIE XYZ color space (assuming input is in sRGB).

    Warning: may have values outside [0, 1] range. Should only be used in
    the process of converting to/from other color spaces.
    """

    def from_rgb(self, imgs):
        # apply gamma correction
        small_values_mask = (imgs < 0.04045).float()
        imgs_corrected = (
            (imgs / 12.92) * small_values_mask +
            ((imgs + 0.055) / 1.055) ** 2.4 * (1 - small_values_mask)
        )

        # linear transformation to XYZ
        r, g, b = imgs_corrected.permute(1, 0, 2, 3)
        x = 0.4124 * r + 0.3576 * g + 0.1805 * b
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        z = 0.0193 * r + 0.1192 * g + 0.9504 * b

        return torch.stack([x, y, z], 1)

    def to_rgb(self, imgs):
        # linear transformation
        x, y, z = imgs.permute(1, 0, 2, 3)
        r = 3.2406 * x - 1.5372 * y - 0.4986 * z
        g = -0.9689 * x + 1.8758 * y + 0.0415 * z
        b = 0.0557 * x - 0.2040 * y + 1.0570 * z

        imgs = torch.stack([r, g, b], 1)

        # apply gamma correction
        small_values_mask = (imgs < 0.0031308).float()
        imgs_clamped = imgs.clamp(min=1e-10)  # prevent NaN gradients
        imgs_corrected = (
            (12.92 * imgs) * small_values_mask +
            (1.055 * imgs_clamped ** (1 / 2.4) - 0.055) *
            (1 - small_values_mask)
        )

        return imgs_corrected

class CIELUVColorSpace(ColorSpace):
    """
    Converts to the 1976 CIE L*u*v* color space.
    """

    def __init__(self, up_white=0.1978, vp_white=0.4683, y_white=1,
                 eps=1e-10):
        self.xyz_cspace = CIEXYZColorSpace()
        self.up_white = up_white
        self.vp_white = vp_white
        self.y_white = y_white
        self.eps = eps

    def from_rgb(self, imgs):
        x, y, z = self.xyz_cspace.from_rgb(imgs).permute(1, 0, 2, 3)

        # calculate u' and v'
        denom = x + 15 * y + 3 * z + self.eps
        up = 4 * x / denom
        vp = 9 * y / denom

        # calculate L*, u*, and v*
        small_values_mask = (y / self.y_white < (6 / 29) ** 3).float()
        y_clamped = y.clamp(min=self.eps)  # prevent NaN gradients
        L = (
            ((29 / 3) ** 3 * y / self.y_white) * small_values_mask +
            (116 * (y_clamped / self.y_white) ** (1 / 3) - 16) *
            (1 - small_values_mask)
        )
        u = 13 * L * (up - self.up_white)
        v = 13 * L * (vp - self.vp_white)

        return torch.stack([L / 100, (u + 100) / 200, (v + 100) / 200], 1)

    def to_rgb(self, imgs):
        L = imgs[:, 0, :, :] * 100
        u = imgs[:, 1, :, :] * 200 - 100
        v = imgs[:, 2, :, :] * 200 - 100

        up = u / (13 * L + self.eps) + self.up_white
        vp = v / (13 * L + self.eps) + self.vp_white

        small_values_mask = (L <= 8).float()
        y = (
            (self.y_white * L * (3 / 29) ** 3) * small_values_mask +
            (self.y_white * ((L + 16) / 116) ** 3) * (1 - small_values_mask)
        )
        denom = 4 * vp + self.eps
        x = y * 9 * up / denom
        z = y * (12 - 3 * up - 20 * vp) / denom

        return self.xyz_cspace.to_rgb(
            torch.stack([x, y, z], 1).clamp(0, 1.1)).clamp(0, 1)
        