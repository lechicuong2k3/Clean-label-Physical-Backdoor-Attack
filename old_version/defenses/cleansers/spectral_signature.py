import numpy as np
import torch
from tqdm import tqdm
from helpers import get_features

def cleanser(inspection_set, model, num_classes, args):
    """
        adapted from : https://github.com/hsouri/Sleeper-Agent/blob/master/forest/filtering_defenses.py
    """
    # Spectral Signature requires an expected poison ratio (we allow the oracle here as a baseline)
    num_poisons_expected = args.poison_rate * len(inspection_set) * 1.5 # allow removing additional 50% (following the original paper)

    feats, class_indices = get_features(inspection_set, model, args.model, num_classes, args.save_rep_path)

    suspicious_indices = []


    for i in range(num_classes):

        if len(class_indices[i]) > 1:

            temp_feats = np.array([feats[temp_idx] for temp_idx in class_indices[i]])
            temp_feats = torch.FloatTensor(temp_feats)

            mean_feat = torch.mean(temp_feats, dim=0)
            temp_feats = temp_feats - mean_feat
            _, _, V = torch.svd(temp_feats, compute_uv=True, some=False)

            vec = V[:, 0]  # the top right singular vector is the first column of V
            vals = []
            for j in range(temp_feats.shape[0]):
                vals.append(torch.dot(temp_feats[j], vec).pow(2))

            k = min(int(num_poisons_expected), len(vals) // 2)
            # default assumption : at least a half of samples in each class is clean

            _, indices = torch.topk(torch.tensor(vals), k)
            for temp_index in indices:
                suspicious_indices.append(class_indices[i][temp_index])

    return suspicious_indices