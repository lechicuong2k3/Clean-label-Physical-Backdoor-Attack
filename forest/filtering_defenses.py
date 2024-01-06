import torch
import numpy as np
import old_version.defenses.cleansers.robust_estimation as robust_estimation

def get_defense(args):
    if args.filter_defense.lower() == 'spectral_signature':
        return _SpectralSignature
    elif args.filter_defense.lower() == 'deepknn':
        return _DeepKNN
    elif args.filter_defense.lower() == 'activation_clustering':
        return _ActivationClustering
    elif args.filter_defense.lower() == 'spectre':
        return _Spectre

def _get_poisoned_features(kettle, victim, poison_delta, dryrun=False):
    class_indices = [[] for _ in range(len(kettle.class_names))]
    feats = []
    layer_cake = list(victim.model.children())
    feature_extractor = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())
    with torch.no_grad():
        for i, (img, source, idx) in enumerate(kettle.trainset):
            lookup = kettle.poison_lookup.get(idx)
            if lookup is not None:
                img += poison_delta[lookup, :, :, :]
            img = img.unsqueeze(0).to(**kettle.setup)
            feats.append(feature_extractor(img))
            class_indices[source].append(i)
            if dryrun and i == 3:  # use a few values to populate these adjancency matrices
                break
    return feats, class_indices

def _DeepKNN(kettle, victim, poison_delta, args, num_classes=10, overestimation_factor=2.0):
    """deepKNN as in Peri et al. "Deep k-NN Defense against Clean-label Data Poisoning Attacks".

    An overestimation factor of 2 is motivated as necessary in that work."""
    clean_indices = []
    target_class = kettle.poison_setup['poison_class']
    num_poisons_expected = int(overestimation_factor * kettle.args.alpha * len(kettle.trainset_distribution[target_class])) if not kettle.args.dryrun else 0
    feats, _ = _get_poisoned_features(kettle, victim, poison_delta, dryrun=kettle.args.dryrun)

    feats = torch.stack(feats, dim=0)
    dist_matrix = torch.zeros((len(feats), len(feats)))
    for i in range(dist_matrix.shape[0]):
        temp_matrix = torch.stack([feats[i] for _ in range(dist_matrix.shape[1])], dim=0)
        dist_matrix[i, :] = torch.norm((temp_matrix - feats).squeeze(1), dim=1)
    for i in range(dist_matrix.shape[0]):
        vec = dist_matrix[i, :]
        point_label, _ = kettle.trainset.get_target(i)
        _, nearest_indices = vec.topk(num_poisons_expected + 1, largest=False)
        count = 0
        for j in range(1, num_poisons_expected + 1):
            neighbor_label, _ = kettle.trainset.get_target(nearest_indices[j])
            if neighbor_label == point_label:
                count += 1
            else:
                count -= 1
        if count >= 0:
            clean_indices.append(i)
    return clean_indices


def _SpectralSignature(kettle, victim, poison_delta, args, num_classes =10, overestimation_factor=1.5):
    """The spectral signature defense proposed by Tran et al. in "Spectral Signatures in Backdoor Attacks"

    https://proceedings.neurips.cc/paper/2018/file/280cf18baf4311c92aa5a042336587d3-Paper.pdf
    The overestimation factor of 1.5 is proposed in the paper.
    """
    clean_indices = []
    target_class = kettle.poison_setup['poison_class']
    num_poisons_expected = kettle.args.alpha * len(kettle.trainset_distribution[target_class])
    feats, class_indices = _get_poisoned_features(kettle, victim, poison_delta, dryrun=kettle.args.dryrun)

    for i in range(len(class_indices)):
        if len(class_indices[i]) > 1:
            temp_feats = []
            for temp_index in class_indices[i]:
                temp_feats.append(feats[temp_index])
            temp_feats = torch.cat(temp_feats)
            mean_feat = torch.mean(temp_feats, dim=0)
            temp_feats = temp_feats - mean_feat
            _, _, V = torch.svd(temp_feats, compute_uv=True, some=False)
            vec = V[:, 0]  # the top right singular vector is the first column of V
            vals = []
            for j in range(temp_feats.shape[0]):
                vals.append(torch.dot(temp_feats[j], vec).pow(2))

            k = min(int(overestimation_factor * num_poisons_expected), len(vals) - 1)
            _, indices = torch.topk(torch.tensor(vals), k)
            bad_indices = []
            for temp_index in indices:
                bad_indices.append(class_indices[i][temp_index])
            clean = list(set(class_indices[i]) - set(bad_indices))
            clean_indices = clean_indices + clean
    return clean_indices

def _ActivationClustering(kettle, victim, poison_delta, args, num_classes=10, clusters=2):
    """This is Chen et al. "Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering" """
    # lazy sklearn import:
    from sklearn.cluster import KMeans

    clean_indices = []
    feats, class_indices = _get_poisoned_features(kettle, victim, poison_delta, dryrun=kettle.args.dryrun)

    for i in range(len(class_indices)):
        if len(class_indices[i]) > 1:
            temp_feats = np.array([feats[temp_idx].squeeze(0).cpu().numpy() for temp_idx in class_indices[i]])
            kmeans = KMeans(n_clusters=clusters).fit(temp_feats)
            if kmeans.labels_.sum() >= len(kmeans.labels_) / 2.:
                clean_label = 1
            else:
                clean_label = 0
            clean = []
            for (bool, idx) in zip((kmeans.labels_ == clean_label).tolist(), list(range(len(kmeans.labels_)))):
                if bool:
                    clean.append(class_indices[i][idx])
            clean_indices = clean_indices + clean
    return clean_indices

def _Spectre(kettle, victim, poison_delta, args, num_classes=10):
    feats, class_indices = _get_poisoned_features(kettle, victim, poison_delta, dryrun=kettle.args.dryrun)

    suspicious_indices = []
    # Spectral Signature requires an expected poison ratio (we allow the oracle here as a baseline)
    # calculate number of image in target class in data.trainset
    budget = int(args.raw_poison_rate * len(kettle.trainset_distribution[kettle.poison_setup['poison_class']]) * 1.5)
    print(budget)
    # allow removing additional 50% (following the original paper)

    max_dim = 2 # 64
    class_taus = []
    class_S = []
    for i in range(num_classes):

        if len(class_indices[i]) > 1:

            # feats for class i in poisoned set
            temp_feats = np.array([feats[temp_idx].cpu().numpy() for temp_idx in class_indices[i]])
            temp_feats = torch.FloatTensor(temp_feats).cuda()

            temp_clean_feats = None

            temp_feats = temp_feats - temp_feats.mean(dim=0) # centered data
            temp_feats = temp_feats.T # feats arranged in column
            print(temp_feats.shape)
            U, _, _ = torch.svd(temp_feats)
            U = U[:, :max_dim]

            # full projection
            projected_feats = torch.matmul(U.T, temp_feats)

            max_tau = -999999
            best_n_dim = -1
            best_to_be_removed = None

            for n_dim in range(2, max_dim+1): # enumarate all possible "reudced dimensions" and select the best

                S_removed, S_left = SPECTRE(U, temp_feats, n_dim, budget, temp_clean_feats)

                left_feats = projected_feats[:, S_left]
                covariance = torch.cov(left_feats)

                L, V = torch.linalg.eig(covariance)
                L, V = L.real, V.real
                L = (torch.diag(L) ** (1 / 2) + 0.001).inverse()
                normalizer = torch.matmul(V, torch.matmul(L, V.T))

                whitened_feats = torch.matmul(normalizer, projected_feats)

                tau = QUEscore(whitened_feats, max_dim).mean()

                if tau > max_tau:
                    max_tau = tau
                    best_n_dim = n_dim
                    best_to_be_removed = S_removed


            print('class=%d, dim=%d, tau=%f' % (i, best_n_dim, max_tau))

            class_taus.append(max_tau)

            suspicious_indices = []
            for temp_index in best_to_be_removed:
                suspicious_indices.append(class_indices[i][temp_index])

            class_S.append(suspicious_indices)

    class_taus = np.array(class_taus)
    median_tau = np.median(class_taus)

    #print('median_tau : %d' % median_tau)
    suspicious_indices = []
    max_tau = -99999
    for i in range(num_classes):
        #if class_taus[i] > max_tau:
        #    max_tau = class_taus[i]
        #    suspicious_indices = class_S[i]
        #print('class-%d, tau = %f' % (i, class_taus[i]))
        #if class_taus[i] > 2*median_tau:
        #    print('[large tau detected] potential poisons! Apply Filter!')
        for temp_index in class_S[i]:
            suspicious_indices.append(temp_index)
    # create clean set
    clean_indices = list(set(range(len(kettle.trainset))) - set(suspicious_indices))

    return clean_indices

def QUEscore(temp_feats, n_dim):

    n_samples = temp_feats.shape[1]
    alpha = 4.0
    Sigma = torch.matmul(temp_feats, temp_feats.T) / n_samples
    I = torch.eye(n_dim).cuda()
    Q = torch.exp((alpha * (Sigma - I)) / (torch.linalg.norm(Sigma, ord=2) - 1))
    trace_Q = torch.trace(Q)

    taus = []
    for i in range(n_samples):
        h_i = temp_feats[:, i:i + 1]
        tau_i = torch.matmul(h_i.T, torch.matmul(Q, h_i)) / trace_Q
        tau_i = tau_i.item()
        taus.append(tau_i)
    taus = np.array(taus)

    return taus

def SPECTRE(U, temp_feats, n_dim, budget, oracle_clean_feats=None):

    projector = U[:, :n_dim].T # top left singular vectors
    temp_feats = torch.matmul(projector, temp_feats)

    if oracle_clean_feats is None:
        estimator = robust_estimation.BeingRobust(random_state=0, keep_filtered=True).fit((temp_feats.T).cpu().numpy())
        clean_mean = torch.FloatTensor(estimator.location_).cuda()
        filtered_feats = (torch.FloatTensor(estimator.filtered_).cuda() - clean_mean).T
        clean_covariance = torch.cov(filtered_feats)
    else:
        clean_feats = torch.matmul(projector, oracle_clean_feats)
        clean_covariance = torch.cov(clean_feats)
        clean_mean = clean_feats.mean(dim = 1)


    temp_feats = (temp_feats.T - clean_mean).T

    # whiten the data
    L, V = torch.linalg.eig(clean_covariance)
    L, V = L.real, V.real
    L = (torch.diag(L)**(1/2)+0.001).inverse()
    normalizer = torch.matmul(V, torch.matmul( L, V.T ) )
    temp_feats = torch.matmul(normalizer, temp_feats)

    # compute QUEscore
    taus = QUEscore(temp_feats, n_dim)

    sorted_indices = np.argsort(taus)
    n_samples = len(sorted_indices)

    budget = min(budget, n_samples//2) # default assumption : at least a half of samples in each class is clean

    suspicious = sorted_indices[-budget:]
    left = sorted_indices[:n_samples-budget]

    return suspicious, left