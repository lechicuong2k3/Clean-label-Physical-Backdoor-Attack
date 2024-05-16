import torch
import numpy as np
import defense.cleansers.robust_estimation as robust_estimation
import random
from .data.datasets import PoisonDataset, normalize
from tqdm import tqdm
from .consts import NON_BLOCKING, NORMALIZE
from sklearn.decomposition import FastICA
from sklearn.metrics import silhouette_score

def get_defense(defense):
    if defense == 'ss':
        return _SpectralSignature
    elif defense== 'deepknn':
        return _DeepKNN
    elif defense == 'ac':
        return _ActivationClustering
    elif defense == 'spectre':
        return _Spectre
    elif defense == 'strip':
        return _Strip
    elif defense== 'scan':
        return _Scan
    elif defense == 'nc':
        return _NeuralCleanse
    else:
        raise NotImplementedError('Defense is not implemented')

def _get_poisoned_features(kettle, victim, poison_delta, dryrun=False):
    class_indices = [[] for _ in range(len(kettle.trainset_class_names))]
    feats = []
    layer_cake = list(victim.model.children())
    feature_extractor = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())
    feature_extractor.eval()
    with torch.no_grad():
        for i, (img, source, idx) in enumerate(kettle.trainset):
            lookup = kettle.poison_lookup.get(idx)
            if lookup is not None and poison_delta is not None:
                img += poison_delta[lookup, :, :, :]
            if NORMALIZE:
                img = normalize(img).to(**kettle.setup)
            else:
                img = img.unsqueeze(0).to(**kettle.setup)
            feats.append(feature_extractor(img))
            class_indices[source].append(i)
            # if dryrun and i == 3:  # use a few values to populate these adjancency matrices
            #     break
    return feats, class_indices

def cluster_metrics(cluster_1, cluster_0):

    num = len(cluster_1) + len(cluster_0)
    features = torch.cat([cluster_1, cluster_0], dim=0)

    labels = torch.zeros(num)
    labels[:len(cluster_1)] = 1
    labels[len(cluster_1):] = 0

    ## Raw Silhouette Score
    raw_silhouette_score = silhouette_score(features, labels)
    return raw_silhouette_score

def _get_cleaned_features(kettle, victim, dryrun=False):
    class_indices = [[] for _ in range(len(kettle.trainset_class_names))]
    feats = []
    layer_cake = list(victim.model.children())
    feature_extractor = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())
    feature_extractor.eval()
    with torch.no_grad():
        for i, (img, source, idx) in enumerate(kettle.validset):
            img = img.unsqueeze(0).to(**kettle.setup)
            feats.append(feature_extractor(img))
            class_indices[source].append(i)
            # if dryrun and i == 3:  # use a few values to populate these adjancency matrices
            #     break
    return feats, class_indices

    
def _DeepKNN(kettle, victim, poison_delta, args, num_classes=10, overestimation_factor=2.0):
    """deepKNN as in Peri et al. "Deep k-NN Defense against Clean-label Data Poisoning Attacks".

    An overestimation factor of 2 is motivated as necessary in that work."""
    clean_indices = []
    target_class = kettle.poison_setup['poison_class']
    num_poisons_expected = int(overestimation_factor * kettle.args.alpha * len(kettle.trainset_dist[target_class])) if not kettle.args.dryrun else 0
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


def _SpectralSignature(kettle, victim, poison_delta, args, num_classes=10, overestimation_factor=1.5):
    """The spectral signature defense proposed by Tran et al. in "Spectral Signatures in Backdoor Attacks"

    https://proceedings.neurips.cc/paper/2018/file/280cf18baf4311c92aa5a042336587d3-Paper.pdf
    The overestimation factor of 1.5 is proposed in the paper.
    """
    clean_indices = []
    target_class = kettle.poison_setup['poison_class']
    num_poisons_expected = kettle.args.alpha * len(kettle.trainset_dist[target_class])
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

def _ActivationClustering(kettle, victim, poison_delta, args, num_classes=10, clusters=2, threshold=0.1):
    """This is Chen et al. "Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering" """
    # lazy sklearn import:
    from sklearn.cluster import KMeans

    suspicious_indices = []
    feats, class_indices = _get_poisoned_features(kettle, victim, poison_delta, dryrun=kettle.args.dryrun)
    
    for target_class in range(len(class_indices)):
        if len(class_indices[target_class]) > 1:
            temp_feats = np.array([feats[temp_idx].squeeze(0).cpu().numpy() for temp_idx in class_indices[target_class]])
            
            ica = FastICA(n_components=10, max_iter=1000, tol=0.005)
            projected_feats = ica.fit_transform(temp_feats)
            kmeans = KMeans(n_clusters=clusters).fit(projected_feats)
            if kmeans.labels_.sum() >= len(kmeans.labels_) / 2.:
                clean_label = 1
            else:
                clean_label = 0
                
            # by default, take the smaller cluster as the poisoned cluster
            if kmeans.labels_.sum() >= len(kmeans.labels_) / 2.:
                clean_label = 1
            else:
                clean_label = 0

            outliers = []
            for (bool, idx) in zip((kmeans.labels_ != clean_label).tolist(), list(range(len(kmeans.labels_)))):
                if bool:
                    outliers.append(class_indices[target_class][idx])

            score = silhouette_score(projected_feats, kmeans.labels_)
            print('[class-%d] silhouette_score = %f' % (target_class, score))
            
            if score > threshold and len(outliers) < len(kmeans.labels_) * 0.35: # if one of the two clusters is abnormally small
            # if len(outliers) < len(kmeans.labels_) * 0.35:
                print(f"Outlier Num in Class {target_class}:", len(outliers))
                suspicious_indices += outliers
    
    clean_indices = list(set(range(len(kettle.trainset))) - set(suspicious_indices))    
    return clean_indices


def _Spectre(kettle, victim, poison_delta, args, num_classes=10):
    """
    Spectre defense method implementation.
    Returns a list of clean indices.
    """
    # Load data using your framework's data loading function
    feats, class_indices = _get_poisoned_features(kettle, victim, poison_delta, dryrun=kettle.args.dryrun)
    clean_feats, clean_class_indices = _get_cleaned_features(kettle, victim, dryrun=kettle.args.dryrun)
    suspicious_indices = []
    raw_poison_rate = args.alpha +args.beta
    budget = int(raw_poison_rate * len(kettle.trainset_dist[kettle.poison_setup['poison_class']]) * 1.5)

    max_dim = 2 # 64
    class_taus = []
    class_S = []
    for i in range(num_classes):

        if len(class_indices[i]) > 1:

            # feats for class i in poisoned set
            temp_feats = np.array([feats[temp_idx].squeeze(0).cpu().numpy() for temp_idx in class_indices[i]])
            temp_feats = torch.FloatTensor(temp_feats).cuda()

            temp_clean_feats = None
            temp_clean_feats = np.array([clean_feats[temp_idx].squeeze(0).cpu().numpy() for temp_idx in clean_class_indices[i]])
            temp_clean_feats = torch.FloatTensor(temp_clean_feats).cuda()
            temp_clean_feats = temp_clean_feats - temp_feats.mean(dim=0)
            temp_clean_feats = temp_clean_feats.T

            temp_feats = temp_feats - temp_feats.mean(dim=0) # centered data
            temp_feats = temp_feats.T # feats arranged in column

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


            # print('class=%d, dim=%d, tau=%f' % (i, best_n_dim, max_tau))

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
    
def _Strip(kettle, victim, poison_delta, args, num_classes=10):
    strip_alpha = 1.0
    N = 200
    defense_fpr = 0.1
    batch_size = 64
    def check(_input, _label, source_set, model):
        _list = []

        samples = list(range(len(source_set)))
        random.shuffle(samples)
        samples = samples[:N]

        with torch.no_grad():

            for i in samples:
                X, _, _ = source_set[i]
                X = X.to(**kettle.setup)
                _test = superimpose(_input, X)
                entro = entropy(_test, model).cpu().detach()
                _list.append(entro)
                # _class = self.model.get_class(_test)

        return torch.stack(_list).mean(0)

    def superimpose(_input1, _input2, alpha = None):
        if alpha is None:
            alpha = strip_alpha

        result = _input1 + alpha * _input2
        return result

    def entropy(_input, model) -> torch.Tensor:
        # p = self.model.get_prob(_input)
        p = torch.nn.Softmax(dim=1)(model(_input)) + 1e-8
        return (-p * p.log()).sum(1)
    
    # choose a decision boundary with the test set
    inspection_set = PoisonDataset(dataset=kettle.trainset, poison_delta=poison_delta, poison_lookup=kettle.poison_lookup)
    clean_entropy = []
    clean_set_loader = torch.utils.data.DataLoader(kettle.validset, batch_size=batch_size, shuffle=False)
    for _input, _label, _ in tqdm(clean_set_loader):
        _input, _label = _input.to(**kettle.setup), _label.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)
        entropies = check(_input, _label, kettle.validset, victim.model)
        for e in entropies:
            clean_entropy.append(e)
    clean_entropy = torch.FloatTensor(clean_entropy)

    clean_entropy, _ = clean_entropy.sort()
    
    threshold_low = float(clean_entropy[int(defense_fpr * len(clean_entropy))])
    threshold_high = np.inf

    # now cleanse the inspection set with the chosen boundary
    inspection_set_loader = torch.utils.data.DataLoader(inspection_set, batch_size=batch_size, shuffle=False)
    all_entropy = []
    for _input, _label, _ in tqdm(inspection_set_loader):
        _input, _label = _input.to(**kettle.setup), _label.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)
        entropies = check(_input, _label, kettle.validset, victim.model)
        for e in entropies:
            all_entropy.append(e)
    all_entropy = torch.FloatTensor(all_entropy)

    suspicious_indices = torch.logical_or(all_entropy < threshold_low, all_entropy > threshold_high).nonzero().reshape(-1)
    
    clean_indices = list(set(range(len(kettle.trainset))) - set(suspicious_indices.tolist()))
    return clean_indices

def _NeuralCleanse(kettle, victim, poison_delta, args, num_classes=10):
    pass

def _Scan(kettle, victim, poison_delta, args, num_classes=10):
    kwargs = {'num_workers': 3, 'pin_memory': True}

    inspection_set = PoisonDataset(dataset=kettle.trainset, poison_delta=poison_delta, poison_lookup=kettle.poison_lookup)
    # main dataset we aim to cleanse
    inspection_split_loader = torch.utils.data.DataLoader(
        inspection_set,
        batch_size=64, shuffle=False, **kwargs)

    # a small clean batch for defensive purpose
    clean_set_loader = torch.utils.data.DataLoader(
        kettle.validset,
        batch_size=64, shuffle=True, **kwargs)
    
    feats_inspection, class_indices_inspection = get_features(inspection_split_loader, victim.model)
    feats_clean, class_indices_clean = get_features(clean_set_loader, victim.model)

    feats_inspection = np.array(feats_inspection)
    class_indices_inspection = np.array(class_indices_inspection)

    feats_clean = np.array(feats_clean)
    class_indices_clean = np.array(class_indices_clean)

    # For MobileNet-V2:
    # from sklearn.decomposition import PCA
    # projector = PCA(n_components=128)
    # feats_inspection = projector.fit_transform(feats_inspection)
    # feats_clean = projector.fit_transform(feats_clean)



    scan = SCAn()

    # fit the clean distribution with the small clean split at hand
    gb_model = scan.build_global_model(feats_clean, class_indices_clean, num_classes)

    size_inspection_set = len(feats_inspection)

    feats_all = np.concatenate([feats_inspection, feats_clean])
    class_indices_all = np.concatenate([class_indices_inspection, class_indices_clean])

    # use the global model to divide samples
    lc_model = scan.build_local_model(feats_all, class_indices_all, gb_model, num_classes)
    
    # statistic test for the existence of "two clusters"
    score = scan.calc_final_score(lc_model)
    threshold = np.e

    suspicious_indices = []

    for target_class in range(num_classes):

        print('[class-%d] outlier_score = %f' % (target_class, score[target_class]) )

        if score[target_class] <= threshold: continue

        tar_label = (class_indices_all == target_class)
        all_label = np.arange(len(class_indices_all))
        tar = all_label[tar_label]

        cluster_0_indices = []
        cluster_1_indices = []

        cluster_0_clean = []
        cluster_1_clean = []

        for index, i in enumerate(lc_model['subg'][target_class]):
            if i == 1:
                if tar[index] > size_inspection_set:
                    cluster_1_clean.append(tar[index])
                else:
                    cluster_1_indices.append(tar[index])
            else:
                if tar[index] > size_inspection_set:
                    cluster_0_clean.append(tar[index])
                else:
                    cluster_0_indices.append(tar[index])


        # decide which cluster is the poison cluster, according to clean samples' distribution
        if len(cluster_0_clean) < len(cluster_1_clean): # if most clean samples are in cluster 1
            suspicious_indices += cluster_0_indices
        else:
            suspicious_indices += cluster_1_indices

    clean_idcs = list(set(range(len(kettle.trainset))) - set(suspicious_indices))
    return clean_idcs
    
import numpy as np
import torch
from tqdm import tqdm

EPS = 1e-5


class SCAn:
    def __init__(self):
        pass

    def calc_final_score(self, lc_model=None):
        if lc_model is None:
            lc_model = self.lc_model
        sts = lc_model['sts']
        y = sts[:, 1]
        ai = self.calc_anomaly_index(y / np.max(y))
        return ai

    def calc_anomaly_index(self, a):
        ma = np.median(a)
        b = abs(a - ma)
        mm = np.median(b) * 1.4826
        index = b / mm
        return index

    def build_global_model(self, reprs, labels, n_classes):
        N = reprs.shape[0]  # num_samples
        M = reprs.shape[1]  # len_features
        L = n_classes

        mean_a = np.mean(reprs, axis=0)
        X = reprs - mean_a

        cnt_L = np.zeros(L)
        mean_f = np.zeros([L, M])
        for k in range(L):
            idx = (labels == k)
            cnt_L[k] = np.sum(idx)
            mean_f[k] = np.mean(X[idx], axis=0)

        u = np.zeros([N, M])
        e = np.zeros([N, M])
        for i in range(N):
            k = labels[i]
            u[i] = mean_f[k]  # class-mean
            e[i] = X[i] - u[i]  # sample-variantion
        Su = np.cov(np.transpose(u))
        Se = np.cov(np.transpose(e))

        # EM
        dist_Su = 1e5
        dist_Se = 1e5
        n_iters = 0
        while (dist_Su + dist_Se > EPS) and (n_iters < 100):
            n_iters += 1
            last_Su = Su
            last_Se = Se

            F = np.linalg.pinv(Se)
            SuF = np.matmul(Su, F)

            G_set = list()
            for k in range(L):
                G = -np.linalg.pinv(cnt_L[k] * Su + Se)
                G = np.matmul(G, SuF)
                G_set.append(G)

            u_m = np.zeros([L, M])
            e = np.zeros([N, M])
            u = np.zeros([N, M])

            for i in range(N):
                vec = X[i]
                k = labels[i]
                G = G_set[k]
                dd = np.matmul(np.matmul(Se, G), np.transpose(vec))
                u_m[k] = u_m[k] - np.transpose(dd)

            for i in range(N):
                vec = X[i]
                k = labels[i]
                e[i] = vec - u_m[k]
                u[i] = u_m[k]

            # max-step
            Su = np.cov(np.transpose(u))
            Se = np.cov(np.transpose(e))

            dif_Su = Su - last_Su
            dif_Se = Se - last_Se

            dist_Su = np.linalg.norm(dif_Su)
            dist_Se = np.linalg.norm(dif_Se)
            # print(dist_Su,dist_Se)

        gb_model = dict()
        gb_model['Su'] = Su
        gb_model['Se'] = Se
        gb_model['mean'] = mean_f
        self.gb_model = gb_model
        return gb_model

    def build_local_model(self, reprs, labels, gb_model, n_classes):
        Su = gb_model['Su']
        Se = gb_model['Se']

        F = np.linalg.pinv(Se)
        N = reprs.shape[0]
        M = reprs.shape[1]
        L = n_classes

        mean_a = np.mean(reprs, axis=0)
        X = reprs - mean_a

        class_score = np.zeros([L, 3])
        u1 = np.zeros([L, M])
        u2 = np.zeros([L, M])
        split_rst = list()

        for k in range(L):
            selected_idx = (labels == k)
            cX = X[selected_idx]
            subg, i_u1, i_u2 = self.find_split(cX, F)
            # print("subg",subg)

            i_sc = self.calc_test(cX, Su, Se, F, subg, i_u1, i_u2)
            split_rst.append(subg)
            u1[k] = i_u1
            u2[k] = i_u2
            class_score[k] = [k, i_sc.squeeze(), np.sum(selected_idx)]

        lc_model = dict()
        lc_model['sts'] = class_score
        lc_model['mu1'] = u1
        lc_model['mu2'] = u2
        lc_model['subg'] = split_rst

        self.lc_model = lc_model
        return lc_model

    def find_split(self, X, F):
        N = X.shape[0]
        M = X.shape[1]
        subg = np.random.rand(N)

        if (N == 1):
            subg[0] = 0
            return (subg, X.copy(), X.copy())

        if np.sum(subg >= 0.5) == 0:
            subg[0] = 1
        if np.sum(subg < 0.5) == 0:
            subg[0] = 0
        last_z1 = -np.ones(N)

        # EM
        steps = 0
        while (np.linalg.norm(subg - last_z1) > EPS) and (np.linalg.norm((1 - subg) - last_z1) > EPS) and (steps < 100):
            steps += 1
            last_z1 = subg.copy()

            # max-step
            # calc u1 and u2
            idx1 = (subg >= 0.5)
            idx2 = (subg < 0.5)
            if (np.sum(idx1) == 0) or (np.sum(idx2) == 0):
                break
            if np.sum(idx1) == 1:
                u1 = X[idx1]
            else:
                u1 = np.mean(X[idx1], axis=0)
            if np.sum(idx2) == 1:
                u2 = X[idx2]
            else:
                u2 = np.mean(X[idx2], axis=0)

            bias = np.matmul(np.matmul(u1, F), np.transpose(u1)) - np.matmul(np.matmul(u2, F), np.transpose(u2))
            e2 = u1 - u2  # (64,1)
            for i in range(N):
                e1 = X[i]
                delta = np.matmul(np.matmul(e1, F), np.transpose(e2))
                if bias - 2 * delta < 0:
                    subg[i] = 1
                else:
                    subg[i] = 0

        return (subg, u1, u2)

    def calc_test(self, X, Su, Se, F, subg, u1, u2):
        N = X.shape[0]
        M = X.shape[1]

        G = -np.linalg.pinv(N * Su + Se)
        mu = np.zeros([1, M])
        SeG = np.matmul(Se,G)
        for i in range(N):
            vec = X[i]
            dd = np.matmul(SeG, np.transpose(vec))
            mu = mu - dd

        b1 = np.matmul(np.matmul(mu, F), np.transpose(mu)) - np.matmul(np.matmul(u1, F), np.transpose(u1))
        b2 = np.matmul(np.matmul(mu, F), np.transpose(mu)) - np.matmul(np.matmul(u2, F), np.transpose(u2))
        n1 = np.sum(subg >= 0.5)
        n2 = N - n1
        sc = n1 * b1 + n2 * b2

        for i in range(N):
            e1 = X[i]
            if subg[i] >= 0.5:
                e2 = mu - u1
            else:
                e2 = mu - u2
            sc -= 2 * np.matmul(np.matmul(e1, F), np.transpose(e2))

        return sc / N

def get_features(data_loader, model):

    class_indices = []
    feats = []

    layer_cake = list(model.children())
    feature_extractor = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())
    feature_extractor.eval()
    with torch.no_grad():
        for i, (ins_data, ins_target, _) in enumerate(tqdm(data_loader)):
            ins_data = ins_data.cuda()
            x_features = feature_extractor(ins_data)

            this_batch_size = len(ins_target)
            for bid in range(this_batch_size):
                feats.append(x_features[bid].cpu().numpy())
                class_indices.append(ins_target[bid].cpu().numpy())

    return feats, class_indices




