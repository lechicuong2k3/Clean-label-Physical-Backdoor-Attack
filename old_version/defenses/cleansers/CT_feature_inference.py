from scan import SCAn
from tqdm import tqdm, trange
import numpy as np
import torch
import os

def get_features(dataset, model, model_name, save_path=None):
    model.eval()
    class_indices = []
    preds_list = []
    feats = []
    
    for iex in trange(len(dataset)):
        sample = dataset[iex]
        x_batch = sample[0].unsqueeze(0).cuda()
        y_batch = sample[1]
        class_indices.append(y_batch)
        with torch.no_grad():
            if model_name == 'VGG16' or model_name == 'DenseNet121':
                inps,outs = [],[]
                def layer_hook(module, inp, out):
                    outs.append(out.data)
                hook = model.features.register_forward_hook(layer_hook)
                outputs = model(x_batch)
                batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
                hook.remove()
                
            elif model_name == 'ResNet50':
                inps,outs = [],[]
                def layer_hook(module, inp, out):
                    outs.append(out.data)
                hook = model.avgpool.register_forward_hook(layer_hook)
                outputs = model(x_batch)
                batch_grads = outs[0].view(outs[0].size(0), -1).squeeze(0)
                hook.remove()
        
        pred = torch.argmax(outputs, dim=1).cpu().numpy()[0]
        preds_list.append(pred)

        feats.append(batch_grads.detach().cpu().numpy())
    
    if save_path is not None:
        feature_path = os.path.join(save_path, 'features.npy')
        indices_path = os.path.join(save_path, 'indices.npy')
        np.save(feature_path, feats)
        np.save(indices_path, class_indices)
        
    return feats, class_indices, preds_list

def cleanser(inspection_set, clean_set, model, num_classes, args):
    feats_inspection, class_indices_inspection, preds_inspection = get_features(inspection_set, model, args.model, args.save_rep_path)
    feats_clean, class_indices_clean, _ = get_features(clean_set, model, args.model, args.save_rep_path)

    feats_inspection = np.array(feats_inspection)
    class_indices_inspection = np.array(class_indices_inspection)
    preds_inspection = np.array(preds_inspection)

    feats_clean = np.array(feats_clean)
    class_indices_clean = np.array(class_indices_clean)


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
    threshold = np.exp(2)

    suspicious_indices = []
    num_samples = len(inspection_set)

    for target_class in range(num_classes):
        print('[class-%d] outlier score = %f.' % (target_class, score[target_class]))
        if score[target_class] <= threshold: continue # omit classes that pass the single-cluster test
        for i in range(num_samples):
            if class_indices_inspection[i] == target_class and preds_inspection[i] == target_class:
                suspicious_indices.append(i)


    return suspicious_indices