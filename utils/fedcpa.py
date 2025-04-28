import torch 
import numpy as np
import scipy.stats

def get_weight(model_or_dict):
    weight_dict = {}
    if isinstance(model_or_dict, dict):  
        weight_dict = model_or_dict
    else:  
        for name, param in model_or_dict.named_parameters():  
            if param.requires_grad:
                weight_dict[name] = param
    return weight_dict

def get_update_static(w_locals, w_glob):
    model_weight_list = []
    global_weight = torch.cat([torch.flatten(val) for val in w_glob.values()])

    for w_local in w_locals:
        local_weight = torch.cat([torch.flatten(w_local[name]) for name in w_glob.keys()])
        model_update = local_weight - global_weight
        model_weight_list.append(model_update)

    model_weight_cat = torch.stack(model_weight_list, dim=0)
    model_std, model_mean = torch.std_mean(model_weight_cat, unbiased=False, dim=0)
    return model_mean, model_std, model_weight_cat, global_weight

def compute_similarity_score(w1, w2, top1, top2, bottom1, bottom2):
    """Compute similarity"""
    # top-k similarity
    topk_intersection = set(top1.cpu().tolist()) & set(top2.cpu().tolist())
    if len(topk_intersection) >= 2:
        topk_intersection = list(topk_intersection)
        topk_corr = (scipy.stats.pearsonr(w1[topk_intersection].cpu().numpy(),
                                            w2[topk_intersection].cpu().numpy())[0] + 1) / 2
    else:
        topk_corr = 0
    topk_jaccard = len(topk_intersection) / (len(top1) + len(top2) - len(topk_intersection))

    # bottom-k
    bottomk_intersection = set(bottom1.cpu().tolist()) & set(bottom2.cpu().tolist())
    if len(bottomk_intersection) >= 2:
        bottomk_intersection = list(bottomk_intersection)
        bottomk_corr = (scipy.stats.pearsonr(w1[bottomk_intersection].cpu().numpy(),
                                                w2[bottomk_intersection].cpu().numpy())[0] + 1) / 2
    else:
        bottomk_corr = 0
    bottomk_jaccard = len(bottomk_intersection) / (len(bottom1) + len(bottom2) - len(bottomk_intersection))

    return (topk_corr + bottomk_corr) / 2 + (topk_jaccard + bottomk_jaccard) / 2

def get_foolsgold_score(total_score, grads, global_weight):
    """计算FoolsGold分数"""
    n_clients = total_score.shape[0]
    norm_score = (total_score - np.min(total_score)) / (np.max(total_score) - np.min(total_score))
    norm_score = np.clip(norm_score, 0.01, 0.99)

    # Logit transfer
    wv = np.log(norm_score / (1 - norm_score)) + 0.5
    wv = np.clip(wv, 0, 1)

    model_weight_list = []
    for i in range(n_clients):
        if wv[i] > 0:
            current_weight = global_weight + wv[i] * grads[i]
            model_weight_list.append(current_weight)

    if model_weight_list:
        fools_gold_weight = torch.stack(model_weight_list).mean(0)
    else:
        fools_gold_weight = global_weight

    return fools_gold_weight, wv

def is_constant(array):
    return np.all(array == array[0])

def cpa_defense(w_locals, w_glob, prev_w_glob, prev_prev_w_glob, net_glob):
    """CPA"""
    n_clients = len(w_locals)


    middle_layer_dims = set()
    for name, param in net_glob.named_parameters():
        if param.requires_grad:  
            middle_layer_dims.add(param.dim())
    if 1 in middle_layer_dims:
        middle_layer_dims.remove(1)

    global_critical_dict = {}
    for name, val in prev_w_glob.items():
        if val.dim() in middle_layer_dims:
            critical_weight = torch.abs((prev_w_glob[name] - prev_prev_w_glob[name]) * prev_w_glob[name])
            global_critical_dict[name] = critical_weight

    global_w_stacked = torch.cat([val.view(-1) for val in global_critical_dict.values()]).view(1, -1)
    global_topk_indices = torch.abs(global_w_stacked).topk(int(global_w_stacked.shape[1] * 0.01)).indices
    global_bottomk_indices = torch.abs(global_w_stacked).topk(int(global_w_stacked.shape[1] * 0.01),
                                                                largest=False).indices


    importance_values = []
    for w_local in w_locals:
        critical_dict = {}
        for name, val in w_local.items():
            if val.dim() in middle_layer_dims:
                critical_weight = torch.abs((val - prev_w_glob[name]) * val)
                critical_dict[name] = critical_weight
        local_weights = torch.cat([val.view(-1) for val in get_weight(critical_dict).values()]).view(1, -1)
        importance_values.append(local_weights)

    w_stacked = torch.stack(importance_values, dim=0)
    w_stacked = w_stacked.squeeze(1)

    # top-k and bottom-k indices
    local_topk_indices = torch.abs(w_stacked).topk(int(w_stacked.shape[1] * 0.01)).indices
    local_bottomk_indices = torch.abs(w_stacked).topk(int(w_stacked.shape[1] * 0.01), largest=False).indices

    # similarity score
    pairwise_score = np.zeros((n_clients, n_clients))
    global_score = np.zeros(n_clients)

    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                pairwise_score[i][j] = 1 
                continue
            elif i < j:
                continue
            topk_intersection = list(
                set(local_topk_indices[i].view(-1).tolist()) & set(local_topk_indices[j].view(-1).tolist())
            )
            if len(topk_intersection) >= 2:
                array_1 = w_stacked[i, topk_intersection].cpu().numpy()
                array_2 = w_stacked[j, topk_intersection].cpu().numpy()
                if is_constant(array_1) or is_constant(array_2):
                    topk_corr_dist = 0  
                else:
                    topk_corr_dist = ((scipy.stats.pearsonr(array_1, array_2)[0]) + 1) / 2
            else:
                topk_corr_dist = 0  

            topk_jaccard_dist = len(topk_intersection) / (
                    len(local_topk_indices[i].view(-1)) + len(local_topk_indices[j].view(-1)) - len(
                topk_intersection)
            )

            bottomk_intersection = list(
                set(local_bottomk_indices[i].view(-1).tolist()) & set(local_bottomk_indices[j].view(-1).tolist())
            )
            if len(bottomk_intersection) >= 2:
                array_1 = w_stacked[i, bottomk_intersection].cpu().numpy()
                array_2 = w_stacked[j, bottomk_intersection].cpu().numpy()
                if is_constant(array_1) or is_constant(array_2):
                    bottomk_corr_dist = 0  
                    bottomk_corr_dist = ((scipy.stats.pearsonr(array_1, array_2)[0]) + 1) / 2
            else:
                bottomk_corr_dist = 0 

            bottomk_jaccard_dist = len(bottomk_intersection) / (
                    len(local_bottomk_indices[i].view(-1)) + len(local_bottomk_indices[j].view(-1)) - len(
                bottomk_intersection)
            )

            pairwise_score[i][j] = (topk_corr_dist + bottomk_corr_dist) / 2 + (
                    topk_jaccard_dist + bottomk_jaccard_dist) / 2
            pairwise_score[j][i] = pairwise_score[i][j]  

    for i in range(n_clients):
        topk_intersection = list(
            set(local_topk_indices[i].tolist()) & set(global_topk_indices[0].tolist())
        )
        if len(topk_intersection) >= 2:
            array_1 = w_stacked[i, topk_intersection].cpu().numpy()
            array_2 = global_w_stacked[0, topk_intersection].cpu().numpy()
            if is_constant(array_1) or is_constant(array_2):
                topk_corr_dist = 0  
            else:
                topk_corr_dist = ((scipy.stats.pearsonr(array_1, array_2)[0]) + 1) / 2
        else:
            topk_corr_dist = 0  
        topk_jaccard_dist = len(topk_intersection) / (
                len(local_topk_indices[i]) + len(global_topk_indices[0]) - len(topk_intersection)
        )

        bottomk_intersection = list(
            set(local_bottomk_indices[i].tolist()) & set(global_bottomk_indices[0].tolist())
        )
        if len(bottomk_intersection) >= 2:
            array_1 = w_stacked[i, bottomk_intersection].cpu().numpy()
            array_2 = global_w_stacked[0, bottomk_intersection].cpu().numpy()
            if is_constant(array_1) or is_constant(array_2):
                bottomk_corr_dist = 0  
            else:
                bottomk_corr_dist = ((scipy.stats.pearsonr(array_1, array_2)[0]) + 1) / 2
        else:
            bottomk_corr_dist = 0  

        bottomk_jaccard_dist = len(bottomk_intersection) / (
                len(local_bottomk_indices[i]) + len(global_bottomk_indices[0]) - len(bottomk_intersection)
        )

        global_score[i] = (topk_corr_dist + bottomk_corr_dist) / 2 + (
                topk_jaccard_dist + bottomk_jaccard_dist) / 2

    total_score = np.mean(pairwise_score, axis=1) + global_score
    update_mean, update_std, update_cat, global_weight = get_update_static(w_locals, w_glob)
    model_weight_foolsgold, wv = get_foolsgold_score(total_score, update_cat, global_weight)

    current_idx = 0

    for key, tensor in w_locals[-1].items():  
        if isinstance(tensor, torch.Tensor):
            length = tensor.numel()  
            if current_idx + length <= len(model_weight_foolsgold):
                sliced_data = model_weight_foolsgold[current_idx:current_idx + length]
                w_glob[key] = sliced_data.reshape(tensor.shape)
                current_idx += length
            else:
                raise ValueError(f"Not enough data to reconstruct tensor for {key}")
        else:
            raise TypeError(f"Expected tensor for {key}, got {type(tensor)}")
    return w_glob