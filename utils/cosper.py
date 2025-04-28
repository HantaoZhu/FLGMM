import copy
import torch
import numpy as np
from models.Fed import FedAvg_0

def cosper_defense(w_locals, w_glob, args, prev_v_locals=None, prev_s_locals=None, momentum_w=None, momentum_v=None):
    """
    CosPer 
    Params:
    w_locals: list of local global model updates
    w_glob: current global model
    args: configuration parameters
    prev_v_locals: local personalized models from previous round (if available)
    prev_s_locals: local aggregation weights from previous round (if available)
    momentum_w: momentum term for global model (if available)
    momentum_v: momentum term for local models (if available)

    Returns:
    w_glob: updated global model
    v_locals: updated list of local personalized models
    s_locals: updated list of local aggregation weights
    momentum_w: updated global model momentum term
    momentum_v: updated local model momentum terms
    excluded_indices: excluded client indices (malicious clients)
    h_locals: list of personalized models
    """
    
    n_clients = len(w_locals)
    beta = args.momentum if hasattr(args, 'momentum') else 0.9  # momentum parameter
    gamma = args.gamma if hasattr(args, 'gamma') else 0.8  
    
    if prev_v_locals is None:
        v_locals = copy.deepcopy(w_locals) 
    else:
        v_locals = prev_v_locals
        
    if prev_s_locals is None:
        s_locals = [0.5] * n_clients  
    else:
        s_locals = prev_s_locals
        
    if momentum_w is None:
        momentum_w = [{}] * n_clients  
    
    if momentum_v is None:
        momentum_v = [{}] * n_clients  
    
    gradients_w = []
    gradients_v = []
    
    for i in range(n_clients):
        grad_w = {}
        for key in w_glob.keys():
            if key in w_locals[i]:
                grad_w[key] = w_locals[i][key] - w_glob[key]
        
        if not momentum_w[i]:  
            momentum_w[i] = copy.deepcopy(grad_w)
        else:
            for key in grad_w.keys():
                if key in momentum_w[i]:
                    momentum_w[i][key] = beta * momentum_w[i][key] + (1 - beta) * grad_w[key]
                else:
                    momentum_w[i][key] = grad_w[key]
        
        gradients_w.append(grad_w)
        
        if isinstance(v_locals[i], dict):
            grad_v = {}
            for key in w_glob.keys():
                if key in v_locals[i] and key in w_locals[i]:
                    h_key = (1 - s_locals[i]) * v_locals[i][key] + s_locals[i] * w_locals[i][key]
                    grad_v[key] = h_key - v_locals[i][key]
            
            if not momentum_v[i]: 
                momentum_v[i] = copy.deepcopy(grad_v)
            else:
                for key in grad_v.keys():
                    if key in momentum_v[i]:
                        momentum_v[i][key] = beta * momentum_v[i][key] + (1 - beta) * grad_v[key]
                    else:
                        momentum_v[i][key] = grad_v[key]
            
            gradients_v.append(grad_v)
    
    cos_similarities = []
    excluded_indices = []
    
    for i in range(n_clients):
        try:
            m_w_flat = torch.cat([momentum_w[i][key].flatten() for key in momentum_w[i].keys()])
            m_v_flat = torch.cat([momentum_v[i][key].flatten() for key in momentum_v[i].keys()])
            cos_sim = torch.nn.functional.cosine_similarity(m_w_flat.unsqueeze(0), m_v_flat.unsqueeze(0))[0].item()
            s_locals[i] = gamma * s_locals[i] + (1 - gamma) * cos_sim
            cos_similarities.append(cos_sim)
        except:
            cos_similarities.append(0)
            s_locals[i] = 0.5  
    
    mean_sim = np.mean(cos_similarities)
    std_sim = np.std(cos_similarities)
    # threshold
    threshold = mean_sim - 2 * std_sim
    
    for i in range(n_clients):
        if cos_similarities[i] < threshold:
            excluded_indices.append(i)
    
    for i in range(n_clients):
        for key in v_locals[i].keys():
            if key in momentum_v[i]:
                v_locals[i][key] = v_locals[i][key] + args.lr * momentum_v[i][key]
    
    h_locals = []
    for i in range(n_clients):
        h_local = {}
        for key in w_glob.keys():
            if key in v_locals[i] and key in w_locals[i]:
                h_local[key] = (1 - s_locals[i]) * v_locals[i][key] + s_locals[i] * w_locals[i][key]
        h_locals.append(h_local)
    
    trusted_w_locals = [w_locals[i] for i in range(n_clients) if i not in excluded_indices]
    
    if len(trusted_w_locals) > 0:
        w_trusted = FedAvg_0(trusted_w_locals)
        for key in w_glob.keys():
            w_glob[key] = w_trusted[key]
    
    print(f"Excluded clients: {excluded_indices}")
    print(f"Aggregation weights: {[round(s, 3) for s in s_locals]}")
    print(f"Average cosine similarity: {round(mean_sim, 3)}")
    return w_glob, v_locals, s_locals, momentum_w, momentum_v, excluded_indices, h_locals