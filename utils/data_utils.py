import torch
import numpy as np
import copy
from torch.utils.data import DataLoader

def prepare_auxiliary_data(dataset, num_classes):
    """准备辅助数据和测试数据的索引
    Args:
        dataset: 完整数据集
        num_classes: 类别数量
    Returns:
        auxiliary_indices: 辅助数据索引列表
        test_indices: 测试数据索引列表
    """
    # 获取数据集的标签
    if hasattr(dataset, 'targets'):
        if isinstance(dataset.targets, list):
            labels = torch.tensor(dataset.targets)
        else:
            labels = dataset.targets
    else:
        labels = []
        for _, target in dataset:
            labels.append(target)
        labels = torch.tensor(labels)
    
    total_size = len(labels)
    aux_size = int(total_size * 0.1)  # 10%作为辅助数据
    
    # 按类别均匀采样辅助数据
    auxiliary_indices = []
    per_class = aux_size // num_classes
    
    class_indices_dict = {}  # 存储每个类别的样本索引
    
    for c in range(num_classes):
        class_indices = (labels == c).nonzero().squeeze().tolist()
        if isinstance(class_indices, int):
            class_indices = [class_indices]
        if len(class_indices) > 0:
            selected = np.random.choice(class_indices, 
                                     min(per_class, len(class_indices)), 
                                     replace=False).tolist()
            auxiliary_indices.extend(selected)
            class_indices_dict[c] = selected
    
    # 剩余数据作为测试集
    test_indices = list(set(range(total_size)) - set(auxiliary_indices))
    
    # 只返回两个值
    return auxiliary_indices, test_indices

def prepare_features(w_locals):
    """从本地模型更新中提取特征
    Args:
        w_locals: 本地模型更新列表
    Returns:
        features: 特征矩阵 [num_clients, feature_dim]
    """
    features = []
    
    for w in w_locals:
        # 直接获取最后一层的权重
        last_layer_name = list(w.keys())[-2]  # -2是权重,-1是偏置
        feature = w[last_layer_name].view(-1)
        features.append(feature)
    
    features = torch.stack(features)
    return features

def update_reputation(net_glob, w_locals, dataset_test, threshold=0.5):
    """更新客户端信誉矩阵
    Args:
        w_locals: 本地模型更新列表
        dataset_test: 测试数据集
        threshold: 性能阈值
    Returns:
        reputation: 信誉矩阵 [1, num_clients]
    """
    num_clients = len(w_locals)
    reputation = np.zeros((1, num_clients))
    
    # 评估每个客户端模型的性能
    test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)
    
    for i, w in enumerate(w_locals):
        model = copy.deepcopy(net_glob)
        model.load_state_dict(w)
        model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        acc = correct / total
        reputation[0, i] = 1 if acc > threshold else 0
        
    return reputation

def aggregate_updates(selected_updates):
    """聚合选中的客户端更新
    Args:
        selected_updates: 选中的客户端更新列表
    Returns:
        w_glob: 聚合后的全局模型参数
    """
    w_avg = copy.deepcopy(selected_updates[0])
    
    for k in w_avg.keys():
        for i in range(1, len(selected_updates)):
            w_avg[k] += selected_updates[i][k]
        w_avg[k] = torch.div(w_avg[k], len(selected_updates))
        
    return w_avg 