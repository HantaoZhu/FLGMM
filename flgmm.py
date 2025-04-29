import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
from utils.geo_med import geometric_median_list_of_array
from torch import nn
from models.krum import krum
from models.median import simple_median
from models.trimmed_mean import trimmed_mean
from utils.cosper import cosper_defense
from utils.fedcpa import cpa_defense
from utils import tool
import matplotlib
from utils.w_noise import  add_noise_to_layers_weights, random_weights
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import random
from torchvision import datasets, transforms
import torch
import seaborn as sns
from utils.sampling import mnist_iid, cifar_iid, fmnist_iid, add_noise_to_client_2, add_gaussian_noise
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, FashionMNISTCNN
from models.Fed import FedAvg_0
from models.test import test_img
from sklearn.mixture import GaussianMixture

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def decompose_normal_distributions(data, n_components=2):
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(data.reshape(-1, 1))
    labels = gmm.predict(data.reshape(-1, 1))
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_cluster_index = unique_labels[np.argmax(counts)]
    max_cluster_data = data[labels == max_cluster_index]
    bounds = (max_cluster_data.min(), max_cluster_data.max())
    return max_cluster_data, bounds, gmm.means_, gmm.covariances_, gmm.weights_

def calculate_std_dev(client_distances, global_mean):
    squared_diffs = [(x - global_mean) ** 2 for x in client_distances]
    variance = sum(squared_diffs) / len(client_distances)
    std_dev = variance ** 0.5
    return std_dev

def plot_control_chart(client_id, client_means, distances_matrix, save_dir, args):
    """
    SPC-based anomaly detection algorithm.
    """
    ano = []
    distances = [distance for client_list in distances_matrix for distance in client_list]
    std = np.std(distances)
    mean = np.mean(distances)
    # the control limit to select clients, in our study LCL is not used
    UCL = mean + args.L *std
    LCL = mean - args.L *std
    
    plt.figure(figsize=(10, 6))
    plt.plot(client_id, client_means, marker='o', linestyle='-', color='blue', label='Average Distance')
    for idx, client_mean in zip(client_id, client_means):
        if client_mean > UCL:
            ano.append(idx)
            plt.plot(idx, client_mean, marker='o', color='red')

    plt.axhline(UCL, color='red', linestyle='--', label='UCL')
    plt.title('Control Chart for All Clients')
    plt.xlabel('Client ID')
    plt.ylabel('Average Distance')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'control_chart_all_clients.png'))
    plt.close()

    return ano, UCL, LCL

def calculate_accuracy(detected_noisy_clients, actual_noisy_clients):
    """
    Calculate the defense metrix of the anomaly detection algorithm.
    """

    detected_set = set(detected_noisy_clients)
    actual_set = set(actual_noisy_clients)

    correct_detections = detected_set.intersection(actual_set)

    # Calculate recall
    if len(actual_set) > 0:
        R = len(correct_detections) / len(actual_set)
    else:
        R = 0.0 

    # Calculate precision
    if len(detected_set) > 0:
        P = len(correct_detections) / len(detected_set)
    else:
        P = 0.0  

    return R,P

def unbiased_selection(p):
    idxs = []
    while(len(idxs) < 2):
        for i in range(len(p)):
            rand = random.random()
            if rand < p[i]:
                idxs.append(i)
    return idxs

def euclidean_distance(local_weights, global_weights):
        distance = 0
        for key in global_weights.keys():
            distance += torch.pow(local_weights[key] - global_weights[key], 2).sum()
        distance = torch.sqrt(distance)
        return distance.item()

if __name__ == '__main__':
    seed = 5
    setup_seed(seed)
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        dict_users = mnist_iid(dataset_train, args.num_users)


    elif args.dataset == 'cifar10':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        dict_users = cifar_iid(dataset_train, args.num_users)
        
    elif args.dataset == 'fmnist':
        dataset_train = datasets.FashionMNIST('data', train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))
                                         ]))

        dataset_test = datasets.FashionMNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
        dict_users = FashionMnist_iid(dataset_train, args.num_users)

    # Initialize nn models
    if args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'cifar10':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.dataset == 'fmnist' and args.model == 'cnn':
        net_glob = FashionMNISTCNN(args=args).to(args.device)
    elif args.model == 'mlp':
        from models.Nets import MLP

        img_size = dataset_train[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)

    else:
        print('Error: unrecognized model')
        exit()

    print(net_glob)
    net_glob.train()

    # copy global weights
    w_glob = net_glob.state_dict()

    if args.method == 'fedcc':
        save_dir = f"./final_{args.method}_{args.dataset}_{args.attack_pattern}_{args.peer_round}_sa_L_{args.L}"
    else:
        save_dir = f"./final_{args.method}_{args.dataset}_{args.attack_pattern}_{args.peer_round}"
    if not args.iid:
        save_dir = f"./final_{args.method}_{args.dataset}_{args.attack_pattern}_{args.peer_round}_noniid"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    global_acctotal = []
    global_losses = []
    control_chart_params = {}
    excluded_clients = []
    distances_matrix = [[] for _ in range(args.num_users)]
    normal_dis = [[] for _ in range(args.num_users)]
    standard_dis = []
    std = []
    client_means = []
    r = []
    p = []
    o = 1
    f1 = []
    ratio = []
    seduce = []

    if args.attack:
        noisy_clients = np.random.choice(range(args.num_users), int(args.peer_round*args.num_users), replace=False)
        print("noisy_clients:", noisy_clients)
        
    for iters in range(args.epochs):
        normal_id = []
        w_locals = []
        local_grads = []
        loss_locals = []
        loss_tests = []
        normal_clients_dis = []
        normal_clients_dis_mean = []
        normal_std = []
        excluded = []
        distances_matrix_this_round = []
        noisy_this_round = []
        attacker = []
        grads = []
        benign_grad = []
        ra = 0 
        byz_grad = []

        
        idxs_users = [i for i in range(args.num_users)]
        for idx in idxs_users:
            if args.attack_pattern == 'lf' and np.random.rand() < args.attack_pos:
                if idx in noisy_clients:
                    fakedata = copy.deepcopy(dataset_train)
                    add_noise_to_client_2(fakedata,dict_users,idx)
                    local = LocalUpdate(args=args, dataset=fakedata, idxs=dict_users[idx])
                    w, loss, grad = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    local_model = copy.deepcopy(net_glob).to(args.device)
                    local_model.load_state_dict(w)
                    w_locals.append(copy.deepcopy(w))
                    # Attackers in the current round
                    noisy_this_round.append(idx)
                    continue
           
            if args.attack_pattern == 'w_noise' and np.random.rand() < args.attack_pos:
                if idx in noisy_clients:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                    w, loss, grad= local.train(net=copy.deepcopy(net_glob).to(args.device))
                    local_model = copy.deepcopy(net_glob).to(args.device)
                    local_model.load_state_dict(w)
                    fake_w = add_noise_to_layers_weights(local_model,args)
                    w_locals.append(copy.deepcopy(fake_w))
                    noisy_this_round.append(idx)
                    continue

            if args.attack_pattern == 'gn' and np.random.rand() < args.attack_pos:
                if idx in noisy_clients:
                    dataset_train_2 = add_gaussian_noise(dataset_train,dict_users,idx)
                    local = LocalUpdate(args=args, dataset=dataset_train_2, idxs=dict_users[idx])
                    w, loss, grad= local.train(net=copy.deepcopy(net_glob).to(args.device))
                    local_model = copy.deepcopy(net_glob).to(args.device)
                    local_model.load_state_dict(w)
                    w_locals.append(copy.deepcopy(w))
                    noisy_this_round.append(idx)
                    continue

            if args.attack_pattern == 'rw' and np.random.rand() < args.attack_pos:
                if idx in noisy_clients:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                    w, loss, grad= local.train(net=copy.deepcopy(net_glob).to(args.device))
                    local_model = copy.deepcopy(net_glob).to(args.device)
                    local_model.load_state_dict(w)
                    fake_w = random_weights(args,w )
                    w_locals.append(copy.deepcopy(fake_w))
                    noisy_this_round.append(idx)
                    continue

            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss, grad= local.train(net=copy.deepcopy(net_glob).to(args.device))


            local_model = copy.deepcopy(net_glob).to(args.device)
            local_model.load_state_dict(w)
            w_locals.append(copy.deepcopy(w))

            loss_test = test_img(local_model, dataset_test, args)[1]
            loss_locals.append(copy.deepcopy(loss))
            loss_tests.append(loss_test)

        print(len(noisy_this_round))



        if args.method == 'flgmm':
            w_glob = FedAvg_0(w_locals)
            for idx, w_local in enumerate(w_locals):
                distance = euclidean_distance(w_local, w_glob)
                distances_matrix_this_round.append(distance)

            distances = distances_matrix_this_round
            distances_array = np.array(distances).reshape(-1, 1)

            # Utilize GMM
            largest_cluster_data, bounds, means, covariances, weights = decompose_normal_distributions(distances_array)
            print(len(largest_cluster_data))
            mean = np.mean(largest_cluster_data)
            std = np.std(largest_cluster_data)

            for idx in range(args.num_users):
                distances_matrix[idx].append((distances[idx]-mean)/std)
                if distances_matrix_this_round[idx] in largest_cluster_data and iters < args.ccepochs:
                    normal_id.append(idx)

            # Plot GMM in round 10
            if iters == 10:
                flat_distances3 = distances_matrix_this_round
                plt.figure(figsize=(10, 6))
                plt.hist(flat_distances3, bins=50, color='grey', edgecolor='black', density=True, alpha=0.6)
                x = np.linspace(min(distances_array), max(distances_array), 1000)
                pdf_1 = weights[0] * (1 / (np.sqrt(2 * np.pi * covariances[0]))) * np.exp(
                    -0.5 * ((x - means[0]) ** 2) / covariances[0])
                pdf_2 = weights[1] * (1 / (np.sqrt(2 * np.pi * covariances[1]))) * np.exp(
                    -0.5 * ((x - means[1]) ** 2) / covariances[1])
                pdf_1 = pdf_1.reshape(-1)
                pdf_2 = pdf_2.reshape(-1)
                plt.plot(x, pdf_1, color='red', linestyle='-.', label='GMM Component 1')
                plt.plot(x, pdf_2, color='green', linestyle='-.', label='GMM Component 2')
                plt.title('Initial Distance Distribution with GMM Components')
                plt.xlabel('Distance')
                plt.ylabel('Density')
                plt.legend()
                plt.savefig(os.path.join(save_dir, 'GMM_distance_distribution_with_GMM_10rounds.png'))
                upper_bound = bounds[1]
                lower_bound = bounds[0]
                for idx, client_distances in enumerate(distances_matrix):
                    normal_dis[idx] = [d for d in client_distances if d <= upper_bound]
                flat_distances1 = [distance for client_list in normal_dis for distance in client_list]
                plt.figure(figsize=(10, 6))
                plt.hist(flat_distances1, bins=40, color='grey', edgecolor='black', density=True)
                sns.kdeplot(flat_distances1, color='red')
                plt.title(f'Final Distance Distribution ')
                plt.xlabel('Distance')
                plt.ylabel('Density')
                plt.savefig(os.path.join(save_dir, f'Final_distance_distribution_10round.png'))
                plt.close()

            if iters == args.ccepochs:
                normalid = []
                f_distances_matrix = [[] for _ in range(args.num_users)]
                all_distances = [distance for client_distances in distances_matrix for distance in client_distances]
                distances_array = np.array(all_distances)

                # Utilize GMM again
                largest_cluster_data, bounds, means, covariances, weights = decompose_normal_distributions(distances_array)
                upper_bound = bounds[1]
                lower_bound = bounds[0]
                for idx, client_distances in enumerate(distances_matrix):
                    normal_dis[idx] = [d for d in client_distances if d <= upper_bound ]
                flat_distances3 = [distance for client_list in distances_matrix for distance in client_list]
                plt.figure(figsize=(10, 6))
                plt.hist(flat_distances3, bins=200, color='grey', edgecolor='black', density=True, alpha=0.6)
                x = np.linspace(min(distances_array), max(distances_array), 1000)
                pdf_1 = weights[0] * (1 / (np.sqrt(2 * np.pi * covariances[0]))) * np.exp(
                    -0.5 * ((x - means[0]) ** 2) / covariances[0])
                pdf_2 = weights[1] * (1 / (np.sqrt(2 * np.pi * covariances[1]))) * np.exp(
                    -0.5 * ((x - means[1]) ** 2) / covariances[1])
                pdf_1 = pdf_1.reshape(-1)
                pdf_2 = pdf_2.reshape(-1)
                plt.plot(x, pdf_1, color='red', linestyle='-.', label='GMM Component 1')
                plt.plot(x, pdf_2, color='green', linestyle='-.', label='GMM Component 2')
                plt.title('Initial Distance Distribution with GMM Components')
                plt.xlabel('Distance')
                plt.ylabel('Density')
                plt.legend()
                plt.savefig(os.path.join(save_dir, 'GMM_distance_distribution_with_GMM.png'))

                flat_distances1 = [distance for client_list in normal_dis for distance in client_list]
                plt.figure(figsize=(10, 6))
                plt.hist(flat_distances1, bins=50, color='grey', edgecolor='black', density=True)
                sns.kdeplot(flat_distances1, color='red')
                plt.title(f'Final Distance Distribution ')
                plt.xlabel('Distance')
                plt.ylabel('Density')
                plt.savefig(os.path.join(save_dir, f'Final_distance_distribution_round.png'))
                plt.close()

                # Erase clients in the component with a lager mean
                largest_cluster_data_2, bounds_2, means, covariances, weights = decompose_normal_distributions(
                    largest_cluster_data)
                upper_bound_2 = bounds[1]
                lower_bound_2 = bounds[0]
                for idx, client_distances in enumerate(distances_matrix):
                    normal_dis[idx] = [d for d in client_distances if d <= upper_bound_2]

                client_means = [np.mean(distances) for distances in distances_matrix]
                excluded_clients, UCL, LCL = plot_control_chart(np.arange(len(client_means)), client_means, normal_dis, save_dir, args)
                print("GMM detects:", excluded_clients)

                r.append(calculate_accuracy(excluded_clients, noisy_clients)[0])
                p.append(calculate_accuracy(excluded_clients, noisy_clients)[1])
                recall = calculate_accuracy(excluded_clients, noisy_clients)[0]
                pre = calculate_accuracy(excluded_clients, noisy_clients)[1]
                f = 2 * recall * pre / (recall + pre)
                f1.append(f)
                print("Initial recall:", r[0])
                print("Initial precision:", p[0])
                print("Initial f1score:", f1[0])

            if iters > args.ccepochs:
                for idx, client_distances in enumerate(distances_matrix):
                    if client_distances[-1] < UCL:
                        normal_id.append(idx)
                    else:
                        excluded.append(idx)
                excluded_clients = excluded
                print("Anomaly:", excluded)
                r.append(calculate_accuracy(excluded_clients, noisy_this_round)[0])
                p.append(calculate_accuracy(excluded_clients, noisy_this_round)[1])
                recall = calculate_accuracy(excluded_clients, noisy_clients)[0]
                pre = calculate_accuracy(excluded_clients, noisy_clients)[1]
                f = 2 * recall * pre / (recall + pre)
                f1.append(f)
                print("Recall:", r[o])
                print("Precision:", p[o])
                print("f1score:", f1[o])
                o += 1

            # Update global model
            if iters < args.ccepochs:
                w_locals_used = [w_locals[i] for i in range(len(w_locals)) if i in normal_id]
                print('numbers of participants:',len(w_locals_used))
    
            else:
                w_locals_used = [w_locals[i] for i in range(len(w_locals)) if i not in excluded_clients]
                print('numbers of participants:', len(w_locals_used))

            if len(w_locals_used) > 0:
                w_glob = FedAvg_0(w_locals_used)
                net_glob.load_state_dict(w_glob)
                global_acc, global_loss = test_img(net_glob, dataset_test, args)
                global_acctotal.append(global_acc)
                global_losses.append(global_loss)
                print('Round {:3d}, global acc {:.3f}'.format(iters, global_acc))
                print('Round {:3d}, global loss {:.3f}'.format(iters, global_loss))
            else:
                print('Round {:3d}, No participant, skip'.format(iters))
                if len(global_acctotal) > 0:
                    global_acctotal.append(global_acctotal[-1])
                    global_losses.append(global_losses[-1])
                else:
                    global_acc, global_loss = test_img(net_glob, dataset_test, args)
                    global_acctotal.append(global_acc)
                    global_losses.append(global_loss)

        if args.method == 'fedavg':
            w_glob = FedAvg_0(w_locals)
            net_glob.load_state_dict(w_glob)
            global_acc, global_loss = test_img(net_glob, dataset_test, args)
            global_losses.append(global_loss)
            global_acctotal.append(float(global_acc))
            print('Round {:3d}, Global acc {:.3f}'.format(iters, global_acc))
            print('Round {:3d}, Global loss {:.3f}'.format(iters, global_loss))

        if args.method == 'median':
            w_glob = simple_median(w_locals)
            net_glob.load_state_dict(w_glob)
            global_acc, global_loss = test_img(net_glob, dataset_test, args)
            global_acctotal.append(float(global_acc))
            global_losses.append(global_loss)
            print('Round {:3d}, global acc {:.3f}'.format(iters, global_acc))
            print('Round {:3d}, global loss {:.3f}'.format(iters, global_loss))

        if args.method == 'krum' :
            w_glob, _ = krum(w_locals, args)
            net_glob.load_state_dict(w_glob)
            global_acc, global_loss = test_img(net_glob, dataset_test, args)
            global_acctotal.append(float(global_acc))
            global_losses.append(global_loss)
            print('Round {:3d}, global acc {:.3f}'.format(iters, global_acc))
            print('Round {:3d}, global loss {:.3f}'.format(iters, global_loss))
            r.append(calculate_accuracy(_, noisy_clients)[0])
            p.append(calculate_accuracy(_, noisy_clients)[1])
            print("查全率：", r[o-1])
            print("查准率：", p[o-1])
            o += 1

        if args.method == 'trimmed_mean':
            w_glob = trimmed_mean(w_locals, 0.2)
            net_glob.load_state_dict(w_glob)
            global_acc, global_loss = test_img(net_glob, dataset_test, args)
            global_acctotal.append(float(global_acc))
            global_losses.append(global_loss)
            print('Round {:3d}, global acc {:.3f}'.format(iters, global_acc))
            print('Round {:3d}, global loss {:.3f}'.format(iters, global_loss))

        if args.method == 'rfa':
            wlist = []
            for i in range(len(w_locals)):
                tmp = [w_locals[i][key] for key in w_locals[i].keys()]
                wlist.append(tmp)
            weights = torch.ones(len(wlist)).to(args.device)
            out, new_weights = geometric_median_list_of_array(wlist, weights, maxiter=1)
            global_weights = copy.deepcopy(w_locals[0])
            for i, key in zip(range(len(out)), global_weights.keys()):
                global_weights[key] = out[i]
            net_glob.load_state_dict(global_weights)
            global_acc, global_loss = test_img(net_glob, dataset_test, args)
            global_losses.append(global_loss)
            global_acctotal.append(float(global_acc))
            print('Round {:3d}, Global acc {:.3f}'.format(iters, global_acc))
            print('Round {:3d}, Global loss {:.3f}'.format(iters, global_loss))

        if args.method == 'cosper':
            if iters == 0:
                v_locals = copy.deepcopy(w_locals)  
                s_locals = [0.5] * args.num_users    # Aggregation weights
                momentum_w = [{}] * args.num_users   # Moment
                momentum_v = [{}] * args.num_users   
                h_locals = copy.deepcopy(w_locals)   
            
                cosper_global_accs = []
                cosper_personalized_accs = []
                cosper_client_weights = []
            
            w_glob, v_locals, s_locals, momentum_w, momentum_v, excluded, h_locals = cosper_defense(
                w_locals, w_glob, args, v_locals, s_locals, momentum_w, momentum_v
            )
            
            net_glob.load_state_dict(w_glob)
            
            global_acc, global_loss = test_img(net_glob, dataset_test, args)
            global_losses.append(global_loss)
            global_acctotal.append(float(global_acc))
            cosper_global_accs.append(float(global_acc))
            
            cosper_client_weights.append(copy.deepcopy(s_locals))
            personalized_accs = []
            for i in range(len(h_locals)):
                if i not in excluded:
                    personalized_model = copy.deepcopy(net_glob)
                    personalized_model.load_state_dict(h_locals[i])
                    p_acc, _ = test_img(personalized_model, dataset_test, args)
                    personalized_accs.append(p_acc)
            
            avg_personalized_acc = np.mean(personalized_accs) if personalized_accs else 0
            cosper_personalized_accs.append(avg_personalized_acc)
            
            if args.attack:
                r.append(calculate_accuracy(excluded, noisy_clients)[0])
                p.append(calculate_accuracy(excluded, noisy_clients)[1])
                print("recall:", r[-1])
                print("precision:", p[-1])
            
            print('Round {:3d}, Global acc {:.3f}, Personalized acc {:.3f}'.format(iters, global_acc, avg_personalized_acc))
            print('Round {:3d}, Global loss {:.3f}'.format(iters, global_loss))
            
            if iters == args.epochs - 1:
                plt.figure()
                plt.plot(range(len(cosper_global_accs)), cosper_global_accs, label='Global Model')
                plt.plot(range(len(cosper_personalized_accs)), cosper_personalized_accs, label='Personalized Model')
                plt.ylabel('Accuracy')
                plt.xlabel('Rounds')
                plt.legend()
                plt.savefig(os.path.join(save_dir, f'cosper_comparison_{args.dataset}_{args.model}.png'))
                
                plt.figure(figsize=(10, 6))
                for i in range(min(10, args.num_users)): 
                    client_weights = [round[i] for round in cosper_client_weights]
                    plt.plot(range(len(client_weights)), client_weights, label=f'Client {i}')
                plt.ylabel('Aggregation Weight')
                plt.xlabel('Rounds')
                plt.legend()
                plt.savefig(os.path.join(save_dir, f'cosper_weights_{args.dataset}_{args.model}.png'))
        
        if args.method == 'cpa':
            if iters == 0 :
                prev_prev_w_glob = copy.deepcopy(w_glob)
                prev_w_glob = copy.deepcopy(w_glob)

                for i,w_local in enumerate(w_locals):
                    if i == 0:
                        for key in w_local:
                            prev_w_glob[key] = w_local[key] /len(w_locals)
                    else:
                        for key in w_local:
                            prev_w_glob[key] += w_local[key] /len(w_locals)
            else:
                prev_prev_w_glob = copy.deepcopy(prev_w_glob)
                prev_w_glob = copy.deepcopy(w_glob)

            w_glob= cpa_defense(w_locals, w_glob, prev_w_glob, prev_prev_w_glob, net_glob)
            net_glob.load_state_dict(w_glob)
            global_acc, global_loss = test_img(net_glob, dataset_test, args)
            global_losses.append(global_loss)
            global_acctotal.append(float(global_acc))
            print('Round {:3d}, Global acc {:.3f}'.format(iters, global_acc))
            print('Round {:3d}, Global loss {:.3f}'.format(iters, global_loss))

    with open(os.path.join(save_dir, 'loss_trainfed.txt'), 'w') as f:
        for item in global_losses:
            f.write("%s\n" % item)

    with open(os.path.join(save_dir, 'global_acc.txt'), 'w') as f:
        for item in global_acctotal:
            f.write("%s\n" % item)

    with open(os.path.join(save_dir, 'recall.txt'), 'w') as f:
        for item in r:
            f.write("%s\n" % item)

    with open(os.path.join(save_dir, 'precision.txt'), 'w') as f:
        for item in p:
            f.write("%s\n" % item)

    with open(os.path.join(save_dir, 'F1score.txt'), 'w') as f:
        for item in f1:
            f.write("%s\n" % item)

    plt.figure()
    plt.figure()
    plt.plot(range(len(global_acctotal)), global_acctotal)
    plt.ylabel('global acc')
    plt.savefig(os.path.join(save_dir, f'fedacc_{args.dataset}_{args.model}_{args.epochs}_C{args.frac}_iid{args.iid}.png'))

    plt.figure()
    plt.figure()
    plt.plot(range(len(global_losses)), global_losses)
    plt.ylabel('global loss')
    plt.savefig(
        os.path.join(save_dir, f'fedloss_{args.dataset}_{args.model}_{args.epochs}_C{args.frac}_iid{args.iid}.png'))
    print("global loss:",global_losses[-1])
    print("global acc:",global_acctotal[-1])












