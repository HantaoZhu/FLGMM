# **Statistical Test-based Adversarial Client Detection in Federated Learning under Poisoning Attacks**

This repository contains the technical implementation of the paper Statistical Test-based Adversarial Client Detection in Federated Learning under Poisoning Attacks, as well as proof of its results.

## **Abstract**
Federated Learning (FL) is an innovative decentralized machine learning paradigm that enables multiple data owners to collaboratively train models while preserving data privacy. However, FL systems are susceptible to adversarial attacks that can significantly degrade the robustness and performance of the global model. Existing defense methods always utilize the similarity measure to detect adversarial clients and prohibit them from participating in forward learning. However, these methods often struggle with false positives and do not adapt to dynamic anomaly detection. In this paper, we introduce a statistical defense mechanism FLGMM designed to detect adversarial clients and enhance the robustness of federated learning against various poisoning attacks. We find that the presence of adversarial clients will lead to a bimodal distribution of distances between the local models and the global model. FLGMM first uses the Gaussian mixture model (GMM) to fit the distance distribution, then leverages a control limit based on statistical process control (SPC) theory to effectively detect and exclude adversarial clients from the model aggregation process by controlling the false positives in each round. Through extensive experiments on MNIST, Fashion-MNIST and CIFAR10 datasets under different attack scenarios, including label-flipping, sample-poisoning, Gaussian-noise and random-weights attacks, we demonstrate that FLGMM significantly enhances the detection accuracy and maintains high model integrity.

## **Contributions**
* We propose a novel method to select benign clients in FL based on statistical modeling. By utilizing the GMM, we can accurately estimate the distributions of model weights for both benign and adversarial clients. This approach provides more informative and effective results compared to most existing distance-based defenses.

* To the best of our knowledge, this is the first work to apply the idea of process control in federated learning. Specifically, we employ the GMM to estimate the distribution of benign clients' distances during the early training phase and use the quantile of the estimated distribution as the threshold to identify and eliminate adversarial clients, thereby mitigating poisoning attacks. This inspiration comes from the process control, which is one of the powerful tools in quality management.

* We conduct extensive experiments on MNIST, Fashion-MNIST and CIFAR10 datasets under four types of poisoning attacks to validate the effectiveness of our FLGMM defense. The results demonstrate that our approach significantly enhances the robustness of the federated learning system against adversaries.

## **Method Overview**
In order to defend against poisoning attacks, FLGMM targets to directly remove clients judged as adversaries before model aggregation. It uses the standardized Euclidean distance between each client's local model parameter vector and the global model's as a statistic to determine whether a client is an adversary.

The workflow of FLGMM is illustrated in the figure. Specifically, due to the distinct difference between the distance distributions of benign and malicious clients, FLGMM employs the Gaussian Mixture Model (GMM) during the initial rounds of communication to approximate the behavior of participating clients. This probabilistic model is instrumental in identifying anomalous behavior by estimating the typical data distribution of benign clients. Based on this estimation, FLGMM defines a normal distance region, which sets the boundaries for expected client behavior under normal operating conditions.
<img src= https://github.com/user-attachments/assets/0b622e21-767f-4c8a-abf9-a7f38003c20f width="200px">

After the normal distance region has been established, FLGMM transitions into a monitoring phase. In this phase, the previously defined normal distance region becomes critical, serving as the basis for setting thresholds for all client distances, similar to the control chart method. This allows FLGMM to promptly flag any subsequent deviations from the norm. The control chart, traditionally used for monitoring manufacturing processes, is adapted here to oversee the activity of clients within the FL system. Anomalies are signaled when client data points fall outside the control limit, indicating potential adversarial behavior or data corruption. Thus, FLGMM is an effective defense against poisoning attacks in FL when adversaries are less than benign clients.

## **Implementation**
*flgmm.py: Contains the main implementation of our defense mechanism.
*models: Contains NN models, local update function, test function, Fedavg and so on.
*utils: Cotains defense algorithms like FedCPA and Cosper, data utils, sample function and so on. 

## **Sample command**
Use command below for experiment: Run FLGMM with hyperparameters L=3 and T_g=50 (ccepochs), in label-flipping (lf) attack with 0.2 attack ratios on MNIST dataset.

```
python flgmm.py --dataset mnist --iid --attack_pattern lf --method flgmm --peer_round 0.2 --num_channels 1 --epochs 200 --ccepochs 50 --L 3 --gpu 0
```

## **Sample Results**


## **Conclusion**
In this paper, we have presented FLGMM, a robust defense mechanism for federated learning to leverage outlier detection methods based on the statistical process control theory to detect and exclude adversarial clients. Our approach utilizes the variant of the control chart to monitor and identify anomalies in the training process, ensuring that only updates from benign clients are aggregated to the global model. We evaluated FLGMM across various datasets and under different attack scenarios, including label-flipping, sample-poisoning, Gaussian-noise, and random-weights attacks. Extensive results demonstrate that FLGMM significantly enhances the detection accuracy of adversarial clients and improves the overall accuracy of the global model.

Our study main focuses on the scenario where adversarial clients launched the same type of attack, and the bimodal distribution is used to model the distance statistics of benign clients and adversarial clients. When the attacks from adversarial clients exhibit heterogeneity, our proposed Gaussian mixture model can be easily extended to the multi-modal case. Only clients with higher modal probability in distance are classified as the benign clients. After identifying the normal control region, we can conduct the SPC-based method to select adversarial clients in the following rounds.

In the future, we plan to extend our work by refining the statistical metrics used in the control chart to improve its sensitivity and specificity in detecting anomalies. Additionally, we aim to apply FLGMM to more complex federated learning scenarios, such as vertical federated learning and federated transfer learning. These scenarios involve evolving data distributions and knowledge transfer across different domains, which present unique challenges and opportunities for enhancing the robustness and efficiency of federated learning systems. By addressing these aspects, we aim to further advance the state-of-the-art in secure and reliable federated learning.
