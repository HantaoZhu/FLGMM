# **Statistical Test-based Adversarial Client Detection in Federated Learning under Poisoning Attacks**

This repository contains the technical implementation of the paper Statistical Test-based Adversarial Client Detection in Federated Learning under Poisoning Attacks, as well as proof of its results.

## **Abstract**
Federated Learning (FL) is an innovative decentralized machine learning paradigm that enables multiple data owners to collaboratively train models while preserving data privacy. However, FL systems are susceptible to adversarial attacks that can significantly compromise the robustness and performance of the global model. Existing defense methods always utilize the similarity measure to detect adversarial clients which struggle with false positives and do not adapt to dynamic anomaly detection. In this study, we introduce Federated Learning Gaussian Mixture Model (FLGMM), a statistical defense method designed to detect adversarial clients and enhance the robustness of FL against various poisoning attacks. We find that the presence of adversarial clients induces a bimodal distribution in the distances between local and global models. Thus, FLGMM first uses the Gaussian Mixture Model (GMM) to fit the distance distribution, then leverages a control limit based on Statistical Process Control (SPC) theory to effectively detect and exclude adversarial clients from the model aggregation process by controlling false positives in each round. Extensive experiments are conducted on both image (MNIST, Fashion-MNIST and CIFAR10) and non-image (Shakespeare) datasets under different attack scenarios, including label-flipping, sample-poisoning, Gaussian-noise and random-weights. The results indicate that FLGMM improves global accuracy by 0.02\% – 46.82\% over competing methods. Our study lays the groundwork for future extensions that integrate advanced SPC techniques (e.g., self-starting control charts) to cope with complicated FL scenarios.

## **Contributions**
* We propose a novel method to select benign clients in FL based on statistical modeling. By utilizing the GMM, we can accurately estimate the distributions of the model parameters for both benign and adversarial clients. This method provides more informative and effective client selection decisions compared to most existing distance-based defense methods.

* To the best of our knowledge, this is the first study to apply the idea of process control in FL. Specifically, we employ the GMM to estimate the distribution of benign client distances during the early training phase and use the quantile of the estimated distribution as the threshold to identify and eliminate adversarial clients, thereby mitigating poisoning attacks. This inspiration comes from SPC, which is one of the powerful tools in quality management.

* We conduct extensive experiments on MNIST, Fashion-MNIST, CIFAR10 and Shakespeare datasets under four types of poisoning attacks to demonstrate the effectiveness of our proposed FLGMM. The results indicate that our method significantly enhances the robustness of FL systems against adversaries.

## **Method Overview**
In order to defend against poisoning attacks, FLGMM targets to directly eliminate clients identified as adversarial before the model aggregation. It uses the standardized Euclidean distance between each client's local model parameter vector and the global model's as a statistical metric for anomaly detection. 

The workflow of FLGMM is illustrated in the figure. Specifically, FLGMM employs the GMM during the initial rounds of communication to capture the distinct statistical differences in the distance distributions of benign and adversarial clients. This probabilistic model is instrumental in identifying anomalous behavior by estimating the typical distance distribution of benign clients. Based on the estimated benign distribution, FLGMM establishes a normal distance region, which defines the expected range of variation for benign client updates under normal conditions.

<img src= https://github.com/HantaoZhu/FLGMM/blob/main/Figure/flowchart_complete.png width="500px">

Once the normal distance region is established, FLGMM transitions into a monitoring phase, where it continuously evaluates client updates using this region to set the control limit. Clients whose distances fall outside the bound are flagged as adversaries. The adaptation of control chart methods enables prompt detection of anomalies, ensuring that adversarial updates or corrupted local data do not influence the global model. Consequently, FLGMM provides an effective and statistically grounded defense against poisoning attacks.

## **Implementation**
* flgmm.py: Contains the main implementation of our defense mechanism.
* models: Contains NN models, local update function, test function, Fedavg and so on.
* utils: Cotains defense algorithms like FedCPA and Cosper, data utils, sample function and so on. 

## **Depdendencies (tentative)**
* python 3.12.3 (Anaconda)
* Pytorch 2.2.2
* torchvision 0.17.2 
* CUDA 12.5
* cuDNN 9.1.1.17 

## **Running Experients**

| Key argument   | Description  |
| :----:  |:----  |
| dataset  | Dataset to use |
| model  | Model to use |
| attack_pattern | The attack type: lf -> label-flipping, gn -> sample-poisoning, wn -> Gaussian-noise, rw -> random-weights|
| method | The defense algorithm to use |
| peer_round | The attacker ratio |
| epochs | The number of FL round |
| ccepochs | The T_g round for GMM |
| L | The width (L sigma) of the control limit |

### **Sample command**
Use command below for experiment: Run FLGMM with hyperparameters L=3 and T_g=50 (ccepochs), in label-flipping (lf) attack with 0.2 attack ratios on MNIST dataset.

```
python flgmm.py --dataset mnist --iid --attack_pattern lf --method flgmm --peer_round 0.2 --num_channels 1 --epochs 200 --ccepochs 50 --L 3 --gpu 0
```

## **Sample Results**
<img width="652" alt="sample results" src="https://github.com/user-attachments/assets/7fef059f-58fd-4efe-ab15-df60689b6306" />

## **Conclusion**
In this study, we present FLGMM, a robust defense method for FL to leverage anomaly detection methods based on the SPC theory to detect and exclude adversarial clients. Our method utilizes a variant of the control chart to monitor and identify anomalies in the training process, ensuring that only updates from benign clients are aggregated to the global model. We evaluated FLGMM across various datasets in different attack scenarios, including label-flipping, sample-poisoning, Gaussian-noise, and random-weights attacks. Extensive results demonstrate that FLGMM significantly enhances the detection accuracy of adversarial clients and improves the overall performance of the global model.

Our study mainly focuses on the scenario where adversarial clients launched the same type of attack, and the bimodal distribution is used to model the distance statistics of benign clients and adversarial clients. When the attacks from adversarial clients exhibit heterogeneity, our proposed client selection by GMM can be easily extended to the multi-modal case. Only clients with a higher modal probability in distance of the smallest mean are classified as the benign clients. After identifying the normal control region, we can conduct the SPC-based method to select adversarial clients in the following rounds.

Currently, our FLGMM depends on the Shewhart control chart with fixed thresholds for anomaly detection, which is already effective in common attack scenarios. However, practical deployments may involve non-stationary client behavior and evolving attack strategies, motivating the need for dynamic thresholding. A particularly suitable extension is the use of self-starting control charts, which eliminate the requirement for an in-control dataset and instead update the control limits as new observations arrive. By automatically recalibrating both the center line and control limits based on cumulative sample statistics, a self-starting chart can adapt to gradual drifts in benign client distributions and maintain accurate anomaly detection without manual retuning. Incorporating this technique into FLGMM would enable continuous and data-driven threshold adjustment throughout training, improving robustness in  complicated FL environments.
    

In the future, we plan to expand our work by integrating adaptive control charts into FLGMM and refining the statistical metrics used in the control chart to improve its sensitivity and specificity in detecting anomalies. Additionally, we aim to apply FLGMM to more complex FL scenarios, such as vertical FL and federated transfer learning. These scenarios involve evolving data distributions and knowledge transfer across different domains, which present unique challenges and opportunities to enhance the robustness and efficiency of FL systems. By addressing these aspects, we aim to further advance the state-of-the-art in secure and reliable FL.
