# MalMiningDetector
This is the source code for the paper "[Malicious mining code detection based on ensemble learning in cloud computing environment](https://authors.elsevier.com/c/1ddLb,ZhUEWLaG)".

## Scripts

There are three scripts as following:

- feature_engineering.py
  - Describe the methods about data analysis and feature engineering including feature vectorizing and visualization.
- model_construction.py
  - Descirbe the methods about model training and evaluation.
- mining_detector.py
  - Outline the main process for detecting malicious mining code in the paper.

## Citation

If you use this data in a publication please cite the following paper:

 Shudong Li, Simulation Modelling Practice and Theory, https://doi.org/10.1016/j.simpat.2021.102391
 
 @article{LI2021102391,
title = {Malicious mining code detection based on ensemble learning in cloud computing environment},
journal = {Simulation Modelling Practice and Theory},
pages = {102391},
year = {2021},
issn = {1569-190X},
doi = {https://doi.org/10.1016/j.simpat.2021.102391},
url = {https://www.sciencedirect.com/science/article/pii/S1569190X21000976},
author = {Shudong Li and Yuan Li and Weihong Han and Xiaojiang Du and Mohsen Guizani and Zhihong Tian},
keywords = {Malicious mining code, Mining virus, Cloud computing, Static analysis, Ensemble learning},
abstract = {Hackers increasingly tend to abuse and nefariously use cloud services by injecting malicious mining code. This malicious code can be spread through infrastructures in the cloud platforms and pose a great threat to users and enterprises. In this study, a method is proposed for detecting malicious mining code in the cloud platforms, which constructs a detection model by fusing the Bagging and Boosting algorithms. By randomly extracting samples and letting models vote together to decide, the variance of model detection can be reduced obviously. Compared with traditional classifiers, the proposed method can obtain higher accuracy and better robustness. The experimental results show that, for the given dataset, the values of AUC and F1-score can reach 0.992 and 0.987 respectively, and the standard deviation of AUC values under different data inputs is only 0.0009.}
}
