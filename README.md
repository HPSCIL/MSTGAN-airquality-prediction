
# MSTGAN 

### Predicting air quality with a multi-scale spatio-temporal graph attention network ###

<font face="Times new roman" size=4>
This repo is the implementation of our manuscript entitled Predicting air quality with a multi-scale spatio-temporal graph attention network. The code is based on Pytorch 1.12.1, and tested on a GeForce RTX 4090 GPU with 24GB memory.


In this study, we present an graph-attention-based approach for air quality prediction at multiple monitoring stations termed the Multi-scale Spatio-Temporal Graph Attention Network (MSTGAN). Experiments with two real-world datasets showed the proposed MSTGAN achieved the highest prediction accuracies for 12,18 and 24-hour prediction time lengths, compared to several state-of-the-art methods.

## Framework

![MSTGAN](./Fig/Fig2.jpg)


## Requirements
MSTGAN uses the following dependencies
 
- Pytorch 1.12.1 and its dependencies
- Numpy and Pandas
- CUDA 11.8 or latest version

## Folder Structure
We list the code of the major modules as follows:<br>
- The main function to train/test our model: [click here](./MSTGAN/code/main.py)<br>
- The source code of our model: [click here](./MSTGAN/code/model/MSTGAN.py)<br>
- Train and test data preporcessing are located at: [click here](./MSTGAN/code/utils/pro_data.py)<br>
- Metric computations: [click here](./MSTGAN/code/utils/All_Metrics.py)<br>

## Arguments
We introduce some major arguments of our main function here.

Training settings:
- train\_rate: rate of train set<br>
- test\_rate: rate pf test set<br>
- lag: time length of hidtorical steps<br>
- pre\_len: time length of future steps<br>
- num\_nodes: the number of stations<br>
- batch\_size: training or testing batch size<br>
- input\_dim: the feature dimension of inputs<br> 
- output\_dim: the feature dimension of outputs<br>
- learning\_rate: the learning rate at the beginning<br>
- epochs: training epochs<br>
- early\_stop_patience: the patience of early stopping<br>
- device: using which GPU to train our model<br>
- seed: the random seed for experiments<br>

Model hyperparameters:<br>
- d\_model: position encoding embedding dimension<br>
- cheb\_k: Chebyshev polynomials order<br>
- block1\_hidden: number of hidden layers in the first block<br>
- block2\_hidden: number of hidden layers in the second block<br>
- time\_strides: time resolution<br>
- nb\_block: number of Multi-Spatio-Temporal_Block (MST\_Block)<br>
- dropout: dropout rate<br>


## Citation
- If you find our work useful in your research, please cite:<br>
Zhou, X., Wang, J., Wang, J. & Guan, Q.* (2024) Predicting air quality using a multi-scale spatiotemporal graph attention network, Information Sciences, 680: 121072. DOI: 10.1016/j.ins.2024.121072


