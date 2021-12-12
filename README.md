# FedMRI

# [Specificity-Preserving Federated Learning for MR Image Reconstruction]( )

> **Authors:** 
> [Chun-Mei Feng](https://scholar.google.com.hk/citations?user=g2nqHBcAAAAJ&hl=zh-CN), 
> [Yunlu Yan](), 
> [Huazhu Fu](http://hzfu.github.io/), 
> [Yong Xu](https://scholar.google.com.hk/citations?user=zOVgYQYAAAAJ&hl=zh-CN), and 
> [Ling Shao](https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=zh-CN).


[[Paper]( )][[Code](https://github.com/chunmeifeng/FedMRI)]

## ⚡ Dependencies
* numpy==1.18.5
* scikit_image==0.16.2
* torchvision==0.8.1
* torch==1.7.0
* runstats==1.8.0
* pytorch_lightning==1.0.6
* h5py==2.10.0
* PyYAML==5.4

## ⚡ Overview

### Motivation
----------
<img src="figs/fig0.png" width="536px"/>

Classical FL algorithm for MR image reconstruction: (a) average all the local client models to obtain a general global model, 
or (b) repeatedly align the latent features between the source and target clients. 
In contrast, we propose a specificity- preserving mechanism (c) to consider both generalized shared information 
as well as  client-specific properities.

### Framework Overview
----------
<img src="figs/fig2.png" width="536px"/>

Overview of the FedMRI framework. Instead of averaging all the local client models, 
a globally shared encoder is used to obtain a generalized representation, 
and a client-specific decoder is used to explore unique domain-specific information. 
We apply the weighted contrastive regularization to better pull the positive pairs together 
and push the negative ones towards the anchor.

### Qualitative Results
----------
<img src="figs/fig1.png" width="536px"/>

T-SNE visualizations of latent features from four datasets, where (a-d) show the distributions of SingleSet, FedAvg [24], FedMRI without Lcon, and our entire FedMRI algorithm, respectively. In SingleSet, each client is trained to use their local data with- out FL. The distribution of points in (a) is clearly differ- entiated because each dataset has its own biases, while the data in (b), (c) and (d) overlap to varying degrees, as these models benefit from the joint training mechanism of FL. However, on the datasets with large differences in distri- bution, e.g., fastMRI and BraTS, FedAvg [24] nearly fails (see Fig. 3 (b)). Notably, even without Lcon, our method still aligns the latent space distribution across the four dif- ferent datasets, which demonstrates that sharing a global en- coder and keeping a client-specific decoder can effectively reduce the domain shift problem (see Fig. 3 (c)). Fig. 3 (d) shows a fully mixed distribution for the latent features of the different clients. This can be attributed to the weighted contrastive regularization, which enables our FedMRI al- gorithm to effectively correct deviations between the client and server during optimization (see Fig. 3 (d))

<img src="figs/fige2.png" width="536px"/>


## ⚡ Data Prepare

Download data from the link https://fastmri.org/dataset/, https://www.med.upenn.edu/sbia/brats2018/data.html 

[[Training code --> FedMRI](https://github.com/chunmeifeng/FedMRI)]

`git clone https://github.com/chunmeifeng/FedMRI.git`

## ⚡ Train
**single gpu train**
"python train.py"
```bash
python train.py
```

**multi gpu train**
"python train_multi_gpu.py"
```bash
python train_multi_gpu.py
```

## ⚡ Citation

```
@inproceedings{feng2021,
  title={Specificity-Preserving Federated Learning for MR Image Reconstruction},
  author={Feng, Chun-Mei and Yan, Yunlu and Fu, Huazhu and Xu, Yong and Ling, Shao },
  journal={arXiv e-prints},
  pages={arXiv--2106},
  year={2021}
}
```


