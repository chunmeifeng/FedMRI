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
<p align="center">
    <img src="imgs/framework-final-min.png"/> <br />
    <em> 
    Figure 1: Overview of the proposed PraNet, which consists of three reverse attention 
    modules with a parallel partial decoder connection. See § 2 in the paper for details.
    </em>
</p>
### Framework Overview
### Results

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


