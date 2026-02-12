# Overparameterization and Adversarial Robustness in Neural Networks: An Overview and Empirical Analysis

This repository is used to perform the security evaluation of models with increasing size of model parameters. 

## Abstract
Thanks to their extensive capacity, over-parameterized neural networks exhibit superior predictive capabilities and generalization. However, having a large parameter space is considered one of the main suspects of the neural networks’ vulnerability to adversarial
examples— input samples crafted ad-hoc to induce a desired misclassification. Relevant literature has claimed contradictory remarks in support of and against the robustness of over-parameterized networks. These contradictory findings might be due to the failure of the attack employed to evaluate the networks’ robustness. Previous research has demonstrated that depending on the considered model, the algorithm employed to generate adversarial examples may not function properly, leading to overestimating the model’s robustness. In this work, we empirically study the robustness of over-parameterized networks against adversarial examples. However, unlike the previous works, we also evaluate the considered attack’s reliability to support the results’ veracity. Our results show that over-parameterized networks are robust against adversarial attacks as opposed to their under-parameterized counterparts.\

**Authors** Srishti Gupta(University of Cagliari, University of Roma La Sapienza), Zhang Chena(Northwestern Polytechnical University, Xi’an, China), Luca Demetrio(University of Cagliari, University of Genova), , Xiaoyi Feng(Northwestern Polytechnical University, Xi’an, China), Zhaoqiang Xia(Northwestern Polytechnical University, Xi’an, China), Antonio Emanuele Cina(University of Cagliari, University of Genova), Maura Pintor(University of Cagliari), Luca Oneto(University of Cagliari, University of Genova), Ambra Demontis(University of Cagliari), Battista Biggio (University of Cagliari), Fabio Roli(University of Cagliari, University of Genova)\

For further details, please refer to our [paper](https://arxiv.org/abs/2406.10090)

## Installation Guide

Before starting the installation process try to obtain the latest version of the `pip` manager by calling:\
`pip install -U pip`

The Python package setuptools manages the setup process. Be sure to obtain the latest version by calling: \
`pip install -U setuptools`

Once the environment is set up, use following commands to install required packages:\
`pip install -r requirements.txt --no-index --find-links file:///tmp/packages`


This script uses PGD L2 attack from `secml` library and AutoAttack from `pytorch` library. The repo contains the necessary conversion.

## Run experiments

### Arguments
**Dataset**: MNIST and CIFAR10\
**Model type**: CNN and FC-RELU for MNIST, Resnet for Cifar10\
**Attack type**: PGD-L2 and AutoAttac

### Command
`python run.py --ds mnist --model cnn --attack pgdl2`\
or \
`python run.py --ds cifar10 --model resnet --attack pgdl2`\


