# JGDN

> Our model is based on the paper of "Exploring a Fine-grained Multiscale Method for Cross-modal Remote Sensing Image Retrieval". Thanks to the great contribution.

## 1. Environment

This is a demo on the datasets of RSITMD, RSICD, Sydney and UCM for our paper. We finish experiments on a server with one NVIDIA GeForce 1080Ti GPU.

We recommended the following dependencies.
```bash
Python 3
PyTorch > 0.3
Numpy
h5py
nltk
yaml
```

## 2. Relevant data

Please preprocess dataset to appropriate the input format or you can download the data we preprocessed from the pan.baidu.com.
```bash
RSICD images (Password:NIST)   https://pan.baidu.com/s/1lH5m047P9m2IvoZMPsoDsQ
RISTMD images (Password:NIST) https://pan.baidu.com/share/init?surl=gDj38mzUL-LmQX32PYxr0Q
```

## 3. Train the new model

Please modify the parameters in the directory of `./option` to suit your situation.

Run the `train.py` to train your own model.

## 4. Test the trained model

Run the `test.py` to test your trained model.
