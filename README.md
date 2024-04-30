# BinVPR
 BinVPR is a BNN designed for VPR and loop closure detection in particular.
 
This repository shares several tool for training and deploying BinVPR.

1. *main.py* is the only file you need for:
    *  Training BinVPR and the other BNNs presented in our paper.
    *  Exporting a model in [Larq-Compute-Engine](https://docs.larq.dev/compute-engine/) format for ARM64 cpus (i.e. [RPI4](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/)).  
    *  Computing an image descriptor

The project has been developed within [Vscode].

## Software Requirements

The main python3 packages required to use the provided code are the following.

* Tensorflow >= 2.3.1
* larq >= 0.10.2
* larq-compute-engine >= 0.4.3
* opencv >= 4.4.0
* prettytable >= 2.0.0
# BinVPR
