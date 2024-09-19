## Introduction：

This repository holds the codebase, dataset and models for the paper:

**Mission: mmWave Radar Person Identification with RGB Cameras**.



## Requirements:

- Python3 (>3.5)
- PyTorch
- Other Python libraries can be installed by `pip install -r requirements.txt`



## Installation:

``` shell
git clone https://github.com/EverRaynor/Mission
```

``` download files
https://drive.google.com/file/d/1DtZ1epsKTfqbcmWJaMZqgq5tbhQbyRGq/view?usp=sharing
```

## Data:

draw_pose.py - Pose Visualization Code

losses.py - Loss Function Definitions

lime_test.py - Network Interpretability Testing Based on LIME

shap_test.py - Network Interpretability Testing Based on SHAP

smpl_utils_extend.py - Extended SMPL Source Code

Resnet.py - ResNet Source Code

gaitpart.py - GaitPart Source Code

Remaining .py files are configuration and open-source project files, which generally do not need to be changed.

loss - Directory for saving training process data

net - Stores some network code

lib - Library directory, includes pre-trained models for HMR and TCMR

log - Directory for saving trained models, ready for direct use

res - Experimental results and visualization display
