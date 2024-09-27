## Introduction：

This repository holds the codebase, dataset and models for the paper:

**Mission: mmWave Radar Person Identification with RGB Cameras**.



## Requirements:

- Python3 (>3.5)
- PyTorch
- Other Python libraries can be installed by `pip install -r requirements.txt`

****Tips:** If you cannot set up the environment locally, you can use the Google's Colab service to run
this [notebook](https://colab.research.google.com/drive/10o0eAFYQMIcq1H70OBD9Z6RtRqX_5cY2?usp=sharing).**



## Installation:

Install project
``` shell
git clone https://github.com/EverRaynor/Mission
```

Download extra files and extract them to the directory.
``` download files
straight_road_npy.zip
https://drive.google.com/file/d/1Z5B_TSyzjd6Kav6mklc4KHjDjJQ3aSta/view?usp=sharing
rgb_mmwave_data.zip
https://drive.google.com/file/d/1DtZ1epsKTfqbcmWJaMZqgq5tbhQbyRGq/view?usp=sharing
```

## Codes:

train_midmodal_tri_nln_loc.py -a training script that also includes testing and visualization features.

network_tcmr.py - the network design file.

dataset_me_rgb.py - the dataset loading script. 

draw_pose.py - Pose Visualization Code

losses.py - Loss Function Definitions

lime_test.py - Network Interpretability Testing Based on LIME

shap_test.py - Network Interpretability Testing Based on SHAP

smpl_utils_extend.py - Extended SMPL Source Code

Resnet.py - ResNet Source Code

gaitpart.py - GaitPart Source Code

Remaining .py files are configuration and open-source project files, which generally do not need to be changed.

## Folders:

loss - Directory for saving training process data

net - Stores some network code

lib - Library directory, includes pre-trained models for HMR and TCMR

log - Directory for saving trained models, ready for direct use

res - Experimental results and visualization display

## Training&Evaluation:
Modify the dataset location on line 8294 of dataset_me_rgb.py.

```
run Mission/train_midmodal_tri_nln_loc.py
``` 
