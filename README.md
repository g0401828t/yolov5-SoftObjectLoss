# Soft-Objectness Loss

## How to use loss_custom in YOLOv5 official code.
1. Used loss_custom.py instead of loss.py
2. Add hyperparameter "n" to hyp.scratch.yaml

## Soft-Objectness Score
<img src="https://latex.codecogs.com/svg.latex?d_i%20%3D%20%5Csqrt%7B%28%5Cfrac%7Bcx%20-%20cx_i%7D%7B%20w%20/%202%7D%29%5E2%20&plus;%20%28%5Cfrac%7Bcy%20-%20cy_i%7D%7B%20h%20/%202%7D%29%5E2%7D" />

## Soft-Objectness Loss
<img src = "https://latex.codecogs.com/svg.latex?score%20%3D%20%28d_i%20-%201%29%5E%7B2m%7D%20%3D%20%28d_i%20-%201%29%20%5En"/>

### Ours1
<img src = "https://latex.codecogs.com/svg.latex?soft%5C%20object%20%5C%20loss1%20%3D%20-%281-s_i%29%5Clog%281-t_i%29"/>
<img width="500" src="https://user-images.githubusercontent.com/55650445/147318712-f488160d-8bdb-49c7-88c0-59e115e2666f.png"/>

### Ours2
<img src = "https://latex.codecogs.com/svg.latex?%5Cbegin%7Baligned%7D%20soft%5C%20object%20%5C%20loss2%20%26%3D%20BCE%28s_i%2C%20t_i%29%20-%20%7B%5Ccolor%7BBlue%7D%20BCE%28s_i%2Cs_i%29%7D%20%5C%5C%20%26%3D%20-s_i%5Clog%28t_i%29%20-%281-s_i%29%5Clog%281-t_i%29%20-%20%7B%5Ccolor%7BBlue%7D%20%28-s_i%5Clog%28s_i%29%20-%281-s_i%29%5Clog%281-s_i%29%29%7D%20%5Cend%7Baligned%7D"/>
<img width="500" src="https://user-images.githubusercontent.com/55650445/147318857-cf4aecf6-f571-48ed-b850-1815796cbb0d.png"/>

### Ours3
<img src = "https://latex.codecogs.com/svg.latex?soft%5C%20object%5C%20loss3%20%3D%20%281-s_i%29soft%5C%20object%5C%20loss2"/>
<img width="500" src="https://user-images.githubusercontent.com/55650445/147319011-4ac7df59-5723-4e76-a873-741944e2eeab.png"/>

## Experiments
### Experiment1: COCO128
Results: https://wandb.ai/wonseokjeong/YOLOv5?workspace=user-wonseokjeong

### Experiment2: COCO
Results: https://wandb.ai/wonseokjeong/train?workspace=user-wonseokjeong

### Experiment3: Pascal VOC
Results: https://wandb.ai/wonseokjeong/VOC/workspace?workspace=user-wonseokjeong
