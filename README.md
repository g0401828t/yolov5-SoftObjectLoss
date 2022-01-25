# Soft-Objectness Loss

## How to use loss_custom in YOLOv5 official code.
1. Add loss_custom.py to the folder (same directory with loss.py) and edit train.py
2. Change import from loss to loss_custom in main.py
  ```
  from utils.loss import ComputeLoss
  ```
  ```
  from utils.loss_custom import ComputeLoss
  ```
3. play with hyperparameter "n", "gamma", "sigma" with parser
  ```
  n: softscore1 assignment
  gamma : softscore1 assignment
  sigma : softscore2 assignment
  ```
- Detail implementation of our method is in loss_custom.py > Compute_Loss > build_targets_custom1
- build_targets_custom: implemented with for loops => very long computation time 
- build_targets_custom1: modified computation into tensor => much faster but still slower than the original wich has less targets (Binary)



## Soft-Objectness Score
<img src="https://latex.codecogs.com/svg.latex?d_i%20%3D%20%5Csqrt%7B%28%5Cfrac%7Bcx%20-%20cx_i%7D%7B%20w%20/%202%7D%29%5E2%20&plus;%20%28%5Cfrac%7Bcy%20-%20cy_i%7D%7B%20h%20/%202%7D%29%5E2%7D" />

### OursScore1 ( 실선 ) 
<img src = "https://latex.codecogs.com/svg.latex?score%20%3D%20%28d_i%20-%201%29%5E%7B2m%7D%20%3D%20%28d_i%20-%201%29%20%5En"/>

### OursScore2 ( 점선 )
<img src = "https://latex.codecogs.com/png.latex?score%20%3D%20%5Cexp%28-%5Cfrac%7Bd_i%5E2%7D%7B2%5Csigma%5E2%7D%29"/>

<img src = "https://user-images.githubusercontent.com/55650445/150890212-107002bf-154f-4d07-ad8b-a148aaaf42f9.png"/>

## Soft-Objectness Loss

### OursLoss1
<img src = "https://latex.codecogs.com/png.latex?soft%5C%20object%5C%20loss1%20%3D%20-%281-s_i%29%5E%5Cgamma%5Clog%281-t_i%29"/>
- graph image for gamma = 1
<img width="500" src="https://user-images.githubusercontent.com/55650445/147318712-f488160d-8bdb-49c7-88c0-59e115e2666f.png"/>

### OursLoss2
<img src = "https://latex.codecogs.com/svg.latex?%5Cbegin%7Baligned%7D%20soft%5C%20object%20%5C%20loss2%20%26%3D%20BCE%28s_i%2C%20t_i%29%20-%20%7B%5Ccolor%7BBlue%7D%20BCE%28s_i%2Cs_i%29%7D%20%5C%5C%20%26%3D%20-s_i%5Clog%28t_i%29%20-%281-s_i%29%5Clog%281-t_i%29%20-%20%7B%5Ccolor%7BBlue%7D%20%28-s_i%5Clog%28s_i%29%20-%281-s_i%29%5Clog%281-s_i%29%29%7D%20%5Cend%7Baligned%7D"/>
<img width="500" src="https://user-images.githubusercontent.com/55650445/147318857-cf4aecf6-f571-48ed-b850-1815796cbb0d.png"/>

### OursLoss3
<img src = "https://latex.codecogs.com/svg.latex?soft%5C%20object%5C%20loss3%20%3D%20%281-s_i%29soft%5C%20object%5C%20loss2"/>
<img width="500" src="https://user-images.githubusercontent.com/55650445/147319011-4ac7df59-5723-4e76-a873-741944e2eeab.png"/>

## Experiments
### Experiment1: COCO128
Results: https://wandb.ai/wonseokjeong/YOLOv5?workspace=user-wonseokjeong

### Experiment2: COCO
Results: https://wandb.ai/wonseokjeong/train?workspace=user-wonseokjeong

### Experiment3: Pascal VOC
Results: https://wandb.ai/wonseokjeong/YOLO_VOC?workspace=user-wonseokjeong

## Conclusion
Helps training in the early stage but interrupts the the model understanding and downgrades the MAP results.
