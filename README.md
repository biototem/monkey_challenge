# MONKEY Challenge Submission
This repository contains the algorithms inference scripts of Docker image  for  the [Monkey Challenge](https://monkey.grand-challenge.org/)  Final test phase

## Prerequisites
- Docker (Ensure that Docker is installed and supports GPU with CUDA 12.1 or newer)
- NVIDIA Docker Toolkit for GPU support
- torch-2.1.2+cu118 
- torchvision-0.16.2+cu118
- tiffslide
- shapely
## Adding the weights
Weight can be downloaded from [here](https://drive.google.com/file/d/13NEbhPwlyS-KYXKRp-KOp-2M-XpWMCUg/view?usp=sharing) :
- The model weight file  needs to be placed in the repository folder. Or you can modify in `main.py` at line 113:
```python
model_path = './115_2.pth'
```

## Running 
```python
python main.py
```

## Input & Output
One input file will be mounted per container at (algorithm job) `/input`.

detected-lymphocytes.json,detected-monocytes.json and detected-inflammatory-cells.json are expected files inside the `/output` directory:
