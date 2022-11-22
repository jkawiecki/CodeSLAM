# CodeSLAM

## Running Model
### Requirements
Code requires the following tools to run: PyTorch, Torchvision, Yacs, Tensorboard, Opencv-python

### Configure Dataset
Use prepare_scenenet.sh bash file to set up dataset file structure to be as follows:
Example folder structure:
```
SCENENET_ROOT
|__ train_0
    |__ train
        |_ depth
        |_ instance
        |_ photo
|__ train_1
    |__ train
        |_ depth
        |_ instance
        |_ photo
|__ ...
```

### Train model
Run the following command to train the model:
```
python3 train.py configs/scenenet.yaml
```

### Test model
Run the following command to test the model:
```
python3 test.py configs/scenenet.yaml
```

## Code Sourcing
### Repository Copy
A good amount of files were copied from this repository (https://github.com/andersfagerli/CodeSLAM/blob/master/README.md). These include:
- Most of the config files
- All the data files
- All the trainer files
- All the utils files
- demo.py

### Modified Code
- prepare_scenenet.sh
- Some config files
- The dataloaders
- unet.py
- cvae.py
- loss.py
- train.py
- codeslam.py

### Students own code
- test.py


## Datasets
### Scenenet RGB-D
A small portion of the Scenenet RGB-D dataset was used and obtained from this site: https://robotvault.bitbucket.io/scenenet-rgbd.html
