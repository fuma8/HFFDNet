# HFFDNet: High Frequency Feature Distillation Network[Pytorch]
[News] This paper is accepted for IJCNN'24!!

This repository provide the model for compressive sensing-based reconstruction.
## Datasets
Training images are cropped to 96*96 pixel size and 89600 images are randomly selected for training the network. You can download it from [here](https://drive.google.com/file/d/1hELlT70R56KIM0VFMAylmRZ5n2IuOxiz/view?usp=sharing).

## Train
Please set the training dataset under ./Datasets/ folder.
You can train HFFDNet with following script:
```
python train.py --sensing_rate 0.1 --layer_num 8
```

## Test
Please set the test dataset under ./Datasets/ folder.
You can test HFFDNet with following script:
```
python test.py --sensing_rate 0.1 --layer_num 8
```
