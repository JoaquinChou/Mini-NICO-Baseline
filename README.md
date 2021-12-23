# Mini-NICO Baseline
The baseline is a reference method for the final exam of machine learning course.

## Requirements

### Installation
we use /python3.7 /torch 1.4.0+cpu /torchvision 0.5.0+cpu for training and evaluation. You can install the pytorch1.4.0 by using this.
``` bash
conda install pytorch==1.4.0 torchvision==0.5.0 cpuonly -c pytorch
```
By the way, you can also use the pytorch with cuda to train this baseline.
### Prepare Datasets
You need to create the `./data/` folder and put the `./mini_nico/train` and `./mini_nico/test` in [Mini-NICO dataset](https://drive.google.com/file/d/1tdgXsaKbjZ9eiIsq0fPgL166WESj4MmN/view?usp=sharing) to the `./data/` directory like
```
data
├── train
│   └── cat
│   └── cow
│   └──  ..
├── test
│   └── 1.jpg 
│   └── 2.jpg 
│   └──  ..
```

## Split the val data
You can use the following command to split the val data from the train data.
```bash
# split the val from the train data and train : val = 7:3
cd utils 
python split_eval_from_train_data.py 
```
## Training
You can use the following command to run for training.
```bash
# you can choose the model such as resnet18, resnet34, resnet50, resnet101
python trainer.py --arch=resnet18
```
If you want to train the method with gpu, you can do this.
```bash
# you can choose the model such as resnet18, resnet34, resnet50, resnet101
python trainer.py --arch=resnet18 --gpu
```
## Testing
You can use the following command to run for testing.
```bash
# you can choose the model such as resnet18, resnet34, resnet50, resnet101
python test.py --arch=resnet18 --ckpt=your model path
```
If you want to test the method with gpu, you can do this.
```bash
# you can choose the model such as resnet18, resnet34, resnet50, resnet101
python test.py --arch=resnet18 --ckpt=your model path --gpu
```


After that, you can get the `test.csv` in the root path `./`. And then upload your result to our Mini_NICO_Leaderboard.