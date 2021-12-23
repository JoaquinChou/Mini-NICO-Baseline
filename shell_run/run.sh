# split the val from the train data and train : val = 7:3
cd utils 
python split_eval_from_train_data.py 

# train the model with cpu
python trainer.py --arch=resnet18
# train the model with gpu
python trainer.py --arch=resnet18 --gpu

# test the model with cpu
python test.py --arch=resnet18 --ckpt=./models/save_resnet18_12-23-00-33/checkpoint_200.th
# test the model with gpu
python test.py --arch=resnet18 --ckpt=./models/save_resnet18_12-23-00-33/checkpoint_200.th --gpu