import os
import sys
import shutil
import random
from utils import Logger

train_path = '../data/train/'
val_path = '../data/val/'


# train : val = 7:3
val_num = 180 / 10 * 3
animals = ['cat', 'cow', 'dog', 'horse', 'rat']

sys.stdout = Logger("./split_eval.txt")

for i in range(len(animals)):
    if not os.path.exists(val_path + '/' + animals[i] + '/'):
        os.makedirs(val_path + '/' + animals[i] + '/')
    count = 0
    imgs = os.listdir(train_path + animals[i] + '/')
    random.shuffle(imgs)
    while count < val_num:
        # shutil.move(train_path + animals[i] + '/' + imgs[count], val_path + animals[i] + '/' + imgs[count])
        # print(str(count + 1) + "____Finishing moving the " + train_path + animals[i] + '/' + imgs[count] + " to " 
        shutil.copy(train_path + animals[i] + '/' + imgs[count], val_path + animals[i] + '/' + imgs[count])
        print(str(count + 1) + "____Finishing copying the " + train_path + animals[i] + '/' + imgs[count] + " to " 
        + val_path + animals[i] + '/' + str(count + 1) + ".jpg !!!")
        count += 1

