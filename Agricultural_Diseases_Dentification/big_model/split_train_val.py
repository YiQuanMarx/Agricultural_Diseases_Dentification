import os
import shutil
import numpy as np
import random


def split_train_val():
    base_root = "./data_set"
    train_set = "./train"
    val_set = "./val"

    if os.path.exists(train_set):
        shutil.rmtree(train_set)
        shutil.rmtree(val_set)
    os.mkdir(train_set)
    os.mkdir(val_set)

    data_names = os.listdir(base_root)
    for file_name in data_names:
        file_path = os.path.join(base_root,file_name)
        save_train = os.path.join(train_set,file_name)
        save_val = os.path.join(val_set,file_name)

        if not os.path.exists(save_train):
            os.mkdir(save_train)
            os.mkdir(save_val)

        imgs_infos = os.listdir(file_path)
        rate = random.sample(imgs_infos, k=int(len(imgs_infos) * 0.2))
        for i in imgs_infos:
            img_path = os.path.join(file_path, i)
            if i in rate:
                shutil.copy(img_path, os.path.join(save_val,i))
            else:
                shutil.copy(img_path,os.path.join(save_train,i))


if __name__ == '__main__':
    split_train_val()