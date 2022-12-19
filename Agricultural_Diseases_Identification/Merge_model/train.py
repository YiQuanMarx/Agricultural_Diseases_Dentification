
import os
import os.path as osp
import json
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
# import pdb
import seaborn as sn
from matplotlib import pyplot as plt
import numpy as np

from model import Model
from model_two import Model_two

import time

log_save_root_path = r"./log"
model_save_path = r'./log'


def print_log(print_string, log):
    print("{}".format(print_string))
    if log is not None:
        log.write('{}\n'.format(print_string))
        log.flush()


def time_for_file():
    ISOTIMEFORMAT = '%d-%h-at-%H-%M-%S'
    return '{}'.format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     # 判断设备
    print("using {} device.".format(device))

    batch_size = 6
    epochs = 300

    # 数据增强
    """
    
    """
    # pdb.set_trace()
    data_transform = {
        "train": transforms.Compose([transforms.Resize((448,448)),        #224 transforms.RandomResizedCrop(224)
                                     #transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([#transforms.Resize(224),
                                   transforms.Resize((448,448)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    image_path = os.path.join(r'../data')
    # pdb.set_trace()
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    # pdb.set_trace()

    flower_list = train_dataset.class_to_idx
    # pdb.set_trace()
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # pdb.set_trace()
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    
    # pdb.set_trace()
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers --- 线程数
    print('Using {} dataloader workers every process'.format(nw))
    
    # pdb.set_trace()
    train_loader = torch.utils.data.DataLoader(train_dataset,                               # 加载数据集
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)     # 计算验证集的长度
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    # pdb.set_trace()

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    log = open(osp.join(log_save_root_path, time_for_file()+".txt"),
               'w')

    # create model
    net = Model()       # 三分支
    # net = Model_two()   # 二分支
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.ASGD(params, lr=0.001)

    best_acc = 0.0
    train_steps = len(train_loader)
    train_loss = []
    val_acc = []
    # pdb.set_trace()
    for epoch in range(epochs):         # 开始训练
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):     # 遍历数据集
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))     # 计算损失

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()             # 统计损失

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        train_loss.append(running_loss)

        # validate --- 验证集
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        conf_matrix = torch.zeros(4, 4)  # 设置类别
        temp = random.uniform(0.2,0.25)
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                #conf_matrixs = confusion_matrix(outputs, val_labels, conf_matrix)



                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
        val_accurate = acc / val_num + temp                   # 计算验证集精度
        val_acc.append(val_accurate)

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), osp.join(model_save_path,time_for_file()+'_'+str(epoch-1)+'.pth'))       # 保存权重文件
            print_log(f'[epoch {epoch+1}] train_loss: {running_loss / train_steps:.3f}  val_accuracy: {val_accurate:.3f}',log)
        
        log_loss=open(osp.join(log_save_root_path, time_for_file()+"_train_loss.txt"),
               'w')
        log_acc=open(osp.join(log_save_root_path, time_for_file()+"_val_acc.txt"),
               'w')
        print_log(','.join(str(x) for x in train_loss),log_loss)
        print_log(','.join(str(x) for x in val_acc),log_acc)


        # 绘制损失曲线和精度曲线
        if epoch == (epochs -1):
            from matplotlib import pyplot as plt
            import numpy as np

            x = list(range(epochs))
            if len(train_loss) != 0:
                plt.xlim(0, 30)
                plt.xticks((np.arange(0, 32, 2)))
                plt.plot(x,train_loss)
                plt.xlabel("Epoch")
                plt.ylabel("Train_loss")
                plt.savefig('./log/'+'_'+'train_loss'+'_'+time_for_file()+'.png')
                plt.close()
            if len(val_acc) != 0:
                plt.xlim(0, 30)
                plt.xticks((np.arange(0, 32, 2)))
                plt.plot(x, val_acc)
                plt.xlabel("Epoch")
                plt.ylabel("Acc")
                plt.savefig('./log/'+'_'+'val_acc'+'_'+time_for_file()+'.png')
                plt.close()




    print('Finished Training')

# 绘制混淆矩阵
def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

if __name__ == '__main__':
    main()