 
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from Demo_Efficientnet import Efficient_BCNN,BCNN
from efficientnet_pytorch import EfficientNet
from vgg16 import BCNN



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    batch_size = 4
    epochs = 30

    train_loss = []
    val_acc = []

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),        #224
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(224),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    #data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    #image_path = os.path.join(data_root, "Data", "data_set")  # flower data set path
    image_path = os.path.join('./small_data/')
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)     # 计算验证集的长度
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # create model --- 加载模型 --- 初始化模型
    #net = EfficientNet.from_pretrained("efficientnet-b7")       # 自动下载模型预训练权重文件
    # net = EfficientNet.from_name('efficientnet-b7')
    # net.load_state_dict(torch.load('./weights/efficientnet-b0-355c32eb.pth'))
    # net._fc = nn.Linear(in_features=net._fc.in_features, out_features=4, bias=True)
    net = BCNN()       # vgg16
    #net = Efficient_BCNN()


    # freeze features weights
    for param in net.features.parameters():
        param.requires_grad = False

    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.001)

    best_acc = 0.0
    save_path = './Efficientnet'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        train_loss.append(running_loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path + str(epoch) + ".pth")

        val_acc.append(val_accurate)

        print('='*100)
        print("acc:",acc)
        print("val_num:",val_num)
        print('='*100)

    print('Finished Training')

    # 绘制损失曲线和精度曲线
    if epoch == (epochs - 1):
        from matplotlib import pyplot as plt
        import numpy as np

        x = list(range(epochs))
        if len(train_loss) != 0:
            plt.xlim(0, 30)
            plt.xticks((np.arange(0, 32, 2)))
            plt.plot(x, train_loss)
            plt.xlabel("Epoch")
            plt.ylabel("Train_loss")
            plt.savefig('train_loss.png')
            plt.close()
        if len(val_acc) != 0:
            plt.xlim(0, 30)
            plt.xticks((np.arange(0, 32, 2)))
            plt.plot(x,val_acc)
            plt.xlabel("Epoch")
            plt.ylabel("Acc")
            plt.savefig('val_acc.png')
            plt.close()
if __name__ == '__main__':
    main()
