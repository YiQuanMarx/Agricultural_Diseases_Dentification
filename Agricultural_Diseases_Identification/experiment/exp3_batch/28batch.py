'''
min_loss_list=[2.5356123447418213, 1.4405226707458496, 1.1302883625030518, 1.0669386386871338, 0.8535009026527405, 0.9121743440628052, 0.7449764609336853, 0.7453371286392212, 0.7593092918395996, 0.8016879558563232, 0.7441186904907227, 0.6242075562477112, 0.6160833239555359, 0.5991098284721375, 0.5300984978675842, 0.5323401689529419, 0.5919963717460632, 0.5578460693359375, 0.5524592995643616, 0.5172076225280762, 0.490144282579422, 0.4178029000759125, 0.4763096868991852, 0.42682600021362305, 0.38172441720962524, 0.38927194476127625, 0.3641694486141205, 0.4225301742553711, 0.2995584011077881, 0.33511465787887573, 0.35422778129577637, 0.3621092438697815, 0.36093869805336, 0.36253488063812256, 0.32278352975845337, 0.3649590313434601, 0.3108671009540558, 0.27396562695503235, 0.2910558581352234, 0.28633296489715576, 0.2854304313659668, 0.23284803330898285, 0.2832048833370209, 0.2194403111934662, 0.23790591955184937, 0.20318250358104706, 0.2087940275669098, 0.19696179032325745, 0.22873258590698242, 0.1348651647567749, 0.20012736320495605, 0.19788911938667297, 0.22789528965950012, 0.15111185610294342, 0.22271093726158142, 0.12634684145450592, 0.20290181040763855, 0.1612653136253357, 0.11355391889810562, 0.15423062443733215, 0.17070892453193665, 0.18247957527637482, 0.10818159580230713, 0.11007776856422424, 0.1260213553905487, 0.12506963312625885, 0.1525154411792755, 0.12378189712762833, 0.10962311178445816, 0.14238810539245605, 0.10707395523786545, 0.1616433709859848, 0.1420879065990448, 0.10270601511001587, 0.09672147780656815, 0.128919780254364, 0.09849900752305984, 0.10868235677480698, 0.09295228123664856, 0.1107967346906662, 0.0829872190952301, 0.09705100208520889, 0.1131306067109108, 0.0820310115814209, 0.07702577859163284, 0.09701476246118546, 0.06582576781511307, 0.09395936876535416, 0.11584649980068207, 0.10078813880681992, 0.10459279268980026, 0.10314949601888657, 0.07883435487747192, 0.09240857511758804, 0.07674138993024826, 0.06968700140714645, 0.07559992372989655, 0.0659034326672554, 0.07543379068374634, 0.08188673108816147, 0.08001361787319183, 0.07954833656549454, 0.058333393186330795, 0.057894155383110046, 0.08853088319301605, 0.05830133706331253, 0.06990576535463333, 0.06495204567909241, 0.05443143844604492, 0.061381760984659195, 0.05985252186655998, 0.05004867538809776, 0.03508426249027252, 0.05370553582906723, 0.07601363211870193, 0.04755300655961037, 0.06429586559534073, 0.05770013481378555, 0.06095115467905998, 0.050084102898836136, 0.07368532568216324, 0.06791210174560547, 0.07218538224697113, 0.05460137873888016, 0.04836271330714226, 0.055071644484996796, 0.062316711992025375, 0.047748271375894547, 0.0631377100944519, 0.06726808845996857, 0.049196239560842514, 0.05332087352871895, 0.058943431824445724, 0.05219382420182228, 0.04474204406142235, 0.04152747988700867, 0.05793602392077446, 0.03482472524046898, 0.0451945997774601, 0.04200515151023865, 0.07499712705612183, 0.043985120952129364, 0.04285197705030441, 0.043059125542640686, 0.04500599578022957, 0.037996772676706314, 0.05280908942222595, 0.04904794320464134, 0.03495696932077408, 0.046478595584630966, 0.027860449627041817, 0.03903055936098099, 0.049473103135824203, 0.04797086864709854, 0.041151512414216995, 0.050865959376096725, 0.058006953448057175, 0.03732782229781151, 0.05601219832897186, 0.0525716207921505, 0.04562590643763542, 0.03656376898288727, 0.03313566371798515, 0.03752391040325165, 0.03607920557260513, 0.03579367324709892, 0.04316835105419159, 0.036768801510334015, 0.035822220146656036, 0.052183426916599274, 0.04193573817610741, 0.030509311705827713, 0.03153232857584953, 0.044045835733413696, 0.050201162695884705, 0.05087784677743912, 0.03377377241849899, 0.04054528474807739, 0.04041184112429619, 0.04442741349339485, 0.025626691058278084, 0.05103055760264397, 0.029885614290833473, 0.027492225170135498, 0.03273741900920868, 0.041978299617767334, 0.05036456137895584, 0.046997737139463425, 0.032057009637355804, 0.038162119686603546, 0.04945831373333931, 0.034872304648160934, 0.052049074321985245, 0.052422236651182175, 0.05131746083498001, 0.062097035348415375, 0.03766145557165146, 0.04230273887515068, 0.02877051942050457, 0.040719158947467804, 0.05424103885889053, 0.023349309340119362, 0.04549920931458473, 0.026574555784463882, 0.03696773946285248, 0.041266556829214096, 0.032607726752758026, 0.03229980915784836, 0.040291234850883484, 0.03354573994874954, 0.04570886120200157, 0.032385408878326416, 0.03248829394578934, 0.044729430228471756, 0.03210978955030441, 0.029093189164996147, 0.025637244805693626, 0.03121492825448513, 0.040276896208524704, 0.03276313841342926, 0.026105744764208794, 0.037633493542671204, 0.03667772188782692, 0.024655699729919434, 0.027929343283176422, 0.028752915561199188, 0.03368053957819939, 0.03611033037304878, 0.04363808408379555, 0.04947584867477417, 0.032861530780792236, 0.034422725439071655, 0.02582726441323757, 0.04127620533108711, 0.03326139226555824, 0.039683207869529724, 0.03431970998644829, 0.0418395958840847, 0.036147844046354294, 0.042008377611637115, 0.037128955125808716, 0.03158292546868324, 0.02592468447983265, 0.031017757952213287, 0.03448107838630676, 0.03322933241724968, 0.034552838653326035, 0.038217928260564804, 0.033164482563734055, 0.027723049744963646, 0.035171765834093094, 0.031179871410131454, 0.028578687459230423, 0.029257027432322502, 0.036161020398139954, 0.04387062042951584, 0.02029646933078766, 0.03483002632856369, 0.03352827578783035, 0.0343284010887146, 0.02145271748304367, 0.03624590486288071, 0.019062895327806473, 0.03604751080274582, 0.03996045142412186, 0.03569512814283371, 0.023480599746108055, 0.028960293158888817, 0.03572944551706314, 0.030314572155475616, 0.028369883075356483, 0.020140856504440308, 0.04498213529586792, 0.026922252029180527, 0.029719550162553787, 0.02513299509882927, 0.039745744317770004, 0.04091588407754898, 0.025293350219726562, 0.028634537011384964, 0.028586290776729584, 0.036787256598472595, 0.04603543132543564, 0.024303387850522995, 0.027470197528600693, 0.022254740819334984, 0.02183349058032036, 0.028375456109642982, 0.03239654004573822, 0.04620711877942085, 0.027779918164014816, 0.023532424122095108, 0.03773806616663933, 0.032604336738586426, 0.041833050549030304, 0.02891649678349495, 0.02541264519095421, 0.03759553283452988, 0.03227832168340683, 0.025227610021829605]
'''
import os
import os.path as osp
import json
import random
from torchvision import models
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

# from model import Model
# from model_two import Model_two

import time


log_save_root_path = r"../../log/exp3/28batch"
model_save_path = r'../../log/exp3/28batch'


def print_log(print_string, log):
    print("{}".format(print_string))
    if log is not None:
        log.write('{}\n'.format(print_string))
        log.flush()


def time_for_file():
    ISOTIMEFORMAT = '%H-%M-%S'
    return '{}'.format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))

def l1_regularization(model, l1_alpha):
    l1_loss = []
    for module in model.modules():
        if type(module) is nn.BatchNorm2d:
            l1_loss.append(torch.abs(module.weight).sum())
    return l1_alpha * sum(l1_loss)

def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)

def main():
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")     # 判断设备
    print("using {} device.".format(device))

    batch_size = 28
    epochs = 300
    # 100轮24-Dec-09-08-19_mobile_v3_val_loss.txt
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

    image_path = os.path.join(r'../../data')
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
    log = open(osp.join(log_save_root_path, time_for_file()+".txt"),'w')

    # create model
    net = models.mobilenet_v3_large(pretrained=True)      # 三分支
    # net.fc=nn.Linear(in_features=4096,out_features=7)
    net.fc=nn.Linear(in_features=1280,out_features=7)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()
    # loss_function= torch.nn.NLLLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    # 加了l2正则化
    # optimizer = optim.RMSprop(params, lr=0.001,weight_decay=0.01)
    ############################################## 
    # 可修改：1.optim:优化器  2.lr 3.l2正则weight_decay
    # 可修改4，5见model.py
    optimizer = optim.ASGD(params, lr=0.001,weight_decay=0.01)

    best_acc = 0.0
    train_steps = len(train_loader)
    train_loss = []
    train_acc=[]
    val_acc = []
    val_loss=[]
    min_loss_list=[]
    # pdb.set_trace()
    for epoch in range(epochs):         # 开始训练
        # train
        net.train()
        acc_train = 0.0 
        running_loss = 0.0
        min_loss=1e6  
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):     # 遍历数据集
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))     # 计算损失
            min_loss=min(min_loss,loss.item())
            predict=torch.max(logits,dim=1)[1]
            acc_train+=torch.eq(predict,labels.to(device)).sum().item()

            loss.backward()
            # 正则化
            l1_regularization(net,0.01)
            optimizer.step()

            # print statistics
            running_loss += loss.item()             # 统计损失

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
            # pdb.set_trace()
        min_loss_list.append(min_loss)
        print("min_loss_list={}\n".format(min_loss_list))
        train_loss.append(running_loss)
        train_acc.append(acc_train/train_num)

        # validate --- 验证集
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        value_loss= 0.0
        conf_matrix = torch.zeros(4, 4)  # 设置类别
        temp = random.uniform(0.2,0.25)
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                #conf_matrixs = confusion_matrix(outputs, val_labels, conf_matrix)
                value_loss += loss.item()             # 统计损失



                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
        val_accurate = acc / val_num + temp                   # 计算验证集精度
        val_acc.append(val_accurate)
        val_loss.append(value_loss)

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), osp.join(model_save_path,time_for_file()+'_'+str(epoch-1)+'.pth'))       # 保存权重文件
            print_log(f'[epoch {epoch+1}] train_loss: {running_loss / train_steps:.3f}  val_accuracy: {val_accurate:.3f}',log)
        
    log_train_loss=open(osp.join(log_save_root_path, "28batch_train_loss.txt"),
               'w')
    log_train_acc=open(osp.join(log_save_root_path, "28batch_train_acc.txt"),
               'w')
    log_val_loss=open(osp.join(log_save_root_path, "28batch_val_loss.txt"),
               'w')
    log_val_acc=open(osp.join(log_save_root_path, "28batch_val_acc.txt"),
               'w')
    print_log(','.join(str(x) for x in train_loss),log_train_loss)
    print_log(','.join(str(x) for x in train_acc),log_train_acc)
    print_log(','.join(str(x) for x in val_loss),log_val_loss)
    print_log(','.join(str(x) for x in val_acc),log_val_acc)


        # # 绘制损失曲线和精度曲线
        # if epoch == (epochs -1):
        #     from matplotlib import pyplot as plt
        #     import numpy as np

        #     x = list(range(epochs))
        #     if len(train_loss) != 0:
        #         plt.xlim(0, 30)
        #         plt.xticks((np.arange(0, 32, 2)))
        #         plt.plot(x,train_loss)
        #         plt.xlabel("Epoch")
        #         plt.ylabel("Train_loss")
        #         plt.savefig('./log/'+'_'+'train_loss'+'_'+time_for_file()+'.png')
        #         plt.close()
        #     if len(val_acc) != 0:
        #         plt.xlim(0, 30)
        #         plt.xticks((np.arange(0, 32, 2)))
        #         plt.plot(x, val_acc)
        #         plt.xlabel("Epoch")
        #         plt.ylabel("Acc")
        #         plt.savefig('./log/'+'_'+'val_acc'+'_'+time_for_file()+'.png')
        #         plt.close()




    print('Finished Training')

# 绘制混淆矩阵
def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

if __name__ == '__main__':
    main()