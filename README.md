# dentification-of-agricultural-diseases-and-pests 
 农业病虫害识别

## Steps To Run the Code
### Step 1: Install Anaconda
Go to the [Anaconda Website](https://www.anaconda.com/products/distribution) and choose a Python 3.x graphical installer.

### Step 2: Clone the Repository
In order to clone the repository, use the following git command in your command line.
```
git clone https://github.com/YiQuanMarx/Agricultural_Diseases_Dentification.git
```
and then move into the project directory with
```
cd Identification-of-agricultural-diseases-and-pests
```
## Step 3: Create a Python Environment
The code requires
* [PyTorch](https://github.com/pytorch/vision):The linker is the corresponding torchvision versions and supported Python versions.The code use torch version 1.4.0.
* torchvision:version->0.4.0~0.7.0.And the code use version 0.5.0.
* Python 3.6 
and the requirements listed are as follow:
* pandas
* numpy  
* seaborn   
* tqdm  

## Step 4: Download the Dataset
In order to download the dataset, download it in: https://pan.baidu.com/s/1VPm3oV4wi9mWD5J2gyaHtg?pwd=6jq7. The key code is: 6jq7   
### big_data
dm = downy mildew, 霜霉病  
pm = powdery mildew, 白粉病  
als = bacterial angular leaf spot, 细菌性角斑病  
tls = target leaf spot, 靶斑病  
gsb = gummy stem blight, ，蔓枯病  
fw = fusarium wilt, 枯萎病  
an = anthracnose, 炭疽病  

霜霉病 - http://www.nongyisheng.com/nongzi?id=uoktntj  
白粉病 - http://www.nongyisheng.com/nongzi?id=uizygcc  
细菌性角斑病 - http://www.nongyisheng.com/nongzi?id=uizezsy  
靶斑病 - http://www.nongyisheng.com/nongzi?id=uuwhllf  
蔓枯病 - http://www.nongyisheng.com/nongzi?id=uoktbkk  
枯萎病 - http://www.nongyisheng.com/nongzi?id=uokqdkq  
炭疽病 - http://www.nongyisheng.com/nongzi?id=uuwhrhl  

文件命名  
qid_#.jpg  
qid是每个问题在农医生的数据库里的id， 每个问题有0-6张照片不等。  

网站：  
"http://www.nongyisheng.com/p/detail.html?qid=" + qid  
json 数据网站  
"http://www.nongyisheng.com/question/detail?fr=pc&qid=" + qid + "&rn=1"  

### small_data
Target 黄瓜靶斑病  
Spot 黄瓜细菌性角斑病  
Powdery 黄瓜白粉病  
Downy 黄瓜霜霉病   


# Repository structure
```
├── Agricultural_Diseases_Dentification
│   ├── big_model
│      ├── mobilenet
│         ├── mobilenet_v2.py
│      ├── model_data
│         ├── efficientnet-b3-5fb5a3c3.pth
│         ├── mobilenet_v2-b0353104.pth
│      ├── class_indices.json
│      ├── Demo_Efficientnet.py
│      ├── ECAAttention.py
│      ├── efficientnet.py
│      ├── main.py
│      ├── model.py
│      ├── model_two.py
│      ├── split_train_val.py
│      ├── train.py
│      ├── vgg16.py
│   ├── small_model
│      ├── mobilenet
│         ├── mobilenet_v2.py
│      ├── class_indices.json
│      ├── ECAAttention.py
│      ├── efficientnet.py
│      ├── model.py
│      ├── model_two.py
│      ├── train.py
├── README.md
```
