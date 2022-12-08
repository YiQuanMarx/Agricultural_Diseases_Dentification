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
cd Agricultural_Diseases_Dentification
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
