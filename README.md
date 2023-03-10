# Agricultural_Diseases_Dentification

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
When you run a resnet etc for comparison experiments, you can use a 30 series graphics card, you can configure the environment with the following command:

```python
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html -i https://pypi.douban.com/simple
```

### Step 3: Create a Python Environment

The code requires

* Cuda:11.3

* Python 3.8.1

  ![imag-1](./pic/pic.png)

  ```python
  pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html -i https://pypi.douban.com/simple
  ```

## Data
### Point1:Data Source
dm = downy mildew,  http://www.nongyisheng.com/nongzi?id=uoktntj  
pm = powdery mildew,  http://www.nongyisheng.com/nongzi?id=uizygcc  
als = bacterial angular leaf spot,  http://www.nongyisheng.com/nongzi?id=uizezsy  
tls = target leaf spot, http://www.nongyisheng.com/nongzi?id=uuwhllf   
gsb = gummy stem blight, http://www.nongyisheng.com/nongzi?id=uoktbkk
fw = fusarium wilt,  http://www.nongyisheng.com/nongzi?id=uokqdkq 
an = anthracnose,  http://www.nongyisheng.com/nongzi?id=uuwhrhl  

Website：  
"http://www.nongyisheng.com/p/detail.html?qid=" + qid  
Json  Data Website
"http://www.nongyisheng.com/question/detail?fr=pc&qid=" + qid + "&rn=1"  

### Point2:Image naming method 

qid_num.jpg:  
qid is the id of each question in Dr. Farmer's database. Each question has 0-6 photos such as '576669_2.jpg'  

## Repository structure

```
├── Agricultural_Diseases_Dentification
│   ├── draw_pic
│   ├── experiment
│      ├── exp1_model
│      ├── exp2_optimizer
│      ├── exp3_batch
│      ├── exp4_flooding
│   ├── log
│      ├── exp1
│      ├── exp2
│      ├── exp3
│      ├── exp4
│   ├── data
│      ├── train
│         ├── als
│         ├── an
│         ├── dm
│         ├── fw
│         ├── gsb
│         ├── pm
│         ├── tls
│      ├── val
│         ├── als
│         ├── an
│         ├── dm
│         ├── fw
│         ├── gsb
│         ├── pm
│         ├── tls
```
