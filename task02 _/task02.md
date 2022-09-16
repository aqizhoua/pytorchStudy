### 第三章 pytorch的主要组成模块与fashionMNIST时装分类

结合fashionMNIST时装分类，讲述pytorch的主要组成模块。

#### 3.1 基本配置

对于一个PyTorch项目，我们需要导入一些Python常用的包来帮助我们快速实现功能。常见的包有os、numpy等，此外还需要调用PyTorch自身一些模块便于灵活使用，比如torch、torch.nn、torch.utils.data.Dataset、torch.utils.data.DataLoader、torch.optimizer等等。

```
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optimizer
```

根据前面我们对深度学习任务的梳理，有如下几个超参数可以统一设置，方便后续调试时修改：

- batch size
- 初始学习率（初始）
- 训练次数（max_epochs）
- GPU配置

```
batch_size = 16
# 批次的大小
lr = 1e-4
# 优化器的学习率
max_epochs = 100
```

GPU的设置有两种常见的方式：

```
# 方案一：使用os.environ，这种情况如果使用GPU不需要设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# 方案二：使用“device”，后续对要使用GPU的变量用.to(device)即可
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
```

#### 3.2 数据读入

通过Dataset+DataLoader的方式完成的，Dataset定义好数据的格式和数据变换形式，DataLoader用iterative的方式不断读入批次数据。

##### 三个函数

- `__init__`: 用于向类中传入外部参数，同时定义样本集
- `__getitem__`: 用于逐个读取样本集合中的元素，可以进行一定的变换，并将返回训练/验证所需的数据
- `__len__`: 用于返回数据集的样本数

构建好Dataset后，就可以使用DataLoader来按批次读入数据了。

参数：

- batch_size：样本是按“批”读入的，batch_size就是每次读入的样本数
- num_workers：有多少个进程用于读取数据
- shuffle：是否将读入的数据打乱
- drop_last：对于样本最后一部分没有达到批次数的样本，使其不再参与训练。

另外，PyTorch中的DataLoader的读取可以使用next和iter来完成。

代码如下：

```
class FMDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.images = df.iloc[:,1:].values.astype(np.uint8)
        self.labels = df.iloc[:, 0].values
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape(28,28,1)
#         print("image的尺寸：",image.shape)
        label = int(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        else: 
            image = torch.tensor(image/255., dtype=torch.float) #因为在0-255之间，归一化
        label = torch.tensor(label, dtype=torch.long)
        return image, label

train_df = pd.read_csv("./FashionMNIST/fashion-mnist_train.csv")
test_df = pd.read_csv("./FashionMNIST/fashion-mnist_test.csv")
train_data = FMDataset(train_df, data_transform)
test_data = FMDataset(test_df, data_transform)
```

#### 3.3 模型构建

Module 类是 nn 模块里提供的一个模型构造类，是所有神经⽹网络模块的基类，我们可以继承它来定义我们想要的模型。

时装秀中的最基本的模型构建如下：

```
class Net(nn.Module): #继承nn.Module
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential( #惯序
            nn.Conv2d(1, 32, 5), #二维卷积
            nn.ReLU(),#激活函数
            nn.MaxPool2d(2, stride=2),#池化
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential( #全连接层
            nn.Linear(64*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64*4*4)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x

model = Net()
model = model.cuda()
```

#### 3.4 模型初始化

在深度学习模型的训练中，权重的初始值极为重要。一个好的权重值，会使模型收敛速度提高，使模型准确率更精确。为了利于训练和减少收敛时间，我们需要对模型进行合理的初始化。PyTorch也在`torch.nn.init`中为我们提供了常用的初始化方法。

#### 3.5 损失函数

在PyTorch中，损失函数是必不可少的。它是数据输入到模型当中，产生的结果与真实标签的评价指标，我们的模型可以按照损失函数的目标来做出改进。

##### 交叉熵损失函数

```
torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
```

**功能**：计算交叉熵函数

**主要参数**：

`weight`:每个类别的loss设置权值。

`size_average`:数据为bool，为True时，返回的loss为平均值；为False时，返回的各样本的loss之和。

`ignore_index`:忽略某个类的损失函数。

`reduce`:数据类型为bool，为True时，loss的返回是标量。

计算公式如下： loss(x, class )=−log⁡(exp⁡(x[ class ])∑jexp⁡(x[j]))=−x[ class ]+log⁡(∑jexp⁡(x[j]))

```
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()
print(output)
tensor(2.0115, grad_fn=<NllLossBackward>)
```

 使用torch.nn模块自带的CrossEntropy损失 PyTorch会自动把整数型的label转为one-hot型，用于计算CE loss 这里需要确保label是从0开始的，同时模型不加softmax层（使用logits计算）,这也说明了PyTorch训练中各个部分不是独立的，需要通盘考虑。

```
criterion = nn.CrossEntropyLoss()
```

#### 3.6 训练和评估

成了上述设定后就可以加载数据开始训练模型了。首先应该设置模型的状态：如果是训练状态，那么模型的参数应该支持反向传播的修改；如果是验证/测试状态，则不应该修改模型参数。在PyTorch中，模型的状态设置非常简便。

##### 训练

```
# 训练和测试（验证）
# 各自封装成函数，方便后续调用
# 关注两者的主要区别：

# 模型状态设置
# 是否需要初始化优化器
# 是否需要将loss传回到网络
# 是否需要每步更新optimizer
# 此外，对于测试或验证过程，可以计算分类准确率‘

def train(epoch):
    model.train()
    train_loss = 0
    for data, label in train_loader:
        data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step() #更新权重
        train_loss += loss.item()*data.size(0)
    train_loss = train_loss/len(train_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
```

##### 测试

```
#验证
def val(epoch):       
    model.eval()
    val_loss = 0
    gt_labels = [] #真实标签 gt ground_truth
    pred_labels = [] #预测标签
    with torch.no_grad(): #不计算梯度
        for data, label in test_loader:
            data, label = data.cuda(), label.cuda()
            output = model(data)
            preds = torch.argmax(output, 1)
            gt_labels.append(label.cpu().data.numpy())
            pred_labels.append(preds.cpu().data.numpy())
            loss = criterion(output, label)
            val_loss += loss.item()*data.size(0)
    val_loss = val_loss/len(test_loader.dataset)
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
    acc = np.sum(gt_labels==pred_labels)/len(pred_labels)
    print('Epoch: {} \tValidation Loss: {:.6f}, Accuracy: {:6f}'.format(epoch, val_loss, acc))
```

输出结果：

```
for epoch in range(1, epochs+1):
    train(epoch)
    val(epoch)
```

out:

```
Epoch: 1 	Training Loss: 0.658712
Epoch: 1 	Validation Loss: 0.428963, Accuracy: 0.841000
Epoch: 2 	Training Loss: 0.416495
Epoch: 2 	Validation Loss: 0.332809, Accuracy: 0.877200
Epoch: 3 	Training Loss: 0.356760
Epoch: 3 	Validation Loss: 0.314435, Accuracy: 0.885100
Epoch: 4 	Training Loss: 0.325156
Epoch: 4 	Validation Loss: 0.289078, Accuracy: 0.896500
Epoch: 5 	Training Loss: 0.301867
Epoch: 5 	Validation Loss: 0.263743, Accuracy: 0.903400
Epoch: 6 	Training Loss: 0.286868
Epoch: 6 	Validation Loss: 0.254227, Accuracy: 0.907200
Epoch: 7 	Training Loss: 0.269632
Epoch: 7 	Validation Loss: 0.247076, Accuracy: 0.909700
Epoch: 8 	Training Loss: 0.260949
Epoch: 8 	Validation Loss: 0.237404, Accuracy: 0.914000
Epoch: 9 	Training Loss: 0.249371
Epoch: 9 	Validation Loss: 0.237191, Accuracy: 0.913600
Epoch: 10 	Training Loss: 0.238720
Epoch: 10 	Validation Loss: 0.217898, Accuracy: 0.920000
Epoch: 11 	Training Loss: 0.229119
Epoch: 11 	Validation Loss: 0.219696, Accuracy: 0.919600
Epoch: 12 	Training Loss: 0.222355
Epoch: 12 	Validation Loss: 0.218665, Accuracy: 0.918300
Epoch: 13 	Training Loss: 0.215519
Epoch: 13 	Validation Loss: 0.211161, Accuracy: 0.921000
Epoch: 14 	Training Loss: 0.209839
Epoch: 14 	Validation Loss: 0.213974, Accuracy: 0.919800
Epoch: 15 	Training Loss: 0.202330
Epoch: 15 	Validation Loss: 0.206288, Accuracy: 0.923900
Epoch: 16 	Training Loss: 0.194416
Epoch: 16 	Validation Loss: 0.203906, Accuracy: 0.925100
Epoch: 17 	Training Loss: 0.190425
Epoch: 17 	Validation Loss: 0.211644, Accuracy: 0.922100
Epoch: 18 	Training Loss: 0.185451
Epoch: 18 	Validation Loss: 0.205757, Accuracy: 0.921100
Epoch: 19 	Training Loss: 0.181223
Epoch: 19 	Validation Loss: 0.210840, Accuracy: 0.923100
Epoch: 20 	Training Loss: 0.177804
Epoch: 20 	Validation Loss: 0.199402, Accuracy: 0.928200
```

