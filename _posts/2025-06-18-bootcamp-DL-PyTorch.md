---
layout: post
title: "[Upstage AI Lab] 위키스레드 - DNN 구현"
description: "[Upstage AI Lab] DNN 구현해보기"
author: "DoorNote"
date: 2025-06-18 10:00:00 +0900
#  permalink: //
categories:
    - AI | 인공지능
    - Upstage AI Lab
tags: [Deep Learning, PyTorch, DNN]
use_math: true
comments: true
pin: false # 고정핀
math: true
mermaid: true
image: /assets/img/DNN.png
---

## 들어가며

> 이번 글은 **Upstage AI Lab** 과정 중 **위키스레드 담당자**로 선정되어 강의 내용을 요약한 내용이다.  
> **PyTorch**를 활용해 **DNN** 구현하면서 **모델의 내부 동작 원리**를 이해하는 데 초점을 맞췄다.
{: .prompt-tip }

<br>
<br>

## PyTorch 실습

---

### 1. PyTorch 작동 구조

> **PyTorch**로 딥러닝 모델을 구성할 때는 **크게 4가지 핵심 구성 요소**를 중심으로 작동한다.  
> 각각은 **PyTorch 내부의 특정 모듈과 연결**되어 있으며, 이 구조를 이해하면 전체 흐름이 훨씬 명확해진다.
{: .prompt-info }

![딥러닝-학습단계](/assets/img/딥러닝-학습단계.png){: width="800" .center}

- **Data**: **`torch.utils.data.Dataset`** 클래스를 상속받아 정의
- **Model**: **`torch.nn.Module`**을 상속하여 구현
- **Loss Function**: **`torch.nn`** 또는 **`torch.nn.functional`** 내에서 선택하여 사용
- **Optimization**: **`torch.optim`** 모듈에서 수행

<br>

#### **Data**

- 이 데이터를 **batch** 단위로 묶어주는 역할은 **DataLoader**가 수행하며
- 학습에 필요한 데이터를 효율적으로 불러오고 전처리할 수 있도록 구성

**Custom Dataset**

> **PyTorch**가 기본으로 제공하는 **Dataset** 클래스를 그대로 쓰면 **기능에 제한이 있다.**  
> 그래서 직접 데이터셋을 정의하는 경우, Dataset을 상속받아 **새로운 클래스를 만들어야 한다.**  
> 이때 반드시 아래 3**가지 메서드를 구현해야 한다.**
{: .prompt-tip }

- **`__init__`**: 객체가 **처음 생성될 때 실행**되며, 파일 경로 설정, 전처리 등 **초기 준비 작업을 수행**
- **`__getitem__`**: 특정 **Index**를 받아 해당 위치의 **데이터 샘플 하나를 반환**
- **`__len__`**: 전체 데이터셋의 샘플 개수 반환

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass
```

**⚠️ Custom Dataset 주의사항**

- **데이터 타입**  
    - **`__getitem__`** 메서드는 데이터를 반드시 **`torch.tensor`** 형태로 반환해야 한다.  
    - **list, tuple, dictionary** 형태도 가능하지만, 내부 값들은 모두 **tensor**여야 함

- **데이터 차원**  
    - **`DataLoader`**는 데이터를 batch로 묶기 때문에, **모든 데이터의 차원이 같아야 한다.**  
    - 예: 이미지 → 동일한 크기 (`height × width × channel`)  
    - 예: 텍스트 → 동일한 길이(`max_len`)로 패딩 필요

**DataLoader**

- **`Dataset`**에서 정의한 데이터를 **batch 단위로 묶어 반환**하는 역할
- 필수 인자로 **`Dataset`**을 받고, 주로 자주 설정하는 인자는 다음과 같다.
  - **`batch_size`**: 한 번에 불러올 데이터 수
  - **`shuffle`**: 학습 시 매 **epoch**마다 데이터 순서를 섞을지 여부

<br>

#### **Model**

- `forward()` 메서드 안에 데이터가 어떻게 흐를지(순전파 구조) 정의한다.
- 신경망의 **뼈대를 구성하는 부분**

**Torchvision**

- **Torchvision** 라이브러리는 이미지 분석에 특화된 다양한 모델을 제공
- **ResNet, VGG, AlexNet, EfficientNet, ViT** 같은 대표적인 모델들을 손쉽게 가져와 쓸 수 있다.
- [**Torchvision**](https://docs.pytorch.org/vision/stable/models.html)에서 여러 가지 모델의 목록을 확인

```python
# 예시: torchvision에서 ResNet50 모델 불러오기
import torchvision
model1 = torchvision.models.resnet50()
```

**PyTorch**

- **CV, 오디오, 생성 모델, NLP** 분야까지 **다양한 도메인의 모델이 공개되어 있다.**
- [**PyTorch Hub**](https://pytorch.org/hub/)에서 여러 모델의 목록을 확인

```python
# 예시: PyTorch Hub에서 ResNet50 모델 불러오기
import torch
model2 = torch.hub.load('pytorch/vision', 'resnet50')
```

<br>

#### **Optimizer**

- **`optimizer.zero_grad()`**: 이전 **gradient를 0으로 설정**
- **`model(data)`**: 데이터를 모델을 통해 연산
- **`loss_function(output, label)`**: loss 값 계산
- **`loss.backward()`**: loss 값에 대한 gradient 계산
- **`optimizer.step()`**: gradient를 이용하여 모델의 파라미터 업데이트

<br>

#### **Inference & Evaluation**

- **`model.eval()`** : 모델을 평가 모드로 전환 → 특정 레이어들이 학습과 추론 과정 각각 다르게 작동해야 하기 때문
- **`torch.no_grad()`** : 추론 과정에서는 **gradient** 계산이 필요하지 않음
- **Pytorch**를 이용하여 평가산식을 직접 구현 or **scikit-learn**을 이용하여 구현

<br>
<br>

### 2. DNN 구현

> 이번 파트에서는 기본적인 **DNN(Deep Neural Network)** 모델을 직접 구현해봤다.    
> **Jutyer** 파일을 **MarkDown**으로 변환하는게 궁금하다면 [**테디노트**](https://teddynote.herokuapp.com/convert)를 이용하자 (매우 편함)  
> 전체 코드는 [**GitHub**](https://github.com/GH-Door/Upstage-AI-lab/blob/main/Part6.%20Deep-Learning/code/DNN-MNIST.ipynb)에서 확인할 수 있다.
{: .prompt-info }

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


#### **Library**

```python
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torchvision 
import torchvision.transforms as T 
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn 
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader 
from tqdm.notebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score 
```


```python
# seed 고정
import random
import torch.backends.cudnn as cudnn

def random_seed(seed_num):
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed_num)

random_seed(42)
```

#### **Data Load**

```python
# Tensor 형태로 변환
mnist_transform = T.Compose([T.ToTensor()])# 이미지를 정규화하고 C, H, W 형식으로 바꿈
download_root = "../data/MNIST_DATASET" # 경로 지정

train_dataset = torchvision.datasets.MNIST(download_root, transform=mnist_transform, train=True, download=True)
test_dataset = torchvision.datasets.MNIST(download_root, transform=mnist_transform, train=False, download=True)
```


```python
for image, label in train_dataset:
    print(image.shape, label) # 여기서 image의 shape은 [C, H, W]로 구성
    break
```

<pre>
torch.Size([1, 28, 28]) 5
</pre>

```python
# train/val 분리
total_size = len(train_dataset)
train_num, valid_num = int(total_size * 0.8), int(total_size * 0.2) # 8 : 2 = train : valid
print(f"Train dataset: {train_num}")
print(f"Validation dataset: {valid_num}")
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_num, valid_num]) # train - valid set 나누기
```

<pre>
Train dataset: 48000
Validation dataset: 12000
</pre>

```python
batch_size = 32

# DataLoader 선언
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

for images, labels in train_dataloader:
  print(images.shape, labels.shape)
  break
```

<pre>
torch.Size([32, 1, 28, 28]) torch.Size([32])
</pre>

```python
grid = vutils.make_grid(images, nrow=8) # 각 행마다 8개의 이미지 배치하여 격자로 구성

# 시각화
plt.figure(figsize=(12,12))
plt.imshow(grid.numpy().transpose((1,2,0)))
plt.title("mini batch visualization")
plt.axis('off')
plt.show()
```

<img src="data:image/png;base64,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61p+4403tLVd0aG2blKz6tKlS1aoS5o0qcinTp2KtfeaSm2j4N5m7dtvv42FEQW3nDlzijxnzhzH+uvBgweLdffu3dNuq3///iKHh4eL3LJlS+32QrGmdtWqVSKHhYVFedujR48WeejQoSKfOXMmyvW5avuYdu3aiTxo0CBtze13333n2B7RVGqLl8cee0x7e7Wllfq5616jO2HCBLHu7t272lps1eeff669zkgoUmtq1esL6Gq///77b+3ju2TJEu3to6NVq1Yi79mzR+QnnnhC5FdeecW1PHfuXMs0HLEFAAAAABiNiS0AAAAAwGhMbAEAAAAARgu5GtuSJUtq++qpNQyIWWqvLvf+vZkzZ9bWO6g1O77U/am1EosWLRI5YUL9W0XtU3v8+HEr1Kk1PmrNlMq9HnLatGkBG1etWrV8un2fPn2sUNe0aVNt7WNs9rpr1qyZ47pjx47F6FhCQenSpbU1zh9//LHXn8FqnZ9aQ6vW5XvqLR+M1O8m91rThz3+Omo9rvtz9bBre9y6dcvyl/Pnz2u/H9Q6QPV6G4888ogVbNTX97p16xz7BNt+++03kRs1auT1vtSetx999JH2M33UqFFWqPP0G3PcuHFe/5atXr26Tz29A2nSpEkijxgxQnvtIdNwxBYAAAAAYDQmtgAAAAAAozGxBQAAAAAYLehrbNW6TLVvWIIECUQ+dOiQY71t165dtTUjau0Los/93P/p06f7VAd49epV7fPn3g9wzZo12noUldo/cd68eSLv3bvXCjVFixYVOUWKFLE2lsSJE4s8efJk13KTJk2091V7WV+7ds0KNWp/UbUnsVo/FJvUntGh3j/a35InT679HFWvP6DrHazWSap1Z+q1CtRej6HI15pata68RYsWruXNmzfHmd6kan/w2rVri7xjxw7H/shqTWCPHj0sE508eVKb9+3b57d9qfXsKVOmFHnFihUi07/9v/2SdTW16vdibNbUplB+e1WuXFnkgQMHau8fm/W//sARWwAAAACA0ZjYAgAAAACMxsQWAAAAAGC0oK+xLV++vE/9SE+cOOFY7/PVV1+JHBERIfKyZctErl+/vsh37971ctR4YObMmY79Ezt27ChyvHjxRG7cuLH28W/YsKHXNbVqX1q1F+r27dutUKPWp7dt21bkdOnSae9/5swZkT/55BO/jS1ZsmRe9zpV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78SLF09bcunJokWLHMs6PH3nxVP2Hew4YgsAAAAAMBoTWwAAAACA0ZjYAgAAAACMFidqbNVLxkenplYVHh4u8pkzZ7SXoHevD9q9e7d222+++Wa0WhV89tlnruWrV69awej3338XOW/evNrbp02bVuSECf/3Eu3cubP2dVOkSBHtttXnevv27a7lUaNGiXXbtm0T+a+//tLWGqktGdSa22rVqjnWH6o1VdeuXbPiIvVv8lRDrlJridu0aSPyoUOHrNig1sXGj6//7325c+cOyrYi/tS2bVttnaauFll9X6tt09R2P2XKlBE5ffr0ruXTp0/7MOrgpX4PutemqtcTUGvnw8LCRK5atar2O9X9M1u9roH63Kk1s2odv9qOT9WhQwfXcrdu3cS6GzduWCZQPz/UVnHuj6c3cubMKfKaNWu8vq9ai6c+XyVLlhQ5RYoUXm9b/TvUcR49etQKdfXr19dey0O1du1akWvXru31e8dX7u+1UKXWzTZv3jzG9q1+z6ntgvA/HLEFAAAAABiNiS0AAAAAwGhMbAEAAAAARosXoRZRxEIfJLX/a7NmzazYcvnyZce6ycSJE2trIz3V2K5cudKx/59a/xksevToIfInn3ziU03uggULXMvDhg3zad9qHZ/a3++bb76xAqV3794iDxo0yLGG8/3334+ztRPudctqPdxjjz3m07bq1q2rrbuMLWovOrX3pic1a9Z0LX/33XdWMHL/Gx/WIzFjxoza+6uv+fv373u9b1/v6/657Otzif8aM2aMtg7Wl9reAQMGaL8P7t69a4WaKlWqiBybnyGeamwDWbM5ZcoUK9Spv1meeOIJ7e1feOEFbc2tP7lfG0StBfb0O6506dJWMFB7dqvXI3C/hkzy5MmtuOLIkSMijxs3TtvzO67OR7z9POKILQAAAADAaExsAQAAAABGY2ILAAAAADBanKixVftX5sqVywoGag9F9x5jti1btlihZs6cOSI3atTIb9vevHmzth+ge71uTFu2bJlruUaNGtrbeuqjGlv1p9GtV0yUKJEVF1Fj6zv3XrG2CRMmaHuRe+pn6st3j9rnuU+fPiJ/8cUXAevlGIzUa0d0795de50EtbfmuXPnRE6WLJlrOXXq1NrnXb2OxWuvvSbyrVu3rGCXPXt2kTdt2qTt4xxI0amF90R97ps0aSLy/PnzrVCTNWtW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xsQWAAAAAGC0OFFj66l2tUOHDtqs07RpU5/2nTVrVscakb59+4ocFhambQHjqabB15qHUDBp0iSRV61a5VreuXOnWJcgQQKRixQp4tO+3FuS5M6d26f7LliwwKfbhyK1lVOxYsVETps2rcjp06ePkRrb4cOHi1y2bFmRN2/erK0/CVb//POPa7lJkyba26r17v7k/hlsy5s3r/Zzk89R/3ryySejdX/3Nngff/yxWHfy5Entfb/++muRr1y5ov2Md3+vqu311HrSYK2xPXHihPYxVtv/uKtfv77I8+fP125b995T98P70mzRqbHt37+/X8eC/15TpmrVqq7l6dOna68jMl1Z76mW3nQcsQUAAAAAGI2JLQAAAADAaExsAQAAAABGi5M1trHp1KlTjuvUHriVKlUSOV++fD71caPPm77O72FZ12tr+/btImfJkkXkt99+W+Ts2bN7PS61X/KBAwe8vm+oeuedd7S1k0eOHBF5x44dARvLe++951pu2bKl9rZqnZnaSzMUBLKGVpUjRw7HuvqH1b/fuHFD5JkzZwZwdKHn0qVLPt3+5s2bInft2tW1PHXqVMuf9uzZI/Lt27cda2xDtcZz5MiRjnW06ndmqVKltFmtsfWlj606DpilX79+Ue6ZC/9r1KiRyDNmzHC8bY8ePUL6vcgRWwAAAACA0ZjYAgAAAACMxsQWAAAAAGC0kKuxrVmzpsjJkiVzrMtUz2kvUaKEyEmSJBGZmtnASpw4sXa9Wjup9h9NmjRplPe9bNkykbdu3RrlbYUK9b3lqS7nzp07ftt3586dHfvqqf2Pz549KzI1mzFbVztmzBixrmDBgo51lLY2bdrEWG12KFLrsdz7JT6sVl6tSfdnXW26dOm0+1bfy+5C9ft406ZNIr/66quu5Tlz5mjrkEuXLq3t8d2gQQOv+9iq44D/+8FWqFDBcd369eu137dqXrduXZTG+LB9wXe5cuUSeeDAgSI3bdpU5Hv37jlez2TUqFHaHripUqXyelzq9696jYu4iCO2AAAAAACjMbEFAAAAABiNiS0AAAAAwGjxIrwsRImrPeHUc8X79OkjslonmzlzZu25577U5aiPia81PX/99ZdruWjRolYoUuuUn332WceaaLVuUn3uAsm9N6Nt3LhxMbZvU6k1Vs8884zIhw4dErlAgQJR3lfz5s1FHjFihMjp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"/>

#### **Model**

```python
class DNN(nn.Module):
  def __init__(self, hidden_dims, num_classes, dropout_ratio, apply_batchnorm, apply_dropout, apply_activation, set_super):
    if set_super:
      super().__init__()

    self.hidden_dims = hidden_dims
    self.layers = nn.ModuleList()

    for i in range(len(self.hidden_dims) - 1):
      self.layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))

      if apply_batchnorm:
        self.layers.append(nn.BatchNorm1d(self.hidden_dims[i+1]))

      if apply_activation:
        self.layers.append(nn.ReLU())

      if apply_dropout:
        self.layers.append(nn.Dropout(dropout_ratio))

    self.classifier = nn.Linear(self.hidden_dims[-1], num_classes)
    self.softmax = nn.LogSoftmax(dim = 1)

  def forward(self, x):
    """
    Input and Output Summary

    Input:
      x: [batch_size, 1, 28, 28]
    Output:
      output: [batch_size, num_classes]

    """
    x = x.view(x.shape[0], -1)  # [batch_size, 784]

    for layer in self.layers:
      x = layer(x)

    x = self.classifier(x) # [batch_size, 10]
    output = self.softmax(x) # [batch_size, 10]
    return output
```


```python
hidden_dim = 128
hidden_dims = [784, hidden_dim * 4, hidden_dim * 2, hidden_dim]
model = DNN(hidden_dims = hidden_dims, num_classes = 10, dropout_ratio = 0.2, apply_batchnorm = True, apply_dropout = True, apply_activation = True, set_super = True)
output = model(torch.randn((32, 1, 28, 28)))
```

#### **Weight Initialization**



```python
# 가중치 초기화
def weight_initialization(model, weight_init_method):
  for m in model.modules():
    if isinstance(m, nn.Linear):
      if weight_init_method == 'gaussian':
        nn.init.normal_(m.weight)
      elif weight_init_method == 'xavier':
        nn.init.xavier_normal_(m.weight)
      elif weight_init_method == 'kaiming':
        nn.init.kaiming_normal_(m.weight)
      elif weight_init_method == 'zeros':
        nn.init.zeros_(m.weight)

      nn.init.zeros_(m.bias)

  return model
```


```python
init_method = 'zeros' # gaussian, xavier, kaiming, zeros
model = weight_initialization(model, init_method)

for m in model.modules():
  if isinstance(m, nn.Linear):
    print(m.weight.data)
    break
```

<pre>
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]])
</pre>

#### **Final Model**

```python
# 최종 모델 코드
class DNN(nn.Module):
  def __init__(self, hidden_dims, num_classes, dropout_ratio, apply_batchnorm, apply_dropout, apply_activation, set_super):
    if set_super:
      super().__init__()

    self.hidden_dims = hidden_dims
    self.layers = nn.ModuleList()

    for i in range(len(self.hidden_dims) - 1):
      self.layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))

      if apply_batchnorm:
        self.layers.append(nn.BatchNorm1d(self.hidden_dims[i+1]))

      if apply_activation:
        self.layers.append(nn.ReLU())

      if apply_dropout:
        self.layers.append(nn.Dropout(dropout_ratio))

    self.classifier = nn.Linear(self.hidden_dims[-1], num_classes)
    self.softmax = nn.LogSoftmax(dim = 1)

  def forward(self, x):
    """
    Input and Output Summary

    Input:
      x: [batch_size, 1, 28, 28]
    Output:
      output: [batch_size, num_classes]

    """
    x = x.view(x.shape[0], -1)  # [batch_size, 784]

    for layer in self.layers:
      x = layer(x)

    x = self.classifier(x) # [batch_size, 10]
    output = self.softmax(x) # [batch_size, 10]
    return output

  def weight_initialization(self, weight_init_method):
    for m in self.modules():
      if isinstance(m, nn.Linear):
        if weight_init_method == 'gaussian':
          nn.init.normal_(m.weight)
        elif weight_init_method == 'xavier':
          nn.init.xavier_normal_(m.weight)
        elif weight_init_method == 'kaiming':
          nn.init.kaiming_normal_(m.weight)
        elif weight_init_method == 'zeros':
          nn.init.zeros_(m.weight)

        nn.init.zeros_(m.bias)

  def count_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)  # numel()은 텐서의 원소 개수를 반환하는 함수
```


```python
model = DNN(hidden_dims = hidden_dims, num_classes = 10, dropout_ratio = 0.2, apply_batchnorm = True, apply_dropout = True, apply_activation = True, set_super = True)
init_method = 'gaussian' # gaussian, xavier, kaiming, zeros
model.weight_initialization(init_method)
print(f'The model has {model.count_parameters():,} trainable parameters')
```

<pre>
The model has 569,226 trainable parameters
</pre>

#### **Loss Function**


```python
criterion = nn.NLLLoss()
```

#### **Optimizer**

```python
lr = 0.001
hidden_dim = 128
hidden_dims = [784, hidden_dim * 4, hidden_dim * 2, hidden_dim]
model = DNN(hidden_dims = hidden_dims, num_classes = 10, dropout_ratio = 0.2, apply_batchnorm = True, apply_dropout = True, apply_activation = True, set_super = True)
optimizer = optim.Adam(model.parameters(), lr = lr)
```

#### **Train**

```python
def training(model, dataloader, train_dataset, criterion, optimizer, device, epoch, num_epochs):
  model.train()  # 모델을 학습 모드로 설정
  train_loss = 0.0
  train_accuracy = 0

  tbar = tqdm(dataloader)
  for images, labels in tbar:
      images = images.to(device)
      labels = labels.to(device)

      # 순전파
      outputs = model(images)
      loss = criterion(outputs, labels)

      # 역전파 및 weights 업데이트
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # 손실과 정확도 계산
      train_loss += loss.item()
      # torch.max에서 dim 인자에 값을 추가할 경우, 해당 dimension에서 최댓값과 최댓값에 해당하는 인덱스를 반환
      _, predicted = torch.max(outputs, 1)
      train_accuracy += (predicted == labels).sum().item()

      # tqdm의 진행바에 표시될 설명 텍스트를 설정
      tbar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}")

  # 에폭별 학습 결과 출력
  train_loss = train_loss / len(dataloader)
  train_accuracy = train_accuracy / len(train_dataset)

  return model, train_loss, train_accuracy

def evaluation(model, dataloader, valid_dataset, criterion, device, epoch, num_epochs):
  model.eval()  # 모델을 평가 모드로 설정
  valid_loss = 0.0
  valid_accuracy = 0

  with torch.no_grad(): # model의 업데이트 막기
      tbar = tqdm(dataloader)
      for images, labels in tbar:
          images = images.to(device)
          labels = labels.to(device)

          # 순전파
          outputs = model(images)
          loss = criterion(outputs, labels)

          # 손실과 정확도 계산
          valid_loss += loss.item()
          # torch.max에서 dim 인자에 값을 추가할 경우, 해당 dimension에서 최댓값과 최댓값에 해당하는 인덱스를 반환
          _, predicted = torch.max(outputs, 1)
          valid_accuracy += (predicted == labels).sum().item()

          # tqdm의 진행바에 표시될 설명 텍스트를 설정
          tbar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Valid Loss: {loss.item():.4f}")

  valid_loss = valid_loss / len(dataloader)
  valid_accuracy = valid_accuracy / len(valid_dataset)

  return model, valid_loss, valid_accuracy


def training_loop(model, train_dataloader, valid_dataloader, criterion, optimizer, device, num_epochs, patience, model_name):
    best_valid_loss = float('inf')  # 가장 좋은 validation loss를 저장
    early_stop_counter = 0  # 카운터
    valid_max_accuracy = -1

    for epoch in range(num_epochs):
        model, train_loss, train_accuracy = training(model, train_dataloader, train_dataset, criterion, optimizer, device, epoch, num_epochs)
        model, valid_loss, valid_accuracy = evaluation(model, valid_dataloader, valid_dataset, criterion, device, epoch, num_epochs)

        if valid_accuracy > valid_max_accuracy:
          valid_max_accuracy = valid_accuracy

        # validation loss가 감소하면 모델 저장 및 카운터 리셋
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f"./model_{model_name}.pt")
            early_stop_counter = 0

        # validation loss가 증가하거나 같으면 카운터 증가
        else:
            early_stop_counter += 1

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}")

        # 조기 종료 카운터가 설정한 patience를 초과하면 학습 종료
        if early_stop_counter >= patience:
            print("Early stopping")
            break

    return model, valid_max_accuracy
```


```python
num_epochs = 100
patience = 3
scores = dict()
device = 'cpu' # cpu 설정 (Mac Loacal 진행)
model_name = 'exp1'
init_method = 'kaiming' # gaussian, xavier, kaiming, zeros

model = DNN(hidden_dims = hidden_dims, num_classes = 10, dropout_ratio = 0.2, apply_batchnorm = True, apply_dropout = True, apply_activation = True, set_super = True)
model.weight_initialization(init_method)
model = model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr = lr)

model, valid_max_accuracy = training_loop(model, train_dataloader, valid_dataloader, criterion, optimizer, device, num_epochs, patience, model_name)
scores[model_name] = valid_max_accuracy
```

<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [1/100], Train Loss: 0.3178, Train Accuracy: 0.9046 Valid Loss: 0.1260, Valid Accuracy: 0.9615
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [2/100], Train Loss: 0.1622, Train Accuracy: 0.9505 Valid Loss: 0.1015, Valid Accuracy: 0.9678
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [3/100], Train Loss: 0.1314, Train Accuracy: 0.9590 Valid Loss: 0.0897, Valid Accuracy: 0.9729
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [4/100], Train Loss: 0.1116, Train Accuracy: 0.9646 Valid Loss: 0.0775, Valid Accuracy: 0.9753
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [5/100], Train Loss: 0.0981, Train Accuracy: 0.9687 Valid Loss: 0.0673, Valid Accuracy: 0.9797
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [6/100], Train Loss: 0.0857, Train Accuracy: 0.9730 Valid Loss: 0.0696, Valid Accuracy: 0.9787
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [7/100], Train Loss: 0.0772, Train Accuracy: 0.9754 Valid Loss: 0.0697, Valid Accuracy: 0.9784
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [8/100], Train Loss: 0.0674, Train Accuracy: 0.9785 Valid Loss: 0.0634, Valid Accuracy: 0.9823
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [9/100], Train Loss: 0.0609, Train Accuracy: 0.9801 Valid Loss: 0.0631, Valid Accuracy: 0.9809
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [10/100], Train Loss: 0.0586, Train Accuracy: 0.9815 Valid Loss: 0.0669, Valid Accuracy: 0.9803
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [11/100], Train Loss: 0.0544, Train Accuracy: 0.9821 Valid Loss: 0.0630, Valid Accuracy: 0.9831
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [12/100], Train Loss: 0.0554, Train Accuracy: 0.9821 Valid Loss: 0.0662, Valid Accuracy: 0.9807
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [13/100], Train Loss: 0.0461, Train Accuracy: 0.9846 Valid Loss: 0.0597, Valid Accuracy: 0.9825
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [14/100], Train Loss: 0.0431, Train Accuracy: 0.9865 Valid Loss: 0.0610, Valid Accuracy: 0.9823
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [15/100], Train Loss: 0.0442, Train Accuracy: 0.9854 Valid Loss: 0.0617, Valid Accuracy: 0.9816
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [16/100], Train Loss: 0.0393, Train Accuracy: 0.9869 Valid Loss: 0.0616, Valid Accuracy: 0.9828
Early stopping
</pre>

#### **Batch normalization 제외**


```python
# Batch normalization을 제외하고 학습 진행
model_name = 'exp2'
init_method = 'kaiming' # gaussian, xavier, kaiming, zeros

model = DNN(hidden_dims = hidden_dims, num_classes = 10, dropout_ratio = 0.2, apply_batchnorm = False, apply_dropout = True, apply_activation = True, set_super = True)
model.weight_initialization(init_method)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr = lr)
model, valid_max_accuracy = training_loop(model, train_dataloader, valid_dataloader, criterion, optimizer, device, num_epochs, patience, model_name)
scores[model_name] = valid_max_accuracy
```

<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [1/100], Train Loss: 0.2839, Train Accuracy: 0.9134 Valid Loss: 0.1365, Valid Accuracy: 0.9599
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [2/100], Train Loss: 0.1316, Train Accuracy: 0.9611 Valid Loss: 0.1105, Valid Accuracy: 0.9678
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [3/100], Train Loss: 0.1022, Train Accuracy: 0.9704 Valid Loss: 0.0892, Valid Accuracy: 0.9738
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [4/100], Train Loss: 0.0855, Train Accuracy: 0.9745 Valid Loss: 0.1010, Valid Accuracy: 0.9711
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [5/100], Train Loss: 0.0743, Train Accuracy: 0.9772 Valid Loss: 0.0817, Valid Accuracy: 0.9782
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [6/100], Train Loss: 0.0636, Train Accuracy: 0.9808 Valid Loss: 0.0889, Valid Accuracy: 0.9778
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [7/100], Train Loss: 0.0575, Train Accuracy: 0.9824 Valid Loss: 0.0840, Valid Accuracy: 0.9780
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [8/100], Train Loss: 0.0523, Train Accuracy: 0.9838 Valid Loss: 0.0817, Valid Accuracy: 0.9785
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [9/100], Train Loss: 0.0465, Train Accuracy: 0.9863 Valid Loss: 0.0806, Valid Accuracy: 0.9801
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [10/100], Train Loss: 0.0450, Train Accuracy: 0.9868 Valid Loss: 0.0900, Valid Accuracy: 0.9786
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [11/100], Train Loss: 0.0421, Train Accuracy: 0.9875 Valid Loss: 0.0995, Valid Accuracy: 0.9778
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [12/100], Train Loss: 0.0393, Train Accuracy: 0.9883 Valid Loss: 0.0861, Valid Accuracy: 0.9792
Early stopping
</pre>

#### **Dropout을 제외**


```python
# Dropout을 제외하고 학습 진행
model_name = 'exp3'
init_method = 'kaiming' # gaussian, xavier, kaiming, zeros

model = DNN(hidden_dims = hidden_dims, num_classes = 10, dropout_ratio = 0.2, apply_batchnorm = True, apply_dropout = False, apply_activation = True, set_super = True)
model.weight_initialization(init_method)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr = lr)
model, valid_max_accuracy = training_loop(model, train_dataloader, valid_dataloader, criterion, optimizer, device, num_epochs, patience, model_name)
scores[model_name] = valid_max_accuracy
```

<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [1/100], Train Loss: 0.2168, Train Accuracy: 0.9348 Valid Loss: 0.1049, Valid Accuracy: 0.9667
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [2/100], Train Loss: 0.1039, Train Accuracy: 0.9671 Valid Loss: 0.0888, Valid Accuracy: 0.9737
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [3/100], Train Loss: 0.0786, Train Accuracy: 0.9747 Valid Loss: 0.0842, Valid Accuracy: 0.9748
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [4/100], Train Loss: 0.0646, Train Accuracy: 0.9802 Valid Loss: 0.0718, Valid Accuracy: 0.9777
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [5/100], Train Loss: 0.0522, Train Accuracy: 0.9828 Valid Loss: 0.0698, Valid Accuracy: 0.9792
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [6/100], Train Loss: 0.0422, Train Accuracy: 0.9864 Valid Loss: 0.0738, Valid Accuracy: 0.9783
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [7/100], Train Loss: 0.0381, Train Accuracy: 0.9878 Valid Loss: 0.0729, Valid Accuracy: 0.9799
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [8/100], Train Loss: 0.0373, Train Accuracy: 0.9876 Valid Loss: 0.0803, Valid Accuracy: 0.9790
Early stopping
</pre>

#### **Activation Function 제외**


```python
# 활성화 함수(activation function)를 제외하고 학습을 진행합니다.
model_name = 'exp4'
init_method = 'kaiming' # gaussian, xavier, kaiming, zeros

model = DNN(hidden_dims = hidden_dims, num_classes = 10, dropout_ratio = 0.2, apply_batchnorm = True, apply_dropout = True, apply_activation = False, set_super = True)
model.weight_initialization(init_method)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr = lr)
model, valid_max_accuracy = training_loop(model, train_dataloader, valid_dataloader, criterion, optimizer, device, num_epochs, patience, model_name)
scores[model_name] = valid_max_accuracy
```

<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [1/100], Train Loss: 0.4584, Train Accuracy: 0.8641 Valid Loss: 0.3394, Valid Accuracy: 0.9050
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [2/100], Train Loss: 0.3789, Train Accuracy: 0.8898 Valid Loss: 0.3528, Valid Accuracy: 0.8998
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [3/100], Train Loss: 0.3584, Train Accuracy: 0.8947 Valid Loss: 0.3263, Valid Accuracy: 0.9114
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [4/100], Train Loss: 0.3506, Train Accuracy: 0.8959 Valid Loss: 0.3134, Valid Accuracy: 0.9120
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [5/100], Train Loss: 0.3430, Train Accuracy: 0.9010 Valid Loss: 0.3208, Valid Accuracy: 0.9101
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [6/100], Train Loss: 0.3347, Train Accuracy: 0.9025 Valid Loss: 0.3184, Valid Accuracy: 0.9096
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [7/100], Train Loss: 0.3344, Train Accuracy: 0.9024 Valid Loss: 0.3160, Valid Accuracy: 0.9113
Early stopping
</pre>

#### **Weight Initialization 0으로**


```python
model_name = 'exp5'
init_method = 'zeros' # gaussian, xavier, kaiming, zeros

model = DNN(hidden_dims = hidden_dims, num_classes = 10, dropout_ratio = 0.2, apply_batchnorm = True, apply_dropout = True, apply_activation = True, set_super = True)
model.weight_initialization(init_method)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr = lr)
model, valid_max_accuracy = training_loop(model, train_dataloader, valid_dataloader, criterion, optimizer, device, num_epochs, patience, model_name)
scores[model_name] = valid_max_accuracy
```

<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [1/100], Train Loss: 2.3017, Train Accuracy: 0.1118 Valid Loss: 2.3007, Valid Accuracy: 0.1133
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [2/100], Train Loss: 2.3015, Train Accuracy: 0.1121 Valid Loss: 2.3007, Valid Accuracy: 0.1133
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [3/100], Train Loss: 2.3015, Train Accuracy: 0.1121 Valid Loss: 2.3008, Valid Accuracy: 0.1133
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [4/100], Train Loss: 2.3015, Train Accuracy: 0.1121 Valid Loss: 2.3008, Valid Accuracy: 0.1133
</pre>
<pre>
  0%|          | 0/1500 [00:00<?, ?it/s]
</pre>
<pre>
  0%|          | 0/375 [00:00<?, ?it/s]
</pre>
<pre>
Epoch [5/100], Train Loss: 2.3015, Train Accuracy: 0.1121 Valid Loss: 2.3007, Valid Accuracy: 0.1133
Early stopping
</pre>

#### **Inference & Evaluation**


```python
# BatchNorm, Dropout을 사용한 모델을 로드
model = DNN(hidden_dims = hidden_dims, num_classes = 10, dropout_ratio = 0.2, apply_batchnorm = True, apply_dropout = True, apply_activation = True, set_super = True)
model.load_state_dict(torch.load("..models/model_exp1.pt"))
model = model.to(device)
```

- Inference

```python
model.eval()
total_labels = []
total_preds = []
total_probs = []
with torch.no_grad():
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels

        outputs = model(images)
        # torch.max에서 dim 인자에 값을 추가할 경우, 해당 dimension에서 최댓값과 최댓값에 해당하는 인덱스를 반환
        _, predicted = torch.max(outputs.data, 1)

        total_preds.extend(predicted.detach().cpu().tolist())
        total_labels.extend(labels.tolist())
        total_probs.append(outputs.detach().cpu().numpy())

total_preds = np.array(total_preds)
total_labels = np.array(total_labels)
total_probs = np.concatenate(total_probs, axis= 0)
```

- Evaluation

```python
# precision, recall, f1를 계산
precision = precision_score(total_labels, total_preds, average='macro')
recall = recall_score(total_labels, total_preds, average='macro')
f1 = f1_score(total_labels, total_preds, average='macro')

# AUC를 계산
# 모델의 출력으로 nn.LogSoftmax 함수가 적용되어 결과물이 출력
total_probs = np.exp(total_probs)
auc = roc_auc_score(total_labels, total_probs, average='macro', multi_class = 'ovr')
print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC: {auc}')
```

<pre>
Precision: 0.9852826409260971, Recall: 0.9851242188042052, F1 Score: 0.985192726704814, AUC: 0.9997577614547168
</pre>