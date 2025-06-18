---
layout: post
title: "[Upstage AI Lab] 13주차 - 딥러닝 구현"
description: "[Upstage AI Lab] 13주차 - DNN, CNN, RNN 구현"
author: "DoorNote"
date: 2025-06-18 10:00:00 +0900
#  permalink: //
categories:
    - AI | 인공지능
    - DL
tags: [Deep Learning, PyTorch, DNN, CNN, RNN]
use_math: true
comments: true
pin: false # 고정핀
math: true
mermaid: true
image: /assets/img/위키라이더-썸네일.png
---

## 들어가며

> 이번 글은 **PyTorch**를 활용해 **DNN, CNN, RNN** 구조를 직접 구현하면서      
> 모델의 내부 동작 원리를 이해하는 데 초점을 맞췄습니다.
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

> 이번 파트에서는 기본적인 **DNN(Deep Neural Network)** 모델을 직접 구현해본다.
{: .prompt-info }

- Library

```python
import torch # PyTorch 
import torch.nn as nn # 모델 구성을 위한 라이브러리
from torch.utils.data import DataLoader # optimizer 설정을 위한 라이브러리
import torchvision # PyTorch의 컴퓨터 비전 라이브러리
import torchvision.transforms as T # 이미지 변환을 위한 모듈
import torchvision.utils as vutils # 이미지를 쉽게 처리하기 위한 유틸리티 모듈
import matplotlib.pyplot as plt
import random
import torch.backends.cudnn as cudnn

# Random_seed 고정
def random_seed(seed_num) :
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed_num)

random_seed(42)
```

- Data Load

```python
mnist_transform = T.Compose([T.ToTensor()]) # 텐서 형식으로 변환
download_root = './MNIST_DATASET'

train_dataset = torchvision.datasets.MNIST(download_root, transform=mnist_transform, train=True, download=True) # train dataset 다운
test_dataset = torchvision.datasets.MNIST(download_root, transform=mnist_transform, train=False, download=True) # test dataset 다운
```

- **28x28**로 구성 돼 있으며 채널이 1개이기에 **흑백 사진**임을 알 수 있다.

```python
for image, label in train_dataset:
  print(image.shape, label)  # 여기서 image의 shape은 [C, H, W]로 구성됨
  break
```

<pre>
torch.Size([1, 28, 28]) 5
</pre>

- Train/Val 분리

```python
total_size = len(train_dataset)
train_num, valid_num = int(total_size * 0.8), int(total_size * 0.2)
print("Train data size: ", train_num)
print("Validation data size: ", valid_num)
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_num, valid_num])
```

<pre>
Train data size:  48000
Validation data size:  12000
</pre>

- DataLoader 정의

```python
batch_szie = 32

train_dataloader = DataLoader(train_dataset, batch_size=batch_szie, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_szie, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_szie, shuffle=False)
```

- Batch 형태 확인

```python
for images, labels in train_dataloader:
  print(images.shape, labels.shape)
  break
```

<pre>
torch.Size([32, 1, 28, 28]) torch.Size([32])
</pre>

- Model

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

- 가중치 초기화

```python
# 가중치 초기화 함수
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

- 최종 Model: 위 과정을 하나의 Class로 정의

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
model = DNN(hidden_dims=hidden_dims, num_classes=10, dropout_ratio=0.2, apply_batchnorm=True, apply_dropout=True, apply_activation=True, set_super=True)
init_method = "gaussian"
model.weight_initialization(init_method)
```

<pre>
DNN(
  (layers): ModuleList(
    (0): Linear(in_features=784, out_features=128, bias=True)
    (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): Dropout(p=0.2, inplace=False)
    (3): ReLU()
    (4): Linear(in_features=128, out_features=128, bias=True)
    (5): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): Dropout(p=0.2, inplace=False)
    (7): ReLU()
    (8): Linear(in_features=128, out_features=128, bias=True)
    (9): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): Dropout(p=0.2, inplace=False)
    (11): ReLU()
  )
  (classifier): Linear(in_features=128, out_features=10, bias=True)
  (softmax): Softmax(dim=1)
)
</pre>

```python
print(f'The model has {model.count_parameters():,} trainable parameters')
```

<pre>
The model has 569,226 trainable parameters
</pre>

- Loss Function

```python
# Label에 맞게 분류 > NLLLoss 사용
criterion = nn.NLLLoss()
```

- Optimizer

```python
lr = 0.001
hidden_dim = 128
hidden_dims = [784, hidden_dim, hidden_dim, hidden_dim]
model = DNN(hidden_dims=hidden_dims, num_classes=10, dropout_ratio=0.2, apply_batchnorm=True, apply_dropout=True, apply_activation=True, set_super=True)
optimizer = optim.Adam(model.parameters(), lr=lr)
```

- Train

```python
def training(model, dataloader, train_dataset, criterion, optimizer, device, epoch, num_epochs) :
    model.train()
    train_loss = 0.0
    train_accuracy = 0

    tbar = tqdm(dataloader)
    for images, labels in tbar :
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_accuracy += (predicted == labels).sum().item()

        tbar.set_description(f'Epoch {epoch+1}/{num_epochs}, Train_Loss: {train_loss:.4f}')

    train_loss /= len(dataloader)
    train_accuracy /= len(train_dataset)

    return model, train_loss, train_accuracy

def evaluation(model, dataloader, valid_dataset, criterion, device, epoch, num_epochs) :
    model.eval()
    valid_loss = 0.0
    valid_accuracy = 0

    with torch.no_grad() :
        tbar = tqdm(dataloader)
        for images, labels in tbar :
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            valid_accuracy += (predicted == labels).sum().item()

            tbar.set_description(f'Epoch {epoch+1}/{num_epochs}, Valid_Loss: {valid_loss:.4f}')

    valid_loss /= len(dataloader)
    valid_accuracy /= len(valid_dataset)

    return model,valid_loss, valid_accuracy

def training_loop(model, train_dataloader, valid_dataloader, criterion, optimizer, device, num_epochs, patience, model_name) :
    best_valid_loss = float('inf')
    early_stop_counter = 0
    valid_max_accuracy = -1

    for epoch in range(num_epochs) :
        model, train_loss, train_accuracy = training(model, train_dataloader, train_dataset, criterion, optimizer, device, epoch, num_epochs)
        model, valid_loss, valid_accuracy = evaluation(model, valid_dataloader, valid_dataset, criterion, device, epoch, num_epochs)
        
        if valid_accuracy > valid_max_accuracy :
            valid_max_accuracy = valid_accuracy

        if valid_loss < best_valid_loss :
            best_valid_loss = valid_loss
            torch.save(model.state_dict()i, f"./model_{model_name}.pt")

        else :
            early_stop_counter += 1

        print(f"Epoch {epoch+1}/{num_epochs}, Train_Loss: {train_loss:.4f}, Train_Accuracy: {train_accuracy:.4f}, Valid_Loss: {valid_loss:.4f}, Valid_Accuracy: {valid_accuracy:.4f}")

        if early_stop_counter >= patience :
            print("Early Stopping")
            break

    return model, valid_max_accuracy
```

```python
num_epochs = 100
patience = 3
scores = dict()
device = 'cuda:0'
model_name = "exp1"
init_method = 'kaiming'

model = DNN(hidden_dims=hidden_dims, num_classes=10, dropout_ratio=0.2, apply_batchnorm=True, apply_dropout=True, apply_activation=True, set_super=True)
model.weight_initialization(init_method)
model = model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

model, valid_max_accuracy = training_loop(model, train_dataloader, valid_dataloader, criterion, optimizer, device, num_epochs, patience, model_name)
scores[model_name] = valid_max_accuracy
```

<pre>
Epoch 1/100, Train_Loss: -1254.1931: 100%|██████████| 1500/1500 [00:08<00:00, 176.03it/s]
Epoch 1/100, Valid_Loss: -350.4877: 100%|██████████| 375/375 [00:01<00:00, 315.23it/s]
Epoch 1/100, Train_Loss: -0.8361, Train_Accuracy: 0.8579, Valid_Loss: -0.9346, Valid_Accuracy: 0.9382
Epoch 2/100, Train_Loss: -1365.1130: 100%|██████████| 1500/1500 [00:08<00:00, 177.13it/s]
Epoch 2/100, Valid_Loss: -355.5040: 100%|██████████| 375/375 [00:01<00:00, 324.88it/s]
Epoch 2/100, Train_Loss: -0.9101, Train_Accuracy: 0.9151, Valid_Loss: -0.9480, Valid_Accuracy: 0.9503
Epoch 3/100, Train_Loss: -1385.8775: 100%|██████████| 1500/1500 [00:08<00:00, 177.68it/s]
Epoch 3/100, Valid_Loss: -357.5532: 100%|██████████| 375/375 [00:01<00:00, 324.07it/s]
Epoch 3/100, Train_Loss: -0.9239, Train_Accuracy: 0.9279, Valid_Loss: -0.9535, Valid_Accuracy: 0.9553
Epoch 4/100, Train_Loss: -1400.4983: 100%|██████████| 1500/1500 [00:08<00:00, 177.52it/s]
Epoch 4/100, Valid_Loss: -358.7989: 100%|██████████| 375/375 [00:01<00:00, 340.53it/s]
Epoch 4/100, Train_Loss: -0.9337, Train_Accuracy: 0.9376, Valid_Loss: -0.9568, Valid_Accuracy: 0.9581
Epoch 5/100, Train_Loss: -1405.4431: 100%|██████████| 1500/1500 [00:08<00:00, 184.06it/s]
Epoch 5/100, Valid_Loss: -359.9167: 100%|██████████| 375/375 [00:01<00:00, 339.84it/s]
Epoch 5/100, Train_Loss: -0.9370, Train_Accuracy: 0.9388, Valid_Loss: -0.9598, Valid_Accuracy: 0.9608
Epoch 6/100, Train_Loss: -1411.7580: 100%|██████████| 1500/1500 [00:07<00:00, 190.95it/s]
Epoch 6/100, Valid_Loss: -360.8559: 100%|██████████| 375/375 [00:01<00:00, 319.57it/s]
Epoch 6/100, Train_Loss: -0.9412, Train_Accuracy: 0.9433, Valid_Loss: -0.9623, Valid_Accuracy: 0.9634
Epoch 7/100, Train_Loss: -1417.0492: 100%|██████████| 1500/1500 [00:08<00:00, 180.71it/s]
Epoch 7/100, Valid_Loss: -361.1227: 100%|██████████| 375/375 [00:01<00:00, 327.59it/s]
Epoch 7/100, Train_Loss: -0.9447, Train_Accuracy: 0.9467, Valid_Loss: -0.9630, Valid_Accuracy: 0.9639
Epoch 8/100, Train_Loss: -1422.2407: 100%|██████████| 1500/1500 [00:08<00:00, 181.93it/s]
Epoch 8/100, Valid_Loss: -360.8939: 100%|██████████| 375/375 [00:01<00:00, 323.78it/s]
Epoch 8/100, Train_Loss: -0.9482, Train_Accuracy: 0.9498, Valid_Loss: -0.9624, Valid_Accuracy: 0.9626
Epoch 9/100, Train_Loss: -1422.8521: 100%|██████████| 1500/1500 [00:08<00:00, 187.39it/s]
Epoch 9/100, Valid_Loss: -360.7753: 100%|██████████| 375/375 [00:01<00:00, 338.03it/s]
Epoch 9/100, Train_Loss: -0.9486, Train_Accuracy: 0.9500, Valid_Loss: -0.9621, Valid_Accuracy: 0.9628
Epoch 10/100, Train_Loss: -1425.4209: 100%|██████████| 1500/1500 [00:08<00:00, 184.94it/s]
Epoch 10/100, Valid_Loss: -363.1457: 100%|██████████| 375/375 [00:01<00:00, 341.63it/s]
Epoch 10/100, Train_Loss: -0.9503, Train_Accuracy: 0.9517, Valid_Loss: -0.9684, Valid_Accuracy: 0.9695
Epoch 11/100, Train_Loss: -1429.4238: 100%|██████████| 1500/1500 [00:08<00:00, 185.38it/s]
Epoch 11/100, Valid_Loss: -362.8645: 100%|██████████| 375/375 [00:01<00:00, 333.20it/s]
Epoch 11/100, Train_Loss: -0.9529, Train_Accuracy: 0.9546, Valid_Loss: -0.9676, Valid_Accuracy: 0.9681
Early Stopping
</pre>

- 평가

```python
model.eval()
total_labels = [] # 실제 값
total_preds = [] # 모델이 예측한 값
total_probs = [] # AUC 구하기 위해 각 Class를 예측한 확률 값

with torch.no_grad() :
    for images, labels in test_dataloader :
        images = images.to(device)
        labels = labels

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total_preds.extend(predicted.detach().cpu().tolist())
        total_labels.extend(labels.tolist())
        total_probs.append(outputs.detach().cpu().numpy())

total_preds = np.array(total_preds)
total_labels = np.array(total_labels)
total_probs = np.concatenate(total_probs, axis=0)
```

```python
precision = precision_score(total_labels, total_preds, average='macro')
recall = recall_score(total_labels, total_preds, average='macro')
f1 = f1_score(total_labels, total_preds, average='macro')
auc = roc_auc_score(total_labels, total_probs, average='macro', multi_class='ovr')

print(f"Precision: {precision}, Recall: {recall}, F1: {f1}, AUC: {auc}")
```

<pre>
Precision: 0.9711098052403464, Recall: 0.9708028372978227, F1: 0.9709088177178395, AUC: 0.9988558476539848
</pre>
