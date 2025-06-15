---
layout: post
title: "[논문 Review] Gradient-Based Learning"
description: "Gradient-Based 기법을 활용한 패턴 및 문서 인식 방법을 정리했습니다."
author: "DoorNote"
date: 2025-04-26 10:00:00 +0900
# permalink: /big-data-10/
categories:
    - Review
    - 논문 Review
tags: ['Deep Learning', 'CNN', 'Gradient Descent']
use_math: true
comments: true
pin: false # 고정핀
math: true
mermaid: true
image: /assets/img/paper-review1.png
---

## 들어가며

> 이번 포스팅에서는 **LeNet-5**라고 알려져 있는 [Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) 논문을 읽고 요약 정리했습니다.
{: .prompt-tip}

<br>

## 핵심 키워드

> - **Neural Networks** : 인간의 뇌 구조를 모방한 연산 시스템, 입력,출력을 연결하는 신경망 구조를 학습에 활용<br>
> - **OCR** : 이미지나 문서 내의 문자 데이터를 인식하고 디지털 텍스트로 변환하는 기술<br>
> - **Document Recognition** : 문서 내의 텍스트, 구조, 레이아웃 등을 인식하여 이해하는 기술<br>
> - **Machine Learning** : 데이터를 이용해 기계가 스스로 패턴을 학습하고 예측하거나 분류하는 기술<br>
> - **Backpropagation**: 신경망 학습을 위해 오차를 거꾸로 전파하는 **역전파 알고리즘**을 의미<br>
> - **Gradient-Based Learning** : 손실 함수의 기울기를 계산해 모델의 파라미터를 업데이트하는 학습 방법<br>
> - **Convolutional Neural Networks (CNN)** : 이미지나 영상 데이터에 특화된 신경망 구조<br>
> - **Graph Transformer Networks (GTN)** : 복잡한 모듈 간의 관계를 그래프 구조로 모델링<br>
> - **Finite State Transducers (FST)** : 입력을 받아 상태를 전이시키고 출력을 생성하는 계산 모델

<br>
<br>

## I. Introduction

> 본 논문은 **전통적인 패턴 인식 시스템**이 가지는 **한계점**을 지적하며, 사람이 설계하는 **feature engineering** 대신, **자동화된 학습(automatic feature learning)** 을 통해 문제를 해결할 수 있음을 강조한다.
{: .prompt-info}

<br>

초기 패턴 인식은 수작업 feature extractor 와 **일반화 가능한 classifier**의 조합으로 구성되었다. 그러나 **feature extractor**의 설계는 과업별로 많은 시간, 노력을 요구하며, 설계자의 **경험에 따라 성능이 좌우되는 문제가 있다.**<br>
**논문에서는 두 가지 문제를 중심으로 접근**하며 내용은 아래와 같다.

- **Character recognition**: 개별 문자를 인식하는 과제, 픽셀 수준에서 직접 학습하는 **CNN** 구조가 적합함을 설명
- **Document understanding**: 문장, 문서를 해석하는 과제, 통합된 모델인 **GTN** 를 통해 접근할 필요성을 제시  

![Gradient-Based-1](/assets/img/Gradient-Based-1.png){: width="500" .center}
_Feature Extraction Module과 Classifier Module의 조합_

<br>

### 전통적 접근 방식의 한계와 변화

전통적인 **feature engineering** 방식은 문제마다 새로운 **feature**를 설계해야 했고, 이로 인해 높은 개발 비용과 제한된 확장성을 초래하였다. 특히 **feature extractor**의 성능이 인식 시스템 전체의 성능을 결정짓는 경우가 많아, 설계자의 역량에 과도하게 의존하는 문제가 존재, 이러한 한계를 극복할 수 있었던 배경은 **다음과 같은 기술적 변화 때문이다.**

1. **컴퓨터 성능의 향상**  
   고성능 하드웨어의 발전으로 복잡한 모델을 **brute-force** 방식으로 학습하는 것이 가능해졌다.

2. **대규모 데이터의 등장**  
   다양한 실제 데이터를 활용하여, **feature**를 수작업으로 설계하지 않고도 자동으로 특징을 학습할 수 있게 되었다.

3. **효과적인 학습 알고리즘의 개발**  
   **Backpropagation** 알고리즘을 이용해 다층 신경망을 학습시킬 수 있게 되면서, 고차원 데이터에 대한 패턴 인식이 현실화되었다.

<br>

본 논문은 이러한 기술적 변화를, 사람이 설계한 **feature extractor**에 의존하지 않고 데이터로부터 자동으로 특징을 학습하는 신경망 기반 시스템을 제안한다. 이를 통해 기존 패턴 인식 시스템이 가진 구조적 한계를 극복하고자 한다.

<br>
<br>

## II. Convolutional Neural Networks for Isolated Character Recognition

> 본 섹션에서는 손글씨 인식 문제를 해결하기 위해 제안된 **Convolutional Neural Network (CNN)** 구조와, 이를 대표하는 모델인 **LeNet-5**의 구조를 다룬다.
{: .prompt-info}

<br>

기존의 패턴 인식 시스템은 수작업 **feature extractor**와 **trainable classifier**의 조합으로 이루어졌지만,  
이 접근 방식은 다음과 같은 한계를 지녔다.

- 입력 이미지가 고해상도일 경우, **feature extractor**를 설계하고 학습하는 데 필요한 파라미터 수가 급격히 증가
- 입력 데이터에 **약간의 변형(shift, scale, distortion)**만 있어도 인식 성능이 급격히 저하
- **Fully-connected network**는 입력의 **공간적 구조를 고려 ❌**, 인접한 픽셀 간 관계를 학습하기 어렵다.

이러한 한계를 극복하기 위해 제안된 것이 바로 **Convolutional Neural Network (CNN)** 이다.

<br>

### A. Convolutional Networks

![Gradient-Based-2](/assets/img/Gradient-Based-2.png){: width="700" .center}

**CNN**은 다음 **세 가지** 핵심 아이디어를 기반으로 설계되었다.

1. **Local Receptive Fields**  
   각 뉴런은 입력 이미지의 일부분만을 바라본다. 이를 통해 국소적(local) 특징을 추출할 수 있으며,  
   **엣지(edge), 코너(corner) 등 저수준(low-level)** 패턴을 감지할 수 있다.

2. **Shared Weights**  
   동일한 feature를 이미지 전체에 적용할 수 있도록 가중치를 공유한다.  
   이 방법은 모델의 파라미터 수를 크게 줄여주고, 입력 데이터의 **shift나 distortion에 대한 강인성을 높인다.**

3. **Subsampling (Pooling)**  
   **feature map**의 해상도를 조금씩 줄여서, 특징의 위치에 대한 민감도를 낮추고 모델의 **일반화 성능을 높인다.**

<br>

이러한 구조 덕분에 **CNN**은 입력 데이터의 구조를 효율적으로 학습할 수 있으며, 입력에 **약간의 변형이 생기더라도 높은 인식 성능을 유지할 수 있다.**

---

### B. LeNet-5 Architecture

![Gradient-Based-3](/assets/img/Gradient-Based-3.png){: width="600" .center}

<br>

**LeNet-5**는 **손글씨 숫자 인식(MNIST 등)**을 목표로 설계된 대표적인 **CNN** 모델이다.  
전체적인 구조는 다음과 같다.

- **Input Layer**: 32x32 크기의 normalized grayscale 이미지
- **C1 Layer**: 6개의 28x28 feature map을 생성하는 Convolution layer
- **S2 Layer**: 6개의 14x14 feature map을 생성하는 Subsampling (Average Pooling) layer
- **C3 Layer**: 16개의 10x10 feature map을 생성하는 Convolution layer
- **S4 Layer**: 16개의 5x5 feature map을 생성하는 Subsampling layer
- **C5 Layer**: 120개의 1x1 feature map (사실상 Fully Connected Layer로 동작)
- **F6 Layer**: 84개의 노드를 가진 Fully Connected Layer
- **Output Layer**: 10개의 Euclidean RBF 노드 (0~9 숫자 분류)

<br>

LeNet-5의 핵심 아이디어는 다음과 같다.

- **특징 추출과 압축을 반복**하면서 점진적으로 고차원 feature를 학습한다.
- **Shared weights**를 통해 parameter 수를 줄이고, **Pooling**을 통해 위치 변화에 강인성을 가진다.
- 최종적으로 Fully Connected Layer를 통해 **전체 feature를 종합하여 분류를 수행한다.**

<br>

이 구조는 이후 발전된 다양한 **CNN 아키텍처(예: AlexNet, VGGNet 등)**의 기반이 되었으며,  
**딥러닝** 기반 **이미지 인식 기술** 발전에 있어 중요한 전환점을 마련했다.

<br>
<br>

## III. Gradient-Based Learning

> 본 섹션에서는 **Gradient-Based Learning**의 원리와 핵심 개념을 다룬다.
{: .prompt-info}

<br>

**Gradient-Based Learning**은 모델의 예측 결과와 실제 정답 간의 오차를 계산하고,  
이 오차를 줄이기 위해 파라미터(parameter)를 미세 조정하는 학습 방법이다.<br>
학습 과정은 다음과 같은 흐름으로 구성된다.

1. **모델 출력**  
   입력 패턴에 대해 모델이 예측 결과(output)를 생성

2. **Loss 계산**  
   모델의 예측 결과와 실제 정답 간의 오차(loss)를 계산
   (예: Mean Squared Error, Cross Entropy 등)

3. **Gradient 계산**  
   오차를 줄이기 위해 각 파라미터가 어느 방향으로, 얼마나 변화해야 하는지를 나타내는 **gradient**를 계산

4. **파라미터 업데이트**  
   계산된 gradient를 이용하여 파라미터를 업데이트
   가장 기본적인 업데이트 방법은 **Gradient Descent**이다.

<br>

### A. 학습 목표

학습의 목표는 전체 training dataset에 대해 평균적인 오차를 최소화하는 것이다.  
Loss function을 E(W)라 할 때, 학습은 다음과 같은 최적화 문제를 푸는 것과 같다.

$$
\min_{W} E(W)
$$

여기서 W는 모델의 trainable parameters를 의미한다.
하지만 실제로 중요한 것은 **training data**에서의 성능만이 아니라, **test data**에 대해서도 오차를 줄이는 것, 즉 일반화를 달성하는 것이다. 이를 위해 **Structural Risk Minimization**이라는 개념이 등장한다.

- 모델의 복잡도(h)가 낮을수록 train/test 간 오차 차이가 줄어든다.
- 너무 복잡한 모델은 training set에서는 오차가 낮지만, unseen data에서는 성능이 저하될 수 있다.

<br>

---

### B. Gradient Descent와 Backpropagation

Gradient Descent는 가장 기본적인 최적화 알고리즘으로, **Loss function**의 **gradient**를 따라 **parameter**를 업데이트하여 점진적으로 loss를 줄여간다.
파라미터 업데이트 식은 다음과 같다.

$$
W := W - \epsilon \nabla E(W)
$$

여기서
- **epsilon**은 learning rate를 의미하며,
- **nabla E(W)**는 **loss function**에 대한 W의 **gradient**이다.

보다 복잡한 모델에서는 각 층(layer)마다 gradient를 계산해야 하는데,  
이 때 필요한 것이 바로 **Backpropagation Algorithm**이다.
Backpropagation은 Chain Rule(연쇄 법칙)을 이용하여, 출력층에서 입력층으로 거꾸로 gradient를 효율적으로 전파하여 모든 파라미터의 업데이트 방향을 계산한다.<br>
이를 **역전파 알고리즘이라고 부른다.**

<br>

### C. Stochastic Gradient Descent (SGD)

전통적인 Gradient Descent는 전체 training set에 대해 gradient를 계산하지만, 이 방식은 데이터셋이 클 경우 계산량이 매우 커진다.
이를 해결하기 위해 **Stochastic Gradient Descent (SGD)** 방법이 제안되었다.

- 데이터 샘플 하나 또는 작은 mini-batch에 대해 gradient를 계산하고 즉시 파라미터를 업데이트한다.
- 이 방법은 계산량을 줄이고, 더 빠른 수렴을 가능하게 한다.
- 다만 업데이트에 noise가 포함되기 때문에, 수렴 과정에 진동이 발생할 수 있다.

<br>

**Gradient-Based Learning**은 CNN을 비롯한 다양한 신경망 구조의 학습을 가능하게 만든 핵심 기반 기술이며, 특히 **Backpropagation** 알고리즘은 **심층 신경망**의 학습을 실용적으로 만든 주요 요인 중 하나이다. **학습 과정에서 Training Error와 Test Error가 어떻게 감소하는지**를 아래 그래프를 통해 확인할 수 있다.

![Gradient-Based-4](/assets/img/Gradient-Based-4.png){: width="500" .center}
_Training Set Iterations에 따른 Error Rate 변화_

<br>
<br>

## IV. Learning in Real Handwriting Recognition Systems

> 본 섹션에서는 실제 손글씨 인식 문제에 **Gradient-Based Learning과 CNN 구조를 적용한 사례를 다룬다.**
{: .prompt-info}

<br>

손글씨 인식 문제는 단순히 문자 하나를 인식하는 문제에 그치지 않고, 문장이나 문서 수준에서 **segmentation**이라는 추가적인 과제를 포함한다.
Segmentation은 연속된 글자 스트림에서 각 문자를 올바르게 분리해내는 작업이다.  
전통적인 방법은 다음과 같은 과정을 거쳤다.

- **Heuristic Over-Segmentation**  
  - 가능한 모든 분할 지점을 탐색하여 후보를 생성하고,
- **Recognizer**  
  - 생성된 후보 중 최적의 조합을 선택하는 구조로 작동하였다.

그러나 이 방식은 다음과 같은 한계를 가진다.

- segmentation 품질에 강하게 의존 → 잘못 분리되면 인식 실패
- 후보군이 많을 경우 최적 조합을 찾는 데 연산량 급증
- 오류가 발생해도 정확한 **레이블링(labeling)**이 어려워, 인식기 학습이 힘듬

전통적인 **segmentation** 기반 방법은 다양한 형태의 손글씨를 다루기에 한계가 있었다.  
아래는 실제 훈련 데이터에 등장하는 다양한 왜곡의 예시이다.

![Gradient-Based-5](/assets/img/Gradient-Based-5.png){: width="500" .center}
_실제 손글씨 데이터에서 발생하는 다양한 변형 예시_

<br>

### CNN 기반 접근 방식의 이점

논문에서는 **CNN 기반 인식기(recognizer)**를 활용하여, feature를 자동으로 학습하고, segmentation에 의존하지 않고도 전체 단어 또는 문서를 인식하는 접근을 제안한다.

이 방식의 핵심은 다음과 같다

- feature extractor와 classifier를 통합하여, 입력 이미지에서 직접적으로 특징을 추출하고 분류
- segmentation 오류에 대한 민감도를 낮추어, 보다 견고한 인식이 가능
- 전체 시스템을 end-to-end로 학습시켜, feature와 classifier를 동시에 최적화

<br>

이를 통해, 기존 heuristic 기반 시스템 대비 손글씨 인식 성능을 크게 향상시킬 수 있었다.  
또한 데이터 레이블링 비용과 segmentation 오류를 줄이는 데에도 실질적인 이점을 제공하였다.

<br>
<br>

## 요약

이 논문은 전통적인 패턴 인식 방식(수작업 feature 설계 + classifier 조합)이 가진 한계를 지적하며,
사람이 직접 feature를 설계하는 대신, 데이터로부터 특징을 자동으로 학습하는 방법을 제안한다.

이를 위해 **Convolutional Neural Network (CNN)** 구조를 도입하고,
**Gradient-Based Learning (특히 Backpropagation)** 을 통해 네트워크를 학습시킨다.

CNN은 입력 이미지의 국소적(local) 특징을 효과적으로 추출하며,
Pooling 과정을 통해 위치 변화나 왜곡에도 강인한 특성을 확보할 수 있다.

논문에서는 이러한 CNN 구조를 손글씨 인식 문제에 적용하여,
전통적인 segmentation 기반 방식에 비해 뛰어난 성능을 달성했음을 보여준다.

결국, 사람이 feature를 일일이 설계하지 않아도,
데이터와 **Gradient** 기반 학습만으로 효과적인 인식이 가능하다는 사실을 실험을 통해 입증한 연구다.

<br>
<br>

## Code 구현

> 위 논문에서 제안한 **LeNet-5** 모델 구조를 기반으로, **MNIST** 데이터셋 분류를 위한 코드를 짧게 구현해봤습니다.<br>
> 전체 코드는 [GitHub 링크](https://github.com/GH-Door/Study/blob/main/DL/code/LeNet-5.ipynb)에서 확인 할 수 있습니다.

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


```python
# Library
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import warnings
from torch.utils.data import DataLoader

# LeNet-5 Model
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)   # (1, 32, 32) -> (6, 28, 28)
        self.pool = nn.AvgPool2d(2, 2)               # (6, 28, 28) -> (6, 14, 14)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) # (6, 14, 14) -> (16, 10, 10)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.pool(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16*5*5)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
```

<pre>
[Epoch 1/5] Loss: 0.2929
[Epoch 2/5] Loss: 0.0955
[Epoch 3/5] Loss: 0.0632
[Epoch 4/5] Loss: 0.0474
[Epoch 5/5] Loss: 0.0386
Test Accuracy: 98.41%
</pre>
<img src="data:image/png;base64,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xYmJiUbuyJEjKk5KSirILsKi16M5d+6ckbtx44bR1q9Df//73xs5vY6aXbti27Ztee6fXntFr3khYh439NopIp51tL799ts89wG5p/9dqtfDEhHZtWuXiu3rFPvvTv3vUruO59ixY412enq6iu3zgn7euHTpkpGzazIVR8xEAQAAAAAAcIGbKAAAAAAAAC6wnCcX7Ond+rbG9pbG9nRvfTvKQ4cOGTmW7/hXxYoVVaxv/SkiUrduXaOtT+O2tzTWp+XbeX16GwKfPp3anjI/YMAAo3327FkVb9682cidOnXKB72Dr+nT4OfNm2fk7GUR+VnCk53jx48bbXvrZH0bXnsJh01frmKf80oqfQmevvWjiOcW5voWyD/++KOR07eY9Be9f/o1S1b27t2rYra4LT70bYubNWtm5OzfYX26vT1Nf8+ePQXfOWRJXy6hLwUX8dyiXl+OGRMTY+T07e3t59nXoblZPqEv3dSXgImYf9/Y5QjWrl1rtGfPnu36PZF/bdu2VbG9NXX9+vWzjEU8txvWjxv2EnV7m2172ZBOX25mX98GwvJgZqIAAAAAAAC4wE0UAAAAAAAAF7iJAgAAAAAA4AI1UXLBrpcxaNAgFXfq1MnI2euJ//znP6vY3gKM+hn+NXToUBU/+uijRi4iIsJo69v8LV261MjZtW2uX79eUF1EMaPXRLHXJetbzomI7Ny5U8XJyclGLhC2fAtUel0MPRYRmT59uor//ve/G7nCOC7otZzstt1328MPP6xi+1yVny0zizNvWxzffvvtRnvNmjUqtreQLQz68UevBybiWZ/n8OHD/ugSCphesyKnmiiXL19W8apVq4wcNVH859q1ayreuHGjkbPrqOm1++zri8GDB2f5OBHPn6ddo8mbFStWqPjFF180cnoNSL3Gm4jnNsvUefOvH374QcXdu3c3cno9HbuuScuWLY22XjutR48eRs4+5+nXOPa1h77lut2fDRs2GG277mRxwEwUAAAAAAAAF7iJAgAAAAAA4AI3UQAAAAAAAFygJkou6GvERETq1KmjYnuv9PXr1xvtn3/+WcXUQPEve41wx44dVVyrVi0jd/HiRaOtry98//33jZy+phUlm74mvVGjRkbOXpOu10w4c+aMbzsG18qUKWO0+/fvb7TLly+v4oyMDCOnHwsK47hQo0YNo92uXTujHRcXp2K7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iohIu3btVPzoo48auddff91o63XevF0L+RozUQAAAAAAAFzgJgoAAAAAAIALJWI5j72NX1xcnIr/93//18jp20+KmNvBzZ8/38ht27atQPqHgqdv8+dtqlffvn29tvPK3ip579692fbHnu7fvn37PL2nvbWhvvzAfs969eoZ7RMnTqh4xowZeXr/QKZvTSriOQ1R34LSXgJm/1wKc+ohsmcvgVm4cKHRrl+/vorDw8ON3MCBA7OMRbz/vL1tfZ7TY/Xf75s3bxo5e9viL7/80vX7oHizl5rpbTt3+fJlo3306FHfdQx5Zi+7spcA68cCe2m5fd1qL/dCyaEvH/3tb39r5KKjo422fh1z6NAhI5ecnKxi+5hi43qnZNN//vp2xyKeSw27du2q4tGjRxu5zz77zGjry+SvXr2a737mFTNRAAAAAAAAXOAmCgAAAAAAgAvcRAEAAAAAAHChRNREsdev16lTR8VNmjQxcvq2XiIiH3zwgYr/85//GLm8bn8L39u4caOK7e0A7foWvnDXXXcZbX0LVJtds0dfQ2hvl+uthoK9le7mzZtVfPbsWSNnt7du3Zrt68JzDNk/3+vXr6vYXvepb4UuUrjrN5E9u67IqlWrjPbw4cNVbG85ap9jfME+3xw/flzFS5YsMXLx8fFG+9SpUz7rF4oWb1th52abbBQd1atXN9r2dat+7EpMTDRy9pbG9jUFSg69JspDDz1k5EJCQoy2fg1tXy+GhoaqODIy0sjZNVLOnTuXt84i4CQlJRnthIQEo/3pp5+q+PHHHzdy48ePN9qvvPKKin/99Vcj589jHDNRAAAAAAAAXOAmCgAAAAAAgAvcRAEAAAAAAHAhIGuiREREGO3u3bsb7VdffVXFaWlpRk6vgSIi8uWXX6r49OnTRo71xEXXlClTVLx//34j16BBA9evExYWpuK4uDgjp+9pLiJSunTpLGMRkR07dqh4+/btRu7ChQtG+/Llyypes2aNkTt69KjR1mukXLlyxcjpbXuc223qdHjS1wjrdZRERJo3b2609XoV3377rZGzfy7e6tqg8Ng/l2PHjhntt99+W8XdunUzcn369FFxw4YNC75zIvLNN98Y7dmzZ6t4w4YNRi45OdloUweh5LBrG+h13uwxbp8HUHTotdL0WhYiItHR0UZb//2+du2akbNrPXHdWnLp9UrKlCnj9bE3btxQ8Z133mnk+vXrp2K7Xs9PP/1ktGfOnJnrfiIw2cceva6biMhXX32l4v79+xu5Xr16GW39+ufkyZNGLjU1NV/9zA1mogAAAAAAALjATRQAAAAAAAAXAnI5T+3atY127969jbY+Fd+e6vjdd98ZbX27OPuxKLr06V2LFi0ycuXKlXP9OvqUWnsrt5iYGKNtb+2mO3/+vIrtqfb6tEkRc4r1mTNnjJy+1EeEqbm+5G1r0IsXLxrtn3/+WcXz5s0zcvr2xyg+7OO9vsX9L7/8YuT0LcJ79Ohh5Nq3b2+09ePGtm3bjNyPP/5otPXlF5s2bTJy+haU9lJTlFwtWrQw2lWrVlWxPd7s6x0UTfb25vaWs5UqVVKxvcUxy0eRF/p5q169ekZOH2OrV682cvq1EOCNvdR98+bNKv7LX/5i5N577z2j3bdvXxXbJRv27NlTQD3MGTNRAAAAAAAAXOAmCgAAAAAAgAvcRAEAAAAAAHAhYGqiVK5cWcWdO3c2cvYa9dDQUBXb9SjsehVsAVj82WuE7TaQFX3bSHsLWX0LbRFzjfq+ffuyfR0UX/pW5Pa25Pp5w66XsmLFCqN92223qdjesnzv3r1GW69nYNdFsNtAVtasWaPiVatWGTl722wUHXodrl27dhm5JUuWGO2ePXuq2D7+cP5BJv289dlnnxm5wYMHG229Hp9e80vErK2k17EQEfn111/z2UuUFHa9pqSkJBUvX77cyD3xxBNGu1OnTipeuXKlkdOvq+y6KwWNmSgAAAAAAAAucBMFAAAAAADAhYBZzlOzZk0Vt23b1sjVr1/faOvTG+2tSu3lPWwhC5RM+u++vczCbqNk05fz2EtC7enOgC/ZU5v1Jck7d+40cvZWuSg69POPvezPXoqhb/F54sQJI8eSdGQ6f/68imfPnm3kjh8/brRPnTqlYnvbYn1J/M2bN40cW2ojr/S/zfWlPSIi06ZNM9pPPfWUiuPi4oxchQoVVMxyHgAAAAAAgCKAmygAAAAAAAAucBMFAAAAAADAhYCpiRIVFaXiKlWqGDm7rsnJkydV/O9//9vI6esARdgeDgAAFA9sWxx47Fp9+/bt89oGsqKPI+q8oSizj3kzZ8402tHR0Sq26/n4sy4PM1EAAAAAAABc4CYKAAAAAACAC9xEAQAAAAAAcCHIsQuGZPfAoCBf9yVf2rVrp+KHH37YyN15551Ge8OGDSqeNGmSkUtJSTHaLr89xY4vP1dRHyvIHV+NFcZJYOGYArc4psANjilwi2MK3OCYArfcjBVmogAAAAAAALjATRQAAAAAAAAXAmY5D3KHKW1wi2mycINjCtzimAI3OKbALY4pcINjCtxiOQ8AAAAAAEAB4SYKAAAAAACAC9xEAQAAAAAAcMF1TRQAAAAAAICSjJkoAAAAAAAALnATBQAAAAAAwAVuogAAAAAAALjATRQAAAAAAAAXuIkCAAAAAADgAjdRAAAAAAAAXOAmCgAAAAAAgAvcRAEAAAAAAHCBmygAAAAAAAAu/D9gromwzIg9HwAAAABJRU5ErkJggg=="/>
