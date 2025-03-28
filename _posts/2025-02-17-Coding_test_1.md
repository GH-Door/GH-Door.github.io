---
layout: post
title: "[알고리즘] 코딩테스트 준비 -1"
description: "[KT 에이블스쿨] AI 트랙 코딩테스트 대비 - 프로그래머스 기반 알고리즘 학습 과정 정리"
author: "DoorNote"
permalink: /coding_test_1/
date: 2025-02-17 10:00:00 +0900
categories:
    - Coding Test
    - Python
tags: [Algorithm, Coding_Test, KT에이블스쿨]
comments: true
pin: False # 고정핀
math: true
mermaid: true
image: /assets/img/KT-AIVLE.png
---

## 코딩테스트 준비

> **KT 에이블스쿨에** AI 트랙에 1차 합격해서 코딩테스트를 준비하려 한다.<br>
> 에이블스쿨은 **프로그래머스**를 통해 코딩테스트를 본다고 하는데<br>
> 난이도는 검색해본 결과 프로그래머스 기준 1~2 정도의 해당하는 문제들이 나온다고 한다.<br> 
> 현재 시간이 없는 관계로... 일주일 남은 지금부터라도 부랴부랴 준비 (참고로 알고리즘 공부를 안한지 꽤 된 거 같다..)<br>
> 자료구조와 기본 함수를 먼저 복습 차원에서 시작해보려 한다.  

<br>

## 1. 자료구조

---

Python에서는 기본적으로 **List, Tuple, Dictionary** 같은 자료구조를 제공하며, 코딩테스트에서도 자주 사용된다.<br>
여기서는 각 자료구조의 **특징과 주요 메서드**를 간략하게 정리한다.<br>

```python
# list
test_list = [1, 2, 3, 4]
test_tuple = (1, 2, 3, 4)
test_dict = {'name':'DoorNote', 'age':99}
```
<br>

### List(리스트)

- 순서가 있는 가변형(mutable) 자료구조, 요소 추가, 삭제, 변경 가능

```python
# 리스트 생성
numbers = [1, 2, 3, 4]

# 요소 추가 및 삭제
numbers.append(5)  # [1, 2, 3, 4, 5]
numbers.remove(2)  # [1, 3, 4, 5]

numbers.sort()  # 오름차순 정렬
numbers.reverse()  # 내림차순 정렬
```

<br>

### Tuple(튜플)

- 순서가 있는 불변형 자료구조, 한번 생성되면 요소 변경 불가

```python
# 튜플 생성
my_tuple = (10, 20, 30)

# 요소 접근
first_value = my_tuple[0]  # 10

# 요소 개수 세기
count_10 = my_tuple.count(10)
```

<br>

### Dictionary(딕셔너리)

- 키-값(key-value) 쌍으로 구성된 자료구조, 순서가 없고 키를 사용해 빠르게 값 조회 가능

```python
# 딕셔너리 생성
user = {'name': 'Alice', 'age': 25}

# 값 추가 및 변경
user['city'] = 'Seoul'
user['age'] = 26

# 값 조회 및 삭제
name = user['name']  # 'Alice'
del user['age']
```

<br>
<br>

## 2. 문제 풀이

---

Python에서 자주 사용되는 내장 함수 및 알고리즘 개념을 함께 정리하려 한다.<br>
이 문제들을 통해 기초부터 다시 잡고 가자 

<br>

### 1번 문제

<img src="/assets/img/Coding_test_1.png" style="box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.3); border-radius: 8px;" width="80%">

위 문제에서 각 자리 숫자의 합을 구하기 위해 문자열을 숫자로 변환한 후 합산하는 방식을 사용했다.<br>
Python에서는 `map()` 함수를 활용하면 문자열을 정수로 변환하는 작업을 간결하게 처리할 수 있다.

```python
def solution(num_str):
    return sum(map(int, num_str))
```

<br>

### Tip: `map()`, `apply()` 비교

필자는 데이터 분석 라이브러리인 **Pandas** 가 훨씬 더 익숙하기에 자주쓰는 `apply()`와 비교해서 정리해봤다.<br>
map() 함수는 리스트 등의 반복 가능한 **객체(iterable)**에 주어진 함수를 적용하는 역할을 한다.<br>
이는 Pandas의 **apply()**와 유사한 기능을 제공하지만, **apply()**는 **Series** 단위로 동작하며 병렬 처리를 지원하는 차이가 있다.


```python
# map() 예시 (Python 기본 함수)
numbers = ["1", "2", "3"]
result = list(map(int, numbers))
print(result)  # [1, 2, 3]

# apply() 예시 (Pandas 함수)
import pandas as pd
df = pd.DataFrame({'numbers': ["1", "2", "3"]})
df['int_numbers'] = df['numbers'].apply(lambda x: int(x))
print(df)
```

<br>

### 2번 문제

<img src="/assets/img/Coding_test_2.png" style="box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.3); border-radius: 8px;" width="80%">

- 이 문제는 정수 n이 주어질 때, n의 약수를 오름차순으로 정렬하여 배열로 반환하는 문제이다.

```python
# 한줄 코드 형식(리스트 컴프리헨션?이라고도 한다.)
def solution(n):
    return [i for i in range(1, n + 1) if n % i == 0]

# 정석 for문
def solution(n):
    result = []  # 약수를 저장할 리스트
    for i in range(1, n + 1):  # 1부터 n까지 반복
        if n % i == 0:  # n을 i로 나누었을 때 나머지가 0이면 약수
            result.append(i)  # 리스트에 추가
    return result
# 이 코드가 정석이라고 생각하고 나도 이게 더 편하다..

# 예제 실행
print(solution(24))  # [1, 2, 3, 4, 6, 8, 12, 24]
print(solution(29))  # [1, 29]
```

<br>
<br>

## 마무리

---

알고리즘 공부를 안한지 너무 오래돼서 그런가 쉬운 문제들도 풀기가 어렵게 느껴진다.<br>
독자분들은 평소에 틈틈히 알고리즘 공부를 하길 권장한다..

<br>
<br>

## 다음 포스팅

---
 
**시리즈** 

- [코딩테스트 준비 -1](/coding_test_1/)
- [코딩테스트 준비 -2](/coding_test_2/) &nbsp;&nbsp; ⬅️ **다음 포스팅**
- [코딩테스트 준비 -3](/coding_test_3/) 
- [코딩테스트 준비 -4](/coding_test_4/) 
- [코딩테스트 준비 -5](/coding_test_5/) 
