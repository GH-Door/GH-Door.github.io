---
layout: post
title: "[알고리즘] 코딩테스트 준비 -4"
description: "[KT 에이블스쿨] AI 트랙 코딩테스트 대비 - 프로그래머스 기반 알고리즘 학습 과정 정리"
author: "DoorNote"
permalink: /coding_test_4/
date: 2025-02-20 10:00:00 +0900
categories:
    - Coding Test
    - Python
tags: [Algorithm, Coding_Test, KT에이블스쿨]
comments: true
pin: False # 고정핀
math: true
mermaid: true
image: /assets/img/python.png
---

## 들어가며

---

이번에는 리스트의 마지막 원소를 처리하는 알고리즘 문제를 풀어보며 리스트 인덱싱과 조건문을 활용하는 연습을 해보려한다.

<br>

## 1. 리스트의 마지막 원소 처리

---

### 문제: 마지막 원소 처리
- 리스트의 마지막 원소를 그전 원소와 비교하여 새로운 값을 추가하는 문제

<img src="/assets/img/Coding_test_7.png" style="box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.3); border-radius: 8px;" width="80%" class="center-image">

#### **문제 설명**
- 정수 리스트 num_list가 주어진다.
- 리스트의 마지막 원소가 그전 원소보다 크면 그전 원소를 뺀 값을 추가
- 리스트의 마지막 원소가 그전 원소보다 크지 않다면 마지막 원소의 두 배 값을 추가

#### **문제 해결 접근**
1. 리스트의 마지막 두 원소 비교
2. 조건에 따라 다른 연산 수행
3. 리스트에 새로운 값 추가 후 반환

#### **처리 규칙**
1.	num_list[-1] > num_list[-2]
- num_list.append(num_list[-1] - num_list[-2])

2.	num_list[-1] <= num_list[-2]
- num_list.append(num_list[-1] * 2)

<br>

### 구현 코드

```python
# 풀이
def solution(num_list):
    last = num_list[-1] # 마지막  
    second_last = num_list[-2] # 마지막 전

    if last > second_last:  # 마지막 요소가 더 크다면
        num_list.append(last - second_last)  # 차이를 추가
    else:  # 마지막 요소가 더 작다면
        num_list.append(last * 2)  # 두 배를 추가

    return num_list  
    
# test
num_list = [5, 2, 1, 7, 5]
solution(num_list) # [5, 2, 1, 7, 5, 10]
```

<br>
<br>

## 2. 수열과 구간 쿼리

---

### 문제: 리스트 원소들의 곱과 합 비교
- 정수로 이루어진 리스트가 주어졌을 때, 리스트의 모든 원소들의 곱과 합의 제곱을 비교하는 문제

<img src="/assets/img/Coding_test_8.png" style="box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.3); border-radius: 8px;" width="80%" class="center-image">

<br>

#### **문제 설명**
- 정수 배열 arr과 2차원 정수 배열 queries가 주어짐
- queries[i] = [s, e, k] 형태로 주어지며, 이는 arr[s]부터 arr[e]까지 중 k보다 크면서 가장 작은 값을 찾아야한다.
- 특정 쿼리에서 조건을 만족하는 값이 없으면 -1을 반환

#### **문제 해결 접근**
1. 쿼리마다 [s, e] 범위를 순회하며 k보다 크면서 가장 작은 값 찾기
2. 조건을 만족하는 값이 없다면 -1을 반환
3. 여러 개의 쿼리를 순서대로 처리

#### **처리 규칙**
1.	각 쿼리 [s, e, k]를 순회하면서 범위 [s, e] 내에서 k보다 큰 최소값을 찾음
2.	k보다 큰 값이 없으면 -1을 저장
3.	모든 쿼리를 처리한 결과를 배열로 반환

<br>

### 구현 코드

```python
# 풀이
def solution(arr, queries):
    result = []  
    for s, e, k in queries:  # queries의 각 쿼리에서 s, e, k를 추출
        candidates = [arr[i] for i in range(s, e+1) if arr[i] > k]  # 조건을 만족하는 값 찾기
        result.append(min(candidates) if candidates else -1)  # 최솟값을 추가 (없으면 -1)

    return result

# test
arr = [0, 1, 2, 4, 3]	
queries = [[0, 4, 2],[0, 3, 2],[0, 2, 2]]
solution(arr, queries) # [5, 9, 7]
```

<br>
<br>

## 마무리

---

이번 포스팅에서는 수열과 구간 쿼리 문제를 살펴보았다.<br>
이 문제를 통해 리스트 슬라이싱, 조건문, 리스트 컴프리헨션 등 Python의 다양한 기능을 활용하는 방법을 복습해봤다.<br>
다음 포스팅에서도 기초적인 알고리즘 문제들을 하나씩 살펴보려한다.

<br>
<br>

## 다음 포스팅

---

**시리즈** 

- [코딩테스트 준비 -1](/coding_test_1/)
- [코딩테스트 준비 -2](/coding_test_2/) 
- [코딩테스트 준비 -3](/coding_test_3/) 
- [코딩테스트 준비 -4](/coding_test_4/) 
- [코딩테스트 준비 -5](/coding_test_5/) &nbsp;&nbsp; ⬅️ **다음 포스팅**
