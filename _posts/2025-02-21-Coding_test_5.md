---
layout: post
title: "[알고리즘] 코딩테스트 준비 -5"
description: "[KT 에이블스쿨] AI 트랙 코딩테스트 대비 - 프로그래머스 기반 알고리즘 학습 과정 정리"
author: "DoorNote"
permalink: /coding_test_5/
date: 2025-02-21 10:00:00 +0900
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

이번 포스팅은 코딩테스트 준비 시리즈의 마지막 글이다.<br>
지금까지 기초적인 알고리즘 문제들을 하나씩 살펴보며 Python의 다양한 기능들을 복습해봤다.<br>

<br>
## 1. 수열과 구간 쿼리

---

### 문제: 수열과 구간 쿼리 -2
- 정수 배열과 쿼리를 처리하여 새로운 배열을 만드는 문제

<img src="/assets/img/Coding_test_9.png" style="box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.3); border-radius: 8px;" width="80%" class="center-image">

#### **문제 설명**
- 정수 배열 arr와 2차원 정수 배열 queries가 주어짐
- queries의 각 원소는 [s, e, k] 형태의 쿼리를 나타냄
- s ≤ i ≤ e 범위의 모든 i에 대해 i가 k의 배수이면 arr[i]에 1을 더함
- 모든 쿼리를 처리한 후의 arr를 return

#### **문제 해결 접근**
1. queries 배열을 순회하며 각 쿼리 처리
2. 각 쿼리에서 주어진 범위(s~e) 확인
3. 범위 내 k의 배수인 인덱스 찾기
4. 해당 인덱스의 arr 값 증가

#### **처리 규칙**
1. queries의 각 쿼리 [s, e, k]에 대해:
- s부터 e까지의 인덱스 i를 확인
- i가 k의 배수이면 arr[i]에 1을 더함

2. 모든 쿼리 처리 후:
- 변경된 arr 배열 반환

<br>

### 구현 코드

```python
# 풀이
def solution(arr, queries):
    for s, e, k, in queries:
        for i in range(s, e + 1): # s ≤ i ≤ e
            if i % k == 0: # i가 k의 배수이면
                arr[i] += 1
    return arr

# test
arr = [0, 1, 2, 4, 3]
queries = [[0, 4, 1],[0, 3, 2],[0, 3, 3]]
solution(arr, queries) # [3, 2, 4, 6, 4]
```

<br>
<br>

## 2. 배열 만들기

---

### 문제: 0과 5로만 이루어진 수 찾기
- 주어진 범위 내에서 0과 5로만 이루어진 모든 정수를 찾는 문제

<img src="/assets/img/Coding_test_10.png" style="box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.3); border-radius: 8px;" width="80%" class="center-image">

<br>

#### **문제 설명**
- 정수 l과 r이 주어짐
- l 이상 r 이하의 정수 중에서 숫자 "0"과 "5"로만 이루어진 모든 정수를 찾기
- 찾은 정수들을 오름차순으로 정렬하여 배열로 반환
- 해당하는 정수가 없다면 [-1] 반환

#### **문제 해결 접근**
1. l부터 r까지의 모든 정수 확인
2. 각 정수가 0과 5로만 이루어졌는지 검사
3. 조건을 만족하는 정수들을 배열에 저장
4. 결과 배열 반환

#### **처리 규칙**
1. 숫자 검사:
- 각 정수를 문자열로 변환하여 검사
- 모든 자릿수가 0 또는 5인지 확인

2. 결과 처리:
- 조건을 만족하는 정수들을 오름차순으로 정렬
- 해당하는 정수가 없으면 [-1] 반환

<br>

### 구현 코드

```python
# 풀이
def solution(l, r):
    answer = []
    for num in range(l, r + 1): # l ≤ num ≤ r
        if set(str(num)) <= {'0', '5'}: # 숫자가 '0'과 '5'로만 이루어져 있는지 판별
            answer.append(num) # 리스트에 추가
        
    return answer if answer else [-1]

# test
l = 5
r = 550
solution(l, r)
```

<br>
<br>

## 마무리

---

지난 5개의 포스팅을 통해 기본적인 코딩테스트 문제들을 다뤄봤다.<br>
프로그래밍의 기초가 되는 개념들을 복습했는데, 평소 자주 쓰지 않는 Python의 내장 함수와<br> 
자료구조를 활용하는 방법도 익힐 수 있었다.
앞으로도 꾸준한 학습과 문제 풀이를 통해 알고리즘 실력을 향상시켜야겠다.

<br>
<br>

## 시리즈

- [코딩테스트 준비 -1](/coding_test_1/)
- [코딩테스트 준비 -2](/coding_test_2/)
- [코딩테스트 준비 -3](/coding_test_3/) 
- [코딩테스트 준비 -4](/coding_test_4/) 
- [코딩테스트 준비 -5](/coding_test_5/) 
