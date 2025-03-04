---
layout: post
title: "[알고리즘] 코딩테스트 준비 -3"
# description: "Python 코딩테스트 문제 풀이, 프로그래머스 Level 1-2 문제 상세 설명"
author: "DoorNote"
date: 2025-02-19 10:00:00 +0900
categories:
    - Coding Test
    - Algorithm
tags: [Python, Coding_Test, KT에이블스쿨]
comments: true
pin: False # 고정핀
math: true
mermaid: true
image:
---

## 들어가며

---

지난 포스팅에서는 상태 전환을 활용한 문자열 처리 문제를 다뤄봤다.<br>
이번에는 주사위 게임과 관련된 알고리즘 문제를 풀어보며 조건문과 수학적 사고력을 기르는 연습을 해보자.

<br>

## 1. 주사위 게임

---

### 문제: 주사위 게임
- 세 개의 주사위를 굴려 나온 값에 따라 특별한 규칙으로 점수를 계산하는 문제

<img src="/assets/img/Coding_test_5.png" style="box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.3); border-radius: 8px;" width="80%" class="center-image">

#### **문제 설명**
- 문자열 1부터 6까지 숫자가 적힌 주사위 3개를 굴린다.
- 각 주사위에서 나온 숫자를 a, b, c라고 한다.
- 각 조건에 맞게 얻는 점수를 **return**

#### **문제 해결 접근**
1. 세 숫자의 동일 여부 확인
2. 조건에 따른 점수 계산 로직 구현
3. 수학적 연산 처리

#### **처리 규칙**
1. 세 숫자가 모두 다른 경우:
- 점수 = a + b + c

2. 두 숫자만 같은 경우:
- 점수 = (a + b + c) × (a² + b² + c²)

3. 세 숫자가 모두 같은 경우:
- 점수 = (a + b + c) × (a² + b² + c²) × (a³ + b³ + c³)

<br>

### 구현 코드

```python
# 풀이
def solution(a, b, c):
    if a != b and b != c and a != c: # 세 숫자가 모두 다르다면 a + b + c 점
        score = a + b + c
        return score
    elif a == b and b != c: # 조건1
        score = (a + b + c) * (a**2 + b**2 + c**2)
        return score
    elif a == c and b != c: # 조건2
        score = (a + b + c) * (a**2 + b**2 + c**2)
        return score
    elif b == c and a != b: # 조건3
        score = (a + b + c) * (a**2 + b**2 + c**2)
        return score
    elif a == b and b == c and a == c: # 세 숫자가 모두 같다면 (a + b + c) × (a2 + b2 + c2 ) × (a3 + b3 + c3 )점
        score = (a + b + c) * (a**2 + b**2 + c**2) * (a**3 + b**3 + c**3)
        return score
    
# test
a = 2
b = 5
c = 4
solution(a, b, c) # 11
```

<br>

### 다른 사람의 풀이

> 문제를 풀다 보니 다른 사람의 풀이를 보는 것이 중요하다고 느꼈다.<br>
> 같은 문제라도 다양한 접근 방식과 해결 방법이 존재하기 때문이다.<br>
> 다른 사람의 코드를 보면서 새로운 알고리즘과 코딩 스타일을 배울 수 있다.<br>
> 특히 아래와 같은 점들을 배울 수 있다.<br>
  1. 더 효율적인 코드 작성 방법
  2. 새로운 파이썬 내장 함수나 메서드 활용법
  3. 문제를 더 단순하고 명확하게 해결하는 방법
  4. 코드의 가독성을 높이는 방법

- 아래는 다른 사람들의 구현 코드이다.

```python
# 해석
"""
set()의 기본적인 사용법은 set(iterable)임으로, 만약 set(a, b, c) 이렇게 했을 경우 에러가 발생한다.
즉 set()은 반드시 하나의 반복 가능한 개체를 인자로 받아야한다.
또한 set()은 집합 자료형으로, 중복을 자동으로 제거하는 기능을 가진다.

ex:
numbers = [1, 2, 2, 3, 3, 3, 4, 5]
unique_numbers = set(numbers)
print(unique_numbers)  # {1, 2, 3, 4, 5}

특히 'check = set(len([a, b, c]))'은 서로 다른 숫자의 개수를 파악해서 중복 여부를 쉽게 판별할 수 있다.
"""

# 풀이
def solution(a, b, c):
    check=len(set([a,b,c])) # set을 이용하여 서로 다른 숫자의 개수를 확인
    if check==1: # 세 숫자가 모두 같다면
        return 3*a*3*(a**2)*3*(a**3)
    elif check==2: # 두 숫자가 모두 같다면
        return (a+b+c)*(a**2+b**2+c**2)
    else: # 세 숫자가 모두 다르다면
        return (a+b+c)

# test
a = 2
b = 5
c = 4
solution(a, b, c) # 11
```

<br>
<br>

## 2. 원소들의 곱과 합

---

### 문제: 리스트 원소들의 곱과 합 비교
- 정수로 이루어진 리스트가 주어졌을 때, 리스트의 모든 원소들의 곱과 합의 제곱을 비교하는 문제

<img src="/assets/img/Coding_test_6.png" style="box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.3); border-radius: 8px;" width="80%" class="center-image">

<br>

#### **문제 설명**
- 정수가 담긴 리스트 `num_list`가 주어짐
- 모든 원소들의 곱과 합의 제곱을 비교
- 곱이 합의 제곱보다 작으면 1, 크면 0을 반환

#### **문제 해결 접근**
1. 리스트의 모든 원소의 곱 계산
2. 리스트의 모든 원소의 합의 제곱 계산
3. 두 값을 비교하여 결과 도출

#### **처리 규칙**
1. 모든 원소의 곱 계산:
- 리스트를 순회하며 각 원소를 곱함

2. 모든 원소의 합의 제곱 계산:
- 리스트의 모든 원소를 더한 후 제곱

<br>

### 구현 코드

```python
# 풀이
def solution(num_list):
    mul = 1
    for num in num_list: # num_list의 각 요소를 순차적으로 가져와 mul 변수에 누적
        mul *= num # # 모든 원소의 곱 계산
    sum_square = sum(num_list) ** 2 # 모든 원소의 합 계산
    return 1 if mul < sum_square else 0 # 곱이 합의 제곱보다 작으면 1, 크면 0 반환

# test
num_list = [3, 4, 5, 2, 1]
solution(num_list) # 1
```

<br>
<br>

## 마무리

---

이번 포스팅에서는 리스트 원소들의 곱과 합을 비교하는 문제를 살펴보았다.<br>
이 문제를 통해 리스트의 기본적인 연산과 Python의 내장 함수를 활용하는 방법을 복습해봤다.<br>
다음 포스팅에서도 기초적인 알고리즘 문제들을 하나씩 살펴보려한다.

<br>
<br>

## 다음 포스팅

---

**시리즈** 

- [코딩테스트 준비 -1](/posts/2025-02-17-Coding_test_1/)
- [코딩테스트 준비 -2](/posts/2025-02-18-Coding_test_2/)
- [코딩테스트 준비 -3](/posts/2025-02-19-Coding_test_3/) 
- [코딩테스트 준비 -4](/posts/2025-02-20-Coding_test_4/) &nbsp;&nbsp; ⬅️ **다음 포스팅**
- [코딩테스트 준비 -5](/posts/2025-02-21-Coding_test_5/) 


