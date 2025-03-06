---
layout: post
title: "[알고리즘] 코딩테스트 준비 -2"
# description: "Python 코딩테스트 문제 풀이 - 자료구조 및 알고리즘 복습"
author: "DoorNote"
permalink: /coding_test_2/
date: 2025-02-18 10:00:00 +0900
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

지난 포스팅에서는 기본적인 자료구조와 간단한 문제풀이를 다뤄봤다.<br>
이번에는 조금 더 심화된 내용의 알고리즘 문제 LV1 ~ LV2 문제들을 다뤄볼 예정이다.

<br>

## 1. 상태 전환을 활용한 문자열 처리

---

### 문제: 문자열 상태 처리
- 주어진 문자열을 특정 규칙에 따라 새로운 문자열로 변환하는 문제

<img src="/assets/img/Coding_test_3.png" style="box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.3); border-radius: 8px;" width="80%" class="center-image">

<br>

#### **문제 설명**
- 문자열 `code`를 순차적으로 읽으며 처리한다.
- `mode`는 0과 1 두 가지 상태가 있으며, 초기값은 0
- "1"을 만나면 `mode`가 전환됨 (0→1 또는 1→0)

#### **문제 해결 접근**
1. 상태(mode) 변수를 활용한 조건 처리
2. 문자열 순회를 통한 인덱스 기반 처리
3. 결과 문자열 동적 생성

#### **처리 규칙**
1. Mode 0인 경우:
- 짝수 인덱스의 문자만 결과 문자열에 추가
- "1"을 만나면 Mode 1로 전환

2. Mode 1인 경우:
- 홀수 인덱스의 문자만 결과 문자열에 추가  
- "1"을 만나면 Mode 0으로 전환

<br>

### 구현 코드

```python
# 풀이
def solution(code):
    mode = 0
    ret = ""
    
    for idx in range(len(code)): # idx를 0 부터 code의 길이 - 1 까지 1씩
        if mode == 0: # mode가 0일 때
            if code[idx] == "1": # code[idx]가 1이면 
                mode = 1 # mode를 1로 변경
            elif idx % 2 == 0: # 1이 아니면 idx가 짝수일 때만
                ret += code[idx] # code[idx] 추가
        else: # mode가 1일 떄
            if code[idx] == "1": # code[idx] 가 1이면
                mode = 0 # mode를 0으로 변경
            elif idx % 2 != 0: # code[idx]가 "1"이 아니면 홀수일 때만
                ret += code[idx] # ret 맨 뒤 code[idx] 추가
                
    return ret if ret else "EMPTY" # ret이 빈 문자라면 "EMPTY" return

# test
code = "abc1abc1abc"
solution(code) # 'acbac'
```

<br>
<br>

## 2. 등차수열 특정 항만 더하기

---

### 문제: 등차수열과 선택적 합계 계산

<img src="/assets/img/Coding_test_4.png" style="box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.3); border-radius: 8px;" width="80%" class="center-image">

#### **문제 설명**
- 등차수열의 첫 항 `a`, 공차 `d`, 그리고 `included` 배열이 주어짐
- `included[i]`가 true이면 i+1항을 더하고, false이면 더하지 않음
- 선택된 항들의 합을 반환

#### **문제 해결 접근**
1. 등차수열 생성
2. included 배열을 통한 선택적 합계 계산
3. 결과값 도출

#### **처리 규칙**
- 등차수열의 i번째 항 = a + (i-1)d
- included[i]가 true인 항만 합계에 포함
- 최종 합계 반환

<br>

### 구현 코드

```python
# 풀이
def solution(a, d, included): # a = 첫쨰항 / d = 공차
    result = 0
    
    for i in range(len(included)): # included의 길이만큼
        term = a + i * d # i번째 등차수열 항 계산
        if included[i]: # 해당 항이 True이면 더하기
            result += term
    return result

# test
a = 3
d = 4
included = [True, False, False, True, True]
solution(a, d, included) # 37
```
<br>
<br>

## 마무리

---

이번 포스팅에서는 등차수열의 특정 항을 선택적으로 더하는 문제를 살펴보았다.<br> 
이 문제를 통해 등차수열의 기본 개념과 Python의 리스트 인덱싱을 활용하는 방법을 복습할 수 있었다.<br>
다음 포스팅에서도 기초적인 알고리즘 문제들을 하나씩 살펴보려한다.

<br>
<br>

## 다음 포스팅

---

**시리즈** 

- [코딩테스트 준비 -1](/coding_test_1/)
- [코딩테스트 준비 -2](/coding_test_2/) 
- [코딩테스트 준비 -3](/coding_test_3/) &nbsp;&nbsp; ⬅️ **다음 포스팅**
- [코딩테스트 준비 -4](/coding_test_4/) 
- [코딩테스트 준비 -5](/coding_test_5/) 