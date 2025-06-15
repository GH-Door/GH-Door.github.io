---
layout: post
title: "[Upstage AI Lab] MLOps Project"
description: "[Upstage AI Lab] MLOps Project 회고"
author: "DoorNote"
date: 2025-06-10 10:00:00 +0900
#  permalink: //
categories:
    - AI & Data
    - Upstage AI Lab
tags: [Upstage AI Lab, MLOps]
use_math: true
comments: true
pin: false # 고정핀
math: true
mermaid: true
image: /assets/img/MLOps-Thumbnail.png
---

## MLOps Project

> **MLOps** Project에 대한 주요 내용과 회고록 
{: .prompt-tip }

---

### 1. 개요

> 본 프로젝트는 **Local Model**을 **API** 로 배포하고 **MLOps pipeline**의 일부 기능을 구현하는 것이 목표
{: .prompt-info }

**Project Overview**

- Project Duration: **2025.05.26 ~ 2025.06.10**
- Team: **5명**
- Goal: 로컬에서 학습한 모델을 **API**로 배포하고, **MLOps 파이프라인**의 주요 구성 요소 일부를 구현

<br>

### 2. Architectur

> 아래 이미지는 **우리 팀**이 구축한 **MLOps pipeline 아키텍처**다.  
{: .prompt-info }

![MLOps-project](/assets/img/MLOps-project.png){: width="800" .center}

<br>

**주요 기능**

- **Amazon S3**: 원시 데이터 및 **모델 아티팩트 저장**
- **Airflow**: DAG 스케줄링으로 **데이터 수집 및 전처리 자동화**
- **MLflow**: 실험 기록 및 **하이퍼파라미터 / 성능 추적**
- **FastAPI**: 날씨 기반 옷차림 추천 **API 구현**  
- **Streamlit**: 예측 결과와 주요 성능 지표를 **웹 대시보드**로 시각화  

<br>

**협업 & CI**

- **GitHub Actions**: `main` 브랜치 변경 시 **Airflow·MLflow·FastAPI Docker 이미지 자동 빌드·테스트**   
- **Docker Hub**: 빌드된 컨테이너 이미지를 중앙 레지스트리에 저장해 **재현 가능한 배포 환경을 보장**  
- **Slack 알림**: CI/CD 각 단계(빌드·테스트·배포) 결과를 **팀 채널에 실시간 전송** 

<br>

### 3. Demo

> 아래는 파이프라인으로 구현된 서비스의 주요 화면 예시다.  
{: .prompt-info }

<img src="/assets/img/MLOps-Demo1.png" style="display:block;margin:0 auto;box-shadow:4px 4px 15px rgba(0,0,0,0.3);border-radius:8px;" width="800">
_Main_

<img src="/assets/img/MLOps-Demo2.png" style="display:block;margin:0 auto;box-shadow:4px 4px 15px rgba(0,0,0,0.3);border-radius:8px;" width="800">
_Detailed view_

<br>

### 4. 주요 역할

> 본 프로젝트는 **팀 프로젝트로 진행**되었으며, 이 섹션에서는 내가 맡았던 **주요 작업**에 대해 설명한다.
{: .prompt-info }

- **Data Preprocessing**  
  **S3 원시 데이터**를 주기적으로 **수집·정제하고, 결측치 보간·스케일링·인코딩** 등 전처리 및 파생변수 **자동화**  

- **Model Training**  
  **LSTM** 기반 시계열 예측 모델 학습, **MLflow**을 사용해 **하이퍼파라미터·메트릭·모델 아티팩트를 자동 기록**  

- **Tracking & Registry**  
  **MLflow Tracking Server** 설정 및 Model Registry 연동으로 **실험 간 성능 비교**  

- **Inference**  
  학습 완료 모델로 예측 수행 후 결과를 **S3에 저장해 후속 서빙·모니터링 및 재학습 파이프라인 재활용 지원**  

<br>
<br>

## 회고

> 이 섹션은 **MLOps Project**를 진행하며 얻은 **문제 해결 및 회고**에 대한 내용을 다룬다.
{: .prompt-tip }

---

### 문제 & 해결 과정

<br>

**문제**

**LSTM 모델**로 **7일 뒤 온도 예측**을 수행하고 **MLflow**로 실험을 모니터링했지만,  
웹 대시보드에 시각화된 결과에서 **새벽 5시~6시**에 온도가 가장 높게 찍히는 **이상 현상**을 발견했다.  
전처리 단계와 학습 로직을 재점검한 결과, 모델 자체에는 문제가 없었으며  
데이터 원본인 **S3 시계열 데이터**를 자세히 살펴본 끝에 특정 구간에 **연속적인 누락**이 있음을 확인했다.  
이 **누락 구간**이 모델에 **시간 순서를 학습하는 데 악영향**을 끼치는 원인으로 판단했다.  

<br>

**해결 과정**

기상청 **API**를 사용해 수집한 데이터를 **S3**에 저장하고 있었기 떄문에 **기상청 데이터 자체에 누락**이 있는걸로 생각했다.  
마감기한이 임박한 상태였기에 우선적으로 모델 추론 결과를 기반으로 웹 대시보드에 배포해 핵심 기능을 유지했다.  
시간이 있었다면 누락값을 결측치로 처리한 후 그 **선형 보간(linear interpolation)** 통해 해결했다면 좋았을 거 같다.

<br>

### 인사이트

**아쉬웠던 점**  

> 프로젝트 마감이 임박한 상태에서 **데이터 누락 문제**를 일시적 보완만으로 문제를 넘어간 점이 아쉬웠다.  
> 시간 부족으로 **MLflow에 하이퍼파라미터 튜닝 로깅**을 구현하지 못해 다양한 실험 결과를 비교할 수 없었다.  

<br>

**알게 된 점**  

> 시계열 모델은 **연속성 유지**가 필수며, 누락 구간을 보완하지 않으면 **모델**에 **악영향**을 끼치는다는 사실을 알았다.
> **MLOps** 전체 워크플로우와 **Airflow**, **MLflow**, **FastAPI** 같은 주요 플랫폼들이 서로 유기적으로 연계되어 **협업·자동화·운영**을 지원하는 구조를 이해할 수 있었다.  