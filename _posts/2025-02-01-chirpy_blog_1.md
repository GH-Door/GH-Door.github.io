---
layout: post
title: "[Jekyll Chirpy] 커스터마이징 -1"
description: "[GitHub Blog] Jekyll Chirpy 테마를 활용한 블로그 커스터마이징"
author: "DoorNote"
date: 2025-02-01 10:00:00 +0900
categories: [GitHub Blog]
tags: [GitHub Pages, Jekyll, Chirpy]
comments: true
pin: False # 고정핀
math: true
mermaid: true
image:
---

## 📑 목차
1. [로컬 환경 설정](#로컬-환경-설정)
2. [Jekyll Chirpy 테마 설정](#jekyll-chirpy-테마-설정)
3. [로컬 환경 연결 및 테스트](#로컬-환경-연결-및-테스트)

<br>

## 들어가며

---

**본론으로 들어가기 전 Github, 터미널 등의 사용법은 이미 숙지했다는 가정하에 진행한다는 점 유의하시길 바랍니다.**

> 이번 포스트에서는 **Jekyll Chirpy** 테마를 사용하여 블로그를 설정하고 커스터마이징하는 방법을 소개합니다.<br> 
> 필자는 하단 블로그를 참고하여 커스터마이징을 진행했으나 현재 **Chirpy** 테마가 업데이트 되어 <br> 
> 많은 시행착오를 겪었습니다..(구조 및 파일명이 전체적으로 변경되어 헷갈림...)<br> 
> 이를 해결하기 위해 공식 문서와 GitHub 이슈 트래커를 참고하며 직접 테스트를 거쳤고<br> 
> 이러한 과정에서 얻은 경험을 바탕으로, **최신 Chirpy 테마**에 맞춤 설정 방법과 커스터마이징을 체계적으로 정리했습니다.<br>
> 이 글은 Mac OS 환경을 기준으로 진행했습니다. 

> **참고 블로그**
> - [하얀눈길 블로그]("https://www.irgroup.org/posts/jekyll-chirpy/)
> - [Dodev 블로그]("https://wlqmffl0102.github.io/posts/Customizing-Blogs/)
> - [JSDevBlog]("https://jason9288.github.io/posts/github_blog_1/)


<br>

## 로컬 환경 설정
---------------------------------

### 1. **Homebrew** 설치 (이미 설치된 경우 생략)

<br>

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

<br>

### 2. **Ruby** 설치하기

Jekyll은 **Ruby**로 작성된 정적 사이트 생성기이므로 Ruby 설치가 필요<br>
Mac에는 기본적으로 Ruby가 내장되어 있지만, 이는 시스템 의존성이 강하고, 최신 Jekyll과의 호환성 문제가 발생할 수 있다.<br>
따라서, 보다 안정적이고 유연한 버전 관리를 위해 **rbenv**를 활용한 Ruby 설치를 권장합니다.

<br>

- 설치

```bash
brew install rbenv
```

- rbenv 초기화

```bash
rbenv init
```

- 설치 가능한 Ruby 버전 확인

```bash
rbenv install -l
```

- 원하는 버전 설치(최신 안정화 버전 권장)

```bash
rbenv install 3.2.2
```

- 기본 버전으로 설정

```bash
rbenv global 3.2.2
```

<br>

### 3. **Jekyll** 및 **Bundler** 설치

<br>

- 설치

```bash
gem install jekyll bundler
```

- 확인

```bash
jekyll -v
bundler -v
```

아래와 같이 출력됨을 확인
<img src="/assets/img/bundle_v.png">

<br>
<br>

## **Jekyll Chirpy** 테마 설정

Chirpy 테마 설치 방법은 아래 3가지 방법이 있으나 [Chirpy 공식 문서]("https://chirpy.cotes.page/)에서 추천하는 방식을 권장한다.<br>
이유는 제일 편하기도 하고, 추후 블로그 배포를 했을때 다양한 문제가 발생할텐데 이를 방지하기 위함<br>
실제로 필자는 로컬에서 모든 테스트를 거치고 배포를 하였으나 업데이트가 되지 않는 현상이 발생함<br>
(Chirpy 테마는 특히 이런 문제가 잦은 편이라고 알려져 있습니다. ⬅️ **정말 번거로움**)<br>

**따라서 공식 문서에서 추천하는 2번째 방법인 "GitHub Fork 방식"으로 진행하는 것을 적극 권장합니다.**<br>
추후 다양한 에러 해결 방법도 추가로 다룰 예정이니, 일단 공식 문서대로 따라가는 것을 추천합니다.

[Chirpy 공식 문서]("https://chirpy.cotes.page/)

#### **설치 방법 (3가지 중 선택 가능)**  
> 1. chirpy starter를 통해 직접 설치
> 2. github에서 소스를 fork 받아서 만들기
> 3. 소스를 zip으로 받아서 설치

**👉 필자 또한 2번째 방법으로 진행**

---

<br>

#### Step 1: 새 레파지토리를 생성

<img src="/assets/img/1. github.png" style="box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.3); border-radius: 8px;" width="80%">

1. [GitHub](https://github.com) 로그인  
2. 좌측 상단의 ➕ 아이콘 클릭 → **"New repository"** 선택  
3. 다음과 같이 설정 후 **"Create repository"** 클릭  
   - **Repository name:** `username.github.io` (ex. `gh-door.github.io`)  
   - **Description:** (선택 사항)  
   - **Public** 선택  
   - ✅ "Initialize this repository with a README" 체크 해제  

<br>

#### Step 2: Fork Chirpy를 사용해 저장소로 fork

<img src="/assets/img/2. github.png" style="box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.3); border-radius: 8px;" width="80%">

1. [Fork Chirpy]("https://github.com/cotes2020/jekyll-theme-chirpy/fork") 로 이동 
2. 우측 상단의 **"Fork"** 버튼 클릭  
3. 본인의 GitHub 계정으로 Fork 완료  

<br>

#### Step 3: 1번에서 새로 생성한 레파지토리에 fork한 레파지토리의 파일을 복사

> ⚠️ **주의:** 아래 코드를 작성하기 전 반드시 작업할 디렉토리로 이동해야 합니다.
> (ex: `cd ~/blog`)

<br>

**1. 새로 만든 레포지토리(username.github.io)를 로컬과 연동**

```bash
git clone https://github.com/username/username.github.io.git
cd username.github.io
```
> 설명:
   - GitHub에서 만든 새 레포지토리를 로컬로 가져옵니다.
   - 이후 작업은 이 디렉토리(username.github.io) 내에서 진행합니다.

<br>

**2. jekyll-theme-chirpy 레파지토리를 로컬과 연동**

```bash
git clone https://github.com/username/jekyll-theme-chirpy.git
```

> 설명:
   - GitHub에서 Fork한 jekyll-theme-chirpy를 로컬로 가져옵니다.
   - 이 폴더에 Chirpy 테마의 모든 파일이 들어있습니다.

<br>

**3. Chirpy 테마의 모든 파일을 새 레포지토리로 복사**

```bash
cp -r jekyll-theme-chirpy/* jekyll-theme-chirpy/.* ./ 2>/dev/null
```

> 설명:
   - jekyll-theme-chirpy 폴더 안의 모든 파일(숨김 파일 포함)을 현재 폴더(username.github.io)로 복사합니다.
   - 2>/dev/null은 오류 메시지(불필요한 경고)를 무시하기 위한 옵션입니다.

<br>

**4. 더 이상 필요 없는 Fork한 Chirpy 폴더 삭제**

```bash
rm -rf jekyll-theme-chirpy
```

설명:
   - 복사가 완료되었으므로 Fork한 폴더는 삭제합니다.
   - **주의:** username.github.io 폴더가 아닌 jekyll-theme-chirpy 폴더만 삭제하세요.

<br>
<br>

## **로컬 환경 연결** 및 테스트

<br>

- Jekyll 서버 실행(로컬에서 확인)

```bash
bundle exec jekyll serve
```

아래와 같이 출력됩니다.

<img src="/assets/img/3. github.png" width="100%">

위 출력 내용에서 `http://127.0.0.1:4000/` ⬅️ 이 주소를 복사<br>
Chorme 열어서 해당 주소를 입력 후 아래와 같은 이미지 확인

<img src="/assets/img/4. github.png" style="box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.3); border-radius: 8px;" width="100%">
![](2025-02-02-16-44-57.png)

<br>

이로써 로컬에서 작업은 모두 마쳤습니다. 아직 배포 전이기에, 지금까지의 작업은 로컬 환경에서만 확인할 수 있습니다.<br>
이제 다음 단계로는 본격적인 **커스터마이징** 방법을 다루게 됩니다.

<br>
<br>

## 마무리

---

여기까지 Jekyll Chirpy 테마 초기 세팅 과정을 함께해주셔서 감사합니다.<br>
처음 접하시는 분들께는 다소 복잡하게 느껴질 수 있지만, 하나씩 따라가다 보면 생각보다 금방 익숙해질 거라 믿습니다.<br>

이번 포스트에서 최대한 자세히 설명하려 했지만, 부족한 부분이 있을 수 있습니다.<br>
혹시 잘못된 부분이나 개선할 점이 보인다면 편하게 댓글이나 이슈로 알려주세요.<br>
함께 고민하고 더 나은 방법을 찾아가는 과정이 더 의미 있다고 생각합니다.<br>

앞으로도 Chirpy 테마의 커스터마이징, 추가 기능, 다양한 활용 방법에 대한 글을 꾸준히 작성할 예정입니다.<br>
이번 글이 조금이라도 도움이 되셨길 바라며, 끝까지 읽어주셔서 진심으로 감사합니다. 🙏

<br>

## 다음 포스팅 

---

- [[Jekyll Chirpy] 커스터마이징 - 2](/posts/2025-02-02-chirpy_blog_2/) &nbsp;&nbsp; ⬅️ 기본 설정 및 사이드바 꾸미기
- [[Jekyll Chirpy] 커스터마이징 - 3](/posts/2025-02-03-chirpy_blog_3/) &nbsp;&nbsp; ⬅️ 댓글, 검색엔진 최적화, 구글 애널리틱스 설정
- [[Jekyll Chirpy] 커스터마이징 - 4](/posts/2025-02-04-chirpy_blog_4/) &nbsp;&nbsp; ⬅️ 카카오톡 공유 기능 추가하기






