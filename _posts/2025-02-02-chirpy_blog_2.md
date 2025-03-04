---
layout: post
title: "[Jekyll Chirpy] 커스터마이징 -2"
description: "Jekyll Chirpy 테마의 기본 설정을 커스터마이징하는 방법을 소개합니다."
author: "DoorNote"
permalink: /Jekyll Chirpy-커스터마이징-2/
date: 2025-02-02 10:00:00 +0900
categories:
    - GitHub Blog
    - Jekyll Chirpy
tags: [GitHub Pages, Jekyll, Chirpy]
comments: true
pin: False # 고정핀
math: true
mermaid: true
image:
---

## 📑  목차
1. [_config.yml 수정](#_configyml-수정)
2. [_sidebar.SCSS 수정](#_sidebarscss-수정)
3. [파비콘 수정](#파비콘-수정)
4. [GitHub Pages 배포](#github-pages-배포)

<br>


## 들어가며

---

> 이번 포스트에서는 Jekyll Chirpy 테마의 기본 설정을 커스터마이징하는 방법을 소개합니다.<br>
> _config.yml 및 _sidebar.scss 파일을 활용해 전반적인 설정 수정과 사이드바 꾸미기를 다룰 예정입니다.<br>
> 추가로 파비콘 변경 방법도 함께 설명하니, 블로그를 더 개성있게 꾸미고 싶은 분들께 추천드립니다.

<br>

## **_config.yml** 수정

---

<img src="/assets/img/config_ 설명_1.png" style="box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.3); border-radius: 8px;" width="100%">

<br>

`_config.yml` 파일은 블로그의 전반적인 설정을 관리하는 파일입니다.<br> 
여기서 사이트 제목, 설명, 언어, 소셜링크, URL 등을 설정할 수 있습니다.<br>
아래 예제 코드를 참고하여 나만의 블로그에 맞게 설정해보시면 됩니다.

```yml
# _config.yml 파일 예제
title: "DoorNote"                        # 블로그의 제목
description: "ex: 기록하는 공간"             # 블로그 설명
url: "https://username.github.io"        # 배포된 블로그 주소
baseurl: ""                              # 공백 권장 "" (하위 폴더에 배포될 경우 사용하는 설정)
lang: "ko"                               # 기본 언어 설정 (예: ko=한국어, en=영어)
timezone: "Asia/Seoul"                   # 포스트 작성 시간의 기준이 되는 시간대 설정 
avatar: "/assets/img/your_img.jpg"       # 프로필 이미지 설정(경로: "/assets/img/"
theme_mode:                              # light, dark 테마를 지원합니다.(공백으로 할시 디폴트 값 적용)

social:
  github: "https://github.com/yourname"  # GitHub 링크
  email: "your_email@example.com"        # 이메일 주소
  linkedin: "https://linkedin.com/in/yourname"  # LinkedIn 프로필

defaults:
    values:
      use_kakao_sdk: true                # 카카오 SDK 사용 여부 설정 - 이후 포스팅에서 다루겠습니다.
      comments: true                     # 댓글 활성화 설정(True: 활성화)

comments:
  active: "utterances"                   # disqus, utterances, giscus 중 선택
  utterances:
    repo: "url/username.github.io"    # <gh-username>/<repo>
    issue_term: pathname                 # < url | pathname | title | ...>

google_analytics:
  id: 'XXX'                              # Google Analytics ID입니다. 이후 포스팅에서 다루겠습니다.
```

<br>
<br>

## **_sidebar.SCCS** 수정

---

<img src="/assets/img/_sidebar.SCSS 설명_1.png" style="box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.3); border-radius: 8px;" width="100%">

<br>

`_sidebar.scss` 파일은 블로그의 사이드바 디자인과 관련된 스타일을 설정하는 곳입니다.<br>
 여기서 메뉴 항목, 아이콘, 폰트, 배경 색상 등을 커스터마이징할 수 있습니다.
기존 Chirpy 테마의 디렉토리 구조와 아키텍처가 업데이트되면서, 사이드바 스타일을 관리하는 파일의 경로가 **`_sass/layout/_sidebar.scss`**로 변경되었습니다.

```scss
// 사이드바 설정
#sidebar {
  @include mx.pl-pr(0);

  position: fixed;
  top: 0;
  left: 0;
  height: 100%;
  overflow-y: auto;
  width: v.$sidebar-width;
  background: url('/assets/img/sidebar.jpg'); // 사이드바 이미지
  background-size: cover; // 배경 비율
  background-position: center; // 배경 위치
  border-right: 1px solid var(--sidebar-border-color);}

// 아바타(프로필 이미지) 설정
#avatar {
  display: block;                           // 블록 요소로 표시 (한 줄 전체 차지)
  width: 6rem;                              // 가로 크기 (6rem = 약 96px)
  height: 6rem;                             // 세로 크기 (정사각형 형태 유지)
  overflow: hidden;                         // 이미지가 영역을 벗어나면 숨김 처리
  border-radius: 50%;                       // 완전한 원형으로 표시 (둥글게)
  box-shadow: var(--avatar-border-color) 0 0 0 2px; // 테두리 효과 (CSS 변수 활용)
  transform: translateZ(0);                 // Safari 브라우저 렌더링 최적화

  @include bp.sm {
    width: 6rem;                            // 작은 화면에서도 동일한 크기 유지
    height: 6rem;
  }

  img {
    width: 100%;                            
    height: 100%;
    transition: transform 0.5s;             // 변환 효과 부드럽게 (0.5초)
    object-fit: cover;                      // 이미지가 잘리지 않고 꽉 차도록 설정

    &:hover {
      transform: scale(1.2);                // 마우스 오버 시 이미지 확대 (1.2배)
    }
  }
}

// 블로그 메인 타이틀 (ex: DoorNote)
.site-title {
@extend %clickable-transition;
@extend %sidebar-link-hover;

font-family: inherit;
font-weight: 900;
font-size: 1.75rem;
line-height: 1.2;
letter-spacing: 0.25px;
margin-top: 1.25rem;
margin-bottom: 0.5rem;
width: fit-content;
color: rgba(246, 254, 255, 99%); // DoorNote 색 변경
}

// 부제목 (ex: 기록하는 공간) 
.site-subtitle {
font-size: 95%;
color: rgba(246, 254, 255, 99%); // 기록하는 공간 색 변경
margin-top: 0.25rem;
word-spacing: 1px;
-webkit-user-select: none;
-moz-user-select: none;
-ms-user-select: none;
user-select: none;
}
```
위와 같이 구조를 이해하는데 참고하시고 직접 커스터마이징을 하시면 됩니다. 추가로 이 항목들 외<br>
**li.nav-item, a.nav-link, sidebar-bottom** 등 이 있으니 각 요소의 역할과 구조를 파악한 후<br>
필요에 맞게 수정하시기 바랍니다. 또한 복사 붙여넣기 할때 들여쓰기 보단 **중괄호{}**를 주의해서 작성하시면 됩니다.

<br>

## **파비콘** 수정

---

<img src="/assets/img/파비콘_설명.png" style="box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.3); border-radius: 8px;" width="100%">

<br>

파비콘이란 위 이미지와 같은 형태로 브라우저 탭, 즐겨찾기, 북마크 등에서 웹사이트를 대표하는 작은 아이콘입니다.<br> 
이 아이콘은 사용자가 여러 개의 탭을 열어둘 때 웹사이트를 빠르게 식별할 수 있도록 도와줍니다.<br>
(귀찮지만..이거는 하는게 좋을 듯 싶습니다.)

<br>

### **1. 이미지 준비**

- 원하는 이미지를 준비합니다.
- 투명 배경을 원한다면 PNG 형식을 권장합니다.

<br>
<br>

### **2. 파비콘 변환하기**
   파비콘은 일반적으로 `.ico` 확장자를 사용합니다. 아래 사이트를 활용해 PNG 이미지를 ICO 파일로 변환할 수 있습니다.

- [**ConvertICO**](https://convertico.com/) - PNG → ICO 변환 지원 (투명 배경 유지)
- [**ICO Convert**](https://icoconvert.com/) - 다양한 아이콘 설정 가능
- [**Favicon.io**](https://favicon.io/) - 간편한 파비콘 생성 지원

<br>

### **3. 파비콘 적용하기**
- 생성된 `favicon.ico` 파일을 블로그의 **`/assets/img/`** 폴더에 업로드합니다. (덮어쓰기 해도 무방합니다.)
- 이후 `_includes/head.html` 파일을 열어 아래와 같이 수정합니다.

```html
<link rel="icon" href="/assets/img/favicon.ico" type="image/x-icon">
<link rel="shortcut icon" href="/assets/img/favicon.ico" type="image/x-icon">
```
<br>

- 변경 사항을 확인하기 위해 로컬 서버에서 확인합니다.

```shell
bundle exec jekyll server
```

```Server address: http://127.0.0.1:4000/```로 접속하여 변경사항을 확인합니다.

<br>
<br>

## **GitHub Pages 배포**

---

앞서 _config.yml, _sidebar.scss, 파일 그리고 파비콘까지 커스터마이징을 완료했습니다.<br>
이제 이 변경 사항들을 GitHub에 업로드하고, GitHub Pages를 통해 실제 블로그에 적용하는 과정을 진행하겠습니다.<br>
필자는 터미널, VScode, GitHub Desktop 모두 사용하기에 독자 분들은 편하신 방법으로 진행하시면 됩니다.<br>
이 글에서는 터미널을 활용하여 설명드리겠습니다.

<br>

### **1. 변경된 파일 확인 및 저장(커밋)**

- 커스터마이징한 내용을 GitHub에 반영하기 위해 먼저 변경된 파일들을 확인하고 **커밋(commit)**해야 합니다.<br>
- 터미널을 열고 아래 명령어를 입력합니다.

```shell
git status # 변경된 파일 확인
git add . # 변경된 파일 영역에 추가
git commit -m "chirpy update" # <-- 메세지와 함께 커밋(저장)
```
<br>

### **2. GitHub에 푸시하기**

- 커밋이 완료되었으면 GitHub 저장소에 변경 사항을 업로드(푸시)합니다.

```shell
git push origin main
```
<br>

### **3. 배포 확인**

<img src="/assets/img/github_ 배포확인.png" style="box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.3); border-radius: 8px;" width="100%">

<br>

1.	GitHub 저장소로 이동하여 Actions 탭을 확인
    - ✅ 녹색 체크 표시가 보이면 배포가 성공적으로 완료
    - ❌ 오류가 발생하면 로그를 확인하여 수정합니다.

2.	설정한 GitHub Pages URL로 접속합니다.
(예: https://username.github.io)

3.	브라우저에서 블로그의 변경 사항이 잘 반영되었는지 확인합니다.
(ex: 파비콘, 사이드바 스타일, 글꼴 변경 등)

<br>
<br>

## 마무리

---

이번 포스트에서는 본격적인 커스터마이징 방법에 대해 살펴보았습니다.<br>
개인적으로 가장 귀찮고 시간이 많이 걸리는 과정이지만, 이 과정을 거치면 블로그의 기본 틀은 거의 완성된다고 볼 수 있습니다.

다음 포스트부터는 더 기능적인 내용을 다룰 예정입니다.<br>
앞으로도 다양한 기능과 커스터마이징 방법을 소개할 예정이니 많은 관심 부탁드립니다.<br>
궁금한 점은 댓글로 남겨주시면 답변드리겠습니다. 감사합니다.🙏

<br>

## 다음 포스팅

---

- [[Jekyll Chirpy] 커스터마이징 - 1](/Jekyll Chirpy-커스터마이징-1/) &nbsp;&nbsp; ⬅️ 기본 설정
- [[Jekyll Chirpy] 커스터마이징 - 2](/Jekyll Chirpy-커스터마이징-2/) &nbsp;&nbsp; ⬅️ 기본 설정 및 사이드바 꾸미기
- [[Jekyll Chirpy] 커스터마이징 - 3](/Jekyll Chirpy-커스터마이징-3/) &nbsp;&nbsp; ⬅️ 댓글, 검색엔진 최적화, 구글 애널리틱스 설정
- [[Jekyll Chirpy] 커스터마이징 - 4](/Jekyll Chirpy-커스터마이징-4/) &nbsp;&nbsp; ⬅️ 카카오톡 공유 기능 추가하기






