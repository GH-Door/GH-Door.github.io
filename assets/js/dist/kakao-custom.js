window.scrapKakao = function () {
    if (!Kakao.isInitialized()) {
        Kakao.init('6eb000f03577bd1c234bd39bda416d13'); // 카카오 앱 키 설정
    }

    // 썸네일 이미지 추출 (og:image 메타 태그 사용)
    const ogImage = document.querySelector('meta[property="og:image"]');
    const image = ogImage ? ogImage.content : 'https://gh-door.github.io/android-chrome-192x192.png';  // 기본 이미지 다시 설정
    
    console.log('공유 이미지:', image); // 디버깅용

    // 제목과 설명 추출 (Open Graph 메타 태그 사용)
    const title = document.querySelector('meta[property="og:title"]') 
        ? document.querySelector('meta[property="og:title"]').content 
        : document.title;

    const description = document.querySelector('meta[property="og:description"]') 
        ? document.querySelector('meta[property="og:description"]').content 
        : '설명이 없습니다.';

    // 카카오톡 공유 API 실행
    Kakao.Link.sendScrap({
        requestUrl: location.origin + location.pathname + 'index.html',
        templateId: 116831,  // 카카오 개발자 콘솔에서 설정한 템플릿 ID
        templateArgs: {
            img1: img1,              // 메시지 템플릿의 ${img1}과 매칭
            img2: img2,              // 메시지 템플릿의 ${img2}과 매칭
            imgRest: imgRest,        // 메시지 템플릿의 ${imgRest}과 매칭
            title: title,            // 메시지 템플릿의 ${title}과 매칭
            description: description, // 메시지 템플릿의 ${description}과 매칭
            pagePathname: location.pathname
        },
        installTalk: true
    });

    console.log("✅ Kakao Share Script Loaded!");
};