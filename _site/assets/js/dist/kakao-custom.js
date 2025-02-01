window.scrapKakao = function () {
    if (!Kakao.isInitialized()) {
        Kakao.init('6eb000f03577bd1c234bd39bda416d13'); // 카카오 앱 키 설정
    }

    
    // ✅ 포스트 내 첫 번째 이미지 가져오기 (없으면 기본 이미지 사용)
    var imgUrl1 = document.querySelector('.page__content img') 
        ? document.querySelector('.page__content img').src 
        : 'https://gh-door.github.io/android-chrome-192x192.png';
        
    var imgUrl2 = '';
    var imgUrlRest = '';
    var $imgs = document.querySelectorAll('.page__content img');
    var imgUrlCnt = $imgs.length + 1;

    if ($imgs.length > 0) {
        imgUrl2 = $imgs[0].src;
    }
    if ($imgs.length > 1) {
        imgUrlRest = $imgs[1].src;
    }

    // 카카오톡 공유 API 실행
    Kakao.Link.sendScrap({
        requestUrl: location.origin + location.pathname,
        templateId: 116831,  // 📌 카카오 개발자 콘솔에서 확인한 템플릿 ID
        templateArgs: {
            img1: imgUrl1,
            img2: imgUrl2,
            imgRest: imgUrlRest,
            imgCnt: imgUrlCnt,
            title: document.title,
            description: document.querySelector('meta[name="description"]') 
                        ? document.querySelector('meta[name="description"]').content 
                        : '설명이 없습니다.',
            pagePathname: location.pathname
        },
        installTalk: true
    });

    console.log("✅ Kakao Share Script Loaded!");
};