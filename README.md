# 졸업 프로젝트
## FIT_EAT_UP_backend
#### 유튜브 영상 : https://www.youtube.com/watch?v=MTsxdkUxlgE&list=LL&index=2&t=244s

[프로젝트 개발 소프트웨어 환경]
<img src="https://user-images.githubusercontent.com/48114924/208365430-54c5fac7-bb34-4653-b2f3-20606c943b41.png" width="700" height="370">

[백엔드 구조 설계]
<img src="https://user-images.githubusercontent.com/48114924/208365223-380dd537-56bb-49c8-806b-658971f0b934.png" width="700" height="370">

#### version 0.0.0
  ##### ⚬ 프로젝트 생성
  
#### version 0.1.0
  ##### ⚬ http://localhost:8000/accounts/signup/ : 유저의 회원가입
  ##### ⚬ http://localhost:8000/accounts/tokenv/ : jwt web token 발급

#### version 0.2.0
  ##### ⚬ 유저 모델 생성

#### version 0.3.0
  ##### ⚬ jwt token 로그인 관련 permission을 allowany -> authorization을 통해 token이 인증된 사용자만 이후 페이지에 접근
  ##### ⚬ http://localhost:8000/accounts/user/: 로그인한 유저 정보 조회
  ##### ⚬ http://localhost:8000/accounts/user/pk/update: 로그인한 유저 정보 수정
  ##### ⚬ http://localhost:8000/accounts/user/list/: 회원가입된 유저 정보 리스트 조회

#### version 0.4.0
  ##### ⚬ 회원가입이나 프로필에서 사용자의 이미지를 추가할 수 있게 avatar 필드를 추가하고 사용자가 개인 사진을 추가하지 않을시             django의 third party인 pydenticon image를 통해서 ID로 구별되는 이미지를 각각 생성
  ##### ⚬ http://localhost:8000/accounts/suggestions/ : 친구 리스트 조회
  ##### ⚬ http://localhost:8000/accounts/follow/ : 친구 추가
  ##### ⚬ http://localhost:8000/accounts/unfollow/ : 친구 삭제
  
#### version 0.5.0
  ##### ⚬ https://localhost:8000/accounts/place/user/like/save/ : 유저가 음식점 상세 정보 페이지에서 좋아요를           클릭시 정보를 저장
  ##### ⚬ https://localhost:8000/accounts/place/user/like/list/ : 유저가 좋아요한 장소들을 리스트로 출력
  ##### ⚬ https://localhost:8000/accounts/place/user/visit/save/ : 유저가 음식점 상세 정보 페이지에서 가본 장           소를 클릭시 정보를 저장
  ##### ⚬ https://localhost:8000/accounts/place/user/visit/list/ : 유저가 가본 장소들을 리스트로 출력
  ##### ⚬ User 모델과 Place 모델을 manytomany 관계를 통해서 likePlace, visitPlace라는 중개 모델을 생성, 해당 모델           에는 User_id, Place_id를 통해 유저와 장소 모델을 ORM을 설정

#### version 0.6.0
  ##### ⚬ Place 모델에 distance 필드에 null or blank=True라고 설정하여서 값을 입력하지 않아도 data를 저장하도록 설정
  ##### ⚬ 해당 음식점에 경도와 위도의 소수점이 길어서 max_digits와 decimal_places 해결
  ##### ⚬ 동일한 장소 A를 좋아요 장소 리스트와 가본 장소 리스트 두 개에 다 저장하려고 하면 Place_id가 이미 존재한다는                 validationError를 customize해서 해당 장소 모델이 이미 존재할시 ORM만  해결

#### version 0.7.0
  ##### ⚬ http://localhost:8000/accounts/place/user/like/delete/ : 유저가 좋아요한 장소를 삭제
  ##### ⚬ http://localhost:8000/accounts/place/user/visit/delete/ : 유저가 가본 장소를 삭제

#### version 0.8.0
  ##### ⚬ http://localhost:8000/accounts/place/user/rating/ : 음식점에 대한 유저의 평점 저장
  ##### ⚬ http://localhost:8000/accounts/place/user/rating/list/ : 음식점에 대한 유저의 평점

#### version 0.8.1
  ##### ⚬ 음식점에 대한 평점 저장 및 평점 리스트 출력 view 작성

#### version 0.9.0
  ##### ⚬ recommands 앱 생성
  ##### ⚬ http://localhost:8000/recommands/recommand/ : SGD를 활용한 행렬 분해기법을 통하여 음식점에 대한 평점 예측을 잠재요인 협업 필터링 방식을 사용
  
#### version 0.9.1
  ##### ⚬ http://localhost:8000/accounts/place/friend/like/list/ : 친구들의 프로필에서 좋아요 장소 리스트 조회
  ##### ⚬ http://localhost:8000/accounts/place/friend/visit/list/ : 친구들의 프로필에서 가본 장소 장소 리스트 조회
  
#### version 0.9.2 ~ 0.9.3
  ##### ⚬ front-end에서 json으로 응답이 가지 않는 문제 해결

#### version 0.9.4
  ##### ⚬ 불필요한 foods app 삭제
  
#### version 0.9.5
  ##### ⚬ 크롤링한 음식점 데이터를 mysql에 저장
  ##### ⚬ 음식점 추천시 추천받고자 하는 지역구 선택을 위한 코드 수정
 
#### version 0.10.0
  ##### ⚬ http://localhost:8000/recommands/surprise/ : 기존 SGD방식의 행렬분해가 평점이 정확하게 예측되지 않는다고 판단하여서 surprise package를 이용하여 SGD를 활용한 행렬분해 이용
  
#### version 0.11.0
  ##### ⚬ http://localhost:8000/recommands/random/ : 크롤링한 음식점 데이터중 평점이 3점이상인 음식점 데이터를 랜덤으로 추천

#### version 0.11.1 ~ 0.11.2
  ##### ⚬ 코드 주석처리 삭제 및 불필요한 내용 


  

