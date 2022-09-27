# 졸업 프로젝트
## FIT_EAT_UP_backend

#### version 0.0.0
  ##### ⚬ 프로젝트 생성
  
#### version 0.1.0
  ##### ⚬ 

#### version 0.1.1
  ##### ⚬ 

#### version 0.1.2
  ##### ⚬ 
  
#### version 0.1.3
  ##### ⚬ 
  
#### version 0.1.4
  ##### ⚬ jwt token 로그인 관련 permission을 allowany -> authorization을 통해 token이 인증된 사용자만 이후 페이지에 접근
  ##### ⚬ http://localhost:8000/accounts/user/: 로그인한 유저 정보 조회
  ##### ⚬ http://localhost:8000/accounts/user/pk/update: 로그인한 유저 정보 수정
  ##### ⚬ http://localhost:8000/accounts/user/list/: 회원가입된 유저 정보 리스트 조회

#### version 0.1.5
  ##### ⚬ 회원가입이나 프로필에서 사용자의 이미지를 추가할 수 있게 avatar 필드를 추가하고 사용자가 개인 사진을 추가하지 않을시             django의 third party인 pydenticon image를 통해서 ID로 구별되는 이미지를 각각 생성
  ##### ⚬ http://localhost:8000/accounts/suggestions/ : 친구 리스트 조회
  ##### ⚬ http://localhost:8000/accounts/follow/ : 친구 추가
  ##### ⚬ http://localhost:8000/accounts/unfollow/ : 친구 삭제
  
#### version 0.1.6
  ##### ⚬ https://localhost:8000/accounts/place/user/like/save/ : 유저가 음식점 상세 정보 페이지에서 좋아요를           클릭시 정보를 저장
  ##### ⚬ https://localhost:8000/accounts/place/user/like/list/ : 유저가 좋아요한 장소들을 리스트로 출력
  ##### ⚬ https://localhost:8000/accounts/place/user/visit/save/ : 유저가 음식점 상세 정보 페이지에서 가본 장           소를 클릭시 정보를 저장
  ##### ⚬ https://localhost:8000/accounts/place/user/visit/list/ : 유저가 가본 장소들을 리스트로 출력
  ##### ⚬ User 모델과 Place 모델을 manytomany 관계를 통해서 likePlace, visitPlace라는 중개 모델을 생성, 해당 모델           에는 User_id, Place_id를 통해 유저와 장소 모델을 ORM을 설정

#### version 0.1.7
  ##### ⚬ Place 모델에 distance 필드에 null or blank=True라고 설정하여서 값을 입력하지 않아도 data를 저장하도록 설정
  ##### ⚬ 해당 음식점에 경도와 위도의 소수점이 길어서 max_digits와 decimal_places 해결
  ##### ⚬ 동일한 장소 A를 좋아요 장소 리스트와 가본 장소 리스트 두 개에 다 저장하려고 하면 Place_id가 이미 존재한다는                 validationError를 customize해서 해당 장소 모델이 이미 존재할시 ORM만  해결

#### version 0.1.8
  ##### ⚬ http://localhost:8000/accounts/place/user/like/delete/ : 유저가 좋아요한 장소를 삭제
  ##### ⚬ http://localhost:8000/accounts/place/user/visit/delete/ : 유저가 가본 장소를 삭제

