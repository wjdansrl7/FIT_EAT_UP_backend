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
  ##### ⚬ 
  ##### ⚬ 
  ##### ⚬ 
  ##### ⚬


#### version 0.1.7
  ##### ⚬ Place 모델에 distance  null or blank=True라고 설정하여서 값을 입력하지 않아도 data를 저장하도록 설정
  ##### ⚬ 해당 음식점에 경도와 위도의 소수점이 길어서 max_digits와 decimal_places 해결
  ##### ⚬ 동일한 장소 A를 좋아요 장소 리스트오 가본 장소 리스트 두 개에 다 저장하려고 하면 Place_id가 이미 존재한다는                 validationError를 customize해서 해결



