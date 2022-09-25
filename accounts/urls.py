from django.urls import path, re_path
from rest_framework_jwt.views import obtain_jwt_token, refresh_jwt_token, verify_jwt_token

from . import views

urlpatterns = [
    # 회원가입
    path('signup/', views.SignupView.as_view(), name='login'),  # 회원가입을 위한 url : localhost:8000/accounts/signup

    # 로그인시 토큰 발급 및 갱신
    path('token/', obtain_jwt_token),  # token 발급 : localhost:8000/accounts/token
    path('token/refresh/', refresh_jwt_token),
    path('token/verify/', verify_jwt_token),

    # 유저에 대한 프로필 관련
    path('user/', views.ProfileAPIView.as_view(), name='profile_user'),
    # 유저 정보 조회: pk, username, nickname / localhost:8000/accounts/user
    path('user/<int:pk>/update/', views.ProfileUpdateAPIView.as_view(), name='profile_user_update'),
    # pk에 따른 유저 정보 정: localhost:8000/accounts/pk/update
    path('suggestions/', views.SuggestionListAPIView.as_view(), name='suggestion_user_list'),  # 친구 리스트 조회: localhost:8000/accounts/suggestions/

    # 친구 추가, 삭제
    path('follow/', views.user_follow, name='user_follow'),  # 친구 추가
    path('unfollow/', views.user_unfollow, name='user_unfollow'),  # 친구 삭제

    # 좋아요한 장소
    path('place/user/like/save/', views.LikePlaceAPIView.as_view(), name='LikePlaceSave'),  # 좋아요한 장소 저장
    path('place/user/like/list/', views.LikePlaceListAPIView.as_view(), name='LikePlaceList'),  # 좋아요한 장소 리스트 출력
    path('place/user/like/delete/', views.likePlace_delete, name='LikePlaceDelete'),  # 좋아요한 장소 리스트 삭제

    # 가본 곳 장소
    path('place/user/visit/save/', views.VisitPlaceAPIView.as_view(), name='VisitPlaceSave'),  # 가본 곳 장소 리스트 저장
    path('place/user/visit/list/', views.VisitPlaceListAPIView.as_view(), name='VisitPlaceList'),  # 가본 곳 장소 리스트 출력
    path('place/user/visit/delete/', views.visitPlace_delete, name='VisitPlaceDelete'),  # 가본 곳 장소 리스트 삭제


]
