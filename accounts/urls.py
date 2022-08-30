from django.urls import path, re_path
from rest_framework_jwt.views import obtain_jwt_token, refresh_jwt_token, verify_jwt_token

from . import views

urlpatterns = [
    path('signup/', views.SignupView.as_view(), name='login'),  # 회원가입을 위한 url : localhost:8000/accounts/signup
    path('token/', obtain_jwt_token),  # token 발급 : localhost:8000/accounts/token
    path('token/refresh/', refresh_jwt_token),
    path('token/verify/', verify_jwt_token),
    path('user/', views.ProfileAPIView.as_view(), name='profile_user'),
    # 유저 정보 조회: pk, username, nickname / localhost:8000/accounts/user
    path('user/<int:pk>/update/', views.ProfileUpdateAPIView.as_view(), name='profile_user_update'),
    # pk에 따른 유저 정보 수정: localhost:8000/accounts/pk/update
    path('suggestions/', views.SuggestionListAPIView.as_view(), name='suggestion_user_list'),  # 친구 리스트 조회: localhost:8000/accounts/suggestions/

    path('follow/', views.user_follow, name='user_follow'),  # 친구 추가
    path('unfollow/', views.user_unfollow, name='user_unfollow'),  # 친구 삭제

]
