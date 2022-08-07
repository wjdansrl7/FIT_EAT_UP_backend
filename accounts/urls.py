from django.urls import path
from rest_framework_jwt.views import obtain_jwt_token, refresh_jwt_token, verify_jwt_token

from . import views

urlpatterns = [
    path('signup/', views.SignupView.as_view(), name='login'), # 회원가입을 위한 url : localhost:8000/accounts/signup
    path('token/', obtain_jwt_token), # token 발급 : localhost:8000/accounts/tokenv
    path('token/refresh/', refresh_jwt_token),
    path('token/verify/', verify_jwt_token),
]