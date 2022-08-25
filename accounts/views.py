from django.contrib.auth import get_user_model
from rest_framework.generics import CreateAPIView, ListAPIView, RetrieveAPIView, UpdateAPIView
from rest_framework.permissions import AllowAny
from .serializers import SignupSerializer, ProfileUserSerializer


# 회원가입 view - POST 요청
class SignupView(CreateAPIView):
    model = get_user_model
    serializer_class = SignupSerializer
    permission_classes = [
        AllowAny,
    ]


# 유저 리스트 조회 view - get 요청
class ProfileListAPIView(ListAPIView):
    queryset = get_user_model().objects.all()
    serializer_class = ProfileUserSerializer


# 유저 조회 view - get 요청
class ProfileAPIView(RetrieveAPIView):
        queryset = get_user_model().objects.all()
        serializer_class = ProfileUserSerializer

        def get_object(self):
            return self.request.user


# 유저 프로필 수정 view - put 요청
class ProfileUpdateAPIView(UpdateAPIView):
    lookup_field = 'pk'
    queryset = get_user_model().objects.all()
    serializer_class = ProfileUserSerializer


