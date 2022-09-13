import json

from django.contrib.auth import get_user_model
from django.http import HttpResponse
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.generics import CreateAPIView, ListAPIView, RetrieveAPIView, UpdateAPIView, get_object_or_404
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from .models import Place
from .serializers import SignupSerializer, ProfileUserSerializer, SuggestionUserSerializer, \
    LikePlaceSerializer, SaveLikePlaceSerializer, SaveVisitPlaceSerializer, VisitPlaceSerializer


# 회원가입 view - POST 요청
class SignupView(CreateAPIView):
    model = get_user_model
    serializer_class = SignupSerializer
    permission_classes = [
        AllowAny,
    ]


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


# 친구 추가가 완료된 유저 리스트 조회 view - get 요청
class SuggestionListAPIView(ListAPIView):
    queryset = get_user_model().objects.all()
    serializer_class = SuggestionUserSerializer

    # 팔로우한 친구들을 출력해준다.
    def get_queryset(self):
        qs = super().get_queryset()
        qs = qs.exclude(pk=self.request.user.pk)
        qs = qs.filter(pk__in=self.request.user.following_set.all())
        return qs


# 친구 추가
@api_view(["POST"])
def user_follow(request):
    username = request.data['username']

    follow_user = get_object_or_404(get_user_model(), username=username)
    request.user.following_set.add(follow_user)
    follow_user.follower_set.add(request.user)
    return Response(status.HTTP_204_NO_CONTENT)


# 친구 삭제
@api_view(["POST"])
def user_unfollow(request):
    username = request.data['username']

    follow_user = get_object_or_404(get_user_model(), username=username, is_active=True)
    request.user.following_set.remove(follow_user)
    follow_user.follower_set.remove(request.user)
    return Response(status.HTTP_204_NO_CONTENT)

# -----------------------------------------------------


# 좋아요한 장소를 Place 모델에 저장 및 해당 유저가 좋아하는 장소 저장
class LikePlaceAPIView(CreateAPIView):
    model = Place
    serializer_class = SaveLikePlaceSerializer


# 좋아요한 장소 리스트 출력
class LikePlaceListAPIView(ListAPIView):
    queryset = Place.objects.all()
    serializer_class = LikePlaceSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        qs = qs.filter(id__in=self.request.user.like_places.all())
        return qs


# 방문한 장소 저장
class VisitPlaceAPIView(CreateAPIView):
    model = Place
    serializer_class = SaveVisitPlaceSerializer


# 방문한 장소 리스트 출력
class VisitPlaceListAPIView(ListAPIView):
    queryset = Place.objects.all()
    serializer_class = VisitPlaceSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        qs = qs.filter(id__in=self.request.user.visit_places.all())
        return qs
