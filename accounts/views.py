import csv
from django.contrib.auth import get_user_model
from django.http import HttpResponse
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.generics import CreateAPIView, ListAPIView, RetrieveAPIView, UpdateAPIView, get_object_or_404, \
    GenericAPIView
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.settings import api_settings

from .models import Place, User, UserRating
from .serializers import SignupSerializer, ProfileUserSerializer, SuggestionUserSerializer, \
    LikePlaceSerializer, SaveLikePlaceSerializer, SaveVisitPlaceSerializer, VisitPlaceSerializer, RatingSerializer


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


# 좋아요한 장소를 Place 모델에 저장 및 해당 유저가 좋아하는 장소 저장
class LikePlaceAPIView(GenericAPIView):
    model = Place
    serializer_class = SaveLikePlaceSerializer
    queryset = Place.objects.all()

    def get_serializer(self, *args, **kwargs):
        serializer_class = self.get_serializer_class()
        kwargs["context"] = self.get_serializer_context()
        draft_request_data = self.request.data.copy()
        kwargs["data"] = draft_request_data
        return serializer_class(*args, **kwargs)

    def create(self, request, *args, **kwargs):
        if Place.objects.filter(id=request.data['id']).exists():
            place = Place.objects.get(id=request.data['id'])
            user = get_object_or_404(User, pk=self.request.data['pk'])
            user.like_places.add(place)
            return Response(status=status.HTTP_200_OK)
        else:
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            self.perform_create(serializer)
            user = get_object_or_404(User, pk=request.data['pk'])
            place = Place.objects.get(id=request.data['id'])
            user.like_places.add(place)
            headers = self.get_success_headers(serializer.data)
            return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    def perform_create(self, serializer):
        serializer.save()

    def get_success_headers(self, data):
        try:
            return {'Location': str(data[api_settings.URL_FIELD_NAME])}
        except (TypeError, KeyError):
            return {}

    def post(self, request, *args, **kwargs):
        return self.create(request, *args, **kwargs)


# 좋아요한 장소 리스트 출력
class LikePlaceListAPIView(ListAPIView):
    queryset = Place.objects.all()
    serializer_class = LikePlaceSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        qs = qs.filter(id__in=self.request.user.like_places.all())
        return qs

# 좋아요한 장소 리스트에서 삭제
@api_view(["POST"])
def likePlace_delete(request):
    id = request.data['id']

    place = get_object_or_404(Place, id=id)
    request.user.like_places.remove(place)
    # place.like_places_set.remove(request.user)
    return Response(status.HTTP_204_NO_CONTENT)


# 방문한 장소 저장
class VisitPlaceAPIView(GenericAPIView):
    model = Place
    serializer_class = SaveVisitPlaceSerializer
    queryset = Place.objects.all()

    def get_serializer(self, *args, **kwargs):
        serializer_class = self.get_serializer_class()
        kwargs["context"] = self.get_serializer_context()

        draft_request_data = self.request.data.copy()
        kwargs["data"] = draft_request_data
        return serializer_class(*args, **kwargs)

    def create(self, request, *args, **kwargs):
        if Place.objects.filter(id=request.data['id']).exists():
            place = Place.objects.get(id=request.data['id'])
            user = get_object_or_404(User, pk=request.data['pk'])
            user.visit_places.add(place)
            return Response(status=status.HTTP_200_OK)
        else:
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            self.perform_create(serializer)
            user = get_object_or_404(User, pk=request.data['pk'])
            place = Place.objects.get(id=request.data['id'])
            user.visit_places.add(place)
            headers = self.get_success_headers(serializer.data)
            return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    def perform_create(self, serializer):
        serializer.save()

    def get_success_headers(self, data):
        try:
            return {'Location': str(data[api_settings.URL_FIELD_NAME])}
        except (TypeError, KeyError):
            return {}

    def post(self, request, *args, **kwargs):
        return self.create(request, *args, **kwargs)

# 방문한 장소 리스트 출력
class VisitPlaceListAPIView(ListAPIView):
    queryset = Place.objects.all()
    serializer_class = VisitPlaceSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        qs = qs.filter(id__in=self.request.user.visit_places.all())
        return qs


# 방문한 장소 리스트 삭제
@api_view(["POST"])
def visitPlace_delete(request):
    id = request.data['id']

    place = get_object_or_404(Place, id=id)
    request.user.visit_places.remove(place)
    # place.visit_places_set.remove(request.user)
    return Response(status.HTTP_204_NO_CONTENT)


# 친구목록에 있는 친구의 좋아요한 장소 리스트 출력
class FriendLikePlaceListAPIView(ListAPIView):
    queryset = Place.objects.all()
    serializer_class = LikePlaceSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        user = get_object_or_404(User, pk=self.request.GET['pk'])
        qs = qs.filter(id__in=user.like_places.all())
        return qs


# 친구목록에 있는 친구의 방문 장소 리스트 출력
class FriendVisitPlaceListAPIView(ListAPIView):
    queryset = Place.objects.all()
    serializer_class = VisitPlaceSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        user = get_object_or_404(User, pk=self.request.GET['pk'])
        qs = qs.filter(id__in=user.visit_places.all())
        return qs


# 음식점에 대한 평점 저장
class RatingView(GenericAPIView):
    model = UserRating
    serializer_class = RatingSerializer
    queryset = UserRating.objects.all()

    def get_serializer(self, *args, **kwargs):
        serializer_class = self.get_serializer_class()
        kwargs["context"] = self.get_serializer_context()

        draft_request_data = self.request.data.copy()
        kwargs["data"] = draft_request_data
        return serializer_class(*args, **kwargs)

    def create(self, request, *args, **kwargs):
        if UserRating.objects.filter(place_id=request.data['place']).exists():
            user = get_object_or_404(User, pk=request.data['user'])
            place = Place.objects.get(id=request.data['place'])
            user.ratings.remove(place)
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            self.perform_create(serializer)
            user.ratings.add(place)
            headers = self.get_success_headers(serializer.data)
            return Response(serializer.data, status=status.HTTP_200_OK, headers=headers)
        else:
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            self.perform_create(serializer)
            user = get_object_or_404(User, pk=request.data['user'])
            place = Place.objects.get(id=request.data['place'])
            user.ratings.add(place)
            headers = self.get_success_headers(serializer.data)
            return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    def perform_create(self, serializer):
        serializer.save()

    def get_success_headers(self, data):
        try:
            return {'Location': str(data[api_settings.URL_FIELD_NAME])}
        except (TypeError, KeyError):
            return {}

    def post(self, request, *args, **kwargs):
        return self.create(request, *args, **kwargs)


# 음식점에 대한 평점 리스트 출력
class RatingListView(ListAPIView):
    queryset = UserRating.objects.all()
    serializer_class = RatingSerializer






















