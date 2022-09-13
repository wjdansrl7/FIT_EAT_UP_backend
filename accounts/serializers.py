from rest_framework import serializers
from django.contrib.auth import get_user_model

# from .models import Place
from rest_framework.generics import get_object_or_404

from .models import Place

User = get_user_model()


# 회원가입
class SignupSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    def create(self, validated_data):
        user = User.objects.create(username=validated_data['username'], nickname=validated_data['nickname'])
        user.set_password(validated_data['password'])
        user.save()
        return user

    class Meta:
        model = User
        fields = ['pk', 'username', 'password', 'nickname', 'avatar_url']


# 프로필 및 유저 정보 확인
class ProfileUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['pk', 'username', 'nickname', 'avatar_url']


# 친구 추천 유저 리스트 목록
class SuggestionUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['pk', 'username', 'nickname', 'avatar_url']


# # 좋아요한 장소 저장
class SaveLikePlaceSerializer(serializers.ModelSerializer):
    def create(self, validated_data):
        place = Place.objects.create(
            address_name=self.data['address_name'],
            category_group_code=self.data['category_group_code'],
            category_group_name=self.data['category_group_name'],
            category_name=self.data['category_name'],
            distance=self.data['distance'],
            id=self.data['id'],
            phone=self.data['phone'],
            place_name=self.data['place_name'],
            place_url=self.data['place_url'],
            road_address_name=self.data['road_address_name'],
            x=self.data['x'],
            y=self.data['y'],
            # image_url=self.data['image_url'],
        )
        place.save()
        user = get_object_or_404(User, pk__in=self.data['pk'])
        user.like_places.add(place)
        return place

    class Meta:
        model = Place
        fields = ['address_name', 'category_group_code', 'category_group_name',
                  'category_name', 'distance', 'id', 'phone', 'place_name',
                  'place_url', 'road_address_name', 'x', 'y', 'image_url', 'pk']


# 좋아요한 장소 리스트
class LikePlaceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Place
        fields = ['address_name', 'category_group_code', 'category_group_name',
                  'category_name', 'distance', 'id', 'phone', 'place_name',
                  'place_url', 'road_address_name', 'x', 'y', 'image_url']


class SaveVisitPlaceSerializer(serializers.ModelSerializer):
    def create(self, validated_data):
        print(self.data)
        place = Place.objects.create(
            address_name=self.data['address_name'],
            category_group_code=self.data['category_group_code'],
            category_group_name=self.data['category_group_name'],
            category_name=self.data['category_name'],
            distance=self.data['distance'],
            id=self.data['id'],
            phone=self.data['phone'],
            place_name=self.data['place_name'],
            place_url=self.data['place_url'],
            road_address_name=self.data['road_address_name'],
            x=self.data['x'],
            y=self.data['y'],
            # image_url=self.data['image_url'],
        )
        place.save()
        user = get_object_or_404(User, pk__in=self.data['pk'])
        user.visit_places.add(place)
        return place

    class Meta:
        model = Place
        fields = ['address_name', 'category_group_code', 'category_group_name',
                  'category_name', 'distance', 'id', 'phone', 'place_name',
                  'place_url', 'road_address_name', 'x', 'y', 'image_url', 'pk']


class VisitPlaceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Place
        fields = ['address_name', 'category_group_code', 'category_group_name',
                  'category_name', 'distance', 'id', 'phone', 'place_name',
                  'place_url', 'road_address_name', 'x', 'y', 'image_url']
