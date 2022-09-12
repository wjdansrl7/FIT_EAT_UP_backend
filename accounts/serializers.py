from rest_framework import serializers
from django.contrib.auth import get_user_model

# from .models import Place
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


# # 들어온 장소 저장
# # create 안에 각각의 변수값들을 저장해야하는지는 찾아봐야함. ex) address_name, 등등
class PlaceSerializer(serializers.ModelSerializer):
    def create(self, validated_data):
        place = Place.objects.create(
            # address_name=self.data['address_name'],
            place_name=self.data['place_name'],
            id=self.data['id'],
            phone=self.data['phone']
                                     )
            # category_group_code=validated_data['category_group_code'],
            # category_group_name=validated_data['category_group_name'],
            # category_name=validated_data['category_name'],
            # distance=validated_data['distance'],
            # id=validated_data['id'],
            # phone=validated_data['phone'],
            # place_name=validated_data['place_name'],
            # place_url=validated_data['place_url'],
            # road_address_name=validated_data['road_address_name'],
            # x=validated_data['x'],
            # y=validated_data['y'],
            # image=validated_data['image'],
        # )
        place.save()
        # User.save(username=validated_data['username'], like_places=validated_data['id'])
        return place

    class Meta:
        model = Place
        fields = ['id', 'place_name', 'image']


# 좋아요한 장소 리스트
class LikePlaceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Place
        fields = ['id', 'pk', 'place_name', 'image']
#
#
# # 가본 곳 장소 리스트
# class VisitPlaceSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Place
#         fields = ['id', 'pk', 'place_name', 'image']




