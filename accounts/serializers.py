from rest_framework import serializers
from django.contrib.auth import get_user_model


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
        fields = ['pk', 'username', 'password', 'nickname']


# 프로필 및 유저 정보 확인
class ProfileUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['pk', 'username', 'nickname']


