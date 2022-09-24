from django.contrib.auth.models import AbstractUser
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.shortcuts import resolve_url
from rest_framework.exceptions import ValidationError


class User(AbstractUser):
    # class GenderChoices(models.TextChoices):
    #     MALE = "M", "남성"
    #     FEMALE = "F", "여성"
    # gender = models.CharField(max_length=1, blank=True, choices=GenderChoices.choices)
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=20)
    nickname = models.CharField(max_length=30, unique=True)
    # phone_number = models.CharField(
    #     max_length=13,
    #     blank=True,
    #     validators=[RegexValidator(r"^010-?[1-9]\d{3}-?\d{4}$")],
    # )

    follower_set = models.ManyToManyField("self", blank=True)  # 상대가 나를 follow
    following_set = models.ManyToManyField("self", blank=True)  # 내가 친구를 follow

    avatar = models.ImageField(
        blank=True,
        upload_to="accounts/avatar/%Y/%m/%d",
        help_text="48px * 48px 크기의 png/jpg 파일을 업로드해주세요.",
    )

    like_places = models.ManyToManyField(
        'Place',
        blank=True,
        related_name='like_users',
    )
    visit_places = models.ManyToManyField(
        'Place',
        blank=True,
        related_name='visit_users',
    )

    # @property는 메소드를 마치 필드인 것처럼 취급할 수 있게 해준다.
    @property
    def avatar_url(self):
        if self.avatar:
            return self.avatar.url
        else:
            return resolve_url("pydenticon_image", self.username)


class Place(models.Model):
    address_name = models.CharField(max_length=200)
    category_group_code = models.CharField(max_length=50)
    category_group_name = models.CharField(max_length=100)
    category_name = models.CharField(max_length=50)
    distance = models.CharField(max_length=50, null=True, blank=True)
    # id = models.AutoField(primary_key=True)  # 음식점 식별 번호
    id = models.CharField(primary_key=True, max_length=50)  # 음식점 식별 번호
    # serial_number = models.CharField(max_length=50)  # 음식점 식별 번호
    phone = models.CharField(max_length=13, blank=True)  # 음식점 전화 번호
    place_name = models.CharField(max_length=200)  # 음식점 상호명
    place_url = models.URLField(blank=True)
    road_address_name = models.CharField(max_length=50)
    x = models.DecimalField(max_digits=18, decimal_places=15)  # 음식점 위도
    y = models.DecimalField(max_digits=18, decimal_places=15)  # 음식점 경도
    image = models.ImageField(
        blank=True,
        upload_to="accounts/image/%Y/%m/%d",
    )  # 음식점 이미지 필드

    @property
    def image_url(self):
        if self.image:
            return self.image.url
        else:
            return resolve_url("pydenticon_image", self.id)

    def __str__(self):
        return self.id