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
        null=True,
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

    ratings = models.ManyToManyField(
        'Place',
        blank=True,
        related_name='rating_users',
        through='UserRating',
    )

    # avatar 필드 관련 확인
    # @property는 메소드를 마치 필드인 것처럼 취급할 수 있게 해준다.
    @property
    def avatar_url(self):
        if self.avatar:
            return self.avatar.url
        else:
            return resolve_url("pydenticon_image", self.username)


class Place(models.Model):
    address_name = models.CharField(max_length=200, null=True, blank=True)
    category_group_code = models.CharField(max_length=50, null=True, blank=True)
    category_group_name = models.CharField(max_length=100, null=True, blank=True)
    category_name = models.CharField(max_length=50, null=True, blank=True)
    distance = models.CharField(max_length=50, null=True, blank=True)
    # id = models.AutoField(primary_key=True)  # 음식점 식별 번호
    id = models.CharField(primary_key=True, max_length=50)  # 음식점 식별 번호
    # serial_number = models.CharField(max_length=50)  # 음식점 식별 번호
    phone = models.CharField(max_length=18, blank=True, null=True)  # 음식점 전화 번호
    place_name = models.CharField(max_length=200)  # 음식점 상호명
    place_url = models.URLField(blank=True)
    road_address_name = models.CharField(max_length=50, null=True, blank=True)
    # x = models.DecimalField(max_digits=18, decimal_places=15)  # 음식점 위도
    # y = models.DecimalField(max_digits=18, decimal_places=15)  # 음식점 경도
    x = models.CharField(max_length=100, null=True, blank=True)
    y = models.CharField(max_length=100, null=True, blank=True)
    image = models.ImageField(
        blank=True,
        upload_to="accounts/image/%Y/%m/%d",
    )  # 음식점 이미지 필드

    # @property
    # def image_url(self):
    #     if self.image:
    #         return self.image.url
    #     else:
    #         return resolve_url("pydenticon_image", self.id)

    def __str__(self):
        return self.id


class UserRating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    place = models.ForeignKey(Place, on_delete=models.CASCADE)
    rating = models.IntegerField(blank=True)

    def __str__(self):
        return f'{self.user}' '_' f'{self.place}'

