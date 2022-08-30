from django.contrib.auth.models import AbstractUser
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.shortcuts import resolve_url


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

    @property
    def avatar_url(self):
        if self.avatar:
            return self.avatar.url
        else:
            return resolve_url("pydenticon_image", self.username)



# class Profile(models.Model):
#     user = models.OneToOneField(User, on_delete=models.CASCADE)
    # user_pk = models.IntegerField(blank=True)
    # nickname = models.CharField(max_length=30, unique=True)


# User 모델로부터 post_save라는 신호, 즉 User 모델 인스턴스 생성에 맞춰 Profile 모델 인스턴스 또한 함께 생성
# @receiver(post_save, sender=User)
# def create_user_profile(sender, instance, created, **kwargs):
#     if created:
#         Profile.objects.create(user=instance, user_pk=instance.id)
#
#
# @receiver(post_save, sender=User)
# def save_user_profile(sender, instance, **kwargs):
#     instance.profile.save()
