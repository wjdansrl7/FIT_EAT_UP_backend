from django.contrib.auth.models import AbstractUser
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver


class User(AbstractUser):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=20)
    nickname = models.CharField(max_length=30, unique=True)


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
