from django.contrib import admin

from .models import User, Place


# 장고 admin 페이지에 User 생성
@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    pass


@admin.register(Place)
class PlaceAdmin(admin.ModelAdmin):
    pass

# @admin.register(Profile)
# class ProfileAdmin(admin.ModelAdmin):
#     pass




