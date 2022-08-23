from django.urls import path

from . import views

app_name = 'foods'

urlpatterns = [
    path('search/', views.search, name='api_search'),  # https://localhost:8000/foods/search 음식점 검색
    path('search_img/', views.image_search, name='api_img_search'),  # https://localhost:8000/foods/search 음식점 이미지 검색


]