from django.urls import path

from . import views

urlpatterns = [
    path('surprise/', views.surprise_train.as_view(), name='surprise_recommand'),
    path('random/', views.random_recomm.as_view(), name='random_recommand'),
]