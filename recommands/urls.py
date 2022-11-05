from django.urls import path

from . import views

urlpatterns = [
    path('index/', views.index),
    path('recommand/', views.Train.as_view(), name='recommand'),
]