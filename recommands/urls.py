from django.urls import path

from . import views

urlpatterns = [
    # path('recommands/', views.Train.as_view(), name='recommands'),
    path('surprise/', views.surprise_train.as_view(), name='surprise_recommand'),
    path('random/', views.random_recomm.as_view(), name='random_recommand'),

]