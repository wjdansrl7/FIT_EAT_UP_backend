from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from django_pydenticon.views import image as pydenticon_image

router = DefaultRouter()

urlpatterns = [
    path('admin/', admin.site.urls), # 장고 서버 관리자 계정 : localhost:8000/admin
    path('accounts/', include('accounts.urls')),
    path('recommands/', include('recommands.urls')),
    path('identicon/image/<path:data>.png/', pydenticon_image, name='pydenticon_image'),
]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

    import debug_toolbar

    urlpatterns += [
        path('__debug__/', include('debug_toolbar.urls')),
    ]

