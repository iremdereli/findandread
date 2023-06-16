# Authentication URLS
from django.urls import path

from finder import views

urlpatterns = [

    path('', views.BookView.as_view(),
         name='home'),
]
