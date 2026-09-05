# library/urls.py

from django.urls import path
from . import views

app_name = 'liked_songs'

urlpatterns = [
    path('favorites/', views.enter_favorites, name='enter_favorites'),
    path('liked-songs/', views.liked_songs, name='liked_songs'),
    path('delete_liked/', views.delete_liked, name='delete_liked'),
    path('favorites/', views.enter_favorites, name='enter_favorites')
]