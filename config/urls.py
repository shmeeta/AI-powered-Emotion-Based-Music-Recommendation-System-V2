
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include(('users.urls', 'users'), namespace='users')),
    path('', include(('liked_songs.urls', 'liked_songs'), namespace='liked_songs')),
    path('', include(('recommendation.urls', 'recommendation'), namespace='recommendation'))
    
]
