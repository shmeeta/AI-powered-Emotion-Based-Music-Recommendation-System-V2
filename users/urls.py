from django.urls import path 
from . import views
from .views import register, user_login, user_logout, user_home 
app_name = 'users'
urlpatterns = [
    path('', user_home, name='home'),
    path('login/', user_login, name='login'),
    path('register/', register, name='register'),
    path('logout/', user_logout, name='logout')
    
]