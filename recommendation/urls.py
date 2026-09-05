from django.urls import path 
from . import views

urlpatterns = [
   
    path('get_recommendation/', views.get_recommendation, name='get_recommendation'),
    path('rating', views.rating, name='rating'),
    
]