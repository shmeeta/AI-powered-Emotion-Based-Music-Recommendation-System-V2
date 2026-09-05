
# Create your models here.
from django.conf import settings
from django.db import models
# Create your models here.
class FavoriteSong(models.Model):

    
    # updated database values
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    rating = models.CharField(max_length=200, default = "unknown")
    artist_name = models.CharField(max_length=200, default ="unknown")
    track_name = models.CharField(max_length=200, default ="unknown")
    popularity = models.CharField(max_length=200, default="unknown")
    year = models.CharField(max_length=200, default="unknown")
    genre = models.CharField(max_length=200, default = "unknown")
    danceability = models.CharField(max_length=200, default="unknown")
    energy = models.CharField(max_length=200, default="unknown")
    loudness = models.CharField(max_length=200, default="unknown")
    speechiness = models.CharField(max_length=200, default="unknown")
    acousticness = models.CharField(max_length=200, default="unknown")
    instrumentalness = models.CharField(max_length=200, default="unknown")
    liveness = models.CharField(max_length=200, default="unknown") 
    valence = models.CharField(max_length=200, default="unknown")
    tempo = models.CharField(max_length=200, default="unknown")
    
    
  
    
    def __str__(self):
        return f"{self.track_name} by {self.artist_name} ({self.user.username})"


class DislikedSongs(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    rating = models.CharField(max_length=200, default = "unknown")
    artist_name = models.CharField(max_length=200, default ="unknown")
    track_name = models.CharField(max_length=200, default ="unknown")
    popularity = models.CharField(max_length=200, default="unknown")
    year = models.CharField(max_length=200, default="unknown")
    genre = models.CharField(max_length=200, default = "unknown")
    danceability = models.CharField(max_length=200, default="unknown")
    energy = models.CharField(max_length=200, default="unknown")
    loudness = models.CharField(max_length=200, default="unknown")
    speechiness = models.CharField(max_length=200, default="unknown")
    acousticness = models.CharField(max_length=200, default="unknown")
    instrumentalness = models.CharField(max_length=200, default="unknown")
    liveness = models.CharField(max_length=200, default="unknown") 
    valence = models.CharField(max_length=200, default="unknown")
    tempo = models.CharField(max_length=200, default="unknown")

    def __str__(self):
        return f"{self.track_name} by {self.artist_name} ({self.user.username})"