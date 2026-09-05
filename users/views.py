from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required

# Create your views here.
# handling registering
def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('liked_songs:enter_favorites') 
    else:
        form = UserCreationForm()
    return render(request, 'users/register.html',{'form': form})




# handling user login
def user_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user is not None: 
            login(request, user)
            return redirect('users:home') 
        else: 
            messages.error(request, "Invalid username or password.")
    return render(request,'users/login.html')



# handling user logouts
def user_logout(request):
    logout(request)
    return redirect('users:home') 


# retun to home
def user_home(request): 
    return render(request, 'users/home.html')