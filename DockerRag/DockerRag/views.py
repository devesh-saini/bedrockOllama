from django.http import HttpResponse
from django.shortcuts import render

def home(request):
    return render(request, 'index.html')

def about(request):
    return HttpResponse("I'm from the hood wanna be a gangsta.")

def contact(request):
    return HttpResponse("If it's not bussiness homie, we ain't talkin' shit.")