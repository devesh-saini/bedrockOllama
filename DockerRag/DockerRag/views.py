from django.http import HttpResponse
from django.shortcuts import render

def home(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

def tech_stack(request):
    return render(request, 'tech_stack.html')