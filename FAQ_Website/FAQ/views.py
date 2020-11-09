import json
from django.shortcuts import render
from django.http import HttpResponse
from . import model


def index(request):
    return render(request, 'index.html')


def search(request):
    if request.method == 'POST':
        post_text = request.POST.get('question')
        answer, answer1, answer2, answer3 = model.data(post_text)

    context = {
        'question': post_text,
        'answer': answer,
        'answer1': answer1,
        'answer2': answer2,
        'answer3': answer3,

    }
    return render(request, 'search.html', context)
