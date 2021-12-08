from django.db import models
from django.http.response import HttpResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import render
import pandas as pd

# Create your views here.

def predict(request):
    return render(request, 'predict.html')


def predict_changes(request):
    if request.method == "POST":
        sepal_length = float(request.POST.get("sepal_length"))
        sepal_width = float(request.POST.get("sepal_width"))
        petal_length = float(request.POST.get("petal_length"))
        petal_width = float(request.POST.get("petal_width"))

        model = pd.read_pickle("model.pickle")
        pipe = pd.read_pickle("pipe.pickle")
        le = pd.read_pickle("LabelEncoder.pickle")
        result = le.inverse_transform(model.predict(pipe.transform([[sepal_length, sepal_width, petal_length, petal_width]])))
        

        return JsonResponse({"result":str(result)})
