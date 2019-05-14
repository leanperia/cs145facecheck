import os
import json
import torch
import torchvision.transforms as transforms
from io import BytesIO
from PIL import Image
from sklearn.externals import joblib

from django.http import HttpResponse, Http404, HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse_lazy, reverse
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView, ListView, DetailView, CreateView, FormView
from django.core.files import File
from django.utils import timezone
from django.core.cache import cache

from .models import RegisteredPerson, SamplePhoto, EndAgent, InferenceRequest, MLModelVersion
from .forms import RawImageUploadForm, RetrainModelForm
from facecheck.settings import MEDIA_ROOT, BASE_DIR #, KNN_CLASSIFIER, BACKBONE_CNN, IMG_TRANSFORM, PNET, RNET, ONET
from facedetect.topleveltools import detect_and_recognize, generate_knn_model
from facedetect.get_nets import PNet, RNet, ONet
from facedetect.model_irse import IR_50
import json
from django.views.decorators.csrf import csrf_exempt

# For storing files to permanent storage
from django.core.files.storage import default_storage
# TODO
# "make a REST api para makakapag CRUD ng RegisteredPerson, SamplePhoto and InferenceRequest by curl."

@csrf_exempt
def helloWorld(request):
    return render(request, "sean.html")

@csrf_exempt
def exampleJSON(request):
    data= {"hello": "wow"}
    return HttpResponse(json.dumps(data))

@csrf_exempt
def AddPeopleREST(request):
    data = {"success": False}
    if request.method == "POST":
        inbound = json.loads(request.body)
        data = []
        for p in inbound:
            RegisteredPerson.objects.create(**p)
            data.append(p)
        data.append({"success": True})
        return HttpResponse(json.dumps(data))

@csrf_exempt
def AddPhotoREST(request, person_name):
    data = {"success": False}
    if request.method == "POST":
        if request.FILES.get('image'): # if there is an image
            print(request.FILES['image'])
            image = Image.open(request.FILES['image'])
            file = request.FILES.get('image')
            targetPerson = RegisteredPerson.objects.filter(first_name=person_name).first()
            targetPerson.numphotos += 1
            filename = str(targetPerson.id) + "_" + str(targetPerson.numphotos) + "." +image.format
            data["image"] = filename
            with default_storage.open('samplephotos/' + filename, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
            targetPerson.save()
            data["success"] = True
        return HttpResponse(json.dumps(data))
