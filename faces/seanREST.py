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

# todo:
# - logging RPi -> export to csv and be able to manually fetch it..?
# - 50 faces

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
            blob = BytesIO()
            image.save(blob, image.format)
            # file = request.FILES.get('image')
            p = RegisteredPerson.objects.filter(first_name=person_name).first()
            filename = str(p.id) + "_" + str(p.samplephoto_set.count() + 1) + "." +image.format
            data["image"] = filename
            s = SamplePhoto.objects.create(person=p)
            s.photo.save(filename, File(blob), save=False)
            p.samplephoto_set.add(s)
            p.numphotos = p.samplephoto_set.count()
            p.save()
            s.save()
            data["success"] = True
        return HttpResponse(json.dumps(data))

@csrf_exempt
# TODO
def DeletePhotoREST(request, person_name):
    photoName = request.POST.get('photo')
    if deletelist:
        targetPerson = RegisteredPerson.objects.filter(first_name=person_name).first()
        for stringid in deletelist:
            target = SamplePhoto.objects.get(pk=int(stringid))
            target.photo.delete(save=True)
            target.delete()
            person.numphotos -= 1
            person.save()
    return HttpResponseRedirect(reverse('view_person', args=(pk,)))
