import os
import json
import torch
import torchvision.transforms as transforms
import numpy as np
from io import BytesIO
from PIL import Image
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from urllib.request import urlopen

from django.http import HttpResponse, Http404, HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse_lazy, reverse
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView, ListView, DetailView, CreateView, FormView
from django.core import files
from django.utils import timezone
from django.core.cache import cache
from django.views.decorators.csrf import csrf_exempt
from django.core.exceptions import ObjectDoesNotExist
from rest_framework.viewsets import ModelViewSet
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import BasicAuthentication

from .models import RegisteredPerson, SamplePhoto, EndAgent, InferenceRequest, MLModelVersion
from .serializers import RegisteredPersonSerializer, SampePhotoSerializer, InferenceRequestSerializer
from .forms import RawImageUploadForm, RetrainModelForm
from facecheck.settings import MEDIA_ROOT, BASE_DIR, MEDIA_URL #, KNN_CLASSIFIER, BACKBONE_CNN, IMG_TRANSFORM, PNET, RNET, ONET
from facedetect.topleveltools import detect_and_recognize, generate_knn_model, generate_embedding
from facedetect.get_nets import PNet, RNet, ONet
from facedetect.model_irse import IR_50

LOCAL_BACKBONE = False

# RPI
# POST
# inference_pk - refers to the INFERENCES
# json['correction'] = "FN"/"FP"/"II" (Incorrect Identification)
# json['sn']
# json['key']


@csrf_exempt
def RESTInferenceCorrection(request):
    output = {'success':False, 'authenticated': False}
    print(request.POST)
    try:
        agent = EndAgent.objects.get(serial_number=request.POST.get('sn', None))
    except ObjectDoesNotExist:
        return HttpResponse(json.dumps(output))
    if request.POST.get('key', None) != agent.secret_key:
        return HttpResponse(json.dumps(output))
    else:
        output['authenticated'] = True

    i = InferenceRequest.objects.get(pk=request.POST.get('inference_pk', None))
    correction = request.POST.get('correction', None)
    agent = i.endagent
    model = MLModelVersion.objects.filter(is_in_use=True)[0]


    if correction == "FN":
        i.false_negative = True
        model.fp_count += 1
        agent.fp_count += 1
    elif correction == "FP":
        i.false_positive = True
        model.fn_count += 1
        agent.fn_count += 1
    elif correction == "II":
        i.incorrect_identification = True
        model.ii_count += 1
        agent.ii_count += 1
    i.save()
    agent.save()
    model.save()

    output['success'] = True

    json_out = json.dumps(output)
    return HttpResponse(json_out)
    """
    This is the format of the curl request:
    curl 'http://localhost:8000/rest/request-inference' -X POST
    -F 'sn=v93nagsd09132nas' -F 'key=Fs9gX@a8pzTl$20m' -F image=@sakura05.jpg
    """
