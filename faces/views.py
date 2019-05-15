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

class HomePageView(TemplateView):
    template_name = 'home.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['mlmodelver'] = cache.get_or_set('deployedmv', MLModelVersion.objects.filter(is_in_use=True)[0])
        context['total_photos'] = SamplePhoto.objects.all().count()
        context['unique_persons'] = RegisteredPerson.objects.filter(numphotos__gt=0).count()
        context['form'] = RetrainModelForm()
        return context

class ListModels(ListView):
    model = MLModelVersion
    template_name = "models_list.html"
    context_object_name = "modelslist"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = RetrainModelForm

        return context

class ListPersons(LoginRequiredMixin, ListView):
    model = RegisteredPerson
    template_name = 'persons_list.html'
    context_object_name = 'targetlist'

class ViewPerson(LoginRequiredMixin, DetailView):
    model = RegisteredPerson
    template_name = 'persons_detail.html'
    context_object_name = 'targetperson'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['samplephotos'] = SamplePhoto.objects.filter(person_id=self.kwargs['pk'])
        return context

class AddPerson(LoginRequiredMixin, CreateView):
    model = RegisteredPerson
    fields = ['first_name', 'last_name', 'is_enrstudent', 'is_faculty', 'studentnum', 'degreeprog', 'department']
    template_name = 'persons_add.html'
    success_url = reverse_lazy('add_person')

class AddSamplePhoto(LoginRequiredMixin, CreateView):
    model = SamplePhoto
    fields = ['photo']
    template_name = 'sampphotos_add.html'

    def form_valid(self, form):
        # The way form_valid works is the form entry is saved, and then we get redirected to the success url
        # Instead of doing return super().form_valid(form), we insert the necessary increment to RegisteredPerson.numphotos before saving
        this_person = RegisteredPerson.objects.get(pk=self.kwargs['pk'])
        form.instance.person = this_person

        img_transform = cache.get("TRANSF")
        backbone_cnn = cache.get('CNN')
        pnet = cache.get('PNET')
        rnet = cache.get('RNET')
        onet = cache.get('ONET')
        if backbone_cnn == None:
            backbone_cnn = IR_50([112,112])
            if LOCAL_BACKBONE:
                print("loading backbone_cnn.pth locally")
                backbone_cnn.load_state_dict(torch.load(os.path.join(MEDIA_ROOT,'models/backbone_cnn.pth')))
            else:
                print('downloading pretrained InceptionResnet from S3 bucket')
                newf = urlopen('https://s3-ap-southeast-1.amazonaws.com/cs145facecheck/media/models/backbone_cnn.pth')
                backbone_cnn.load_state_dict(torch.load(BytesIO(newf.read()), map_location='cpu'))
            backbone_cnn.eval()
            print('CNN loaded successfully')
            cache.set('CNN', backbone_cnn)
        if pnet == None:
            pnet = PNet(MEDIA_URL+'models/')
            pnet.eval()
            cache.set('PNET', pnet)
        if rnet == None:
            rnet = RNet(MEDIA_URL+'models/')
            rnet.eval()
            cache.set('RNET', rnet)
        if onet == None:
            onet = ONet(MEDIA_URL+'models/')
            onet.eval()
            cache.set('ONET', onet)
        if img_transform == None:
            img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])])
            cache.set('TRANSF', img_transform)

        uploaded_img = Image.open(self.request.FILES['photo'])
        embedding = generate_embedding(uploaded_img, backbone_cnn, img_transform, pnet, rnet, onet)
        form.instance.embedding = embedding.tolist()[0]

        self.object = form.save()
        this_person.numphotos += 1
        this_person.save()
        return HttpResponseRedirect(self.get_success_url())

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['person'] = RegisteredPerson.objects.get(pk=self.kwargs['pk'])
        return context

    def get_success_url(self):
        return reverse('add_samplephoto', args=(self.kwargs['pk'],))


class EditSamplePhotos(LoginRequiredMixin, TemplateView):
    template_name = 'sampphotos_list.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['person'] = RegisteredPerson.objects.get(pk=self.kwargs['pk'])
        context['samplephotos'] = SamplePhoto.objects.filter(person_id=self.kwargs['pk'])
        return context

def DeleteSamplePhotos(request, pk):
    deletelist = request.POST.getlist('photos')
    if deletelist:
        person = RegisteredPerson.objects.get(pk=pk)
        for stringid in deletelist:
            target = SamplePhoto.objects.get(pk=int(stringid))
            target.photo.delete(save=True)
            target.delete()
            person.numphotos -= 1
            person.save()
    return HttpResponseRedirect(reverse('view_person', args=(pk,)))


class ListAgents(LoginRequiredMixin, ListView):
    model = EndAgent
    template_name = 'agents_list.html'
    context_object_name = 'targetlist'

class AddAgent(LoginRequiredMixin, CreateView):
    model = EndAgent
    fields = ['name', 'location', 'serial_number', 'secret_key']
    template_name = 'agents_add.html'
    success_url = reverse_lazy('list_agents')

    def form_valid(self, form):
        form.instance.manager = self.request.user
        return super().form_valid(form)


class ListInferences(LoginRequiredMixin, ListView):
    model = InferenceRequest
    template_name = 'inferences_list.html'
    context_object_name = 'targetlist'
    ordering = ['-timestamp']

class RequestInference(LoginRequiredMixin, FormView):
    form_class = RawImageUploadForm
    template_name = 'inferences_test.html'

class RetryInference(LoginRequiredMixin, FormView):
    form_class = RawImageUploadForm
    template_name = 'inferences_test.html'

def RunInference(request):
    webagent = cache.get_or_set('webagent',EndAgent.objects.get(pk=2))
    mv = cache.get_or_set('deployedmv', MLModelVersion.objects.filter(is_in_use=True)[0])
    img = Image.open(request.FILES['raw_image'])


    knn_classifier = cache.get('KNNCLF')
    if knn_classifier == None:
        newfile = urlopen(mv.model.url).read()
        knn_classifier = joblib.load(BytesIO(newfile))
        cache.set('KNNCLF', knn_classifier)

    img_transform = cache.get("TRANSF")
    backbone_cnn = cache.get('CNN')
    pnet = cache.get('PNET')
    rnet = cache.get('RNET')
    onet = cache.get('ONET')
    if backbone_cnn == None:
        backbone_cnn = IR_50([112,112])
        if LOCAL_BACKBONE:
            print("loading backbone_cnn.pth locally")
            backbone_cnn.load_state_dict(torch.load(os.path.join(MEDIA_ROOT,'models/backbone_cnn.pth')))
        else:
            print('downloading pretrained InceptionResnet from S3 bucket')
            newf = urlopen('https://s3-ap-southeast-1.amazonaws.com/cs145facecheck/media/models/backbone_cnn.pth')
            backbone_cnn.load_state_dict(torch.load(BytesIO(newf.read()), map_location='cpu'))
        backbone_cnn.eval()
        print('CNN loaded successfully')
        cache.set('CNN', backbone_cnn)
    if pnet == None:
        pnet = PNet(MEDIA_URL+'models/')
        pnet.eval()
        cache.set('PNET', pnet)
    if rnet == None:
        rnet = RNet(MEDIA_URL+'models/')
        rnet.eval()
        cache.set('RNET', rnet)
    if onet == None:
        onet = ONet(MEDIA_URL+'models/')
        onet.eval()
        cache.set('ONET', onet)
    if img_transform == None:
        img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])])
        cache.set('TRANSF', img_transform)


    prediction, distance, resultimg, extra_faces = detect_and_recognize(
        img, knn_classifier, backbone_cnn, img_transform, pnet, rnet, onet)

    if prediction==None:
        instance = InferenceRequest.objects.create(
            endagent = webagent,
            timestamp = timezone.now(),
            face_detected = False,
        )
        blob = BytesIO()
        img.save(blob, 'JPEG')
        instance.inference.save('newinference.jpg', File(blob), save=False)
        instance.save()
        return HttpResponseRedirect(reverse('retry_inference', args=(0,)))

    if distance > mv.threshold:
        instance = InferenceRequest.objects.create(
            endagent = webagent,
            timestamp = timezone.now(),
            unknown_detected = True,
            l2_distance = distance,
        )
    else:
        person = RegisteredPerson.objects.get(pk=prediction)
        instance = InferenceRequest.objects.create(
            endagent = webagent,
            person = person,
            timestamp = timezone.now(),
            l2_distance = distance,
            )
    blob = BytesIO()
    resultimg.save(blob, 'JPEG')
    instance.inference.save('newinference.jpg', files.File(blob), save=False)
    if extra_faces: instance.too_many_faces = True
    instance.save()

    webagent.inf_count += 1
    webagent.save()
    mv.inf_count += 1
    mv.save()

    return HttpResponseRedirect(reverse('view_inference', args=(instance.id,)))

class ViewInference(LoginRequiredMixin, DetailView):
    model = InferenceRequest
    template_name = 'inferences_detail.html'
    context_object_name = 'targetitem'

def EditInference(request, pk):
    instance = InferenceRequest.objects.get(pk=pk)
    agent = cache.get_or_set('webagent',EndAgent.objects.get(pk=2))
    mv = cache.get_or_set('deployedmv', MLModelVersion.objects.filter(is_in_use=True)[0])
    values = request.POST.getlist('correction')
    print('the POST dict contains ', values)
    if "1" in values:
        if instance.false_positive==True:
            agent.fp_count -= 1
            mv.fp_count -= 1
            instance.false_positive = False
    elif "2" in values:
        if instance.false_positive==False:
            agent.fp_count += 1
            mv.fp_count += 1
            instance.false_positive = True
    elif "3" in values:
        if instance.false_negative==True:
            agent.fn_count -= 1
            mv.fn_count -= 1
            instance.false_negative = False
    elif "4" in values:
        if instance.false_negative==False:
            agent.fn_count += 1
            mv.fn_count += 1
            instance.false_negative = True
    elif "5" in values:
        if instance.incorrect_identification==False:
            agent.ii_count += 1
            mv.ii_count += 1
            instance.incorrect_identification = True
    elif "6" in values:
        if instance.incorrect_identification==True:
            agent.ii_count -= 1
            mv.ii_count -= 1
            instance.incorrect_identification = False
    instance.save()
    agent.save()
    mv.save()
    return HttpResponseRedirect(reverse('view_inference', args=(pk,)))

def SwitchModel(request):
    id = int(request.POST.getlist('select model')[0])
    mv = cache.get_or_set('deployedmv', MLModelVersion.objects.filter(is_in_use=True)[0])
    mv.is_in_use = False
    mv.save()

    mv = MLModelVersion.objects.get(pk=id)
    mv.is_in_use = True
    mv.save()
    cache.set('deployedmv', mv)

    return HttpResponseRedirect(reverse_lazy('home'))

def RetrainMLmodel(request):
    k = int(request.POST['k_neighbors'])
    thresh = float(request.POST['threshold'])

    # Retrieve the currently used model first, train new model THEN change is_in_use to False
    mv = cache.get_or_set('deployedmv', MLModelVersion.objects.filter(is_in_use=True)[0])


    img_transform = cache.get("TRANSF")
    backbone_cnn = cache.get('CNN')
    pnet = cache.get('PNET')
    rnet = cache.get('RNET')
    onet = cache.get('ONET')
    if backbone_cnn == None:
        backbone_cnn = IR_50([112,112])
        if LOCAL_BACKBONE:
            print("loading backbone_cnn.pth locally")
            backbone_cnn.load_state_dict(torch.load(os.path.join(MEDIA_ROOT,'models/backbone_cnn.pth')))
        else:
            print('downloading pretrained InceptionResnet from S3 bucket')
            newf = urlopen('https://s3-ap-southeast-1.amazonaws.com/cs145facecheck/media/models/backbone_cnn.pth')
            backbone_cnn.load_state_dict(torch.load(BytesIO(newf.read()), map_location='cpu'))
        backbone_cnn.eval()
        print('CNN loaded successfully')
        cache.set('CNN', backbone_cnn)
    if pnet == None:
        pnet = PNet(MEDIA_URL+'models/')
        pnet.eval()
        cache.set('PNET', pnet)
    if rnet == None:
        rnet = RNet(MEDIA_URL+'models/')
        rnet.eval()
        cache.set('RNET', rnet)
    if onet == None:
        onet = ONet(MEDIA_URL+'models/')
        onet.eval()
        cache.set('ONET', onet)
    if img_transform == None:
        img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])])
        cache.set('TRANSF', img_transform)

    X, y = [], []
    print("Retrieving all face embeddings and training a classifier.")
    samplephotos = SamplePhoto.objects.all()
    for instance in samplephotos:
        X.append(np.array(instance.embedding))
        y.append(instance.person.id)
    X = np.array(X)
    #X = np.squeeze(X, axis=1)
    new_clf = KNeighborsClassifier(n_neighbors = k)
    new_clf.fit(X,y)

    cache.set('KNNCLF', new_clf)
    print('Trained a new kNN classifier!')
    new_mv = MLModelVersion.objects.create(
        time_trained = timezone.now(),
        total_photos = SamplePhoto.objects.all().count(),
        unique_persons = RegisteredPerson.objects.filter(numphotos__gt=0).count(),
        k_neighbors = k,
        threshold = thresh,
        is_in_use = True
    )
    newf = BytesIO()
    joblib.dump(new_clf, newf)
    new_mv.model.save('newclf.pkl', files.File(newf), save=False)
    new_mv.save()
    cache.set('deployedmv', new_mv)

    mv.fa_rate = (mv.fp_count/float(mv.inf_count))*100
    mv.fr_rate = (mv.fn_count/float(mv.inf_count))*100
    mv.ii_rate = (mv.ii_count/float(mv.inf_count))*100
    mv.is_in_use = False
    mv.save()
    return HttpResponseRedirect(reverse_lazy('home'))


@csrf_exempt
def RESTRunInference(request):
    output = {'success':False, 'authenticated': False}
    try:
        agent = EndAgent.objects.get(serial_number=request.POST.get('sn', None))
    except ObjectDoesNotExist:
        return HttpResponse(json.dumps(output))
    if request.POST.get('key', None) != agent.secret_key:
        return HttpResponse(json.dumps(output))
    else:
        output['authenticated'] = True

    mv = cache.get_or_set('deployedmv', MLModelVersion.objects.filter(is_in_use=True)[0])
    img = Image.open(request.FILES['image'])

    knn_classifier = cache.get('KNNCLF')
    if knn_classifier == None:
        newfile = urlopen(mv.model.url).read()
        knn_classifier = joblib.load(BytesIO(newfile))
        cache.set('KNNCLF', knn_classifier)

    img_transform = cache.get("TRANSF")
    backbone_cnn = cache.get('CNN')
    pnet = cache.get('PNET')
    rnet = cache.get('RNET')
    onet = cache.get('ONET')
    if backbone_cnn == None:
        backbone_cnn = IR_50([112,112])
        if LOCAL_BACKBONE:
            print("loading backbone_cnn.pth locally")
            backbone_cnn.load_state_dict(torch.load(os.path.join(MEDIA_ROOT,'models/backbone_cnn.pth')))
        else:
            print('downloading pretrained InceptionResnet from S3 bucket')
            newf = urlopen('https://s3-ap-southeast-1.amazonaws.com/cs145facecheck/media/models/backbone_cnn.pth')
            backbone_cnn.load_state_dict(torch.load(BytesIO(newf.read()), map_location='cpu'))
        backbone_cnn.eval()
        print('CNN loaded successfully')
        cache.set('CNN', backbone_cnn)
    if pnet == None:
        pnet = PNet(MEDIA_URL+'models/')
        pnet.eval()
        cache.set('PNET', pnet)
    if rnet == None:
        rnet = RNet(MEDIA_URL+'models/')
        rnet.eval()
        cache.set('RNET', rnet)
    if onet == None:
        onet = ONet(MEDIA_URL+'models/')
        onet.eval()
        cache.set('ONET', onet)
    if img_transform == None:
        img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])])
        cache.set('TRANSF', img_transform)


    prediction, distance, resultimg, extra_faces = detect_and_recognize(
        img, knn_classifier, backbone_cnn, img_transform, pnet, rnet, onet)

    if prediction==None:
        instance = InferenceRequest.objects.create(
            endagent = agent,
            timestamp = timezone.now(),
            face_detected = False,
        )
        blob = BytesIO()
        img.save(blob, 'JPEG')
        instance.inference.save('newinference.jpg', File(blob), save=False)
        instance.save()
        return HttpResponse(json.dumps(output))

    if distance > mv.threshold:
        instance = InferenceRequest.objects.create(
            endagent = agent,
            timestamp = timezone.now(),
            unknown_detected = True,
            l2_distance = distance,
        )
    else:
        person = RegisteredPerson.objects.get(pk=prediction)
        instance = InferenceRequest.objects.create(
            endagent = agent,
            person = person,
            timestamp = timezone.now(),
            l2_distance = distance,
            )
    blob = BytesIO()
    resultimg.save(blob, 'JPEG')
    instance.inference.save('newinference.jpg', files.File(blob), save=False)
    if extra_faces: instance.too_many_faces = True
    instance.save()

    agent.inf_count += 1
    agent.save()
    mv.inf_count += 1
    mv.save()

    output['success'] = True
    if distance > mv.threshold:
        output['decision'] = "reject"
        output['name'] = "Unknown"
        output['idnumber'] = "n/a"
    else:
        output['decision'] = "accept"
        output['name'] = f"{instance.person.first_name} {instance.person.last_name}"
        output['idnumber'] = f"{instance.person.studentnum}"
        output['inference_pk'] = f"{instance.id}"
        # FP
        # FN


    json_out = json.dumps(output)
    return HttpResponse(json_out)
    """
    This is the format of the curl request:
    curl 'http://localhost:8000/rest/request-inference' -X POST
    -F 'sn=v93nagsd09132nas' -F 'key=Fs9gX@a8pzTl$20m' -F image=@sakura05.jpg
    """

@csrf_exempt
def InferenceCorrection(request):
    output = {'success': False}

    return HttpResponse(json.dumps(output))

@csrf_exempt
def RESTAddPhoto(request):
    output = {'success': False}

    return HttpResponse(json.dumps(output))

class RegisteredPersonViewSet(ModelViewSet):
    queryset = RegisteredPerson.objects.all()
    serializer_class = RegisteredPersonSerializer
    permission_classes = (IsAuthenticated,)
    authentication_classes = (BasicAuthentication,)

"""
class SamplePhotoViewSet(ModelViewSet):
    queryset = SamplePhoto.objects.all()
    serializer_class = SamplePhoto
    permission_classes = (IsAuthenticated,)
    authentication_classes = (BasicAuthentication,)

class InferenceRequestViewSet(ModelViewSet):
    queryset =  InferenceRequest.objects.all()
    serializer_class = InferenceRequestSerializer
    permission_classes = (IsAuthenticated,)
    authentication_classes = (BasicAuthentication,)
"""
