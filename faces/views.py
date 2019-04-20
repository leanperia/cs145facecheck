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
from .forms import RawImageUploadForm, RawIntegerForm
from facecheck.settings import MEDIA_ROOT, BASE_DIR #, KNN_CLASSIFIER, BACKBONE_CNN, IMG_TRANSFORM, PNET, RNET, ONET
from facedetect.topleveltools import detect_and_recognize, generate_knn_model
from facedetect.get_nets import PNet, RNet, ONet
from facedetect.model_irse import IR_50

class HomePageView(FormView):
    template_name = 'home.html'
    form_class = RawIntegerForm

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        orderedmodelversions = MLModelVersion.objects.all().order_by('-time_trained')
        context['mlmodelver'] = orderedmodelversions[0]
        context['total_photos'] = SamplePhoto.objects.all().count()
        context['unique_persons'] = RegisteredPerson.objects.filter(numphotos__gt=0).count()
        if len(orderedmodelversions) > 1:
            context['lastmodelver'] = orderedmodelversions[1]
        else: context['lastmodelver'] = None
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
    fields = ['first_name', 'last_name', 'is_enrstudent', 'is_faculty', 'studentnum', 'degreeprog']
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
        this_person.numphotos += 1
        this_person.save()
        form.instance.person = this_person
        self.object = form.save()
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
    fields = ['name', 'location']
    template_name = 'agents_add.html'
    success_url = reverse_lazy('list_agents')

    def form_valid(self, form):
        form.instance.manager = self.request.user
        return super().form_valid(form)


class ListInferences(LoginRequiredMixin, ListView):
    model = InferenceRequest
    template_name = 'inferences_list.html'
    context_object_name = 'targetlist'

class RequestInference(LoginRequiredMixin, FormView):
    form_class = RawImageUploadForm
    template_name = 'inferences_test.html'

class RetryInference(LoginRequiredMixin, FormView):
    form_class = RawImageUploadForm
    template_name = 'inferences_test.html'

def RunInference(request):
    webagent = cache.get_or_set('webagent',EndAgent.objects.get(pk=2))
    img = Image.open(request.FILES['raw_image'])

    knn_classifier = cache.get_or_set('KNNCLF', joblib.load(os.path.join(MEDIA_ROOT,json.load(open('media/knnclf_init.json','r'))['filepath'])))

    img_transform = cache.get("TRANSF")
    backbone_cnn = cache.get('CNN')
    pnet = cache.get('PNET')
    rnet = cache.get('RNET')
    onet = cache.get('ONET')
    if backbone_cnn == None:
        backbone_cnn = IR_50([112,112])
        backbone_cnn.load_state_dict(torch.load(os.path.join(MEDIA_ROOT,'models/backbone_cnn.pth')))
        backbone_cnn.eval()
    if pnet == None:
        pnet = PNet(os.path.join(MEDIA_ROOT,'models/'))
        pnet.eval()
    if rnet == None:
        rnet = RNet(os.path.join(MEDIA_ROOT,'models/'))
        rnet.eval()
    if onet == None:
        onet = ONet(os.path.join(MEDIA_ROOT,'models/'))
        onet.eval()
    if img_transform == None:
        img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])])

    prediction, resultimg, extra_faces = detect_and_recognize(
        img, knn_classifier, backbone_cnn, img_transform, pnet, rnet, onet)

    if prediction==None:
        instance = InferenceRequest.objects.create(
            endagent = webagent,
            timestamp = timezone.now(),
        )
        blob = BytesIO()
        img.save(blob, 'JPEG')
        instance.inference.save('newinference.jpg', File(blob), save=False)
        instance.save()
        return HttpResponseRedirect(reverse('retry_inference', args=(0,)))

    person = RegisteredPerson.objects.get(pk=prediction)
    fname, lname = person.first_name, person.last_name
    instance = InferenceRequest.objects.create(
        endagent = webagent,
        person = person,
        timestamp = timezone.now(),
        face_detected=True
        )
    blob = BytesIO()
    resultimg.save(blob, 'JPEG')
    instance.inference.save('newinference.jpg', File(blob), save=False)
    if extra_faces: instance.too_many_faces = True
    instance.save()

    webagent.inf_count += 1
    webagent.save()
    mv = MLModelVersion.objects.all().order_by('-time_trained')[0]
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
    mv = cache.get_or_set('mostrecentmv', MLModelVersion.objects.all().order_by('-time_trained')[0])
    values = request.POST.getlist('correction')
    for value in values:
        if value=="1":
            if instance.false_positive==True:
                agent.fp_count -= 1
                mv.fp_count -= 1
            instance.false_positive = False
        elif value=="2":
            if instance.false_positive==False:
                agent.fp_count += 1
                mv.fp_count += 1
            instance.false_positive = True
        elif value=="3":
            if instance.false_negative==True:
                agent.fn_count -= 1
                mv.fn_count -= 1
            instance.false_negative = False
        elif value=="4":
            if instance.false_negative==False:
                agent.fn_count += 1
                mv.fn_count += 1
            instance.false_negative = True
        elif value=="5":
            instance.face_detected = False
        elif value=="6":
            instance.face_detected = True
    instance.save()
    agent.save()
    mv.save()
    return HttpResponseRedirect(reverse('view_inference', args=(pk,)))

def RetrainMLmodel(request):
    mv = cache.get_or_set('mostrecentmv', MLModelVersion.objects.all().order_by('-time_trained')[0])
    mv.fa_rate = (mv.fp_count/float(mv.inf_count))*100
    mv.fr_rate = (mv.fn_count/float(mv.inf_count))*100
    mv.save()

    img_transform = cache.get("TRANSF")
    backbone_cnn = cache.get('CNN')
    pnet = cache.get('PNET')
    rnet = cache.get('RNET')
    onet = cache.get('ONET')
    if backbone_cnn == None:
        backbone_cnn = IR_50([112,112])
        backbone_cnn.load_state_dict(torch.load(os.path.join(MEDIA_ROOT,'models/backbone_cnn.pth')))
        backbone_cnn.eval()
    if pnet == None:
        pnet = PNet(os.path.join(MEDIA_ROOT,'models/'))
        pnet.eval()
    if rnet == None:
        rnet = RNet(os.path.join(MEDIA_ROOT,'models/'))
        rnet.eval()
    if onet == None:
        onet = ONet(os.path.join(MEDIA_ROOT,'models/'))
        onet.eval()
    if img_transform == None:
        img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])])

    new_clf = generate_knn_model(os.path.join(MEDIA_ROOT,'samplephotos/'),int(request.POST['k']),
        backbone_cnn, img_transform, pnet, rnet, onet)
    cache.set('KNNCLF', new_clf)
    print('Trained a new kNN classifier!')
    new_mv = MLModelVersion.objects.create(
        time_trained = timezone.now(),
        total_photos = SamplePhoto.objects.all().count(),
        unique_persons = RegisteredPerson.objects.filter(numphotos__gt=0).count(),
        k_neighbors = request.POST['k']
    )
    filename = "knnclf_%i.sav"%new_mv.id
    new_mv.filepath = os.path.join('models/',filename)
    new_mv.save()
    cache.set('mostrecentmv', new_mv)

    joblib.dump(new_clf, os.path.join(MEDIA_ROOT, new_mv.filepath))

    new_init = {'time_trained':new_mv.time_trained, 'filepath':new_mv.filepath}
    with open('media/knnclf_init.json', 'w') as f:
        json.dump(new_init, f)
    return HttpResponseRedirect(reverse_lazy('home'))
