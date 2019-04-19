from django.http import HttpResponse, Http404, HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse_lazy, reverse
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView, ListView, DetailView, CreateView
from .models import RegisteredPerson, SamplePhoto, EndAgent, InferenceRequest

class HomePageView(TemplateView):
    template_name = 'home.html'

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

class MakeInference(LoginRequiredMixin, TemplateView):
    template_name = 'inferences_test.html'
