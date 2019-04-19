import os
from django.db import models
from django.db.models import Model
from django.contrib.auth.models import AbstractUser

class CustomUser(AbstractUser):
    def __str__(self):
        return str(self.username)

class RegisteredPerson(Model):
    first_name = models.CharField('First name', max_length=40)
    last_name = models.CharField('Last name', max_length=25)
    is_enrstudent = models.BooleanField('Is an enrolled UP student', default=True)
    is_faculty = models.BooleanField('Is a faculty member', default=False)
    studentnum = models.CharField('Student/faculty number', max_length=9, null=True, blank=True)
    degreeprog = models.CharField('Degree program', max_length=50, null=True, blank=True)
    numphotos = models.PositiveIntegerField('Number of sample photos', default=0)

    def __str__(self):
        return str(self.first_name)+" "+str(self.last_name)

def samplephotopath(instance, filename):
    path = 'samplephotos/'
    format = 'person'+str(instance.person.id)+'_'+str(instance.person.numphotos)+'.jpg'
    return os.path.join(path, format)

def infphotopath(instance, filename):
    path = 'infrequests/'
    format = 'inference_'+str(instance.timestamp)+'.jpg'
    return os.path.join(path, format)

class SamplePhoto(Model):
    person = models.ForeignKey(RegisteredPerson, on_delete=models.CASCADE)
    photo = models.ImageField(upload_to=samplephotopath, default='default.png')

    def __str__(self):
        return str(self.person.first_name) + " " + str(self.person.last_name)+ " " + str(self.photo)

class EndAgent(Model):
    manager = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    name = models.CharField('Name of End Agent', max_length=30, blank=False)
    location = models.CharField('Location of Agent', max_length=50, blank=True)
    inf_count = models.PositiveIntegerField("Inferences performed count", default=0)
    fp_count = models.PositiveIntegerField("False positives count", default=0)
    fn_count = models.PositiveIntegerField("False negatives count", default=0)

    def __str__(self):
        return self.name

class InferenceRequest(Model):
    endagent = models.ForeignKey(EndAgent, on_delete=models.CASCADE)
    person = models.ManyToManyField(RegisteredPerson, blank=True)
    inference = models.ImageField(upload_to=infphotopath, default='defaultinference.jpg')
    timestamp = models.DateTimeField()
    inf_count = models.PositiveIntegerField("Number of faces detected", default=0)
    fp_count = models.PositiveIntegerField("Number of false positives in this inference", default=0)
    fn_count = models.PositiveIntegerField("Number of false negatives in this inference", default=0)


    def __str__(self):
        return str(self.timestamp) + " " + str(self.endagent)
