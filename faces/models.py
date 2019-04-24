import os
from django.db import models
from django.db.models import Model
from django.contrib.auth.models import AbstractUser

class CustomUser(AbstractUser):
    def __str__(self):
        return "%i - %s"%(self.id, self.username)

class RegisteredPerson(Model):
    first_name = models.CharField('First name', max_length=40)
    last_name = models.CharField('Last name', max_length=25)
    is_enrstudent = models.BooleanField('Is an enrolled UP student', default=True)
    is_faculty = models.BooleanField('Is a faculty member', default=False)
    studentnum = models.CharField('Student/faculty number', max_length=9, null=True, blank=True)
    degreeprog = models.CharField('Degree program', max_length=50, null=True, blank=True)
    numphotos = models.PositiveIntegerField('Number of sample photos', default=0)

    def __str__(self):
        return "%i - %s %s"%(self.id, self.first_name,self.last_name)

def samplephotopath(instance, filename):
    path = 'samplephotos/'
    # NOTE: Be careful if you change this because the filename parsing in the ML model generator
    # will need to be modified accordingly
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
        return "%i - %s %s - %s"%(self.id, self.person.first_name, self.person.last_name)

class EndAgent(Model):
    manager = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    name = models.CharField('Name of End Agent', max_length=30, blank=False)
    location = models.CharField('Location of Agent', max_length=50, blank=True)
    # These are intended to be cumulative counts from the very start of the database deployment
    inf_count = models.PositiveIntegerField("Inferences performed count", default=0)
    ii_count = models.PositiveIntegerField("Incorrect identifications count", default=0)
    fp_count = models.PositiveIntegerField("False positives count", default=0)
    fn_count = models.PositiveIntegerField("False negatives count", default=0)

    def __str__(self):
        return "%i - %s"%(self.id, self.name)

class InferenceRequest(Model):
    endagent = models.ForeignKey(EndAgent, on_delete=models.CASCADE)
    person = models.ForeignKey(RegisteredPerson, on_delete=models.SET_NULL, blank=True, null=True)
    inference = models.ImageField(upload_to=infphotopath, default='defaultinference.jpg')
    timestamp = models.DateTimeField()
    face_detected = models.BooleanField('A face was detected', default=True)
    unknown_detected = models.BooleanField('An unknown face was detected', default=False)
    incorrect_identification = models.BooleanField(default=False)
    false_positive = models.BooleanField(default=False)
    false_negative = models.BooleanField(default=False)
    too_many_faces = models.BooleanField(default=False)
    l2_distance = models.FloatField("L2 distance between target face and nearest face in dataset", blank=True, null=True)

    def __str__(self):
        return "%i - %s by %s"%(self.id, self.timestamp, self.endagent.name)

def modelpath(instance, filename):
    path = 'models/'
    format = 'knnclf_%i.pkl'%instance.id
    return os.path.join(path,format)

class MLModelVersion(Model):
    time_trained = models.DateTimeField()
    model = models.FileField(upload_to=modelpath, default='defaultclf.pkl')
    total_photos = models.PositiveIntegerField(default=1)
    unique_persons = models.PositiveIntegerField(default=1)
    k_neighbors = models.PositiveIntegerField("k parameter for kNN algorithm", default=1)
    threshold = models.FloatField('Threshold for distinguishing registered and unregistered faces',default=1.1)
    is_in_use = models.BooleanField(default=False, blank=False, null=False)
    # These are intended to be cumulative counts since the next most recently trained model
    inf_count = models.PositiveIntegerField("Inferences performed", default=0)
    fp_count = models.PositiveIntegerField("False positives incurred", default=0)
    fn_count = models.PositiveIntegerField("False negatives incurred", default=0)
    ii_count = models.PositiveIntegerField("Incorrect identifications incurred", default=0)
    # frozen rates - are meant to be null and only calculated when a new model is deployed
    fa_rate = models.FloatField("False acceptance rate", default=None, blank=True, null=True)
    fr_rate = models.FloatField("False rejection rate", default=None, blank=True, null=True)
    ii_rate = models.FloatField("Incorrect identification rate", default=None, blank=True, null=True)


    def __str__(self):
        return "%i - %s - %s"%(self.id, self.model.url, self.time_trained)
