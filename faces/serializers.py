from rest_framework import serializers
from .models import RegisteredPerson, SamplePhoto, InferenceRequest, EndAgent

class RegisteredPersonSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = RegisteredPerson
        fields = ('id','first_name', 'last_name', 'is_enrstudent',
            'is_faculty', 'studentnum', 'degreeprog', 'department')

class SamplePhotoSerializer(serializers.HyperlinkedModelSerializer):
    person = serializers.StringRelatedField()
        #view_name='registeredperson-detail',
        #allow_null = True)
    photo = serializers.ImageField(use_url=True, allow_empty_file=True)
    class Meta:
        model = SamplePhoto
        fields = ('id', 'person', 'photo',)

class InferenceRequestSerializer(serializers.HyperlinkedModelSerializer):
    person = serializers.HyperlinkedRelatedField(view_name='registeredperson-detail',
        queryset = RegisteredPerson.objects.all())
    endagent = serializers.HyperlinkedRelatedField(view_name='endagent-detail',
        queryset = EndAgent.objects.all())
    inference = serializers.ImageField(use_url=True, allow_empty_file=True)
    class Meta:
        model = InferenceRequest
        fields = ('id','endagent', 'person',  'inference','face_detected', 'unknown_detected',
            'incorrect_identification', 'false_positive', 'false_negative',
            'too_many_faces', 'l2_distance')

class EndAgentSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = EndAgent
        fields = ('name', 'location', 'inf_count', 'ii_count', 'fp_count',
            'fn_count', 'serial_number', 'secret_key')
