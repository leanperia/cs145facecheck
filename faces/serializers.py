from rest_framework import serializers
from .models import RegisteredPerson, SamplePhoto, InferenceRequest

class RegisteredPersonSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = RegisteredPerson
        fields = ('id','first_name', 'last_name', 'is_enrstudent',
            'is_faculty', 'studentnum', 'degreeprog', 'department', 'numphotos')

class SampePhotoSerializer(serializers.HyperlinkedModelSerializer):
    person = serializers.HyperlinkedRelatedField(
        many=True,
        read_only=True,
        view_name='registeredperson-detail'
    )
    class Meta:
        model = SamplePhoto
        fields = ('id', 'person', 'photo')

class InferenceRequestSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = InferenceRequest
        fields = ('endagent', 'person', 'face_detected', 'unknown_detected',
            'incorrect_identification', 'false_positive', 'false_negative',
            'too_many_faces', 'l2_distance')
