from django import forms
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from .models import CustomUser, MLModelVersion

class CustomUserCreationForm(UserCreationForm):
	class Meta(UserCreationForm.Meta):
		model = CustomUser
		fields = ('username', 'password', 'first_name', 'last_name', 'email')


class CustomUserChangeForm(UserChangeForm):
	class Meta:
		model = CustomUser
		fields = ('first_name', 'last_name', 'email')

class RawImageUploadForm(forms.Form):
	raw_image = forms.ImageField()

class RetrainModelForm(forms.ModelForm):
	class Meta:
		model = MLModelVersion
		fields = ('threshold', 'k_neighbors')
