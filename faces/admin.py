from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .forms import CustomUserCreationForm, CustomUserChangeForm
from .models import CustomUser, RegisteredPerson, SamplePhoto, InferenceRequest, EndAgent, MLModelVersion

class CustomUserAdmin(UserAdmin):
	add_form = CustomUserCreationForm
	form = CustomUserChangeForm
	list_display = ['username', 'first_name', 'last_name', 'email']
	model = CustomUser

admin.site.register(CustomUser)
admin.site.register(RegisteredPerson)
admin.site.register(SamplePhoto)
admin.site.register(InferenceRequest)
admin.site.register(EndAgent)
admin.site.register(MLModelVersion)
