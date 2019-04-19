from django.urls import path
from . import views

urlpatterns = [
    path('', views.HomePageView.as_view(), name='home'),
    path('persons', views.ListPersons.as_view(), name='list_persons'),
    path('persons/<int:pk>', views.ViewPerson.as_view(), name='view_person'),
    path('persons/add', views.AddPerson.as_view(), name='add_person'),
    path('samplephotos/add/<int:pk>', views.AddSamplePhoto.as_view(), name='add_samplephoto'),
    path('samplephotos/edit/<int:pk>', views.EditSamplePhotos.as_view(), name='edit_samplephotos'),
    path('samplephotos/delete/<int:pk>', views.DeleteSamplePhotos, name='delete_samplephotos'),
    path('agents', views.ListAgents.as_view(), name='list_agents'),
    path('agents/add', views.AddAgent.as_view(), name='add_agent'),
    path('inferences/history', views.ListInferences.as_view(), name='list_inferences'),
    path('inferences/test', views.MakeInference.as_view(), name='make_inference'),
]
