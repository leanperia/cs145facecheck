from django.urls import path
from . import views
from .import apps
urlpatterns = [
    path('', views.HomePageView.as_view(), name='home'),
    path('retrain', views.RetrainMLmodel, name='retrain_model'),
    path('persons', views.ListPersons.as_view(), name='list_persons'),
    path('persons/<int:pk>', views.ViewPerson.as_view(), name='view_person'),
    path('persons/add', views.AddPerson.as_view(), name='add_person'),
    path('samplephotos/add/<int:pk>', views.AddSamplePhoto.as_view(), name='add_samplephoto'),
    path('samplephotos/edit/<int:pk>', views.EditSamplePhotos.as_view(), name='edit_samplephotos'),
    path('samplephotos/delete/<int:pk>', views.DeleteSamplePhotos, name='delete_samplephotos'),
    path('agents', views.ListAgents.as_view(), name='list_agents'),
    path('agents/add', views.AddAgent.as_view(), name='add_agent'),
    path('inferences/history', views.ListInferences.as_view(), name='list_inferences'),
    path('inferences/test', views.RequestInference.as_view(), name='make_inference'),
    path('inferences/test/error<int:pk>', views.RetryInference.as_view(), name='retry_inference'),
    path('inferences/execute', views.RunInference, name='run_inference'),
    path('inferences/result/<int:pk>', views.ViewInference.as_view(), name='view_inference'),
    path('inferences/result/<int:pk>/correction', views.EditInference, name='correct_inference'),
    path('models/', views.ListModels.as_view(), name='list_models'),
    path('models/switch', views.SwitchModel, name='switch_model'),
    path('request-inference',views.requestInference, name='request_inference'),

    path('test_path/', views.exampleJSON),
    path('make_people/', views.AddPeopleREST),
    path('add_photo/<slug:person_name>/', views.AddPhotoREST),
    path('delete_photo/<slug:person_name>/', views.DeletePhotoREST)

]
