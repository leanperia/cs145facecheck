# How to setup to use facecheck on a local machine

0. download: https://s3-ap-southeast-1.amazonaws.com/cs145facecheck/media/bitnami-lappstack-7.1.24-0-linux-x64-installer.run
-    download the backbone_cnn.pth https://s3-ap-southeast-1.amazonaws.com/cs145facecheck/media/models/backbone_cnn.pth
1. install using bitnami-lappstack-7.1.24-0-linux-x64-installer.run
2. take note of superuser password! you will need it
3. enter into the directory of lappstack
4. ./use_lappstack
5. psql -U postgres (use the password you gave at installation time)
6. create database facecheck;
7. create user leanperia with password 'sakura123';
8. open a new terminal. 
-   (on any chosen directory outside of facecheck-amazon-complete directory)
-   virtualenv djangoenv
-   source djangoenv/bin/activate
-   then enter into facecheck-amazon-complete directory
-   pip install -r requirements.txt
9. rename facecheck/settings.py to settings_deployed.py and rename settings_local.py to settings.py
-   change faces/views.py LOCAL_BACKBONE=False to True
-   put the backbone_cnn.pht file into media/models
10. series of steps: 
- python manage.py makemigrations faces
- python manage.py migrate faces
- python manage.py migrate
- python manage.py createsuperuser tester1 (any password - don't forget it)
11. (extra steps so the database's assumptions will be followed):
- python manage.py shell
- (inside shell)
- from faces.models import MLModelVersion as ml
- from django.utils import timezone as t
- x = ml.objects.create(is_in_use=True, time_trained=t.now())
- x.save()
- quit()
13. you can now use runserver. you will have to populate the local postgresql databse in your machine but all images will be uploaded the S3 bucket. before performing inferences locally, you need to create two end agents in the browser interface. the second  one is the 'web agent' (i will change this later so the first one is the web agent)

# Web server for FaceCheck IoT project

by Team kawAI


The Facial Recognition engine of FaceCheck is built taking from the following repo:
https://github.com/ZhaoJ9014/face.evoLVe.PyTorch

Specifically, we took his:
1. Pretrained MTCNN and InceptionResnet for generating facial embeddings
2. MTCNN code for facial detection
3. Auxiliary code for face alignment and visualization

Our strategy is to use the pretrained neural networks to generate 512-dimensional
embedding vectors and use the k-nearest neighbors machine learning model to
classify embedding vectors for registered individuals. We use an adjustable threshold 
L2 norm distance between the target embedding vector and the nearest embedding vector in 
the database to determine target embedding vector really describes someone in the database
or the person in the target image is an unknown person. This threshold value will need to be tuned.

Our recommendation is to upload at least 3 sample photos per registered individual.
When retraining the kNN model, make sure to use an odd k (1,3,5) and that most individuals
must have at least k sample photos uploaded. Obviously, registered individuals with
no sample photos will not be included when retraining the model.

Download the IRnet (166MB) from the link below then place it in media/models/backbone_cnn.pth
https://s3-ap-southeast-1.amazonaws.com/cs145facecheck/media/models/backbone_cnn.pth
