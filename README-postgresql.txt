0.) download: https://s3-ap-southeast-1.amazonaws.com/cs145facecheck/media/bitnami-lappstack-7.1.24-0-linux-x64-installer.run
1.) install using bitnami-lappstack-7.1.24-0-linux-x64-installer.run
2.) take note of superuser password! you will need it
3.) enter into the directory of lappstack
4.) ./use_lappstack
5.) psql -U postgres (use the password you gave at installation time)
6.) create database facecheck;
7.) create user leanperia with password 'sakura123';
8.) open a new terminal. use a virtualenv with requirements installed already. then enter into facecheck-amazon-complete directory
9.) series of steps: 
  python manage.py makemigrations faces
  python manage.py migrate faces
  python manage.py migrate
  python manage.py createsuperuser tester1 (any password - don't forget it)
10.) (extra steps so the database's assumptions will be followed):
  python manage.py shell
  (inside shell)
  from faces.models import MLModelVersion as ml
  from django.utils import timezone as t
  x = ml.objects.create(is_in_use=True, time_trained=t.now())
  x.save()
  quit()
11.) create two end agents. the second  one is the 'web agent' (i will change this later so the first one is the web agent)
