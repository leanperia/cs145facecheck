0.) download: https://s3-ap-southeast-1.amazonaws.com/cs145facecheck/media/bitnami-lappstack-7.1.24-0-linux-x64-installer.run
1.) install using bitnami-lappstack-7.1.24-0-linux-x64-installer.run
2.) take note of superuser password! you will need it
3.) enter into the directory of lappstack
4.) ./use_lappstack
5.) psql -U postgres (use password you gave at installation time)
6.) create database facecheck
7.) create user leanperia with password 'sakura123'
8.) enter into facecheck-amazon-complete directory
9.) python manage.py makemigrations and migrate
10.) runserver
