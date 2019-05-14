from faces.models import CustomUser
CustomUser.objects.create_user('sean', 'email@email.com', 'password')
CustomUser.objects.create_user('user', 'user@email.com', 'password')
