import requests
from PIL import Image

username = 'tester1'
password = 'sakura123'
hostname = 'https://cs145facecheck.com'

#response = requests.post(hostname+'get-auth-token/', data={'username':username, 'password':password},#
#    headers = {'Accept': 'application/json'})
#token = response.json()['token']

print('uploading image')
img = open('sakura01.jpg', 'rb')
response = requests.post(hostname+'rest/sample_photos/', files={'image':img}, data={'person_pk':1},
    headers={'Accept':'application/json'})#, 'Authorization':f'Token {token}'})
print(f"status = {response.status_code}")
print(response.content)

# but this works: curl localhost:8000/rest/sample_photos/ -X POST
#                 -F image=@sakura01.jpg -F person_pk=1 -H 'Accept: application/json'



response = requests.post(hostname+'rest/registered_persons/',
    data = {
        'first_name': 'Eunbi',
        'last_name': 'Kwon',
        'is_enrstudent': True,
        'is_faculty': False,
        'studentnum': '201710101'
    },
    headers={'Accept':'application/json'}    #,'Authorization':f'Token {token}'}
)
print(response) # HTTP 201 means a new resource has been successfully created
f = open('output.html', 'wb')
f.write(response.content)
f.close()
