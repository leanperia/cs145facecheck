# USE LITERALS

# InfCorrection
curl -XPOST -F 'correction=FN' \
-F  'inference_pk=39' \
 -F 'key=Fs9gX@a8pzTl$20m' \
  -F 'sn=v93nagsd09132nas' \
 'https://cs145facecheck.com/rest/inference-correction' > error.html

# addPerson
curl -XPOST -H "Content-type: application/json" -d '
  {
    "first_name": "Sean",
    "last_name": "Chan",
    "is_enrstudent": true
  }
' 'https://cs145facecheck.com/rest/add-person'

curl -XPOST -H "Content-type: application/json" -d '
  {
    "first_name": "Sean",
    "last_name": "Chan",
    "is_enrstudent": true
  }
' 'localhost:8000/rest/add-person'


# addPhoto
curl -XPOST -F 'image=@49261918_10215945516643331_1396781707494948864_o.jpg' \
-F  'first_name=Sean' \
'https://cs145facecheck.com/rest/add-photo'

curl -XPOST -F 'image=@49261918_10215945516643331_1396781707494948864_o.jpg' \
-F  'first_name=Sean' \
'localhost:8000/rest/add-photo' > error.html


# request inference
 curl 'https://cs145facecheck.com/rest/request-inference' -X POST \
   -F 'sn=v93nagsd09132nas' -F 'key=Fs9gX@a8pzTl$20m' -F 'image=@lean_test.jpg'
