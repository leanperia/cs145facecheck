# USE LITERALS

curl -XPOST -F 'correction=FN' \
-F  'inference_pk=39' \
 -F 'key=Fs9gX@a8pzTl$20m' \
  -F 'sn=v93nagsd09132nas' \
 'https://cs145facecheck.com/rest/inference-correction' > error.html

 curl 'https://cs145facecheck.com/rest/request-inference' -X POST \
   -F 'sn=v93nagsd09132nas' -F 'key=Fs9gX@a8pzTl$20m' -F 'image=@lean_test.jpg'
