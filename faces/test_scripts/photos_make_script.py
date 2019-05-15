import os
dirs = os.listdir(".")
print ([name for name in dirs if os.path.isdir(name)])

# sort alphabetically
# increasing SN
# BS Computer Science
# Department of Computer Science

# InfCorrection
# curl -XPOST -F 'correction=FN' \
# -F  'inference_pk=39' \
#  -F 'key=Fs9gX@a8pzTl$20m' \
#   -F 'sn=v93nagsd09132nas' \
#  'https://cs145facecheck.com/rest/inference-correction' > error.html
#
# # addPerson
# curl -XPOST -H "Content-type: application/json" -d '
#   {
#     "first_name": "Ryan",
#     "last_name": "CURL1",
#     "is_enrstudent": true
#   }
# ' 'https://cs145facecheck.com/rest/add-person'
#
# # addPhoto
# curl -XPOST -F 'image=@FILENAME.jpg' \
# -F  'first_name=!!!first_name' \
# 'https://cs145facecheck.com/rest/add-photo'
