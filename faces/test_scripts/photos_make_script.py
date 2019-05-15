import os
dirs = os.listdir(".")
print ([name for name in dirs if os.path.isdir(name)])

program = "BS Computer Science"
dept = "Department of Computer Science"

# sort alphabetically
# increasing SN
# BS Computer Science
# Department of Computer Science

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
