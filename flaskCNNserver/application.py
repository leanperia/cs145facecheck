# torch
#  
import torch

import numpy as np
import torchvision.transforms as transforms
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
from urllib.request import urlopen

from get_nets import PNet, RNet, ONet
from model_irse import IR_50
from detector import detect_faces
from align_trans import warp_and_crop_face

application = Flask(__name__)

#variables
cnn = None
pnet = None
rnet = None
onet = None

# print a nice greeting.
def say_hello(username = "World"):
    return '<p>Hello %s!</p>\n' % username

# some bits of text for the page.
header_text = '''
    <html>\n<head> <title>EB Flask Test</title> </head>\n<body>'''
instructions = '''
    <p><em>Hint</em>: This is a RESTful web service! Append a username
    to the URL (for example: <code>/Thelonious</code>) to say hello to
    someone specific.</p>\n'''
home_link = '<p><a href="/">Back</a></p>\n'
footer_text = '</body>\n</html>'

application.add_url_rule('/', 'index', (lambda: header_text +
    say_hello() + instructions + footer_text))
application.add_url_rule('/<username>', 'hello', (lambda username:
    header_text + say_hello(username) + home_link + footer_text))

def load_models():
    global cnn, pnet, rnet, onet
    print('loading DL models')
    cnn = IR_50([112,112])

    #newf = urlopen('https://s3-ap-southeast-1.amazonaws.com/cs145facecheck/media/models/backbone_cnn.pth')
    #cnn.load_state_dict(torch.load(BytesIO(newf.read()), map_location='cpu'))
    #cnn.load_state_dict(torch.load('backbone_cnn.pth'))
    cnn.eval()
    pnet = PNet('')
    rnet = RNet('')
    onet = ONet('') 
    pnet.eval()
    rnet.eval()
    onet.eval()
    print('finished loading.')

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])])

def generate_embedding(img, cnn, pnet, rnet, onet):
    _, landmarks = detect_faces(img, pnet, rnet, onet)
    facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
    warped_face = warp_and_crop_face(src_img=np.array(img), facial_pts=facial5points, crop_size=(112,112), align_type='similarity')
    imgtensor = transform(warped_face)
    imgtensor.unsqueeze_(0)
    vector = l2_norm(cnn(imgtensor))
    return vector.detach().numpy()

# application route
# 'generate-embedding' rule is abound to the predict() function
# if you visit http://localhost:5000/generate-embedding,
# the output of the predict() function will be rendered in the
# browser
@application.route("/generate-embedding", methods=["POST"])
def predict():
    data = {"success": False}
    if not cnn and not pnet and not rnet and not onet:
        load_models()
    if request.method == "POST":
        if request.files.get("image"):
            print('checking the retrieved image')
            print("type(request.files['image']) = ", type(request.files['image']))
            image = request.files["image"].read()
            print('type of image: ', type(image))
            image = Image.open(BytesIO(image))
            print('type of processed image: ', type(image))
            print(type(cnn))
            vector = generate_embedding(image, cnn, pnet, rnet, onet)
            data["embedding"] = vector.tolist()
            """
            use nparray.tolist()
            reference: https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable

            To unjsonify:
            obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
            b_new = json.loads(obj_text)
            a_new = np.array(b_new)
            """
            data["success"] = True
    return jsonify(data)

if __name__ == '__main__':
    application.run(debug=True)
    # standard request format:
    # curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
