import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from PIL import Image
from facedetect.detector import detect_faces
from facedetect.visualization_utils import show_results
from facedetect.align_trans import warp_and_crop_face

import torch


def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def detect_and_recognize(pil_img, classifier, cnn, transform, pnet, rnet, onet):
    bounding_boxes, landmarks = detect_faces(pil_img, pnet, rnet, onet)
    extra_faces = False
    if landmarks == []:
        return None, None, None
    elif landmarks.shape[0] > 1:
        extra_faces = True

    resultimg = show_results(pil_img, bounding_boxes, landmarks)
    facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
    warped_face = warp_and_crop_face(src_img=np.array(pil_img), facial_pts=facial5points, crop_size=(112,112), align_type='similarity')
    imgtensor = transform(warped_face)
    imgtensor.unsqueeze_(0)
    vector = l2_norm(cnn(imgtensor))
    prediction = classifier.predict(vector.detach().numpy())
    return int(prediction[0]), resultimg, extra_faces

def generate_knn_model(path, n_neighbors):
    X, y = [], []
    for filename in os.listdir(path):
        X.append(generate_embedding(os.path.join(path, filename)))
        y.append(int(filename[6:filename.index('_')]))
    X = np.array(X)
    X = np.squeeze(X, axis=1)
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X,y)
    return clf
