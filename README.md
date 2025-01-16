Web server for FaceCheck IoT project

by Team kawAI
Winner - 1st Place, Nokia IoT Cup on May 17, 2019 at UP Alumni Engineers Centennial Hall

(NOTE: The final version deployed in AWS is in the amazon-complete branch)

The Facial Recognition engine of FaceCheck is built taking from the following repo:
https://github.com/ZhaoJ9014/face.evoLVe.PyTorch

Specifically, we took his:
1. Pretrained MTCNN and InceptionResnet for generating facial embeddings
2. MTCNN code for facial detection
3. Auxiliary code for face alignment and visualization

Our strategy is to use the pretrained neural networks to generate 512-dimensional
embedding vectors and use the k-nearest neighbors machine learning model to
distinguish embedding vectors for registered individuals.

Our recommendation is to upload at least 3 sample photos per registered individual.
When retraining the kNN model, make sure to use an odd k (1,3,5) and that most individuals
must have at least k sample photos uploaded. Obviously, registered individuals with
no sample photos will not be included when retraining the model.

Download the IRnet (166MB) from the link below then place it in media/models/backbone_cnn.pth
https://s3-ap-southeast-1.amazonaws.com/cs145facecheck/media/models/backbone_cnn.pth


